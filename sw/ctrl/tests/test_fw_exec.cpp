// test_fw_exec.cpp — ctests fw_exec_host, fw_exec_rv and fw_exec_perf (P2.4): the graph executor
// (sw/ctrl/kernels/graph.cpp) against docs/design/06-software-verification.md "SW" §7.2, §7.3, §7.8,
// §4.7 and §10.2.
//
// One driver, two builds of the executor:
//   host  (TSR_EXEC_TEST_HOST, target test_fw_exec_host): graph.cpp compiled for the host
//         (tsr_fw_host, SW §6.8) with the test kinds of sw/ctrl/tests/fw/exec_kinds.cpp, run by
//         tsr::host_run on a simulator (no RISC-V; the driver adds the runtime's R7 / R9);
//   rv    (target test_fw_exec_rv): the test image fw_exec (production sources + exec_kinds.cpp,
//         tsr_fw_image) booted through the host CSR port and launched through the LQ / LCQ like
//         the transport will (EX §3.3, §4.8) on tsr_rv: the reference (SW §6.8).
// Both run the same scenarios with the same checks (kinds 0x30-0x3F are exec_kinds.cpp's):
//   codes      every return code 0x8000_0001-7 (and the layout / record / dep variants that map to
//              them), handler failures, a list rejected after a tile error (the tile error wins)
//   chains     copies through DRAM with RAW and WAR dependencies, JOIN chains for > 2 predecessors
//   deps       the deps the executor pushes: after(ticket[dep0], ticket[dep1]) (JOIN NOPs, a COPY),
//              NOP records (ticket none, no pushes)
//   sizes      n_ops 1 and 4096 (the 4096-op list is 4097 lines: streamed)
//   stream     a 6.6k-line list with records crossing chunk boundaries, deps across chunks, with
//              and without VERIFY_CRC; an injected tile error in chunk 2; a chunk load that fails;
//              a 700-record NOP list (5.5 chunks) in which only the executor's joins order the
//              RISC-V's reads of each chunk after its load (the race detector checks that); these
//              launches and the 4096-op list are captured and their trace digests (push records,
//              DONE, expect regions; fw_trace_digest.h) must equal the W2a image's (W2a's push order)
//   errors     the failing-op index for injected tile errors at positions 0, 1, 15, 16, 17, 40,
//              63 (NOP neighbours: empty ops share a push index), DEP_FAILED behind the error,
//              failed executor loads (list, preload) -> RET 0
//   stop       STOP_ON_ERROR: RET and the error record are schedule-independent; the pushes stop
//   preload    four entries (4 / 100 / 64 / 300 bytes), one join; read by the RISC-V and by STs
//   flags      MARK (NODE_BEGIN markers in order, before each op's pushes), the PROF launch flag,
//              REC on the LD / ST pushes, SERIAL (no op starts before the previous one is done)
//   plan       plan.hpp on the RISC-V (PLAN kind) = plan.hpp on the host = tsr_ref geometry, for the
//              Qwen2.5-0.5B / 32B and Gemma 4 12B MUL_MAT shapes and edge shapes
//   mulmat     MUL_MAT records through the real kind_mulmat (a 0.5B-like layer), RET 0; a 48-layer
//              decode list (T = 1: the executor's fast path) that wraps the scratch ring and fills
//              its table (exec_state diagnostics checked)
// fw_exec_rv (--full) runs the deterministic part under the fixed and 7 random schedules with
// TSR_SIM_RACE=fatal and compares SW §4.7's observables (op streams, RISC-V SPM stores, final
// written DRAM / SPM) and every DONE record r0-r6, and asserts PERF PUSH_STALL = 0 (no push waited
// for a credit, CC §7.3) and that a list rejected before step 2 reports no ring counters of the
// previous launch; the STOP_ON_ERROR part compares the DONE
// records; a race-detector probe (a missing dep inside a test kind, an unwritten preload line)
// shows the detector sees the executor's pushes and the RISC-V's reads. fw_exec_perf
// measures RISC-V cycles per graph op (SW §7.8 target <= 400) on 0.5B-like lists.
//
// Usage: test_fw_exec_rv <image.elf> [--full|--quick|--perf]   test_fw_exec_host
//   (no option: 8 schedules without the big lists, which run under the fixed one; --quick: 2)
#undef _GNU_SOURCE
#include "test_sim_obs.h"   // test_sim_util.h, test_util.h (CHECK), tsr_sim_int.h, tsr_bus.h

#include "tsr_race.h"
#include "tsr_ref.h"
#include "tsr_rv.h"

#include <array>
#include <string>
#include <vector>

#include "fw_prof.h"
#include "fw_trace_digest.h"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_trace.h"

#ifdef TSR_EXEC_TEST_HOST
#include "tsr/exec.hpp"
#include "tsr/kernel.hpp"
#include "tsr/rt.hpp"
extern "C" int32_t graph(const tsr_args * args);
#endif

#ifdef TSR_EXEC_HAVE_LOWER
#include "tsr-lower.h"     // the reference tiling (sw/ggml-tesseract): the per-token line counts
#endif

using tsr::opl::writer;
namespace opl = tsr::opl;
namespace plan = tsr::plan;

// ---------------------------------------------------------------------------------------------
// Configuration and DRAM plan (beats; ch0 = list / IO, ch1 = sources)
// ---------------------------------------------------------------------------------------------
static const uint32_t NCH = 2;
static const uint64_t CH_BYTES = 1ull << 30;              // 2^24 beats per channel
static const uint32_t LIST_BEAT = 0x10000;                 // ch0, OPL region: up to 4 MiB lists
static const uint32_t PRE_BEAT = 0x30000;                  // ch0: preload sources
static const uint32_t PLAN_BEAT = 0x38000;                 // ch0: PLAN tables / results
static const uint32_t OUT_BEAT = 0x100000;                 // ch0 IO: outputs
static const uint32_t SRC_BEAT = 0x100000;                 // ch1: sources
static const uint32_t W_BEAT = (uint32_t)(TSR_REGION_WEIGHT_OFF / 64u);   // every channel: weights

static const uint32_t TK_COPY = 0x30, TK_STUB5 = 0x31, TK_FAIL = 0x32, TK_PRE = 0x33, TK_PREST = 0x34,
                      TK_RET = 0x35, TK_PLAN = 0x36, TK_RACE = 0x37;
static const uint32_t FAIL_BEAT = 0x0FFFFFC0u;            // exec_kinds.cpp FAIL: LD from ch0 here

static inline uint32_t h0(uint32_t beat) { return beat; }                  // ch0 handle
static inline uint32_t h1(uint32_t beat) { return (1u << 28) | beat; }     // ch1 handle

static tsr_config test_cfg() {
    tsr_config c = TSR_CONFIG_U200;
    c.num_ch = NCH;
    c.ch_bytes = CH_BYTES;
    c.t_max = 8;
    c.caps = TSR_CAPS_VU | TSR_CAPS_ATT;       // the M6 CAPS (as fw_m6_e2e; W2b: VU test kinds and M7 kinds)
    return c;
}
static plan::dev_cfg dev_of(const tsr_config & c) {
    return plan::dev_cfg{c.spm_bytes / 64u, c.t_max, c.k_max};
}

// ---------------------------------------------------------------------------------------------
// Launch environments
// ---------------------------------------------------------------------------------------------
struct done_t {
    uint32_t r[8];
    uint32_t status() const { return TSR_GET(r[0], TSR_LCQ_R0_STATUS); }
    uint32_t err() const { return TSR_GET(r[0], TSR_LCQ_R0_ERR_CODE); }
    uint32_t ret() const { return r[2]; }
};

struct env {
    tsr_sim * sim = nullptr;
    std::vector<std::array<uint32_t, 7>> dones;   // every DONE r0-r6 (determinism)
    std::string tag;
    virtual ~env() {}
    virtual done_t launch(uint32_t a0, uint32_t a1, uint32_t a2, bool prof) = 0;
    virtual bool rv() const = 0;
    // Executor state after the last launch (the DTCM tables / exec_state), when readable.
    virtual bool exec_state(opl::exec_state * st) = 0;
    // The executor's DTCM tables after the last launch: ticket[i] and pidx[i] of ops 0 .. n-1.
    virtual bool exec_tables(uint32_t * ticket, uint32_t * pidx, uint32_t n) = 0;
    done_t run(uint32_t a0, uint32_t a1, uint32_t a2 = 0, bool prof = false) {
        done_t d = launch(a0, a1, a2, prof);
        std::array<uint32_t, 7> x;
        for (int i = 0; i < 7; ++i) x[i] = d.r[i];
        dones.push_back(x);
        return d;
    }
};

#ifdef TSR_EXEC_TEST_HOST
struct host_env : env {
    uint32_t seq = 0;
    explicit host_env(tsr_sim * s) {
        sim = s;
        tsr::host_bind(s);
    }
    ~host_env() override { tsr::host_bind(nullptr); }
    bool rv() const override { return false; }
    done_t launch(uint32_t a0, uint32_t a1, uint32_t a2, bool prof) override {
        tsr_args args{};
        args.a[0] = a0;
        args.a[1] = a1;
        args.a[2] = a2;
        args.seq = seq;
        args.flags = prof ? TSR_ARGS_FLAG_PROF : 0u;
        const tsr::host_result h = tsr::host_run(&graph, args, nullptr, 0u);
        // R7 (EX §4.4 precedence; no PLAT_ERR / STOPPED here) and R9.
        uint32_t st = TSR_LCQ_R0_STATUS_OK;
        if (h.err_code == TSR_ERR_ABORTED) st = TSR_LCQ_R0_STATUS_ABORTED;
        else if (h.err_code != 0u) st = TSR_LCQ_R0_STATUS_TILE_ERR;
        else if (h.ret < 0) st = TSR_LCQ_R0_STATUS_KERNEL_ERR;
        done_t d{};
        d.r[0] = TSR_LCQ_R0_TYPE_DONE | st << 8 | (h.err_code & 0xFFu) << 16 | (seq & 0xFFu) << 24;
        d.r[1] = seq;
        d.r[2] = (uint32_t)h.ret;
        d.r[3] = h.err_code ? h.err_ticket : TSR_TICKET_NONE;
        d.r[4] = h.err_code ? h.err_info : 0u;
        d.r[5] = h.err_code ? h.err_w0 : 0u;
        d.r[6] = h.err_code ? h.err_pidx : 0u;
        if (tsr::err_units() != 0u) tsr::join(tsr::qclear());
        ++seq;
        return d;
    }
    bool exec_state(opl::exec_state * st) override {
        *st = __tsr_exec_state;
        return true;
    }
    bool exec_tables(uint32_t * ticket, uint32_t * pidx, uint32_t n) override {
        memcpy(ticket, __tsr_exec_ticket, 4u * n);
        memcpy(pidx, __tsr_exec_pidx, 4u * n);
        return true;
    }
};
#endif

// The RISC-V image, booted and driven through the host CSR port (EX §3.3, §4.8).
struct rv_env : env {
    uint32_t seq = 0, lcq_rd = 0, log_rd = 0, kidx = 0;
    uint32_t sym_state = 0, sym_ticket = 0, sym_pidx = 0;
    bool ok = false;
    std::vector<done_t> recs;

    static uint32_t crc32z(const uint8_t * p, size_t n) {
        uint32_t c = 0xFFFFFFFFu;
        for (size_t i = 0; i < n; ++i) {
            c ^= p[i];
            for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (TSR_CRC32_POLY & (0u - (c & 1u)));
        }
        return ~c;
    }
    bool boot(const char * path) {
        tsr_rv_elf elf;
        if (tsr_rv_elf_read(path, &elf) != 0) {
            CHECK(0, "%s: cannot read %s", tag.c_str(), path);
            return false;
        }
        for (uint32_t s = 0; s < elf.nseg; ++s) {
            const tsr_rv_elf_seg * g = &elf.seg[s];
            if (g->filesz == 0) continue;
            const uint32_t bytes = (g->filesz + 3u) & ~3u;
            std::vector<uint8_t> buf(bytes, 0);
            memcpy(buf.data(), g->data, g->filesz);
            tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
            for (uint32_t i = 0; i < bytes; i += 4) tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, get_u32_le(&buf[i]));
            CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == crc32z(buf.data(), bytes), "%s: load CRC", tag.c_str());
        }
        // The graph kernel's index: its entry in .tsr_kernels (EX §3.1; entry size from the table
        // and __tsr_n_kernels: SW §6.2's fields, 36 bytes, rt.hpp).
        uint32_t ks = 0, ke = 0, nk = 0;
        bool found = false;
        if (tsr_rv_elf_symbol(&elf, "__tsr_kernels_start", &ks) == 0 && tsr_rv_elf_symbol(&elf, "__tsr_kernels_end", &ke) == 0 &&
            tsr_rv_elf_symbol(&elf, "__tsr_n_kernels", &nk) == 0 && nk != 0 && ke > ks) {
            const uint32_t es = (ke - ks) / nk;
            for (uint32_t s = 0; s < elf.nseg && !found; ++s) {
                const tsr_rv_elf_seg * g = &elf.seg[s];
                if (ks < g->paddr || ke > g->paddr + g->filesz) continue;
                for (uint32_t a = ks, i = 0; a + es <= ke; a += es, ++i) {
                    const char * name = (const char *)g->data + (a - g->paddr) + 12;
                    if (strncmp(name, "graph", 24) == 0) {
                        kidx = i;
                        found = true;
                        break;
                    }
                }
            }
        }
        CHECK(found, "%s: kernel 'graph' in the image's .tsr_kernels", tag.c_str());
        tsr_rv_elf_symbol(&elf, "__tsr_exec_state", &sym_state);
        tsr_rv_elf_symbol(&elf, "__tsr_exec_ticket", &sym_ticket);
        tsr_rv_elf_symbol(&elf, "__tsr_exec_pidx", &sym_pidx);
        tsr_rv_elf_free(&elf);
        if (!found) return false;
        tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, TSR_LOG_HDR_LEVEL_INFO));
        tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0u);
        tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
        // HELLO (EX §4.7 R1).
        for (int it = 0; it < 2000 && recs.empty(); ++it) {
            tsr_sim_advance(sim, 100000);
            if (!consume()) return false;
        }
        ok = !recs.empty() && TSR_GET(recs[0].r[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_HELLO;
        CHECK(ok, "%s: HELLO after boot", tag.c_str());
        recs.clear();
        return ok;
    }
    bool consume() {
        const uint32_t rt = tsr_sim_csr_read(sim, TSR_CSR_RT_STATE);
        if (TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) {
            CHECK(0, "%s: the firmware trapped (RT_INFO 0x%08x, pc 0x%08x)", tag.c_str(),
                  tsr_sim_csr_read(sim, TSR_CSR_RT_INFO), tsr_sim_rv_pc(sim));
            return false;
        }
        const uint32_t wr = tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR);
        for (uint32_t k = lcq_rd; k != wr; ++k) {
            done_t d;
            for (uint32_t j = 0; j < 8u; ++j) d.r[j] = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, j));
            recs.push_back(d);
        }
        lcq_rd = wr;
        tsr_sim_csr_write(sim, TSR_CSR_LCQ_RD, wr);
        const uint32_t lwr = tsr_sim_csr_read(sim, TSR_CSR_LOG_WR);
        log_rd = lwr;                                          // log content is never compared (SW §4.7)
        tsr_sim_csr_write(sim, TSR_CSR_LOG_RD, lwr);
        return true;
    }
    bool rv() const override { return true; }
    done_t launch(uint32_t a0, uint32_t a1, uint32_t a2, bool prof) override {
        done_t res{};
        if (!ok) return res;
        uint32_t d[8] = {TSR_DESC_TYPE_LAUNCH | (prof ? TSR_DESC_PROF : 0u) | TSR_PUT(TSR_DESC_KERNEL, kidx), a0, a1, a2,
                         0, 0, 0, 0};
        for (uint32_t k = 0; k < 8u; ++k) tsr_sim_csr_write(sim, TSR_CSR_LQ_W(k), d[k]);
        tsr_sim_csr_write(sim, TSR_CSR_LQ_PUSH, 1u);
        const uint32_t s = seq++;
        for (int it = 0; it < 200000; ++it) {
            for (size_t i = 0; i < recs.size(); ++i) {
                const uint32_t ty = TSR_GET(recs[i].r[0], TSR_LCQ_R0_TYPE);
                if ((ty == TSR_LCQ_R0_TYPE_DONE || ty == TSR_LCQ_R0_TYPE_TRAP) && recs[i].r[1] == s) {
                    res = recs[i];
                    recs.erase(recs.begin() + (long)i);
                    CHECK(ty == TSR_LCQ_R0_TYPE_DONE, "%s: launch %u answered with a TRAP record", tag.c_str(), s);
                    if (ty != TSR_LCQ_R0_TYPE_DONE) ok = false;
                    return res;
                }
            }
            tsr_sim_advance(sim, 200000);
            if (!consume()) {
                ok = false;
                return res;
            }
        }
        CHECK(0, "%s: launch %u: no DONE record", tag.c_str(), s);
        ok = false;
        return res;
    }
    bool dtcm_read(uint32_t addr, void * dst, size_t n) {
        uint32_t size = 0;
        const uint8_t * dt = tsr_sim_dtcm(sim, &size);
        if (addr < (uint32_t)TSR_RV_DTCM_BASE || addr - (uint32_t)TSR_RV_DTCM_BASE + n > size) return false;
        memcpy(dst, dt + (addr - (uint32_t)TSR_RV_DTCM_BASE), n);
        return true;
    }
    bool exec_tables(uint32_t * ticket, uint32_t * pidx, uint32_t n) override {
        return sym_ticket && sym_pidx && dtcm_read(sym_ticket, ticket, 4u * n) && dtcm_read(sym_pidx, pidx, 4u * n);
    }
    bool exec_state(opl::exec_state * st) override {
        if (sym_state == 0) return false;
        uint32_t n = 0;
        const uint8_t * dt = tsr_sim_dtcm(sim, &n);
        const uint32_t off = sym_state - (uint32_t)TSR_RV_DTCM_BASE;
        if (off + sizeof(opl::exec_state) > n) return false;
        memcpy(st, dt + off, sizeof(*st));
        return true;
    }
};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------
static void dram_put(env & e, uint32_t ch, uint32_t beat, const void * p, size_t n) {
    CHECK(tsr_sim_dram_write(e.sim, ch, (uint64_t)beat * 64u, p, n) == 0, "dram write ch%u beat 0x%x", ch, beat);
}
static std::vector<uint8_t> dram_get(env & e, uint32_t ch, uint32_t beat, size_t n) {
    std::vector<uint8_t> v(n);
    tsr_sim_dram_read(e.sim, ch, (uint64_t)beat * 64u, v.data(), n);
    return v;
}
static std::vector<uint8_t> rand_bytes(uint64_t seed, size_t n) {
    rng_t r = {seed};
    std::vector<uint8_t> v(n);
    rng_bytes(&r, v.data(), n);
    return v;
}
static void fill(env & e, uint32_t ch, uint32_t beat, size_t n, uint8_t b) {
    std::vector<uint8_t> v(n, b);
    dram_put(e, ch, beat, v.data(), n);
}
// Puts the list at ch0 beat `beat` and launches graph on it.
static done_t run_list(env & e, const std::vector<uint8_t> & blob, uint32_t a2 = 0, bool prof = false,
                       uint32_t beat = LIST_BEAT, uint32_t a1 = 0) {
    dram_put(e, 0, beat, blob.data(), blob.size());
    return e.run(h0(beat), a1 ? a1 : (uint32_t)blob.size(), a2, prof);
}
static uint32_t rd32le(const std::vector<uint8_t> & b, size_t off) { return get_u32_le(&b[off]); }
static void wr32le(std::vector<uint8_t> & b, size_t off, uint32_t v) { writer::wr32(&b[off], v); }

static bool expect_ok(env & e, const done_t & d, const char * what) {
    const bool ok = d.status() == TSR_LCQ_R0_STATUS_OK && d.ret() == 0u && d.err() == 0u && d.r[3] == TSR_TICKET_NONE;
    CHECK(ok, "%s: %s: DONE OK expected: status %u ret 0x%08x err %u", e.tag.c_str(), what, d.status(), d.ret(), d.err());
    return ok;
}
static void expect_kerr(env & e, const done_t & d, uint32_t code, const char * what) {
    CHECK(d.status() == TSR_LCQ_R0_STATUS_KERNEL_ERR && d.ret() == code && d.err() == 0u,
          "%s: %s: KERNEL_ERR 0x%08x expected: status %u ret 0x%08x err %u", e.tag.c_str(), what, code, d.status(),
          d.ret(), d.err());
}
// TILE_ERR whose first error is exec_kinds.cpp's FAIL LD (DRAM_RANGE at FAIL_BEAT).
static void expect_fail(env & e, const done_t & d, uint32_t ret, const char * what) {
    CHECK(d.status() == TSR_LCQ_R0_STATUS_TILE_ERR && d.err() == TSR_ERR_DRAM_RANGE && d.ret() == ret &&
              (d.r[3] >> 28) == TSR_UNIT_DMA && d.r[4] == FAIL_BEAT && d.r[5] == tsr_w0(TSR_OP_LD, 0, 0, 0),
          "%s: %s: TILE_ERR (FAIL LD) with RET %u expected: status %u ret 0x%08x err %u ticket 0x%08x info 0x%08x w0 0x%08x",
          e.tag.c_str(), what, ret, d.status(), d.ret(), d.err(), d.r[3], d.r[4], d.r[5]);
}
// The executor's push stream pinned to W2a's (SW §7.3; executor optimisation review, 2026-10-30):
// the launch between pin_begin and pin_end is captured (tsr_sim_capture_launch, RV only: the host
// build's launches do not go through the LQ) and its digest (fw_trace_digest.h: push records,
// DONE, expect regions) compared with the value the W2a image (92f0b17) gives. A deliberate change
// of the pushes changes it: record the decision (SW §7.3) and update the value from the output.
static void pin_begin(env & e) {
    if (e.rv()) tsr_sim_capture_launch(e.sim, 1);
}
static void pin_end(env & e, const char * what, uint64_t w2a) {
    if (!e.rv()) return;
    tsr_sim_capture_launch(e.sim, 0);
    tsr_trace2 t;
    tsr_trace2_init(&t);
    const int rc = tsr_sim_capture_take(e.sim, &t);
    CHECK(rc == 0, "%s: %s: no captured launch trace (%d)", e.tag.c_str(), what, rc);
    if (rc == 0) {
        const uint64_t d = fw_trace_digest(&t);
        CHECK(d == w2a, "%s: %s: launch trace digest %016llx (%zu pushes, DONE ret %d, error ticket / pidx +%u / +%u), "
              "W2a's %016llx: the push stream changed", e.tag.c_str(), what, (unsigned long long)d, t.n_push, t.done.ret,
              t.done.err_ticket_rel, t.done.err_pidx_rel, (unsigned long long)w2a);
        tsr_trace2 extra;
        tsr_trace2_init(&extra);
        CHECK(tsr_sim_capture_take(e.sim, &extra) == 1, "%s: %s: more than one launch captured", e.tag.c_str(), what);
        tsr_trace2_clear(&extra);
    }
    tsr_trace2_clear(&t);
}
// W2a's digests (92f0b17 fw_exec.elf under this harness; see pin_end).
static const uint64_t PIN_SIZES_4096 = 0x67e12f5afa4f9a49ull, PIN_STREAM = 0x6f121c56b38e44dfull,
                      PIN_STREAM_NOP = 0x85145e8fd762e827ull, PIN_STREAM_CRC = 0x0185d6f07b611a15ull,
                      PIN_STREAM_FAIL = 0x80a61be7db450c0full, PIN_STREAM_CHUNK_FAIL = 0x5dcf7a79d9d44147ull;

static uint32_t copy_args(uint32_t * a, uint32_t src, uint32_t dst, uint32_t lines) {
    a[0] = src; a[1] = dst; a[2] = lines;
    return 3;
}
static uint32_t add_copy(writer & w, uint32_t src, uint32_t dst, uint32_t lines, std::initializer_list<uint32_t> deps = {},
                         uint32_t n_slots = 0) {
    uint32_t a[3];
    copy_args(a, src, dst, lines);
    std::vector<uint32_t> d(deps);
    return w.add(TK_COPY, a, 3, d.data(), (uint32_t)d.size(), 0, n_slots);
}
static uint32_t add_fail(writer & w, uint32_t lines = 4, std::initializer_list<uint32_t> deps = {}) {
    const uint32_t a[1] = {lines};
    std::vector<uint32_t> d(deps);
    return w.add(TK_FAIL, a, 1, d.data(), (uint32_t)d.size());
}

// ---------------------------------------------------------------------------------------------
// Listener: markers, REC bits, SERIAL ordering (per launch window)
// ---------------------------------------------------------------------------------------------
struct watch {
    bool on = false;
    std::vector<uint32_t> marks;       // MARK_PUSH values
    uint32_t pushes = 0, rec_pushes = 0, done = 0;
    uint32_t serial_violations = 0;    // a NODE_BEGIN pushed while ops of the launch were outstanding
    uint32_t ops_before_first_mark = 0;
    std::vector<uint32_t> push_mark;   // for each push: index of the last mark (or ~0)
    std::vector<uint32_t> push_w0;
    std::vector<uint32_t> push_w1;
    std::vector<uint8_t> push_rec;
    std::vector<std::array<uint32_t, 4>> push_info;   // pidx, ticket, dep0, dep1
    void reset() { *this = watch{}; on = true; }
};
static watch g_watch;
static void watch_q(void * ctx, tsr_sim * sim, const tsr_q_event * ev) {
    (void)ctx; (void)sim;
    watch & w = g_watch;
    if (!w.on) return;
    if (ev->kind == TSR_QEV_MARK_PUSH) {
        if (w.pushes != w.done) w.serial_violations++;
        w.marks.push_back(ev->w[0]);
    } else if (ev->kind == TSR_QEV_PUSH) {
        w.pushes++;
        if (ev->rec) w.rec_pushes++;
        w.push_mark.push_back(w.marks.empty() ? ~0u : (uint32_t)w.marks.size() - 1u);
        w.push_w0.push_back(ev->w[0]);
        w.push_w1.push_back(ev->w[1]);
        w.push_rec.push_back((uint8_t)(ev->rec ? 1 : 0));
        w.push_info.push_back(std::array<uint32_t, 4>{ev->pidx, ev->ticket, ev->dep[0], ev->dep[1]});
    } else if (ev->kind == TSR_QEV_DONE) {
        w.done++;
    }
}

// =============================================================================================
// Scenarios
// =============================================================================================

// --- return codes ------------------------------------------------------------------------------
static void sc_codes(env & e) {
    const char * T = e.tag.c_str();
    // A valid two-copy list to corrupt.
    auto base = [&](uint32_t flags = opl::F_STOP_ON_ERROR) {
        writer w(flags, 0xC0DE0001u);
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        add_copy(w, h0(OUT_BEAT), h0(OUT_BEAT + 16), 2, {0});
        return w.finish();
    };
    std::vector<uint8_t> src = rand_bytes(11, 128);
    dram_put(e, 1, SRC_BEAT, src.data(), src.size());
    {   // the valid list itself, with and without VERIFY_CRC
        fill(e, 0, OUT_BEAT, 2048, 0xEE);
        expect_ok(e, run_list(e, base()), "valid list");
        expect_ok(e, run_list(e, base(), opl::LF_VERIFY_CRC), "valid list, VERIFY_CRC");
        CHECK(dram_get(e, 0, OUT_BEAT + 16, 128) == src, "%s: valid list: the chained copy", T);
    }
    {   // BAD_MAGIC: magic, and a blob shorter than a header
        std::vector<uint8_t> b = base();
        b[0] ^= 1;
        expect_kerr(e, run_list(e, b), opl::RET_BAD_MAGIC, "magic");
        expect_kerr(e, run_list(e, base(), 0, false, LIST_BEAT, 32), opl::RET_BAD_MAGIC, "a1 = 32");
    }
    {   // BAD_VERSION: version, hdr_bytes, unknown flag bit
        std::vector<uint8_t> b = base();
        wr32le(b, 4, 2u | 64u << 16);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_VERSION, "version 2");
        b = base();
        wr32le(b, 4, 1u | 128u << 16);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_VERSION, "hdr_bytes 128");
        b = base();
        wr32le(b, 20, rd32le(b, 20) | 0x10u);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_VERSION, "flag bit 4");
        b = base();
        wr32le(b, 60, 1);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_VERSION, "reserved header word 15");
    }
    {   // TOO_MANY_OPS, and n_ops = 0 (BAD_RECORD)
        std::vector<uint8_t> b = base();
        wr32le(b, 8, 4097);
        expect_kerr(e, run_list(e, b), opl::RET_TOO_MANY_OPS, "n_ops 4097");
        wr32le(b, 8, 0xFFFFFFFFu);
        expect_kerr(e, run_list(e, b), opl::RET_TOO_MANY_OPS, "n_ops 2^32-1");
        wr32le(b, 8, 0);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "n_ops 0");
    }
    {   // BAD_RECORD: layout
        std::vector<uint8_t> b = base();
        wr32le(b, 40, rd32le(b, 40) - 4);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "total_bytes not a multiple of 64");
        b = base();
        wr32le(b, 40, rd32le(b, 40) + 64);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "total_bytes > a1");
        b = base();
        wr32le(b, 12, rd32le(b, 12) + 1);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "n_slots inconsistent");
        b = base();
        wr32le(b, 16, 96);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "ops_off unaligned");
        b = base();
        wr32le(b, 32, 3);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "pre_lines without preload entries");
    }
    {   // BAD_RECORD: records (after two copies that ran)
        fill(e, 0, OUT_BEAT, 2048, 0xEE);
        writer w;
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT + 4), 2);
        w.add_raw(0x20, nullptr, 0, opl::DEP_NONE, opl::DEP_NONE);
        expect_kerr(e, run_list(e, w.finish()), opl::RET_BAD_RECORD, "unknown kind 0x20");
        CHECK(dram_get(e, 0, OUT_BEAT + 4, 128) == src, "%s: the ops before the bad record ran", T);
        writer w2;
        add_copy(w2, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        w2.add_raw(0x7F, nullptr, 0, opl::DEP_NONE, opl::DEP_NONE);
        expect_kerr(e, run_list(e, w2.finish()), opl::RET_BAD_RECORD, "kind 0x7F (>= N_KINDS)");
        writer w3;
        const uint32_t a11[11] = {0};
        w3.add_raw(opl::K_MUL_MAT, a11, 11, opl::DEP_NONE, opl::DEP_NONE);
        expect_kerr(e, run_list(e, w3.finish()), opl::RET_BAD_RECORD, "MUL_MAT with 11 args");
        writer w4;
        const uint32_t a2[2] = {h1(SRC_BEAT), h0(OUT_BEAT)};
        w4.add_raw(TK_COPY, a2, 2, opl::DEP_NONE, opl::DEP_NONE);
        expect_kerr(e, run_list(e, w4.finish()), opl::RET_BAD_RECORD, "COPY with 2 args");
        writer w5;
        uint32_t a3[3];
        copy_args(a3, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        w5.add_raw(TK_COPY, a3, 3, opl::DEP_NONE, opl::DEP_NONE, 0, 0, 0x4);
        expect_kerr(e, run_list(e, w5.finish()), opl::RET_BAD_RECORD, "word0 [15:12] != 0");
        writer w6;
        add_copy(w6, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        w6.add_raw(opl::K_NOP, nullptr, 0, opl::DEP_NONE, opl::DEP_NONE);
        std::vector<uint8_t> b = w6.finish();
        const uint32_t ops_off = rd32le(b, 16);
        wr32le(b, ops_off + 64, opl::make_w0(opl::K_NOP, 4, 0, 0));      // claims 4 slots, 1 left
        writer::patch_crc(b);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "slot overflow");
        b = w6.finish();
        wr32le(b, ops_off + 64, opl::make_w0(opl::K_NOP, 1, 0, 15));     // 15 args in a 1-slot NOP
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "NOP with 15 args");
        b = w6.finish();
        wr32le(b, 8, 3);                                                 // 3 ops, records for 2
        wr32le(b, 12, 2);
        expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "n_ops > n_slots");
    }
    {   // BAD_DEP
        writer w;
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        uint32_t a[3];
        copy_args(a, h1(SRC_BEAT), h0(OUT_BEAT + 4), 2);
        w.add_raw(TK_COPY, a, 3, 1, opl::DEP_NONE);                      // dep0 = own index
        expect_kerr(e, run_list(e, w.finish()), opl::RET_BAD_DEP, "dep0 = own index");
        writer w2;
        add_copy(w2, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        add_copy(w2, h1(SRC_BEAT), h0(OUT_BEAT + 4), 2);
        w2.add_raw(opl::K_JOIN, nullptr, 0, 0, 7);                       // dep1 beyond
        expect_kerr(e, run_list(e, w2.finish()), opl::RET_BAD_DEP, "dep1 = 7 at op 2");
    }
    {   // SCRATCH: a handler failure after pushes (they stay joinable), a ring overflow
        writer w;
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        const uint32_t a[2] = {opl::RET_SCRATCH, 3};
        w.add(TK_RET, a, 2);
        expect_kerr(e, run_list(e, w.finish()), opl::RET_SCRATCH, "handler failure after 3 NOPs");
        writer w2;
        add_copy(w2, h1(SRC_BEAT), h0(OUT_BEAT), 70000);                 // more lines than the ring
        expect_kerr(e, run_list(e, w2.finish()), opl::RET_SCRATCH, "COPY larger than the ring");
        // MUL_MAT whose one token does not fit the ring (y = 125,000 lines): kind_mulmat -> SCRATCH.
        writer w3;
        w3.mul_mat(h0(OUT_BEAT), 2, h0(OUT_BEAT + 64), 125000, W_BEAT, plan::FMT_Q8_0, 2000000, 32, 1);
        expect_kerr(e, run_list(e, w3.finish()), opl::RET_SCRATCH, "MUL_MAT N = 2,000,000");
        writer w4;
        w4.mul_mat(h0(OUT_BEAT), 2, h0(OUT_BEAT + 64), 1, W_BEAT, 7, 16, 32, 1);
        expect_kerr(e, run_list(e, w4.finish()), opl::RET_BAD_RECORD, "MUL_MAT fmt 7");
    }
    {   // CRC: a corrupted padding word (unused by the record) fails only with VERIFY_CRC
        std::vector<uint8_t> b = base();
        const uint32_t ops_off = rd32le(b, 16);
        b[ops_off + 60] ^= 0x40;                                          // word 15 of slot 0
        expect_kerr(e, run_list(e, b, opl::LF_VERIFY_CRC), opl::RET_CRC, "corrupted byte, VERIFY_CRC");
        expect_ok(e, run_list(e, b, 0), "corrupted padding byte without VERIFY_CRC");
        b = base();
        b[b.size() - 1] ^= 1;                                             // last byte of the blob
        expect_kerr(e, run_list(e, b, opl::LF_VERIFY_CRC), opl::RET_CRC, "last byte, VERIFY_CRC");
        b = base();
        wr32le(b, 44, rd32le(b, 44) ^ 0x80000000u);
        expect_kerr(e, run_list(e, b, opl::LF_VERIFY_CRC), opl::RET_CRC, "header crc32 field");
    }
    {   // a tile error before a bad record: the tile error wins (STATUS TILE_ERR, RET = op + 1)
        writer w(0);
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 2);
        add_fail(w);
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT + 4), 2);
        w.add_raw(0x21, nullptr, 0, opl::DEP_NONE, opl::DEP_NONE);
        expect_fail(e, run_list(e, w.finish()), 2, "FAIL at op 1, bad record at op 3");
    }
}

// --- the M7 kinds at the executor (W2b infra; SW §7.2, §7.3) ------------------------------------
static const uint32_t TK_SREG = 0x3A;
static void sc_m7(env & e) {
    const char * T = e.tag.c_str();
    static const uint32_t kinds[] = {opl::K_ADD, opl::K_MUL, opl::K_SCALE, opl::K_RMS_NORM, opl::K_GLU, opl::K_ROPE,
                                     opl::K_GET_ROWS, opl::K_SET_ROWS_KV, opl::K_FLASH_ATTN, opl::K_CPY, opl::K_UNARY};
    // 1. A wrong argument count of every M7 kind is the record check's BAD_RECORD (the kind table's
    //    n_args, before any handler); the op before it ran.
    std::vector<uint8_t> src = dram_get(e, 1, SRC_BEAT, 128);
    for (uint32_t k : kinds) {
        for (int delta = -1; delta <= 1; delta += 2) {
            fill(e, 0, OUT_BEAT, 128, 0xEE);
            const uint32_t n = uint32_t(int(opl::kind_nargs(k)) + delta);
            const std::vector<uint32_t> z(n + 1u, 0u);
            writer w;
            add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 2);
            w.add_raw(k, z.data(), n, opl::DEP_NONE, opl::DEP_NONE);
            char what[64];
            snprintf(what, sizeof(what), "kind 0x%02x with %u arguments", k, n);
            expect_kerr(e, run_list(e, w.finish()), opl::RET_BAD_RECORD, what);
            CHECK(dram_get(e, 0, OUT_BEAT, 128) == src, "%s: %s: the op before the bad record ran", T, what);
        }
    }
    {   // FLASH_ATTN's 22 words in one slot: BAD_RECORD (arguments that do not fit the slots)
        const std::vector<uint32_t> z(opl::fa::NARGS, 0u);
        writer w;
        w.add_raw(opl::K_FLASH_ATTN, z.data(), opl::fa::NARGS, opl::DEP_NONE, opl::DEP_NONE, 0, 1);
        expect_kerr(e, run_list(e, w.finish()), opl::RET_BAD_RECORD, "FLASH_ATTN in one slot");
    }
    // 2. The s-register shadow (kind_ctx::sreg_*, k_fast.hpp sreg_movi; test kind 0x3A): y = x * s[i]
    //    exact, SOP.MOVI only where the launch's pushes do not already leave the value in s[i], again
    //    after a RED TO_S overwrote it, and again in the next launch (the shadow is per launch).
    {
        const uint32_t m = 40u, L = 3u;                               // one row, three lines
        std::vector<float> x(16u * L, 0.0f);
        for (uint32_t j = 0; j < m; ++j) x[j] = 1.0f + float(j) / 64.0f;
        dram_put(e, 1, SRC_BEAT + 0x200, x.data(), 4u * x.size());
        struct rec { uint32_t si; float v; uint32_t fl; };
        const rec rs[7] = {{2, 0.5f, 0}, {2, 0.5f, 0}, {2, 3.0f, 0}, {2, 3.0f, 1}, {2, 3.0f, 0}, {5, 0.5f, 0}, {2, 3.0f, 0}};
        for (int launch = 0; launch < 2; ++launch) {
            fill(e, 0, OUT_BEAT, 64u * L * 7u, 0xEE);
            writer w;
            uint32_t prev = opl::DEP_NONE;
            for (uint32_t r = 0; r < 7u; ++r) {
                const uint32_t a[6] = {h1(SRC_BEAT + 0x200), h0(OUT_BEAT + L * r), m, rs[r].si,
                                       writer::f32_bits(rs[r].v), rs[r].fl};
                const uint32_t dd[1] = {prev};
                prev = w.add(TK_SREG, a, 6, dd, prev == opl::DEP_NONE ? 0u : 1u);
            }
            g_watch.reset();
            const done_t d = run_list(e, w.finish());
            g_watch.on = false;
            expect_ok(e, d, "s-register shadow list");
            uint32_t movi = 0, sop = 0;
            for (uint32_t w0 : g_watch.push_w0)
                if ((w0 & 0xFFu) == TSR_OP_SOP) {
                    ++sop;
                    movi += ((w0 >> 8) & 0xFu) == TSR_SOP_SUB_MOVI ? 1u : 0u;
                }
            CHECK(sop == 4u && movi == 4u, "%s: s-register shadow launch %d: %u SOP pushes (%u MOVI), want 4 MOVI "
                  "(records 0, 2, 4 after the clobber, 5 on s5)", T, launch, sop, movi);
            for (uint32_t r = 0; r < 7u; ++r) {
                const std::vector<uint8_t> got = dram_get(e, 0, OUT_BEAT + L * r, 64u * L);
                bool ok = true;
                for (uint32_t j = 0; j < 16u * L; ++j) {
                    const float want = j < m ? x[j] * rs[r].v : 0.0f;   // exact products; the tail rule zeroes the rest
                    uint32_t gw, ww;
                    memcpy(&gw, &got[4u * j], 4);
                    memcpy(&ww, &want, 4);
                    ok = ok && gw == ww;
                }
                CHECK(ok, "%s: s-register shadow launch %d record %u: y = x * %g", T, launch, r, (double)rs[r].v);
            }
        }
    }
#ifdef TSR_EXEC_TEST_HOST
    // 3. kind_ctx::pre_ok = opl::pre_range_ok (the host reader's rule), over offsets, counts, element
    //    sizes and alignments around the area's end.
    {
        tsr::kind_ctx c{};
        bool same = true;
        for (uint32_t pl : {0u, 1u, 3u, 256u})
            for (uint32_t esz = 0; esz <= 3u && same; ++esz)
                for (uint32_t al : {1u, 4u, 8u, 64u})
                    for (uint32_t off = 0; off <= 64u * pl + 130u && same; off += (off < 200u ? 1u : 61u))
                        for (uint32_t n : {0u, 1u, 2u, 15u, 16u, 17u, 4096u, 0x40000000u, 0xFFFFFFFFu}) {
                            c.pre_lines = pl;
                            const bool a = c.pre_ok(off, n, esz, al - 1u);
                            const bool b = opl::pre_range_ok(off, uint64_t(n) << esz, al, pl);
                            same = same && a == b;
                            if (!same)
                                fprintf(stderr, "pre_ok(%u, %u, %u, %u) pre_lines %u: %d vs %d\n", off, n, esz, al, pl, a, b);
                        }
        CHECK(same, "%s: kind_ctx::pre_ok = opl::pre_range_ok", T);
    }
#endif
}

// --- chains and JOINs ------------------------------------------------------------------------
static void sc_chains(env & e) {
    const char * T = e.tag.c_str();
    const uint32_t L[8] = {1, 3, 7, 16, 17, 33, 2, 40};
    std::vector<std::vector<uint8_t>> A(8);
    for (uint32_t k = 0; k < 8; ++k) {
        A[k] = rand_bytes(100 + k, (size_t)L[k] * 64);
        dram_put(e, 1, SRC_BEAT + 64 * k, A[k].data(), A[k].size());
    }
    fill(e, 0, OUT_BEAT, (size_t)64 * 64 * 16, 0xEE);
    auto B = [](uint32_t k) { return h0(OUT_BEAT + 64 * k); };          // per-k 64-line slots
    writer w(opl::F_STOP_ON_ERROR, 0x00C4A1Au);
    uint32_t op[8];
    for (uint32_t k = 0; k < 8; ++k) op[k] = add_copy(w, h1(SRC_BEAT + 64 * k), B(k), L[k]);   // A_k -> B_k
    const uint32_t c0 = add_copy(w, B(0), B(8), L[0], {op[0]});                                // RAW: B0 -> C0
    // > 2 predecessors: the writer inserts JOINs (a chain of them for 7)
    const uint32_t d0 = add_copy(w, B(8), B(9), L[0], {c0, op[1], op[2], op[3], op[4], op[5], op[6]});
    const uint32_t f1 = add_copy(w, B(1), B(10), L[1], {op[1]});                               // reads B1
    const uint32_t w1 = add_copy(w, h1(SRC_BEAT + 64 * 7), B(1), L[1], {f1});                  // WAR: B1 <- A7
    // a JOIN record with JOIN predecessors, and an op depending on it
    const uint32_t j1 = w.add(opl::K_JOIN, {}, {d0, w1});
    const uint32_t j2 = w.add(opl::K_JOIN, {}, {j1, op[7]});
    w.add(opl::K_NOP, {}, {j2});
    const uint32_t last = add_copy(w, B(9), B(11), L[0], {j2});
    CHECK(w.ok(), "%s: writer: %s", T, w.error());
    std::vector<uint8_t> blob = w.finish();
    // The writer emitted JOINs for the 7-way op: count JOIN records before d0.
    opl::reader r;
    CHECK(r.open(blob.data(), (uint32_t)blob.size(), true) == opl::RET_OK, "%s: reader accepts the list", T);
    uint32_t joins = 0, n = 0;
    opl::record rec{};
    while (r.next(rec, true) == opl::RET_OK) {
        if (rec.kind == opl::K_JOIN && rec.idx < d0) ++joins;
        ++n;
    }
    CHECK(n == w.n_ops() && joins == 5, "%s: %u records, %u JOINs before the 7-way op (5 expected)", T, n, joins);
    expect_ok(e, run_list(e, blob), "chains");
    for (uint32_t k = 0; k < 8; ++k)
        if (k != 1) CHECK(dram_get(e, 0, OUT_BEAT + 64 * k, A[k].size()) == A[k], "%s: B%u = A%u", T, k, k);
    CHECK(dram_get(e, 0, OUT_BEAT + 64 * 8, A[0].size()) == A[0], "%s: C0 = A0 (RAW)", T);
    CHECK(dram_get(e, 0, OUT_BEAT + 64 * 9, A[0].size()) == A[0], "%s: D0 = A0 (JOIN chain)", T);
    CHECK(dram_get(e, 0, OUT_BEAT + 64 * 10, A[1].size()) == A[1], "%s: F = old B1 (WAR read first)", T);
    std::vector<uint8_t> a7(A[7].begin(), A[7].begin() + (long)A[1].size());
    CHECK(dram_get(e, 0, OUT_BEAT + 64 * 1, A[1].size()) == a7, "%s: B1 = A7 (WAR write second)", T);
    CHECK(dram_get(e, 0, OUT_BEAT + 64 * 11, A[0].size()) == A[0], "%s: last copy (op %u) after the JOINs", T, last);
    opl::exec_state st{};
    if (e.exec_state(&st))
        CHECK(st.n_ops == w.n_ops() && st.n_done == w.n_ops() && st.graph_id == 0x00C4A1Au && st.chunks == 0 && !st.stopped,
              "%s: exec_state n_ops %u n_done %u id 0x%x chunks %u", T, st.n_ops, st.n_done, st.graph_id, st.chunks);
}

// --- the deps the executor pushes: after(ticket[dep0], ticket[dep1]) (JOIN records' NOPs) -------
static void sc_deps(env & e) {
    const char * T = e.tag.c_str();
    std::vector<uint8_t> a = rand_bytes(9, 64 * 4);
    dram_put(e, 1, SRC_BEAT, a.data(), a.size());
    writer w;
    const uint32_t c0 = add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 1);
    const uint32_t c1 = add_copy(w, h1(SRC_BEAT + 1), h0(OUT_BEAT + 1), 1);
    const uint32_t j0 = w.add(opl::K_JOIN, {}, {c0, c1});         // same unit: the later ST
    const uint32_t j1 = w.add(opl::K_JOIN, {}, {c0, j0});         // DMA + SYNC
    const uint32_t n0 = w.add(opl::K_NOP, {}, {j1});              // ticket none
    const uint32_t j2 = w.add(opl::K_JOIN, {}, {n0, j1});         // none + SYNC
    const uint32_t j3 = w.add(opl::K_JOIN, {});                   // no deps
    const uint32_t c2 = add_copy(w, h1(SRC_BEAT + 2), h0(OUT_BEAT + 2), 1, {j3, c1});
    g_watch.reset();
    expect_ok(e, run_list(e, w.finish()), "deps");
    g_watch.on = false;
    uint32_t tk[16], px[16];
    if (!e.exec_tables(tk, px, w.n_ops())) {
        CHECK(0, "%s: deps: executor tables not readable", T);
        return;
    }
    auto push_at = [&](uint32_t pidx) -> const std::array<uint32_t, 4> * {
        for (const auto & p : g_watch.push_info)
            if (p[0] == pidx) return &p;
        return nullptr;
    };
    struct want { uint32_t op, d0, d1; };
    const want ws[] = {{j0, tk[c1], TSR_TICKET_NONE}, {j1, tk[c0], tk[j0]}, {j2, tk[j1], TSR_TICKET_NONE},
                       {j3, TSR_TICKET_NONE, TSR_TICKET_NONE}};
    for (const want & x : ws) {
        const auto * p = push_at(px[x.op]);
        CHECK(p != nullptr && (*p)[1] == tk[x.op] && (*p)[2] == x.d0 && (*p)[3] == x.d1,
              "%s: deps: JOIN op %u pushed with deps 0x%08x 0x%08x (expected 0x%08x 0x%08x), ticket 0x%08x / 0x%08x", T, x.op,
              p ? (*p)[2] : 0u, p ? (*p)[3] : 0u, x.d0, x.d1, p ? (*p)[1] : 0u, tk[x.op]);
    }
    CHECK(tk[n0] == TSR_TICKET_NONE && px[n0] == px[j2], "%s: deps: a NOP record has ticket none and no pushes", T);
    CHECK((tk[c0] >> 28) == TSR_UNIT_DMA && (tk[j0] >> 28) == TSR_UNIT_SYNC, "%s: deps: tickets of COPY (ST) and JOIN (NOP)", T);
    // the COPY after {j3, c1}: its first push (the LD, or a join2 NOP folding its deps) waits for both
    const auto * p = push_at(px[c2]);
    CHECK(p != nullptr && ((*p)[2] == tk[j3] || (*p)[3] == tk[j3]), "%s: deps: COPY c2's first push depends on j3", T);
}

// --- n_ops 1 and 4096 ---------------------------------------------------------------------------
static void sc_sizes(env & e) {
    const char * T = e.tag.c_str();
    std::vector<uint8_t> a = rand_bytes(7, 64 * 64);
    dram_put(e, 1, SRC_BEAT, a.data(), a.size());
    fill(e, 0, OUT_BEAT, (size_t)64 * 600, 0xEE);
    {
        writer w;
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 5);
        expect_ok(e, run_list(e, w.finish()), "n_ops 1");
        std::vector<uint8_t> x(a.begin(), a.begin() + 5 * 64);
        CHECK(dram_get(e, 0, OUT_BEAT, x.size()) == x, "%s: n_ops 1: copy", T);
    }
    {
        writer w;
        uint32_t copies = 0;
        for (uint32_t i = 0; i < 4096; ++i) {
            if (i % 8 == 0) {
                add_copy(w, h1(SRC_BEAT + (copies % 64)), h0(OUT_BEAT + copies), 1, {i >= 8 ? i - 8 : opl::DEP_NONE});
                ++copies;
            } else if (i % 8 == 3) {
                w.add(opl::K_JOIN, {}, {i - 3, i - 1});
            } else {
                w.add(opl::K_NOP, {}, {i - 1});
            }
        }
        CHECK(w.n_ops() == 4096 && w.ok(), "%s: 4096 ops", T);
        const std::vector<uint8_t> blob = w.finish();
        CHECK(blob.size() == 64u * 4097u, "%s: 4096 one-slot records = 4097 lines (%zu bytes)", T, blob.size());
        pin_begin(e);
        expect_ok(e, run_list(e, blob), "n_ops 4096");
        pin_end(e, "n_ops 4096", PIN_SIZES_4096);
        bool all = true;
        for (uint32_t c = 0; c < copies; ++c) {
            std::vector<uint8_t> x(a.begin() + 64 * (c % 64), a.begin() + 64 * (c % 64) + 64);
            all = all && dram_get(e, 0, OUT_BEAT + c, 64) == x;
        }
        CHECK(all, "%s: n_ops 4096: %u copies", T, copies);
        opl::exec_state st{};
        if (e.exec_state(&st)) CHECK(st.n_done == 4096 && st.chunks == 3, "%s: 4096: n_done %u chunks %u", T, st.n_done, st.chunks);
        // 4097 ops: TOO_MANY_OPS (the builder must split, SW-29)
        writer w2;
        for (uint32_t i = 0; i < 4097; ++i) w2.add(opl::K_NOP, {});
        CHECK(w2.finish().empty() && !w2.ok(), "%s: the writer refuses 4097 ops", T);
    }
}

// --- streaming (> 256 KiB) -----------------------------------------------------------------------
// Records of 1-4 slots (padding) so that chunk boundaries fall inside records; deps across chunks.
static std::vector<uint8_t> stream_list(uint32_t n_ops, uint32_t fail_at, uint32_t flags, uint32_t * n_copies) {
    writer w(flags, 0x57AEA4u);
    uint32_t copies = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
        const uint32_t ns = 1u + (i * 7u + i / 5u) % 4u;
        if (i == fail_at) {
            const uint32_t a[1] = {2};
            w.add(TK_FAIL, a, 1, nullptr, 0, 0, ns);
            continue;
        }
        uint32_t a[3];
        copy_args(a, h1(SRC_BEAT + (i % 64)), h0(OUT_BEAT + copies), 1);
        const uint32_t d[2] = {i >= 1500 ? i - 1500 : opl::DEP_NONE, i >= 1 ? i - 1 : opl::DEP_NONE};
        w.add(TK_COPY, a, 3, d, 2, 0, ns);
        ++copies;
    }
    *n_copies = copies;
    return w.finish();
}
static void sc_stream(env & e, bool with_crc_variants) {
    const char * T = e.tag.c_str();
    std::vector<uint8_t> a = rand_bytes(31, 64 * 64);
    dram_put(e, 1, SRC_BEAT, a.data(), a.size());
    uint32_t copies = 0;
    const std::vector<uint8_t> blob = stream_list(3000, ~0u, opl::F_STOP_ON_ERROR, &copies);
    const uint32_t lines = (uint32_t)(blob.size() / 64);
    CHECK(lines > 4096 && lines > 2 * 2048 + 2048, "%s: stream list %u lines (> 3 chunks)", T, lines);
    auto check_out = [&](uint32_t upto, const char * what) {
        bool all = true;
        uint32_t c = 0;
        for (uint32_t i = 0; i < 3000 && c < upto; ++i, ++c) {
            std::vector<uint8_t> x(a.begin() + 64 * (i % 64), a.begin() + 64 * (i % 64) + 64);
            all = all && dram_get(e, 0, OUT_BEAT + c, 64) == x;
        }
        CHECK(all, "%s: %s: %u copies", T, what, upto);
    };
    fill(e, 0, OUT_BEAT, (size_t)64 * 3000, 0xEE);
    pin_begin(e);
    g_watch.reset();
    expect_ok(e, run_list(e, blob), "stream");
    g_watch.on = false;
    pin_end(e, "stream", PIN_STREAM);
    check_out(copies, "stream");
    opl::exec_state st{};
    if (e.exec_state(&st))
        CHECK(st.chunks == (lines + 2047) / 2048 && st.n_done == 3000 && st.late_chunks == 0,
              "%s: stream: chunks %u n_done %u late %u", T, st.chunks, st.n_done, st.late_chunks);
    // pidx[i] is the push index of op i's first push (SW §7.3 2.3), also for the op before which a
    // chunk LD is pushed: here every op's first push is its COPY's LD into the ring, never a chunk
    // LD into the OPL area.
    {
        const uint32_t opl_line = tsr_sim_config(e.sim)->spm_bytes / 64u - plan::ARG_LINES - plan::OPL_AREA_LINES;
        std::vector<uint32_t> tk(3000), px(3000);
        if (e.exec_tables(tk.data(), px.data(), 3000)) {
            uint32_t bad = 0, chunk_lds = 0, first_bad = ~0u;
            for (size_t k = 0; k < g_watch.push_info.size(); ++k) {
                const bool ld = (g_watch.push_w0[k] & 0xFFu) == TSR_OP_LD;
                if (ld && g_watch.push_w1[k] >= opl_line) ++chunk_lds;
            }
            size_t k = 0;                               // pushes come in pidx order, px is non-decreasing
            for (uint32_t i = 0; i < 3000; ++i) {
                while (k < g_watch.push_info.size() && g_watch.push_info[k][0] - px[i] >= 0x80000000u) ++k;
                const bool ok = k < g_watch.push_info.size() && g_watch.push_info[k][0] == px[i] &&
                                (g_watch.push_w0[k] & 0xFFu) == TSR_OP_LD && g_watch.push_w1[k] < opl_line;
                if (!ok && bad++ == 0) first_bad = i;
            }
            CHECK(bad == 0 && chunk_lds == st.chunks,
                  "%s: stream: %u ops whose pidx is not their first push (first: op %u); %u LDs into the OPL area "
                  "(expected one per chunk, %u)", T, bad, first_bad, chunk_lds, st.chunks);
        } else {
            CHECK(0, "%s: stream: executor tables not readable", T);
        }
    }
    // NOP records of 16 slots (4 MiB-class list, no pushes and no joins in the op stream): every
    // chunk is read only after the executor's own join of its load (the race detector sees a
    // missing one: nothing else orders the RISC-V's reads after the DMA's writes; STOP_ON_ERROR
    // is off because its QSTATUS reads with QIDLE = 1 would observe every completion, SW §4.6).
    {
        writer w(0, 0x57AEA5u);
        for (uint32_t i = 0; i < 700; ++i) w.add(opl::K_NOP, nullptr, 0, nullptr, 0, 0, 16);
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT + 3000), 1, {699});
        const std::vector<uint8_t> nb = w.finish();
        fill(e, 0, OUT_BEAT + 3000, 64, 0xEE);
        pin_begin(e);
        expect_ok(e, run_list(e, nb), "stream of NOP records");
        pin_end(e, "stream of NOP records", PIN_STREAM_NOP);
        std::vector<uint8_t> x(a.begin(), a.begin() + 64);
        CHECK(dram_get(e, 0, OUT_BEAT + 3000, 64) == x, "%s: stream of NOP records: the last op", T);
        if (e.exec_state(&st))
            CHECK(st.chunks == (uint32_t)(nb.size() / 64 + 2047) / 2048 && st.n_done == 701 && st.late_chunks == 0,
                  "%s: NOP stream: chunks %u n_done %u late %u", T, st.chunks, st.n_done, st.late_chunks);
    }
    if (with_crc_variants) {
        fill(e, 0, OUT_BEAT, (size_t)64 * 3000, 0xEE);
        pin_begin(e);
        expect_ok(e, run_list(e, blob, opl::LF_VERIFY_CRC), "stream, VERIFY_CRC");
        pin_end(e, "stream, VERIFY_CRC", PIN_STREAM_CRC);
        check_out(copies, "stream, VERIFY_CRC");
        std::vector<uint8_t> b = blob;
        b[64u * 5000u + 60u] ^= 0x01;                          // a padding word in chunk 2
        expect_kerr(e, run_list(e, b, opl::LF_VERIFY_CRC), opl::RET_CRC, "stream, corrupted chunk 2, VERIFY_CRC");
    }
    // A tile error in chunk 2 (op index 2400: blob line > 4096), STOP_ON_ERROR = 0.
    {
        uint32_t c2 = 0;
        const std::vector<uint8_t> bf = stream_list(3000, 2400, 0, &c2);
        pin_begin(e);
        expect_fail(e, run_list(e, bf), 2401, "stream, FAIL at op 2400");
        pin_end(e, "stream, FAIL at op 2400", PIN_STREAM_FAIL);
    }
    // A chunk load that fails: the list placed so that chunk 2 lies beyond the channel.
    {
        const uint32_t beat = (uint32_t)(CH_BYTES / 64u) - 4096u - 8u;
        const std::vector<uint8_t> head(blob.begin(), blob.begin() + (long)(4096u + 8u) * 64);
        dram_put(e, 0, beat, head.data(), head.size());
        pin_begin(e);
        const done_t d = e.run(h0(beat), (uint32_t)blob.size(), 0, false);
        pin_end(e, "chunk 2 beyond the channel", PIN_STREAM_CHUNK_FAIL);
        CHECK(d.status() == TSR_LCQ_R0_STATUS_TILE_ERR && d.err() == TSR_ERR_DRAM_RANGE && d.ret() == 0u &&
                  d.r[5] == tsr_w0(TSR_OP_LD, 0, 0, 0) && d.r[4] == beat + 2u * 2048u,
              "%s: chunk 2 beyond the channel: TILE_ERR, RET 0 (status %u ret 0x%08x err %u info 0x%08x)", T, d.status(),
              d.ret(), d.err(), d.r[4]);
    }
    // A record whose words end before a failing chunk but whose padding slot lies in it still runs:
    // only the lines that hold a record's words are waited for (W2a); the next record, which starts
    // in that chunk, stops the launch. NOP records (no pushes): exec_state's n_done shows it.
    {
        writer w(0, 0x57AEA6u);
        for (uint32_t i = 0; i < 4096; ++i) w.add(opl::K_NOP, nullptr, 0, nullptr, 0, 0, i == 4094u ? 2u : 1u);
        const std::vector<uint8_t> nb = w.finish();
        CHECK(rd32le(nb, 16) == 64u && nb.size() == 64u * 4098u, "%s: padding stream: ops at line 1, 4098 lines (%zu B)", T,
              nb.size());
        const uint32_t beat = (uint32_t)(CH_BYTES / 64u) - 4096u;        // line 4096 (chunk 2) beyond the channel
        dram_put(e, 0, beat, nb.data(), (size_t)4096u * 64u);
        const done_t d = e.run(h0(beat), (uint32_t)nb.size(), 0, false);
        CHECK(d.status() == TSR_LCQ_R0_STATUS_TILE_ERR && d.err() == TSR_ERR_DRAM_RANGE && d.ret() == 0u &&
                  d.r[4] == beat + 2u * 2048u,
              "%s: padding slot in a failing chunk: TILE_ERR, RET 0 (status %u ret 0x%08x err %u info 0x%08x)", T,
              d.status(), d.ret(), d.err(), d.r[4]);
        if (e.exec_state(&st))
            CHECK(st.n_done == 4095u && st.stopped == 1u,
                  "%s: padding slot in a failing chunk: n_done %u (expected 4095: op 4094, whose padding slot is in "
                  "chunk 2, runs), stopped %u", T, st.n_done, st.stopped);
    }
}

// --- failing-op index -------------------------------------------------------------------------
static void sc_errors(env & e) {
    const char * T = e.tag.c_str();
    std::vector<uint8_t> a = rand_bytes(41, 64 * 64);
    dram_put(e, 1, SRC_BEAT, a.data(), a.size());
    const uint32_t pos[] = {0, 1, 15, 16, 17, 40, 63};
    for (uint32_t p : pos) {
        writer w(0);                                          // STOP_ON_ERROR = 0: deterministic streams
        for (uint32_t i = 0; i < 64; ++i) {
            if (i == p) add_fail(w, 3, {i > 0 ? i - 1 : opl::DEP_NONE});
            else if (i + 1 == p || i == p + 1) w.add(opl::K_NOP, {});             // empty neighbours
            else if (i % 5 == 2) w.add(opl::K_JOIN, {}, {i - 1, i - 2});
            else add_copy(w, h1(SRC_BEAT + i), h0(OUT_BEAT + i), 1, {i > 0 ? i - 1 : opl::DEP_NONE});
        }
        char what[64];
        snprintf(what, sizeof(what), "FAIL at op %u", p);
        expect_fail(e, run_list(e, w.finish()), p + 1, what);
    }
    // DEP_FAILED behind the error does not move the first error. The DMA unit is in error state
    // after the failing LD, so its later entries complete ABORTED (EX §8.2), dependent or not; the
    // op before the error ran.
    {
        fill(e, 0, OUT_BEAT, 64 * 64, 0xEE);
        writer w(0);
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 1);
        const uint32_t f = add_fail(w, 2);
        add_copy(w, h1(SRC_BEAT + 1), h0(OUT_BEAT + 1), 1, {f});              // DEP_FAILED
        add_copy(w, h1(SRC_BEAT + 2), h0(OUT_BEAT + 2), 1);                   // ABORTED
        expect_fail(e, run_list(e, w.finish()), 2, "FAIL at op 1 with a dependent op");
        std::vector<uint8_t> x0(a.begin(), a.begin() + 64);
        CHECK(dram_get(e, 0, OUT_BEAT, 64) == x0, "%s: the op before the error ran", T);
        CHECK(dram_get(e, 0, OUT_BEAT + 1, 128) == std::vector<uint8_t>(128, 0xEE), "%s: the ops after it did not", T);
    }
    // Executor loads that fail: the list itself, a preload entry: TILE_ERR, RET 0.
    {
        const done_t d = e.run(h0(0x00FFFFF0u), 64u * 64u, 0, false);
        CHECK(d.status() == TSR_LCQ_R0_STATUS_TILE_ERR && d.err() == TSR_ERR_DRAM_RANGE && d.ret() == 0u &&
                  d.r[5] == tsr_w0(TSR_OP_LD, 0, 0, 0),
              "%s: list beyond the channel: status %u ret 0x%08x err %u", T, d.status(), d.ret(), d.err());
        writer w;
        w.add_preload(h0(0x00FFFFFFu), 128);
        add_copy(w, h1(SRC_BEAT), h0(OUT_BEAT), 1);
        const done_t d2 = run_list(e, w.finish());
        CHECK(d2.status() == TSR_LCQ_R0_STATUS_TILE_ERR && d2.err() == TSR_ERR_DRAM_RANGE && d2.ret() == 0u &&
                  d2.r[4] == 0x00FFFFFFu,
              "%s: preload beyond the channel: status %u ret 0x%08x err %u info 0x%08x", T, d2.status(), d2.ret(),
              d2.err(), d2.r[4]);
    }
}

// --- STOP_ON_ERROR (schedule-dependent pushes; only the DONE record is compared) -----------------
static void sc_stop(env & e, bool fixed_sched) {
    const char * T = e.tag.c_str();
    std::vector<uint8_t> a = rand_bytes(51, 64 * 64);
    dram_put(e, 1, SRC_BEAT, a.data(), a.size());
    for (uint32_t serial = 0; serial < 2; ++serial) {
        writer w(opl::F_STOP_ON_ERROR | (serial ? opl::F_SERIAL : 0u));
        for (uint32_t i = 0; i < 600; ++i) {
            if (i == 37) add_fail(w, 2);
            else add_copy(w, h1(SRC_BEAT + i % 64), h0(OUT_BEAT + i), 1);
        }
        expect_fail(e, run_list(e, w.finish()), 38, serial ? "STOP_ON_ERROR + SERIAL, FAIL at op 37" : "STOP_ON_ERROR, FAIL at op 37");
        opl::exec_state st{};
        if (e.exec_state(&st)) {
            printf("%s: STOP_ON_ERROR%s: pushed %u of 600 ops before stopping (stopped %u)\n", T, serial ? " + SERIAL" : "",
                   st.n_done, st.stopped);
            if (fixed_sched || serial)
                CHECK(st.stopped == 1 && st.n_done < 600 && st.n_done >= 38, "%s: STOP_ON_ERROR stopped (n_done %u, stopped %u)",
                      T, st.n_done, st.stopped);
            if (serial) CHECK(st.n_done == 38, "%s: SERIAL + STOP_ON_ERROR stops at the failing op (n_done %u)", T, st.n_done);
        }
    }
}

// --- preload ---------------------------------------------------------------------------------
static void sc_preload(env & e) {
    const char * T = e.tag.c_str();
    const uint32_t sizes[4] = {4, 100, 64, 300};
    std::vector<std::vector<uint8_t>> P(4);
    uint32_t pb[4];
    writer w(opl::F_STOP_ON_ERROR, 0x9E10ADu);
    for (uint32_t k = 0; k < 4; ++k) {
        P[k] = rand_bytes(60 + k, 320);                          // the source rows (whole lines in DRAM)
        dram_put(e, 0, PRE_BEAT + 8 * k, P[k].data(), P[k].size());
        pb[k] = w.add_preload(h0(PRE_BEAT + 8 * k), sizes[k]);
    }
    CHECK(pb[0] == 0 && pb[1] == 64 && pb[2] == 192 && pb[3] == 256 && w.pre_lines() == 9,
          "%s: preload layout %u %u %u %u, %u lines", T, pb[0], pb[1], pb[2], pb[3], w.pre_lines());
    fill(e, 0, OUT_BEAT, 64 * 32, 0xEE);
    const uint32_t a0[3] = {pb[1], 16, h0(OUT_BEAT)};
    w.add(TK_PRE, a0, 3);                                        // RISC-V reads entry 1
    const uint32_t a1[3] = {pb[0], 1, h0(OUT_BEAT + 1)};
    w.add(TK_PRE, a1, 3);                                        // RISC-V reads entry 0
    const uint32_t a2[3] = {pb[2], 1, h0(OUT_BEAT + 2)};
    w.add(TK_PREST, a2, 3);                                      // ST of entry 2
    const uint32_t a3[3] = {pb[3], 5, h0(OUT_BEAT + 8)};
    w.add(TK_PREST, a3, 3);                                      // ST of entry 3 (5 lines)
    g_watch.reset();
    expect_ok(e, run_list(e, w.finish()), "preload");
    g_watch.on = false;
    std::vector<uint8_t> o = dram_get(e, 0, OUT_BEAT, 64 * 16);
    bool r1 = true;
    for (uint32_t k = 0; k < 16; ++k) r1 = r1 && rd32le(o, 4 * k) == (get_u32_le(&P[1][4 * k]) ^ (0xA5A50000u + k));
    CHECK(r1, "%s: RISC-V read of preload entry 1", T);
    CHECK(rd32le(o, 64) == (get_u32_le(&P[0][0]) ^ 0xA5A50000u) && rd32le(o, 68) == 0u, "%s: RISC-V read of entry 0", T);
    CHECK(memcmp(&o[128], P[2].data(), 64) == 0, "%s: device read of entry 2", T);
    CHECK(memcmp(&o[512], P[3].data(), 300) == 0 && memcmp(&o[512 + 300], &P[3][300], 20) == 0,
          "%s: device read of entry 3 (300 bytes + the source line's tail)", T);
    // The pushes: list LD, then the 4 preload LDs back to back (one join), then the ops.
    uint32_t lds = 0;
    for (size_t i = 0; i < g_watch.push_w0.size() && (g_watch.push_w0[i] & 0xFFu) == TSR_OP_LD; ++i) ++lds;
    CHECK(lds == 5, "%s: the list LD and 4 preload LDs come first (%u LDs)", T, lds);
    // Preload validation: an entry beyond pre_lines, an unaligned pre_byte.
    writer wb;
    wb.add_preload(h0(PRE_BEAT), 64);
    wb.add(opl::K_NOP, {});
    std::vector<uint8_t> b = wb.finish();
    wr32le(b, 64 + 8, 32);                                       // pre_byte 32
    expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "preload pre_byte unaligned");
    b = wb.finish();
    wr32le(b, 64 + 4, 65);                                       // 65 bytes in a 1-line area
    expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "preload entry beyond pre_lines");
    b = wb.finish();
    wr32le(b, 32, 257);                                          // pre_lines > 256
    expect_kerr(e, run_list(e, b), opl::RET_BAD_RECORD, "pre_lines 257");
}

// --- flags: MARK, PROF, REC, SERIAL --------------------------------------------------------------
static void sc_flags(env & e) {
    const char * T = e.tag.c_str();
    std::vector<uint8_t> a = rand_bytes(71, 64 * 64);
    dram_put(e, 1, SRC_BEAT, a.data(), a.size());
    auto list = [&](uint32_t flags) {
        writer w(flags);
        for (uint32_t i = 0; i < 20; ++i) {
            if (i % 4 == 1) w.add(opl::K_NOP, {});
            else if (i % 4 == 3) w.add(opl::K_JOIN, {}, {i - 1, i - 3});
            else add_copy(w, h1(SRC_BEAT + i), h0(OUT_BEAT + i), 1 + i % 3);
        }
        return w.finish();
    };
    for (int variant = 0; variant < 2; ++variant) {             // MARK flag, PROF launch flag
        g_watch.reset();
        expect_ok(e, run_list(e, list(variant == 0 ? opl::F_MARK : 0u), 0, variant == 1), variant == 0 ? "MARK" : "PROF");
        g_watch.on = false;
        // the runtime adds LAUNCH_BEGIN / END markers on the RV; keep the NODE_BEGIN ones
        std::vector<uint32_t> nodes;
        for (uint32_t m : g_watch.marks)
            if ((m >> 28) == 3u) nodes.push_back(m);
        bool in_order = nodes.size() == 20;
        for (uint32_t i = 0; i < nodes.size() && in_order; ++i) in_order = nodes[i] == (3u << 28 | i);
        CHECK(in_order, "%s: %s: NODE_BEGIN markers 0..19 in order (%zu)", T, variant == 0 ? "MARK" : "PROF", nodes.size());
        // each op's pushes come right after its own marker: the LD / ST of COPY i between NODE_BEGIN i
        // and NODE_BEGIN i + 1 (all markers count for the index: map it back to the NODE number)
        std::vector<uint32_t> ld_ops, st_ops;
        for (size_t k = 0; k < g_watch.push_w0.size(); ++k) {
            const uint32_t m = g_watch.push_mark[k];
            if (m == ~0u || (g_watch.marks[m] >> 28) != 3u) continue;
            const uint32_t opi = g_watch.marks[m] & 0x0FFFFFFFu, opc = g_watch.push_w0[k] & 0xFFu;
            if (opc == TSR_OP_LD) ld_ops.push_back(opi);
            if (opc == TSR_OP_ST) st_ops.push_back(opi);
        }
        std::vector<uint32_t> want;
        for (uint32_t i = 0; i < 20; ++i)
            if (i % 4 == 0 || i % 4 == 2) want.push_back(i);
        CHECK(ld_ops == want && st_ops == want, "%s: each COPY's LD and ST follow its own marker (%zu LDs, %zu STs)", T,
              ld_ops.size(), st_ops.size());
    }
    {   // REC: every push of the launch after the list's own load carries ISSUE_PUSH.REC
        g_watch.reset();
        expect_ok(e, run_list(e, list(opl::F_REC)), "REC");
        g_watch.on = false;
        // every LD / ST after the first push (the list's own load, pushed before the header is read)
        uint32_t ldst = 0, ldst_rec = 0;
        for (size_t k = 1; k < g_watch.push_w0.size(); ++k) {
            const uint32_t opc = g_watch.push_w0[k] & 0xFFu;
            if (opc != TSR_OP_LD && opc != TSR_OP_ST) continue;
            ++ldst;
            ldst_rec += g_watch.push_rec[k];
        }
        CHECK(ldst == 20 && ldst_rec == ldst && g_watch.push_rec[0] == 0, "%s: REC: %u of %u LD / ST pushes with REC", T,
              ldst_rec, ldst);
        g_watch.reset();
        expect_ok(e, run_list(e, list(0)), "no REC");
        g_watch.on = false;
        CHECK(g_watch.rec_pushes == 0, "%s: no REC flag: %u pushes with REC", T, g_watch.rec_pushes);
    }
    {   // SERIAL (+ MARK to delimit the ops): no op's first push while an earlier one is outstanding
        g_watch.reset();
        expect_ok(e, run_list(e, list(opl::F_SERIAL | opl::F_MARK)), "SERIAL");
        g_watch.on = false;
        CHECK(g_watch.serial_violations == 0, "%s: SERIAL: %u markers pushed with ops outstanding", T, g_watch.serial_violations);
    }
}

// --- plan.hpp parity (RISC-V vs host vs tsr_ref) --------------------------------------------------
struct shape { uint32_t fmt, N, K, T; };
static std::vector<shape> plan_shapes() {
    std::vector<shape> v;
    // Qwen2.5-0.5B (896 / kv 128 / ffn 4864 / vocab 151936), 32B (5120 / 1024 / 27648 / 152064),
    // Gemma 4 12B (3840; q 4096 / 8192, kv 2048 / 512, ffn 15360, vocab 262144; SW §8).
    const uint32_t mk[][2] = {{896, 896}, {128, 896}, {4864, 896}, {896, 4864}, {151936, 896},
                              {5120, 5120}, {1024, 5120}, {27648, 5120}, {5120, 27648}, {152064, 5120},
                              {4096, 3840}, {8192, 3840}, {2048, 3840}, {512, 3840}, {3840, 4096}, {3840, 8192},
                              {15360, 3840}, {3840, 15360}, {262144, 3840}};
    const uint32_t Ts[3] = {1, 8, 512};
    for (const auto & s : mk)
        for (uint32_t fmt = 0; fmt < 2; ++fmt)
            for (uint32_t T : Ts) v.push_back(shape{fmt, s[0], s[1], T});
    // edges: bad fmt, N = 0, K not a block multiple, K beyond K_MAX / the field, T = 0, SCRATCH
    const shape edge[] = {{2, 64, 256, 1}, {0, 0, 256, 1}, {1, 64, 32, 1}, {0, 64, 48, 1}, {0, 64, 0, 1},
                          {0, 64, 32768, 1}, {0, 64, 32800, 1}, {1, 64, 65536, 1}, {0, 64, 256, 0},
                          {0, 2000000, 32, 1}, {0, 16, 32, 7}, {1, 1, 256, 9}};
    for (const auto & s : edge) v.push_back(s);
    return v;
}
static void sc_plan(env & e) {
    const char * T = e.tag.c_str();
    const plan::dev_cfg dc = dev_of(*tsr_sim_config(e.sim));
    const std::vector<shape> sh = plan_shapes();
    uint32_t mismatches = 0, ok_count = 0, n_done = 0;
    for (size_t base = 0; base < sh.size(); base += 64) {
        const uint32_t n = (uint32_t)std::min<size_t>(64, sh.size() - base);
        std::vector<uint8_t> tab(64u * ((16u * n + 63u) / 64u), 0);
        for (uint32_t i = 0; i < n; ++i) {
            wr32le(tab, 16 * i, sh[base + i].fmt); wr32le(tab, 16 * i + 4, sh[base + i].N);
            wr32le(tab, 16 * i + 8, sh[base + i].K); wr32le(tab, 16 * i + 12, sh[base + i].T);
        }
        dram_put(e, 0, PLAN_BEAT, tab.data(), tab.size());
        fill(e, 0, PLAN_BEAT + 64, 64u * n, 0xEE);
        writer w;
        const uint32_t a[3] = {h0(PLAN_BEAT), n, h0(PLAN_BEAT + 64)};
        w.add(TK_PLAN, a, 3);
        expect_ok(e, run_list(e, w.finish()), "PLAN");
        const std::vector<uint8_t> out = dram_get(e, 0, PLAN_BEAT + 64, 64u * n);
        for (uint32_t i = 0; i < n; ++i) {
            const shape & s = sh[base + i];
            plan::mm_tiling m{};
            const uint32_t st = plan::mm_plan(dc, s.fmt, s.N, s.K, s.T, &m);
            const uint32_t rec[12] = {0, s.K / 16u, 0x10000000u, (s.N + 15u) / 16u, 0, s.fmt, s.N, s.K, s.T, opl::HANDLE_NONE, 0, 0};
            const uint32_t chk = plan::check_op(dc, opl::K_MUL_MAT, rec, 12);
            const uint32_t want[12] = {st, m.kind, m.x_lines, m.qa_lines, m.y_lines, m.lpt, m.t_tile, m.n_tiles,
                                       plan::ring_budget(dc), chk, dc.t_max, dc.k_max};
            bool same = true;
            for (uint32_t k = 0; k < 12; ++k) same = same && rd32le(out, 64 * i + 4 * k) == want[k];
            if (!same) {
                if (++mismatches < 5)
                    CHECK(0, "%s: plan of fmt %u N %u K %u T %u differs: device status 0x%08x t_tile %u, host 0x%08x %u", T,
                          s.fmt, s.N, s.K, s.T, rd32le(out, 64 * i), rd32le(out, 64 * i + 24), st, m.t_tile);
            }
            if (st == opl::RET_OK) ++ok_count;
            ++n_done;
        }
    }
    CHECK(mismatches == 0, "%s: plan parity: %u mismatches of %u shapes", T, mismatches, n_done);
    printf("%s: plan parity: %u shapes (%u plannable), device = host\n", T, n_done, ok_count);
}

// Host-only: plan.hpp geometry against tsr_ref (independent C code) and the reference tiling.
static void plan_host_checks() {
    uint32_t bad = 0;
    for (uint32_t fmt = 0; fmt < 2; ++fmt)
        for (uint32_t K = 32; K <= 65535; K += 32) {
            const uint32_t kind = fmt == 0 ? TSR_QA_A32 : TSR_QA_A256;
            const bool blk = K % tsr_fmt_block_size(fmt) == 0;
            if (plan::row_beats(fmt, K) != (blk ? tsr_row_beats(fmt, K) : 0u)) ++bad;
            if (plan::qa_lines(kind, K) != (blk ? tsr_qa_lines(kind, K) : 0u)) ++bad;
        }
    CHECK(bad == 0, "plan.hpp qa_lines / row_beats = tsr_ref for every K: %u differences", bad);
    // dev_cfg U200 and the budget: L - 16 - 4096 - 256.
    CHECK(plan::ring_budget(plan::U200_K4) == 65536u - 16u - 4096u - 256u - 1024u, "ring budget (the one-tile block: W2b)");
    plan::exec_layout l{};
    CHECK(plan::layout(65536, 9, &l) && l.arg_line == 65520 && l.opl_line == 61424 && l.pre_line == 61415 && l.one_lines == 1024 &&
              l.one_line == 60391 && l.ring_lines == 60391,
          "exec layout");
#ifdef TSR_EXEC_HAVE_LOWER
    // Reference tiling (sw/ggml-tesseract tsr-lower): same per-token line counts; its SPM-based shape check
    // accepts everything plan.hpp accepts.
    tsr_config c = test_cfg();
    uint32_t diff = 0, n = 0;
    for (const shape & s : plan_shapes()) {
        if (s.T == 0 || s.fmt > 1 || s.N == 0) continue;
        tsr_mm_tiling v1{};
        const bool v1ok = tsr_mm_plan_tiling(c, s.fmt, s.N, s.K, 1, &v1);
        plan::mm_tiling m{};
        const bool ok = plan::mm_plan(dev_of(c), s.fmt, s.N, s.K, 1, &m) == opl::RET_OK;
        ++n;
        if (ok && (!v1ok || v1.x_lines != m.x_lines || v1.qa_lines != m.qa_lines || v1.y_lines != m.y_lines ||
                   tsr_mm_lines_per_token(s.fmt, s.N, s.K) != m.lpt))
            ++diff;
        if (v1ok && !ok && (uint64_t)v1.x_lines + v1.qa_lines + v1.y_lines <= plan::ring_budget(dev_of(c))) ++diff;
    }
    CHECK(diff == 0, "plan.hpp vs tsr_mm_plan_tiling: %u differences over %u shapes", diff, n);
#endif
}

// --- MUL_MAT through kind_mulmat (a 0.5B-like layer, Q8_0, T tokens) -----------------------------
struct mm_layer {
    uint32_t w_q, w_k, w_v, w_o, w_g, w_u, w_d;   // W_OFF beats
};
static void put_rand_weights(env & e, uint32_t fmt, uint32_t N, uint32_t K, uint32_t w_off, uint64_t seed) {
    rng_t r = {seed};
    const size_t row_bytes = (size_t)(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    std::vector<uint8_t> blocks(row_bytes * N);
    for (uint32_t n = 0; n < N; ++n) gen_row_blocks(&r, fmt, K, &blocks[row_bytes * n]);
    put_weights(e.sim, fmt, N, K, w_off, blocks.data());
}
static uint32_t rows_beats(uint32_t fmt, uint32_t N, uint32_t K) {
    return tsr_ceil_div_u32(N, NCH) * tsr_row_beats(fmt, K);
}
static mm_layer setup_layer(env & e, uint32_t fmt) {
    mm_layer L{};
    uint32_t at = W_BEAT;
    auto alloc = [&](uint32_t N, uint32_t K, uint64_t seed) {
        const uint32_t w = at;
        put_rand_weights(e, fmt, N, K, w, seed);
        at += rows_beats(fmt, N, K) + 64;
        return w;
    };
    L.w_q = alloc(896, 896, 1); L.w_k = alloc(128, 896, 2); L.w_v = alloc(128, 896, 3); L.w_o = alloc(896, 896, 4);
    L.w_g = alloc(4864, 896, 5); L.w_u = alloc(4864, 896, 6); L.w_d = alloc(896, 4864, 7);
    return L;
}
// One layer's MUL_MATs with the graph's dependency shape (x -> q, k, v -> o (+residual) -> gate, up
// -> down (+residual)); activations in ch0 IO. layer l reads the previous layer's output.
static void add_layer(writer & w, const mm_layer & L, uint32_t fmt, uint32_t T, uint32_t l, uint32_t * prev, bool stub) {
    const uint32_t A = OUT_BEAT + 0x40000u * (l % 2);         // per-layer activation area
    const uint32_t X = l == 0 ? OUT_BEAT + 0x80000u : OUT_BEAT + 0x40000u * ((l + 1) % 2) + 0x30000u;
    const uint32_t xs = 896 / 16, fs = 4864 / 16;
    auto mm = [&](uint32_t x, uint32_t xst, uint32_t y, uint32_t yst, uint32_t woff, uint32_t N, uint32_t K,
                  uint32_t add, std::initializer_list<uint32_t> deps) {
        std::vector<uint32_t> d(deps);
        if (stub) {
            const uint32_t a[2] = {h0(x), h0(y)};
            return w.add(TK_STUB5, a, 2, d.data(), (uint32_t)d.size());
        }
        return w.mul_mat(h0(x), xst, h0(y), yst, woff, fmt, N, K, T, add == ~0u ? opl::HANDLE_NONE : h0(add), add == ~0u ? 0u : yst,
                         0, d.data(), (uint32_t)d.size());
    };
    const uint32_t q = mm(X, xs, A, xs, L.w_q, 896, 896, ~0u, {*prev});
    const uint32_t k = mm(X, xs, A + 0x1000, 8, L.w_k, 128, 896, ~0u, {*prev});
    const uint32_t v = mm(X, xs, A + 0x2000, 8, L.w_v, 128, 896, ~0u, {*prev});
    const uint32_t o = mm(A, xs, A + 0x3000, xs, L.w_o, 896, 896, X, {q, k, v});
    const uint32_t g = mm(A + 0x3000, xs, A + 0x8000, fs, L.w_g, 4864, 896, ~0u, {o});
    const uint32_t u = mm(A + 0x3000, xs, A + 0x10000, fs, L.w_u, 4864, 896, ~0u, {o});
    const uint32_t dn = mm(A + 0x8000, fs, A + 0x30000, xs, L.w_d, 896, 4864, A + 0x3000, {g, u});
    *prev = dn;
}
static void sc_mulmat(env & e) {
    const mm_layer L = setup_layer(e, TSR_FMT_Q8_0);
    std::vector<uint8_t> x = rand_bytes(5, 896 * 4 * 2);
    for (size_t i = 0; i + 3 < x.size(); i += 4) x[i + 3] = (uint8_t)((x[i + 3] & 0x83u) | 0x3C);   // finite values
    dram_put(e, 0, OUT_BEAT + 0x80000u, x.data(), x.size());
    writer w;
    uint32_t prev = opl::DEP_NONE;
    for (uint32_t l = 0; l < 2; ++l) add_layer(w, L, TSR_FMT_Q8_0, 2, l, &prev, false);
    expect_ok(e, run_list(e, w.finish()), "MUL_MAT layers (kind_mulmat)");
}

// A long decode list (T = 1: every MUL_MAT record takes the executor's fast path, k_mulmat.hpp):
// `layers` layers of 7 MUL_MATs (~1.6 K scratch lines each) wrap the ~61 K-line ring and fill its
// 128-record table, so spm_ring::alloc_dma_first's table-full drop and the wrap fallback to
// kind_mulmat / alloc run on the device; the race detector (fatal) checks every reuse is ordered.
static void sc_mulmat_long(env & e, uint32_t layers) {
    const mm_layer L = setup_layer(e, TSR_FMT_Q8_0);
    std::vector<uint8_t> x = rand_bytes(6, 896 * 4);
    for (size_t i = 0; i + 3 < x.size(); i += 4) x[i + 3] = (uint8_t)((x[i + 3] & 0x83u) | 0x3C);   // finite values
    dram_put(e, 0, OUT_BEAT + 0x80000u, x.data(), x.size());
    writer w;
    uint32_t prev = opl::DEP_NONE;
    for (uint32_t l = 0; l < layers; ++l) add_layer(w, L, TSR_FMT_Q8_0, 1, l, &prev, false);
    expect_ok(e, run_list(e, w.finish()), "MUL_MAT decode list, T = 1 (ring wraps, table full)");
    opl::exec_state st{};
    if (e.exec_state(&st))
        CHECK(st.n_done == w.n_ops() && st.ring_joins > 0u && st.ring_slow > 0u && st.mm_fallback > 0u &&
                  st.mm_fallback < layers,
              "%s: decode list: %u ops done; ring: %u software joins (table full), %u full-algorithm allocations "
              "(wraps); %u of %u MUL_MAT records fell back to kind_mulmat",
              e.tag.c_str(), st.n_done, st.ring_joins, st.ring_slow, st.mm_fallback, layers * 7u);
    // The next launch, rejected before its ring exists (BAD_MAGIC), reports its own counters (0), not
    // this launch's (fix W2a-opt review: exec_state carries nothing from launch to launch).
    writer wb;
    wb.add(opl::K_NOP, {});
    std::vector<uint8_t> bad = wb.finish();
    bad[0] ^= 0x01u;
    expect_kerr(e, run_list(e, bad), opl::RET_BAD_MAGIC, "BAD_MAGIC after the decode list");
    if (e.exec_state(&st))
        CHECK(st.ring_joins == 0u && st.ring_slow == 0u && st.mm_fallback == 0u && st.n_done == 0u,
              "%s: a list rejected before step 2 reports ring joins %u, full-algorithm allocations %u, fallbacks %u, "
              "n_done %u (expected 0: nothing from the previous launch)", e.tag.c_str(), st.ring_joins, st.ring_slow,
              st.mm_fallback, st.n_done);
}

// =============================================================================================
// Runs
// =============================================================================================
struct run_result {
    std::vector<std::array<uint32_t, 7>> dones;
    obs o;
    bool have_obs = false;
};

static env * make_env(tsr_sim * sim, const char * image, const std::string & tag) {
    env * e = nullptr;
#ifdef TSR_EXEC_TEST_HOST
    (void)image;
    e = new host_env(sim);
    e->tag = tag;
#else
    rv_env * r = new rv_env();
    r->sim = sim;
    r->tag = tag;
    r->boot(image);
    e = r;
#endif
    return e;
}

enum suite { SUITE_DET = 1, SUITE_STOP = 2, SUITE_BIG = 4 };

// One simulator, one boot, the scenarios of `suites` as consecutive launches. SUITE_DET's pushes are
// schedule-independent (all of SW §4.7's observables compare); SUITE_STOP's are not (only the DONE
// records compare); SUITE_BIG adds n_ops 4096 and the streamed list's VERIFY_CRC variants.
static void one_run(const char * image, const char * sched, uint32_t race, uint32_t suites, run_result * out) {
    const tsr_config cfg = test_cfg();
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = sched;
    o.race = race;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    CHECK(sim != NULL, "create (%s)", sched);
    if (!sim) return;
    std::string tag = std::string(
#ifdef TSR_EXEC_TEST_HOST
                          "host "
#else
                          "rv "
#endif
                          ) + tsr_sim_opt_sched(sim) + ((suites & SUITE_STOP) ? " stop" : "");
    if (race) CHECK(tsr_sim_race_active(sim) == 1, "%s: the race detector is active", tag.c_str());
    obs_attach(&out->o, sim);
    out->have_obs = true;
    tsr_sim_add_listener(sim, watch_q, nullptr);
    env * e = make_env(sim, image, tag);
    if (suites & SUITE_DET) {
        sc_codes(*e);
        sc_chains(*e);
        sc_deps(*e);
        sc_errors(*e);
        sc_preload(*e);
        sc_flags(*e);
        sc_plan(*e);
        sc_mulmat(*e);
        sc_mulmat_long(*e, 48);
        sc_stream(*e, (suites & SUITE_BIG) != 0);
        if (suites & SUITE_BIG) sc_sizes(*e);
        sc_m7(*e);                                  // last: the pinned launches above keep their state
    }
    if (suites & SUITE_STOP) sc_stop(*e, strcmp(tsr_sim_opt_sched(sim), "fixed") == 0);
    obs_finish(&out->o, sim);
    out->dones = e->dones;
    {   // CC §7.3: the firmware reads the credit before it pushes; no push ever waited in hardware
        const uint64_t stall = tsr_sim_perf(sim, TSR_PERF_PUSH_STALL(cfg.num_ch));
        CHECK(stall == 0u, "%s: PERF PUSH_STALL %llu RV cycles: a push waited for a credit in hardware", tag.c_str(),
              (unsigned long long)stall);
    }
    printf("%s: %zu launches, MTIME %llu, %llu RV instructions\n", tag.c_str(), e->dones.size(),
           (unsigned long long)tsr_sim_now(sim), (unsigned long long)tsr_sim_rv_instret(sim));
    delete e;
    tsr_sim_destroy(sim);
}

// The detector sees the executor's pushes (RV only: on the host the pushes are the harness's):
// a list with a missing RAW dependency and an op reading preload lines no entry wrote are reported
// (report mode), and nothing else is.
static void race_probe(const char * image) {
    const tsr_config cfg = test_cfg();
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = "fixed";
    o.race = 1;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    env * e = make_env(sim, image, "rv race probe");
    std::vector<uint8_t> a = rand_bytes(3, 64 * 8);
    dram_put(*e, 1, SRC_BEAT, a.data(), a.size());
    const tsr_race_report * rep = nullptr;
    // A QACT reading ring lines that a LD writes, without / with the dep (exec_kinds.cpp RACE).
    for (uint32_t with_dep = 0; with_dep < 2; ++with_dep) {
        writer w;
        const uint32_t ra[2] = {h1(SRC_BEAT), with_dep};
        w.add(TK_RACE, ra, 2);
        expect_ok(*e, run_list(*e, w.finish()), with_dep ? "RACE with the dep" : "RACE without the dep");
        const size_t n = tsr_sim_race_reports(sim, &rep);
        bool raw = false;
        for (size_t i = 0; i < n; ++i)
            raw = raw || (rep[i].kind == TSR_RACE_RAW && rep[i].space == TSR_RACE_SPACE_SPM && rep[i].first.agent == TSR_UNIT_DMA &&
                          rep[i].second.agent == TSR_UNIT_QACT);
        if (with_dep) CHECK(n == 0, "race probe: no report with the dependency (%zu)", n);
        else CHECK(raw, "race probe: the missing dependency is reported (%zu reports)", n);
        tsr_sim_race_clear(sim);
    }
    {   // SPM_STALE: an ST of preload lines past the one entry (pre_lines 2, entry of 64 bytes)
        writer w;
        w.add_preload(h0(PRE_BEAT), 64);
        const uint32_t pa[3] = {0, 1, h0(OUT_BEAT)};
        w.add(TK_PREST, pa, 3);
        std::vector<uint8_t> b = w.finish();
        wr32le(b, 32, 2);                                     // pre_lines 2: line 1 is never written
        const uint32_t pb[3] = {64, 1, h0(OUT_BEAT + 1)};
        const uint32_t ops_off = rd32le(b, 16);
        for (int k = 0; k < 3; ++k) wr32le(b, ops_off + 8 + 4 * k, pb[k]);
        expect_ok(*e, run_list(*e, b), "stale preload line");
        const size_t n = tsr_sim_race_reports(sim, &rep);
        bool stale = false;
        for (size_t i = 0; i < n; ++i) stale = stale || rep[i].kind == TSR_RACE_SPM_STALE;
        CHECK(stale, "race probe: SPM_STALE for the unwritten preload line (%zu reports)", n);
        tsr_sim_race_clear(sim);
    }
    delete e;
    tsr_sim_destroy(sim);
}

static void compare_runs(const run_result & a, const run_result & b, bool all_obs, const char * what) {
    bool same = a.dones.size() == b.dones.size();
    size_t first_diff = 0;
    for (size_t i = 0; i < a.dones.size() && same; ++i)
        if (a.dones[i] != b.dones[i]) {
            same = false;
            first_diff = i;
        }
    CHECK(same, "%s: DONE records r0-r6 identical (%zu / %zu launches, first difference at %zu)", what, a.dones.size(),
          b.dones.size(), first_diff);
    if (all_obs) obs_equal(&a.o, &b.o, what);
}

// ---------------------------------------------------------------------------------------------
// fw_exec_perf: RISC-V cycles per graph op (SW §7.8)
// ---------------------------------------------------------------------------------------------
#ifndef TSR_EXEC_TEST_HOST
struct busy_ctx {
    tsr_rv * rv;
    uint64_t last, busy, insns;
    uint64_t work, work_insns;                 // busy minus spin-wait iterations (fw_prof.h)
    fwprof::spin_filter spin;
    fwprof::prof * prof;                       // per-PC / per-class profile (nullptr: off)
};
static int busy_hook(void * ctx, const tsr_rv_retire * r) {
    busy_ctx * b = (busy_ctx *)ctx;
    const uint64_t now = tsr_rv_cycles(b->rv);
    const uint64_t d = now - b->last;
    b->last = now;
    const bool spin = b->spin.step(r);
    if (d <= 256u) {                           // larger steps are sleeps / fast-forwarded waits
        b->busy += d;
        if (!spin) {
            b->work += d;
            b->work_insns++;
        }
        if (b->prof) b->prof->add(r, d, spin);
    }
    b->insns++;
    return 0;
}
static const char * g_perf_image = nullptr;    // for the profile's symbols
// One measurement: busy RV cycles and instructions of a list built by `build(w, scale)` at scale 1
// and `big`, the difference divided by the difference in ops (the launch's fixed cost cancels).
// Also the executor's work (spin-wait iterations excluded) and, with TSR_FW_PROF set, its profile
// per op (fw_prof.h; TSR_FW_PROF_DUMP adds the per-PC histogram).
template <class F>
static double perf_case(rv_env & e, busy_ctx & b, const char * what, uint32_t big, F build, double * ins_per_op,
                        double * work_per_op = nullptr) {
    double cyc[2] = {0, 0}, ins[2] = {0, 0}, wk[2] = {0, 0}, wki[2] = {0, 0};
    uint32_t nops[2] = {0, 0};
    fwprof::prof prof[2];
    const bool profile = getenv("TSR_FW_PROF") != nullptr;
    for (int k = 0; k < 2; ++k) {
        writer w;
        build(w, k ? big : 1u);
        const uint64_t b0 = b.busy, i0 = b.insns, x0 = b.work, xi0 = b.work_insns;
        if (profile) {
            prof[k].load_symbols(g_perf_image);
            b.prof = &prof[k];
        }
        expect_ok(e, run_list(e, w.finish()), what);
        b.prof = nullptr;
        nops[k] = w.n_ops();
        cyc[k] = (double)(b.busy - b0);
        ins[k] = (double)(b.insns - i0);
        wk[k] = (double)(b.work - x0);
        wki[k] = (double)(b.work_insns - xi0);
    }
    const double dn = (double)(nops[1] - nops[0]);
    const double c = (cyc[1] - cyc[0]) / dn, i = (ins[1] - ins[0]) / dn;
    const double wc = (wk[1] - wk[0]) / dn, wi = (wki[1] - wki[0]) / dn;
    printf("%s: %-34s %4u vs %3u ops: %7.1f RV busy cycles / op, %7.1f instructions / op (launch fixed cost %.0f cycles); "
           "work (spin waits excluded) %.1f cycles, %.1f instructions / op\n",
           e.tag.c_str(), what, nops[1], nops[0], c, i, cyc[0] - c * nops[0], wc, wi);
    if (profile) {
        char title[200];
        snprintf(title, sizeof(title), "%s: profile per op, %s", e.tag.c_str(), what);
        static int n_case = 0;
        char tag[16];
        snprintf(tag, sizeof(tag), ".case%d", n_case++);
        fwprof::prof::diff(prof[1], prof[0]).report(stdout, title, dn, 12, tag);
    }
    if (ins_per_op) *ins_per_op = i;
    if (work_per_op) *work_per_op = wc;
    return c;
}
// SW §7.8's target (an *estimate*, checked in P3 / P6) is printed, not asserted (fw_m6_perf prints
// the executor's work on the 0.5B decode lists). Asserted: a regression ceiling (6000 busy cycles,
// ~1.3x the MUL_MAT list at P2.4), and, in perf(), the executor optimisation's ceilings on the
// spin-free work.
static void report_target(const env & e, const char * what, double c) {
    printf("%s: %s: %.1f RV cycles per graph op; SW §7.8 target <= 400: %s\n", e.tag.c_str(), what, c,
           c <= 400.0 ? "met" : "NOT met");
    CHECK(c <= 6000.0, "%s: %s: regression ceiling 6000 RV cycles per op (%.1f)", e.tag.c_str(), what, c);
}
// --- short VU records (W2b infra; SW §7.3 "Hot path", §7.8) -----------------------------------------
// One decode-shaped record per kind (Qwen2.5-0.5B, T = 1; rows of n = 896 = the hidden size for the
// row kinds, GLU and UNARY included: the FFN width 4,864 takes the same one-tile path, but its LD / VOP
// / ST keep the units busier than the issuer, the queues fill and the credit waits' first reads would
// count as work), in a chain (record i reads record i - 1's output and depends on it, as in a layer):
// the reference FASTVU (test kind 0x39: LD, one VOP, ST in k_fast.hpp's form; tests/fw/exec_fast.cpp)
// and every M7 short kind of the image. Work per record =
// the spin-free RISC-V cycles (fw_prof.h) of a 384-record list minus a 16-record one, divided by the
// record difference (perf_case), with deep queues (the RISC-V's own cost; SW §7.8's metric); the
// same with the U200 queue depths for information. W2b's requirement: <= 350 RV cycles of work per
// short record. A kind whose handler is still zz_kind_stubs.cpp's weak stub is reported as not in
// the image; one whose first record is refused, as refused (with a limit that fails). Each kind
// package measures its own kind with
//     test_fw_exec_rv <fw_exec.elf> --perf-m7 [--kind 0x14] [--strict]     (TSR_FW_PROF=1: profile)
// fw_exec_perf prints them and asserts FASTVU's regression ceiling; fw_exec_perf_m7 (--strict) asserts
// every kind present <= 350; fw_exec_perf_ref (--ceiling) the reference kind of the fw_exec_ref image
// (tests/fw/exec_ref_unary.cpp: FASTVU's code as a production table kind, no test-kind lookup).
// Measured at W2b infra (maxperf, deep queues): the reference 3-push record (LD, VOP, ST) costs 365
// RV cycles of work, 112 of them the executor's (record read and checks, tables, RT_INFO, pidx, the
// handler call) and 73 the pushes' 24 control-window accesses; FASTVU 408 (+ the lookup).
static const uint32_t ACT_BEAT = OUT_BEAT + 0x40000;      // ch0: activation buffers, 1,024 lines apart
static const uint32_t PAR_BEAT = SRC_BEAT + 0x40000;      // ch1: weight / bias / table rows (56 lines apart)
static const uint32_t SHORT_POS_BEAT = PRE_BEAT + 0x100;  // ch0: preload source: position 7 (I32)
static const uint32_t SHORT_IDS_BEAT = PRE_BEAT + 0x101;  // ch0: preload source: id 3 (I32)
static const uint32_t TK_FASTVU = 0x39;
static uint32_t act(uint32_t k) { return h0(ACT_BEAT + 1024u * k); }
static uint32_t f32b(float f) { return writer::f32_bits(f); }

struct short_kind {
    uint32_t kind;
    const char * name;
    const char * handler;   // demangled-name prefix of its handler (symbol binding: stub or not)
    uint32_t variant = 0;   // SCALE: 1 = with a bias (W2b fix: the review's SCALE + b record)
};
static const short_kind SHORT_KINDS[] = {
    {TK_FASTVU, "FASTVU reference 896 (LD, VOP, ST)", "tsr::exec_test_fastvu("},
    {opl::K_ADD, "ADD 896 (residual)", "tsr::kind_add("},
    {opl::K_MUL, "MUL 896 (norm weight)", "tsr::kind_mul("},
    {opl::K_SCALE, "SCALE 896", "tsr::kind_scale("},
    {opl::K_SCALE, "SCALE 896 with bias", "tsr::kind_scale(", 1u},
    {opl::K_RMS_NORM, "RMS_NORM 896 + weight", "tsr::kind_rms_norm("},
    {opl::K_GLU, "GLU SWIGLU 896", "tsr::kind_glu("},
    {opl::K_ROPE, "ROPE NEOX 16 heads x 64 (q + k)", "tsr::kind_rope("},
    {opl::K_GET_ROWS, "GET_ROWS 1 x 56 lines", "tsr::kind_get_rows("},
    {opl::K_CPY, "CPY 56 lines", "tsr::kind_cpy("},
    {opl::K_UNARY, "UNARY SILU 896", "tsr::kind_unary("},
};

// Record i of a chain of `kind` (deps: prev). pos / ids: the list's preload byte offsets.
static uint32_t add_short(writer & w, uint32_t kind, uint32_t i, uint32_t prev, uint32_t pos, uint32_t ids,
                          uint32_t variant = 0) {
    const uint32_t in = act(i & 1u), out = act((i + 1u) & 1u), par = h1(PAR_BEAT);
    const uint32_t dep[1] = {prev};
    const uint32_t nd = prev == opl::DEP_NONE ? 0u : 1u;
    switch (kind) {
        case opl::K_ADD:
        case opl::K_MUL: {
            const opl::ew_args a{out, 56, in, 56, par, 56, 896, 1, 1, 0};
            return kind == opl::K_ADD ? w.add_add(a, dep, nd) : w.add_mul(a, dep, nd);
        }
        case opl::K_SCALE:
            return w.add_scale(opl::scale_args{out, 56, in, 56, 896, 1, f32b(0.5f), variant == 1u ? f32b(1.0f) : 0u}, dep, nd);
        case opl::K_RMS_NORM:
            return w.add_rms_norm(opl::rms_norm_args{out, 56, in, 56, 896, 1, f32b(1e-6f), f32b(1.0f / 896.0f), par, 1},
                                  dep, nd);
        case opl::K_GLU: return w.add_glu(opl::glu_args{out, 56, in, 56, act(2), 56, 896, 1, opl::glu::SWIGLU}, dep, nd);
        case opl::K_ROPE:
            return w.add_rope(opl::rope_args{out, 64, in, 64, pos, opl::HANDLE_NONE, 0x3F2B5A7Bu, f32b(1.0f), f32b(1.0f), 64,
                                             64, 16, 1, opl::rope::MODE_NEOX},
                              dep, nd);
        case opl::K_GET_ROWS: return w.add_get_rows(opl::get_rows_args{out, 56, par, 56, ids, 1, 56}, dep, nd);
        case opl::K_CPY: return w.add_cpy(opl::cpy_args{out, 56, in, 56, 56, 1}, dep, nd);
        case opl::K_UNARY: return w.add_unary(opl::unary_args{out, 56, in, 56, 896, 1, opl::unary::SILU}, dep, nd);
        default: {
            const uint32_t a[4] = {in, out, 896, 0};
            return w.add(TK_FASTVU, a, 4, dep, nd);
        }
    }
}
static void build_short(writer & w, uint32_t kind, uint32_t n_rec, uint32_t variant = 0) {
    const uint32_t pos = w.pre_positions(h0(SHORT_POS_BEAT), 1), ids = w.pre_ids(h0(SHORT_IDS_BEAT), 1);
    uint32_t prev = opl::DEP_NONE;
    for (uint32_t i = 0; i < n_rec; ++i) prev = add_short(w, kind, i, prev, pos, ids, variant);
}

// The scenarios on one environment (deep: assert, else print). only: one kind (~0u: all). limit: the
// work per record every kind of the image must meet (0: print only; --strict 350, --ceiling N);
// FASTVU, whose record also pays the test-kind lookup (tsr_exec_test_kind, ~40 RV cycles), always
// has the regression ceiling FASTVU_CEILING.
static const double FASTVU_CEILING = 450.0;
// --strict holds every short kind to 350 (W2b fix: the integration's regression ceilings for RMS_NORM
// and ROPE are gone; their hot entries and the staged-word reuse, k_fast.hpp push_s, meet it).
static void perf_short(rv_env & e, busy_ctx & b, bool deep, double limit, uint32_t only) {
    const bool strict = limit > 0.0;
    const uint32_t pos[16] = {7u}, ids[16] = {3u};
    dram_put(e, 0, SHORT_POS_BEAT, pos, sizeof(pos));
    dram_put(e, 0, SHORT_IDS_BEAT, ids, sizeof(ids));
    fwprof::prof syms;
    const bool have_syms = syms.load_symbols(g_perf_image);
    for (const short_kind & k : SHORT_KINDS) {
        if (only != ~0u && k.kind != only) continue;
        if (have_syms && syms.binding_of(k.handler) == 2) {
            printf("%s: short record %-40s kind 0x%02x: not in the image (zz_kind_stubs.cpp's weak stub)\n", e.tag.c_str(),
                   k.name, k.kind);
            continue;
        }
        writer w1;
        build_short(w1, k.kind, 1, k.variant);
        const done_t d = run_list(e, w1.finish());
        if (d.status() != TSR_LCQ_R0_STATUS_OK || d.ret() != 0u) {
            printf("%s: short record %-40s kind 0x%02x: refused (DONE status %u ret 0x%08x err %u)\n", e.tag.c_str(),
                   k.name, k.kind, d.status(), d.ret(), d.err());
            CHECK(!strict || !deep, "%s: %s: the kind refuses its decode record (ret 0x%08x)", e.tag.c_str(), k.name,
                  d.ret());
            continue;
        }
        double work = 0, ins = 0;
        char what[96];
        snprintf(what, sizeof(what), "short %s", k.name);
        perf_case(e, b, what, 24, [&](writer & w, uint32_t n) { build_short(w, k.kind, 16u * n, k.variant); }, &ins, &work);
        const bool met = work <= 350.0;
        printf("%s: short record %-40s kind 0x%02x: %.1f RV cycles of work per record; W2b requirement <= 350: %s\n",
               e.tag.c_str(), k.name, k.kind, work, met ? "met" : "NOT met");
        if (!deep) continue;
        if (k.kind == TK_FASTVU)
            CHECK(work <= FASTVU_CEILING, "%s: the k_fast.hpp reference record (test-kind path): regression ceiling %.0f "
                  "RV cycles of work (%.1f)", e.tag.c_str(), FASTVU_CEILING, work);
        else if (strict)
            CHECK(work <= limit, "%s: %s: <= %.0f RV cycles of work per short record (%.1f)", e.tag.c_str(), k.name, limit,
                  work);
    }
}

// mode 0: the M6 scenarios and the short records (fw_exec_perf); 1: the short records only (--perf-m7).
static void perf(const char * image, int mode = 0, double limit = 0.0, uint32_t only = ~0u) {
    g_perf_image = image;
    const tsr_config cfg = test_cfg();
    for (int real = 0; real < 2; ++real) {
        tsr_sim_opts o;
        memset(&o, 0, sizeof(o));
        o.sched = "fixed";
        static const uint8_t deep[6] = {31, 31, 31, 31, 0, 31};
        if (!real) o.q_depth = deep;           // no credit waits: the RISC-V alone sets the pace
        tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
        rv_env e;
        e.sim = sim;
        e.tag = real ? "perf (U200 queues)" : "perf (deep queues)";
        if (!e.boot(image)) {
            tsr_sim_destroy(sim);
            return;
        }
        busy_ctx b{};
        b.rv = tsr_rv_agent_rv(tsr_sim_rv_agent(sim));
        b.last = tsr_rv_cycles(b.rv);
        tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), busy_hook, &b);
        perf_short(e, b, !real, limit, only);
        if (mode == 1) {
        } else if (!real) {
            double w0 = 0, w1 = 0;
            perf_case(e, b, "NOP records (no pushes)", 192, [](writer & w, uint32_t n) {
                for (uint32_t i = 0; i < 8 * n; ++i) w.add(opl::K_NOP, {}, {i ? i - 1 : opl::DEP_NONE});
            }, nullptr, &w0);
            // The executor's own work per record (read, check, deps, tables, RT_INFO, ISSUE_PIDX):
            // the part of SW §7.8's 400 cycles that is graph.cpp's; a regression guard (executor
            // optimisation: 90 / 106 cycles on Ibex maxperf / small; W2a: 156 / 191).
            CHECK(w0 <= 130.0, "%s: executor overhead per record <= 130 RV cycles (%.1f)", e.tag.c_str(), w0);
            perf_case(e, b, "JOIN records (1 push)", 192, [](writer & w, uint32_t n) {
                for (uint32_t i = 0; i < 8 * n; ++i) w.add(opl::K_JOIN, {}, {i ? i - 1 : opl::DEP_NONE, i > 1 ? i - 2 : opl::DEP_NONE});
            }, nullptr, &w1);
            CHECK(w1 <= 200.0, "%s: JOIN record (one push) <= 200 RV cycles (%.1f)", e.tag.c_str(), w1);
            const double c = perf_case(e, b, "STUB5 0.5B-like layers (5 pushes)", 24, [](writer & w, uint32_t n) {
                uint32_t prev = opl::DEP_NONE;
                for (uint32_t l = 0; l < n; ++l) add_layer(w, mm_layer{}, TSR_FMT_Q8_0, 1, l, &prev, true);
            }, nullptr);
            report_target(e, "STUB5 list", c);
        } else {
            const mm_layer L = setup_layer(e, TSR_FMT_Q8_0);
            double wm = 0;
            const double c = perf_case(e, b, "MUL_MAT 0.5B layers (kind_mulmat)", 4, [&](writer & w, uint32_t n) {
                uint32_t prev = opl::DEP_NONE;
                for (uint32_t l = 0; l < n; ++l) add_layer(w, L, TSR_FMT_Q8_0, 1, l, &prev, false);
            }, nullptr, &wm);
            report_target(e, "MUL_MAT list", c);
            CHECK(wm <= 700.0, "%s: MUL_MAT list: regression ceiling 700 RV cycles of work per op (%.1f)", e.tag.c_str(), wm);
        }
        tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), NULL, NULL);
        const uint64_t stall = tsr_sim_perf(sim, TSR_PERF_PUSH_STALL(cfg.num_ch));
        CHECK(stall == 0u, "%s: PERF PUSH_STALL %llu RV cycles: a push waited for a credit in hardware (CC §7.3)",
              e.tag.c_str(), (unsigned long long)stall);
        tsr_sim_destroy(sim);
    }
}
#endif

int main(int argc, char ** argv) {
    const char * image = nullptr;
    bool full = false, quick = false, do_perf = false, perf_m7 = false;
    double limit = 0.0;
    uint32_t only = ~0u;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--full")) full = true;
        else if (!strcmp(argv[i], "--quick")) quick = true;
        else if (!strcmp(argv[i], "--perf")) do_perf = true;
        else if (!strcmp(argv[i], "--perf-m7")) perf_m7 = true;
        else if (!strcmp(argv[i], "--strict")) limit = 350.0;                       // W2b's requirement
        else if (!strcmp(argv[i], "--ceiling") && i + 1 < argc) limit = strtod(argv[++i], nullptr);
        else if (!strcmp(argv[i], "--kind") && i + 1 < argc) only = (uint32_t)strtoul(argv[++i], nullptr, 0);
        else image = argv[i];
    }
#ifdef TSR_EXEC_TEST_HOST
    (void)quick; (void)do_perf; (void)full; (void)image; (void)perf_m7; (void)limit; (void)only;
    plan_host_checks();
    run_result a, b;
    one_run(nullptr, "fixed", 2, SUITE_DET | SUITE_BIG, &a);
    one_run(nullptr, "fixed", 2, SUITE_STOP, &b);
    if (a.have_obs) obs_free(&a.o);
    if (b.have_obs) obs_free(&b.o);
    return test_finish("fw_exec_host");
#else
    if (image == nullptr) {
        fprintf(stderr, "usage: %s <fw_exec image.elf> [--quick|--full|--perf|--perf-m7 [--strict|--ceiling <c>] [--kind <k>]]\n",
                argv[0]);
        return 2;
    }
    if (do_perf || perf_m7) {
        perf(image, perf_m7 ? 1 : 0, limit, only);
        return test_finish(perf_m7 ? "fw_exec short-record perf" : "fw_exec_perf");
    }
    plan_host_checks();
    race_probe(image);
    // SW §4.7 / §10.2: the fixed schedule and 7 random ones with the race detector fatal. The
    // deterministic suite's observables and DONE records must be identical under every schedule,
    // the STOP_ON_ERROR suite's DONE records too. Default: the big lists (n_ops 4096, the streamed
    // list's CRC variants) under the fixed schedule only; --full: everything under all 8.
    const char * scheds[8] = {"fixed", "random:1", "random:2", "random:3", "random:0x5eed", "random:77",
                              "random:1234567", "random:9,0.1,8"};
    const int n_sched = quick ? 2 : 8;
    const uint32_t det = SUITE_DET | (full ? (uint32_t)SUITE_BIG : 0u);
    run_result ref_det, ref_stop;
    for (int s = 0; s < n_sched; ++s) {
        run_result d, st;
        one_run(image, scheds[s], 2, det, &d);
        one_run(image, scheds[s], 2, SUITE_STOP, &st);
        if (s == 0) {
            ref_det = d;
            ref_stop = st;
            continue;
        }
        char what[96];
        snprintf(what, sizeof(what), "%s vs fixed", scheds[s]);
        compare_runs(ref_det, d, true, what);
        snprintf(what, sizeof(what), "%s vs fixed (STOP_ON_ERROR)", scheds[s]);
        compare_runs(ref_stop, st, false, what);
        obs_free(&d.o);
        obs_free(&st.o);
    }
    if (!full) {
        run_result big;
        one_run(image, "fixed", 2, SUITE_DET | SUITE_BIG, &big);
        obs_free(&big.o);
    }
    obs_free(&ref_det.o);
    obs_free(&ref_stop.o);
    return test_finish("fw_exec_rv");
#endif
}
