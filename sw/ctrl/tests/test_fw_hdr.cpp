// test_fw_hdr.cpp — ctests fw_hdr_host and fw_hdr_rv of P2.3, the tile/VU header library and the
// host backend (docs/v2/06-software-verification.md "SW" §6.7, §6.8, §10.2; CC §5, §7-§9, §12;
// EX §4.9). One source, two executables (sw/ctrl/tests/hdr.cmake):
//
// fw_hdr_host (TSR_HDR_TEST_HOST; the ASan / UBSan variant of tsr_fw_host):
//   builders    every LD / ST / QACT / MVM / NOP / FENCE and every generated VU / AT builder
//               (vu.hpp, at.hpp) against the simulator's checker (tsr_sim_op): words equal to the
//               C encoders / the generated field macros the units decode with (TSR_INSN_GET), code 0
//               for legal operands, the documented code (and ERR_INFO) for illegal ones, ILLEGAL_OP
//               for a field value that does not fit (the builders' poison bit), saturation of MVM K /
//               T / fmt to their K_RANGE / T_RANGE / BAD_FMT codes;
//   tickets     issue / done / join / join_irq / join2 / join_all / error / first_error / qclear /
//               pidx_now / issued_seq / QABORT / QHOLD / TRACE_MARK / the WMASK and REC of the push,
//               deps as pushed, the unit of refused opcodes (missing CAPS, unit not built: no hang);
//   group       folding of 1..6 units with the documented NOP count and deps, the SYNC subsumption,
//               idempotence, as_ticket;
//   ring        spm_ring against a line-level model over random alloc / release sequences with real
//               tickets of every unit: placement (padding, parity, wrap), failure exactly when an
//               unreleased block is in the way, and every older release ticket of a reused line
//               covered by must_wait or done (joined in software) when alloc returns; the same
//               sequence under the fixed and two random schedules gives the same deps;
//   kernels     the test kernels (tests/fw/hdr_kernels.cpp) through tsr::host_run: results against
//               direct references (tsr_qact, tsr_dot_row, tsr_ref_vop), queue occupancy never above
//               Q_DEPTH and reaching it on every unit (credit back-pressure), no push without credit.
//
// fw_hdr_rv (TSR_HDR_TEST_RV; the image fw_hdr = production sources + hdr_kernels.cpp on tsr_rv):
//   the kernels launched through the LQ / LCQ under the fixed and 7 random schedules with
//   TSR_SIM_RACE=fatal (a missing dep, e.g. in spm_ring's must_wait, is a race: the process aborts),
//   the DONE records (r0-r6), the outputs and the per-launch push streams (words, push values, deps
//   and tickets relative to the launch's base sequences) identical across the schedules and
//   identical to the host build's (SW §6.8: the same kernel pushes the same stream on both),
//   PERF PUSH_STALL = 0 (the firmware never relied on the hardware's credit wait, CC §7.3), queue
//   occupancy as above, and a soft stop (LQ_STOP) seen by stop_requested() (DONE STOPPED).
#include "test_sim_obs.h"   // test_sim_util.h (CHECK, rng, put_weights), tsr_sim_int.h, tsr_bus.h

#include "tsr_race.h"
#include "tsr_ref.h"
#include "tsr_ref_vu.h"
#include "tsr_rv.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "tsr/kernel.hpp"
#include "tsr/rt.hpp"

extern "C" {
int32_t hdr_stream(const tsr_args *);
int32_t hdr_credit(const tsr_args *);
int32_t hdr_err(const tsr_args *);
int32_t hdr_qclear(const tsr_args *);
int32_t hdr_irq(const tsr_args *);
int32_t hdr_ring(const tsr_args *);
int32_t hdr_stop(const tsr_args *);
int32_t hdr_xring(const tsr_args *);
}

namespace {

constexpr uint32_t NCH = 2u;
constexpr uint32_t SEQ = 0x0FFF'FFFFu;
constexpr uint32_t BAD_BEAT = 0x0FFF'FFF0u;          // beyond a 1 GiB channel (DRAM_RANGE)
// DRAM layout (beats) of the kernel scenarios.
constexpr uint32_t X_BEAT = 4096u;                   // ch0: hdr_stream x
constexpr uint32_t Y_BEAT = 4096u;                   // ch1: hdr_stream y
constexpr uint32_t SRC_BEAT = 65536u;                // ch0: hdr_ring source pattern, hdr_irq source
constexpr uint32_t DST_BEAT = 65536u;                // ch1: hdr_ring destination
constexpr uint32_t CRED_BEAT = 131072u;              // ch0: hdr_credit source
constexpr uint32_t GOOD_BEAT = 200000u;              // ch0: hdr_err / hdr_qclear
constexpr uint32_t W_OFF = 300000u;                  // every channel: weights (hdr_stream, hdr_err)
constexpr uint32_t W_OFF2 = 400000u;                 // every channel: hdr_xring's weights (Q8_0, 32 x 64)
constexpr uint32_t XR_BEAT = 8192u;                  // ch1: hdr_xring y

uint32_t h0(uint32_t beat) { return beat; }
uint32_t h1(uint32_t beat) { return (1u << 28) | beat; }

// -----------------------------------------------------------------------------------------------
// Simulators
// -----------------------------------------------------------------------------------------------
tsr_sim * mk_sim(const char * sched, uint32_t race, uint32_t caps, uint32_t t_max = 1, const uint8_t * q_depth = nullptr,
                 uint32_t num_ch = NCH, uint64_t ch_bytes = 1ull << 30) {
    tsr_config c = TSR_CONFIG_U200_V1;
    c.num_ch = num_ch;
    c.ch_bytes = ch_bytes;
    c.caps = caps;
    c.t_max = t_max;
    tsr_sim_opts o;
    memset(&o, 0, sizeof(o));
    o.sched = sched;
    o.race = race;
    o.q_depth = q_depth;
    tsr_sim * s = tsr_sim_create_opts(&c, &o);
    if (s == nullptr) {
        fprintf(stderr, "tsr_sim_create_opts failed (sched %s)\n", sched ? sched : "-");
        exit(1);
    }
    return s;
}

// -----------------------------------------------------------------------------------------------
// Push-stream recorder and queue-occupancy check (tsr_sim_int.h §5 queue events)
// -----------------------------------------------------------------------------------------------
struct pushrec {
    uint32_t kind, unit, ticket, rpidx, w[8], dep0, dep1, push;
    bool operator==(const pushrec & o) const {
        return kind == o.kind && unit == o.unit && ticket == o.ticket && rpidx == o.rpidx && memcmp(w, o.w, sizeof w) == 0 &&
               dep0 == o.dep0 && dep1 == o.dep1 && push == o.push;
    }
};

// Queue occupancy (CC §7.3): a slot belongs to an entry from its push until its unit accepts it
// (START) or it completes without running; the model takes that slot back at the head decision,
// one cycle before the DONE event of a non-running entry (tsr_q.c). The occupancy seen by the push
// of the k-th entry of unit u at time t is k minus the slots freed at or before t; it is computed
// at the end (check()), when every free time is known.
struct recorder {
    uint32_t issued[TSR_NUM_UNITS] = {};   // ISSUED_SEQ, tracked from the events (0 at create)
    uint32_t base[TSR_NUM_UNITS] = {};     // ISSUED_SEQ at the window start
    uint32_t depth[TSR_NUM_UNITS] = {};
    uint32_t maxocc[TSR_NUM_UNITS] = {};
    uint32_t over = 0;                     // pushes that found the queue full (must never happen)
    uint32_t pidx_base = 0;
    std::vector<uint64_t> push_t[TSR_NUM_UNITS], free_t[TSR_NUM_UNITS];
    bool on = false, by_marks = false;     // by_marks: window = LAUNCH_BEGIN .. LAUNCH_END (RV)
    std::vector<pushrec> v;

    void attach(tsr_sim * sim) {
        const uint32_t qd = tsr_sim_csr_read(sim, TSR_CSR_Q_DEPTH);
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) depth[u] = (qd >> (5u * u)) & 31u;
        tsr_sim_add_listener(sim, &recorder::ev, this);
    }
    void begin(uint32_t pidx) {
        v.clear();
        memcpy(base, issued, sizeof base);
        pidx_base = pidx;
        on = true;
    }
    uint32_t rel(uint32_t t) const {
        const uint32_t u = t >> 28;
        if (u >= TSR_NUM_UNITS) return t;
        return (u << 28) | ((t - base[u]) & SEQ);
    }
    void check() {
        over = 0;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
            std::vector<uint64_t> f = free_t[u];
            std::sort(f.begin(), f.end());
            maxocc[u] = 0;
            for (size_t k = 0; k < push_t[u].size(); ++k) {
                const size_t freed = (size_t)(std::upper_bound(f.begin(), f.end(), push_t[u][k]) - f.begin());
                const uint32_t o = (uint32_t)(k + 1u - freed);
                if (o > maxocc[u]) maxocc[u] = o;
                if (o > depth[u]) ++over;
            }
        }
    }
    static void ev(void * ctx, tsr_sim * sim, const tsr_q_event * e) {
        (void)sim;
        recorder & r = *(recorder *)ctx;
        switch (e->kind) {
        case TSR_QEV_PUSH: {
            if (r.on) {
                pushrec p{};
                p.kind = e->kind;
                p.unit = e->unit;
                p.ticket = r.rel(e->ticket);
                p.rpidx = e->pidx - r.pidx_base;
                memcpy(p.w, e->w, sizeof p.w);
                p.dep0 = r.rel(e->dep[0]);
                p.dep1 = r.rel(e->dep[1]);
                p.push = e->push;
                r.v.push_back(p);
            }
            if (e->unit < TSR_NUM_UNITS) {
                r.issued[e->unit] = (r.issued[e->unit] + 1u) & SEQ;
                r.push_t[e->unit].push_back(e->time);
            }
            break;
        }
        case TSR_QEV_QCTRL: {
            if (r.on) {
                pushrec p{};
                p.kind = e->kind;
                p.unit = e->unit;
                p.ticket = r.rel(e->ticket);
                p.rpidx = e->pidx - r.pidx_base;
                p.w[0] = e->w[0];
                p.dep0 = p.dep1 = TSR_TICKET_NONE;
                r.v.push_back(p);
            }
            if (e->unit == TSR_UNIT_SYNC) {
                r.issued[TSR_UNIT_SYNC] = (r.issued[TSR_UNIT_SYNC] + 1u) & SEQ;
                r.push_t[TSR_UNIT_SYNC].push_back(e->time);
            }
            break;
        }
        case TSR_QEV_MARK_PUSH: {
            const uint32_t cls = e->w[0] >> 28;
            if (r.by_marks && cls == tsr::MARK_CLASS_LAUNCH_BEGIN) {
                r.begin(e->pidx);
            } else if (r.by_marks && cls == tsr::MARK_CLASS_LAUNCH_END) {
                r.on = false;
            } else if (r.on) {
                pushrec p{};
                p.kind = e->kind;
                p.unit = TSR_UNIT_MARKER;
                p.ticket = TSR_TICKET_NONE;
                p.rpidx = e->pidx - r.pidx_base;
                p.w[0] = e->w[0];
                p.dep0 = p.dep1 = TSR_TICKET_NONE;
                r.v.push_back(p);
            }
            break;
        }
        case TSR_QEV_START:
            if (e->unit < TSR_NUM_UNITS) r.free_t[e->unit].push_back(e->time);
            break;
        case TSR_QEV_DONE:
            if (!e->ran && e->unit < TSR_NUM_UNITS) r.free_t[e->unit].push_back(e->time - 1u);
            break;
        default:
            break;
        }
    }
};

bool streams_equal(const std::vector<pushrec> & a, const std::vector<pushrec> & b, const char * what) {
    if (a.size() != b.size()) {
        CHECK(0, "%s: %zu vs %zu pushes", what, a.size(), b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!(a[i] == b[i])) {
            CHECK(0, "%s: push %zu differs: unit %u/%u ticket %08x/%08x w0 %08x/%08x deps %08x,%08x/%08x,%08x push %08x/%08x",
                  what, i, a[i].unit, b[i].unit, a[i].ticket, b[i].ticket, a[i].w[0], b[i].w[0], a[i].dep0, a[i].dep1,
                  b[i].dep0, b[i].dep1, a[i].push, b[i].push);
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------------------------
// Scenario data (shared by the host and RV runs)
// -----------------------------------------------------------------------------------------------
struct stream_case {
    uint32_t fmt, N, K, tiles, ring;
    std::vector<uint32_t> want;    // tiles x N words: SILU(W . QACT(x_t))
};

uint32_t qa_lines_ref(uint32_t kind, uint32_t K) { return tsr_qa_lines(kind, K); }

// Writes x and the weights of case c and computes the expected outputs.
void stream_setup(tsr_sim * sim, stream_case & c, uint64_t seed) {
    rng_t r = {seed};
    const uint32_t kind = tsr_qa_kind_for_fmt(c.fmt);
    const size_t row_bytes = (size_t)(c.K / tsr_fmt_block_size(c.fmt)) * tsr_fmt_ggml_block_bytes(c.fmt);
    std::vector<uint8_t> blocks(row_bytes * c.N);
    for (uint32_t n = 0; n < c.N; ++n) gen_row_blocks(&r, c.fmt, c.K, blocks.data() + row_bytes * n);
    put_weights(sim, c.fmt, c.N, c.K, W_OFF, blocks.data());
    const uint32_t xl = tsr::f32_lines(c.K);
    std::vector<float> x((size_t)c.tiles * xl * 16u, 0.0f);
    for (uint32_t t = 0; t < c.tiles; ++t) gen_f32(&r, &x[(size_t)t * xl * 16u], c.K);
    tsr_sim_dram_write(sim, 0, (uint64_t)X_BEAT * 64u, x.data(), x.size() * 4u);
    std::vector<uint8_t> qa((size_t)qa_lines_ref(kind, c.K) * 64u), row((size_t)tsr_row_beats(c.fmt, c.K) * 64u);
    const uint32_t yl = tsr::f32_lines(c.N);
    c.want.assign((size_t)c.tiles * c.N, 0u);
    for (uint32_t t = 0; t < c.tiles; ++t) {
        tsr_qact(kind, &x[(size_t)t * xl * 16u], c.K, qa.data());
        std::vector<uint32_t> yw((size_t)yl * 16u, 0u), zw((size_t)yl * 16u, 0u);
        for (uint32_t n = 0; n < c.N; ++n) {
            tsr_repack_row(c.fmt, blocks.data() + row_bytes * n, c.K, row.data());
            yw[n] = f2u(tsr_dot_row(c.fmt, row.data(), c.K, qa.data()));
        }
        tsr_ref_vop(TSR_VOP_FUNC_SILU, TSR_VOP_CMODE_NONE, 1, c.N, zw.data(), yl, yw.data(), yl, NULL, 0, NULL);
        for (uint32_t n = 0; n < c.N; ++n) c.want[(size_t)t * c.N + n] = zw[n];
    }
    std::vector<uint8_t> ee((size_t)c.tiles * yl * 64u, 0xEE);
    tsr_sim_dram_write(sim, 1, (uint64_t)Y_BEAT * 64u, ee.data(), ee.size());
}
void stream_args(const stream_case & c, uint32_t a[7]) {
    a[0] = h0(X_BEAT);
    a[1] = W_OFF;
    a[2] = h1(Y_BEAT);
    a[3] = c.N;
    a[4] = c.K;
    a[5] = c.fmt | (c.tiles << 8);
    a[6] = c.ring;
}
void stream_check(tsr_sim * sim, const stream_case & c, const char * what) {
    const uint32_t yl = tsr::f32_lines(c.N);
    std::vector<uint32_t> got((size_t)c.tiles * yl * 16u);
    tsr_sim_dram_read(sim, 1, (uint64_t)Y_BEAT * 64u, got.data(), got.size() * 4u);
    uint32_t bad = 0;
    for (uint32_t t = 0; t < c.tiles; ++t)
        for (uint32_t n = 0; n < c.N; ++n)
            if (got[(size_t)t * yl * 16u + n] != c.want[(size_t)t * c.N + n]) ++bad;
    CHECK(bad == 0u, "%s: %u output words differ from SILU(W . QACT(x)) (tsr_ref)", what, bad);
}

// hdr_ring: the source pattern and the replay of the kernel's draws.
struct ring_case {
    uint32_t first, lines, nblk, seed, maxb;
};
void ring_setup(tsr_sim * sim) {
    std::vector<uint32_t> p(1024u * 16u);
    for (uint32_t i = 0; i < p.size(); ++i) p[i] = 0xA5000000u ^ (i * 2654435761u);
    tsr_sim_dram_write(sim, 0, (uint64_t)SRC_BEAT * 64u, p.data(), p.size() * 4u);
}
uint32_t ring_expect(const ring_case & rc, std::vector<uint32_t> & srcoff, std::vector<uint32_t> & nlines) {
    uint32_t s = rc.seed ? rc.seed : 1u, out = 0;
    auto next = [&]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; };
    for (uint32_t i = 0; i < rc.nblk; ++i) {
        const uint32_t n = 1u + next() % rc.maxb;
        next();                    // parity
        next();                    // keep
        const uint32_t soff = next() % 256u;
        srcoff.push_back(soff);
        nlines.push_back(n);
        out += n;
    }
    return out;
}
void ring_args(const ring_case & rc, uint32_t a[7]) {
    a[0] = h0(SRC_BEAT);
    a[1] = h1(DST_BEAT);
    a[2] = rc.first;
    a[3] = rc.lines;
    a[4] = rc.nblk;
    a[5] = rc.seed;
    a[6] = rc.maxb;
}
void ring_check(tsr_sim * sim, const ring_case & rc, int32_t ret, const char * what) {
    std::vector<uint32_t> so, nl;
    const uint32_t total = ring_expect(rc, so, nl);
    CHECK(ret == (int32_t)total, "%s: RET %d, expected %u lines", what, ret, total);
    std::vector<uint8_t> src(1024u * 64u), dst((size_t)total * 64u);
    tsr_sim_dram_read(sim, 0, (uint64_t)SRC_BEAT * 64u, src.data(), src.size());
    tsr_sim_dram_read(sim, 1, (uint64_t)DST_BEAT * 64u, dst.data(), dst.size());
    uint32_t out = 0, bad = 0;
    for (size_t i = 0; i < so.size(); ++i) {
        if (memcmp(&dst[(size_t)out * 64u], &src[(size_t)so[i] * 64u], (size_t)nl[i] * 64u) != 0) ++bad;
        out += nl[i];
    }
    CHECK(bad == 0u, "%s: %u of %zu blocks did not arrive intact", what, bad, so.size());
}

// hdr_xring: weights and the expected outputs y_t = W . QACT(64 x f32(c ^ (t << 4))).
struct xring_case {
    uint32_t ring, tiles, c;
};
const xring_case k_xring[2] = {{10, 24, 0x3F800000u}, {64, 24, 0xBE000000u}};
void xring_setup(tsr_sim * sim) {
    rng_t r = {0xC0FFEEu};
    const size_t row_bytes = (size_t)(64u / 32u) * tsr_fmt_ggml_block_bytes(TSR_FMT_Q8_0);
    std::vector<uint8_t> blocks(row_bytes * 32u);
    for (uint32_t n = 0; n < 32u; ++n) gen_row_blocks(&r, TSR_FMT_Q8_0, 64, blocks.data() + row_bytes * n);
    put_weights(sim, TSR_FMT_Q8_0, 32, 64, W_OFF2, blocks.data());
}
void xring_args(const xring_case & c, uint32_t a[7]) {
    a[0] = c.ring;
    a[1] = c.tiles;
    a[2] = W_OFF2;
    a[3] = h1(XR_BEAT);
    a[4] = c.c;
}
void xring_check(tsr_sim * sim, const xring_case & c, const char * what) {
    rng_t r = {0xC0FFEEu};
    const size_t row_bytes = (size_t)(64u / 32u) * tsr_fmt_ggml_block_bytes(TSR_FMT_Q8_0);
    std::vector<uint8_t> blocks(row_bytes * 32u), row((size_t)tsr_row_beats(TSR_FMT_Q8_0, 64) * 64u), qa(2u * 64u);
    for (uint32_t n = 0; n < 32u; ++n) gen_row_blocks(&r, TSR_FMT_Q8_0, 64, blocks.data() + row_bytes * n);
    std::vector<uint32_t> got((size_t)c.tiles * 32u);
    tsr_sim_dram_read(sim, 1, (uint64_t)XR_BEAT * 64u, got.data(), got.size() * 4u);
    uint32_t bad = 0;
    for (uint32_t t = 0; t < c.tiles; ++t) {
        float x[64];
        for (float & v : x) v = u2f(c.c ^ (t << 4));
        tsr_qact(TSR_QA_A32, x, 64, qa.data());
        for (uint32_t n = 0; n < 32u; ++n) {
            tsr_repack_row(TSR_FMT_Q8_0, blocks.data() + row_bytes * n, 64, row.data());
            if (got[(size_t)t * 32u + n] != f2u(tsr_dot_row(TSR_FMT_Q8_0, row.data(), 64, qa.data()))) ++bad;
        }
    }
    CHECK(bad == 0u, "%s: %u hdr_xring output words differ from W . QACT(c) (tsr_ref)", what, bad);
}

const stream_case k_stream[3] = {
    {TSR_FMT_Q8_0, 48, 256, 6, 40, {}},      // ring 40 lines: wraps, shrinks, software joins
    {TSR_FMT_Q4_K, 40, 512, 5, 64, {}},      // a tile per lap: every block reused each tile
    {TSR_FMT_Q8_0, 33, 128, 9, 4096, {}},    // no reuse
};
const ring_case k_ring[3] = {
    {0, 64, 120, 7, 24},          // small ring: wraps, failures and retries
    {1001, 257, 200, 99, 40},     // odd start
    {33, 4000, 300, 5, 4},        // many small blocks: the record table overflows (joined, forgotten)
};

// -----------------------------------------------------------------------------------------------
// Results of one launch, in the form both runs report
// -----------------------------------------------------------------------------------------------
struct result {
    uint32_t status = 0, err = 0;
    int32_t ret = 0;
    uint32_t err_ticket = TSR_TICKET_NONE, err_info = 0, err_w0 = 0, err_rpidx = 0;
    std::vector<pushrec> stream;
};

}  // namespace

// ===============================================================================================
// Host-only tests
// ===============================================================================================
#if defined(TSR_HDR_TEST_HOST)
namespace {

using namespace tsr;

// --- builders against the checker ---------------------------------------------------------------
uint32_t op_code(tsr_sim * s, const op & o, uint32_t * info = nullptr) {
    uint32_t inf = 0;
    const uint32_t e = tsr_sim_op(s, o.w, &inf);
    if (info) *info = inf;
    return e;
}
bool words_eq(const op & o, const tsr_insn & i) { return memcmp(o.w, i.w, sizeof o.w) == 0; }
op with_bits(op o, uint32_t word, uint32_t bits) {
    o.w[word] |= bits;
    return op_finish(o);
}
void expect(tsr_sim * s, const op & o, uint32_t code, uint32_t info, const char * what) {
    uint32_t inf = 0xDEADBEEFu;
    const uint32_t e = op_code(s, o, &inf);
    CHECK(e == code && (code == TSR_ERR_NONE || inf == info), "%s: code %u info 0x%x, expected %u 0x%x", what, e, inf, code,
          info);
}
// op::used = the encoding's words that hold a field (<OP>_WORDS), and every nonzero word is in it.
bool used_ok(const op & o, uint32_t words) {
    for (uint32_t i = 1; i < 8u; ++i)
        if (o.w[i] != 0u && !((o.used >> i) & 1u)) return false;
    return o.used == words;
}

void test_builders() {
    // U200 SPM (65,536 lines), 4 channels of 64 MiB (1 Mi beats), T_MAX 8, every CAPS bit the
    // simulator implements (VU, ATT, TOPK, DQROW).
    const uint32_t caps = TSR_CAPS_VU | TSR_CAPS_ATT | TSR_CAPS_TOPK | TSR_CAPS_DQROW;
    tsr_sim * s = mk_sim("fixed", 0, caps, 8, nullptr, 4, 64ull << 20);
    const uint32_t N = 65536u, LIMIT = 1u << 20;
    const uint32_t IL = TSR_ERR_ILLEGAL_OP, OP = TSR_ERR_OPERAND, SR = TSR_ERR_SPM_RANGE, DR = TSR_ERR_DRAM_RANGE;
    const uint32_t KR = TSR_ERR_K_RANGE, TR = TSR_ERR_T_RANGE, BF = TSR_ERR_BAD_FMT;

    // LD / ST / QACT / MVM / NOP / FENCE: the C encoders of tsr_isa.h, then the checker.
    {
        const op ld = LD(spm{10}, dram{3, 1000}, 64);
        CHECK(words_eq(ld, tsr_enc_ld(3, 10, 1000, 64)) && used_ok(ld, LD_WORDS), "LD = tsr_enc_ld");
        CHECK(words_eq(ST(spm{7}, dram{1, 5}, 2), tsr_enc_st(1, 7, 5, 2)), "ST = tsr_enc_st");
        CHECK(words_eq(QACT(qa::a256, spm{4}, spm{9}, 512, 3, 40), tsr_enc_qact(TSR_QA_A256, 4, 9, 512, 3, 40)), "QACT");
        CHECK(words_eq(MVM(spm{1}, spm{2}, wref{300, TSR_FMT_Q4_K}, 77, 1024, 8, 9, true),
                       tsr_enc_mvm(TSR_FMT_Q4_K, TSR_MVM_FLAG_ACC, 1, 2, 300, 77, 1024, 8, 9)), "MVM");
        CHECK(words_eq(NOP(), tsr_enc_nop()) && words_eq(FENCE(), tsr_enc_fence()), "NOP / FENCE");
        expect(s, ld, 0, 0, "LD legal");
        expect(s, LD(spm{N - 64}, dram{3, LIMIT - 64}, 64), 0, 0, "LD ends at both limits");
        expect(s, LD(spm{N - 63}, dram{0, 0}, 64), SR, N - 63, "LD SPM_RANGE");
        expect(s, LD(spm{0}, dram{0, LIMIT - 63}, 64), DR, LIMIT - 63, "LD DRAM_RANGE");
        expect(s, LD(spm{0}, dram{4, 0}, 1), IL, 0, "LD channel >= NUM_CH");
        expect(s, LD(spm{0}, dram{16, 0}, 1), IL, 0, "LD channel 16 (does not fit: poison)");
        expect(s, ST(spm{0}, dram{200, 0}, 1), IL, 0, "ST channel 200 (poison)");
        CHECK((LD(spm{0}, dram{16, 0}, 1).w[0] & 0xF00u) == 0u, "LD channel 16: not masked into channel 0 silently (poison set)");
        expect(s, with_bits(ld, 0, 1u << 16), IL, 0, "LD x16");
        expect(s, with_bits(ld, 4, 1u), IL, 0, "LD w4");
        expect(s, QACT(qa::a32, spm{0}, spm{100}, 4096, 8), 0, 0, "QACT legal (T 8)");
        expect(s, QACT(qa::a32, spm{0}, spm{100}, 4096 + 16, 1), KR, 0, "QACT K not a multiple of 32");
        expect(s, QACT(qa::a256, spm{0}, spm{100}, 4096 + 32, 1), KR, 0, "QACT A256 K not a multiple of 256");
        expect(s, QACT(qa::a32, spm{0}, spm{100}, 0, 1), KR, 0, "QACT K 0");
        expect(s, QACT(qa::a32, spm{0}, spm{100}, 64, 0), TR, 0, "QACT T 0");
        expect(s, QACT(qa::a32, spm{0}, spm{100}, 64, 9), TR, 0, "QACT T 9");
        expect(s, QACT(qa(2), spm{0}, spm{100}, 64, 1), IL, 0, "QACT kind 2");
        expect(s, QACT(qa(17), spm{0}, spm{100}, 64, 1), IL, 0, "QACT kind 17 (poison)");
        expect(s, QACT(qa::a32, spm{N - 1}, spm{100}, 64, 1), SR, N - 1, "QACT dst range");
        const op mv = MVM(spm{0}, spm{100}, wref{0, TSR_FMT_Q8_0}, 64, 256, 2);
        expect(s, mv, 0, 0, "MVM legal");
        expect(s, MVM(spm{0}, spm{100}, wref{0, 2}, 64, 256, 2), BF, 0, "MVM fmt 2");
        expect(s, MVM(spm{0}, spm{100}, wref{0, 200}, 64, 256, 2), BF, 0, "MVM fmt 200 saturates to 15: BAD_FMT");
        expect(s, MVM(spm{0}, spm{100}, wref{0, 0}, 64, 0x10020, 1), KR, 0, "MVM K 65568 (wider than 16 bits): K_RANGE");
        CHECK((MVM(spm{0}, spm{100}, wref{0, 0}, 64, 0x10020, 1).w[5] & 0xFFFFu) == 0xFFFFu, "MVM K saturates");
        expect(s, MVM(spm{0}, spm{100}, wref{0, 0}, 64, 256, 17), TR, 0, "MVM T 17 (wider than 4 bits): T_RANGE");
        expect(s, MVM(spm{0}, spm{100}, wref{0, 0}, 64, 256, 0), TR, 0, "MVM T 0");
        expect(s, MVM(spm{0}, spm{100}, wref{0, 1}, 64, 288, 1), KR, 0, "MVM Q4_K K not a multiple of 256");
        expect(s, MVM(spm{0}, spm{100}, wref{0, 0}, 64, 32768 + 32, 1), KR, 0, "MVM K > K_MAX");
        expect(s, MVM(spm{0}, spm{100}, wref{LIMIT, 0}, 64, 256, 1), DR, LIMIT, "MVM weight range");
        expect(s, NOP(), 0, 0, "NOP");
        expect(s, FENCE(), 0, 0, "FENCE");
        expect(s, with_bits(NOP(), 1, 1u), IL, 0, "NOP w1");
    }

    // VU builders (vu.hpp): fields read back with the generated macros tsr_vu.c decodes with.
    {
        const op v = VOP({.cmode = VOP_CMODE_SREG, .func = VOP_FUNC_MULC, .si = 3, .d = 100, .a = 200, .m = 896, .rows = 2,
                          .d_stride = 56, .a_stride = 60});
        CHECK(TSR_INSN_GET(v.w, TSR_VOP_CMODE) == 1u && TSR_INSN_GET(v.w, TSR_VOP_FUNC) == TSR_VOP_FUNC_MULC &&
              TSR_INSN_GET(v.w, TSR_VOP_SI) == 3u && TSR_INSN_GET(v.w, TSR_VOP_D) == 100u && TSR_INSN_GET(v.w, TSR_VOP_A) == 200u &&
              TSR_INSN_GET(v.w, TSR_VOP_M) == 896u && TSR_INSN_GET(v.w, TSR_VOP_ROWS) == 2u &&
              TSR_INSN_GET(v.w, TSR_VOP_D_STRIDE) == 56u && TSR_INSN_GET(v.w, TSR_VOP_A_STRIDE) == 60u && used_ok(v, VOP_WORDS),
              "VOP fields");
        expect(s, v, 0, 0, "VOP MULC legal");
        expect(s, VOP({.func = VOP_FUNC_ADD, .d = 1, .a = 2, .b = 3, .m = 16, .rows = 1}), 0, 0, "VOP ADD legal");
        expect(s, VOP({.func = 25, .d = 1, .a = 2, .m = 16, .rows = 1}), IL, 0, "VOP function 25");
        expect(s, VOP({.cmode = 3, .func = VOP_FUNC_COPY, .d = 1, .a = 2, .m = 16, .rows = 1}), IL, 0, "VOP cmode 3");
        expect(s, VOP({.func = VOP_FUNC_COPY, .si = 2, .d = 1, .a = 2, .m = 16, .rows = 1}), IL, 0, "VOP si without SREG");
        expect(s, VOP({.func = VOP_FUNC_COPY, .d = 1, .a = 2, .m = 32, .rows = 2, .d_stride = 1}), OP, 1, "VOP d_stride < L(m)");
        expect(s, VOP({.func = VOP_FUNC_COPY, .d = N - 1, .a = 2, .m = 32, .rows = 1}), SR, N - 1, "VOP d range");
        expect(s, VOP({.func = VOP_FUNC_COPY, .d = 0x1000000u, .a = 2, .m = 16, .rows = 1}), IL, 0,
               "VOP d = 2^24 (does not fit: poison, not line 0)");
        expect(s, VOP({.func = VOP_FUNC_COPY, .d = 1, .a = 2, .m = 16, .rows = 0x10000u}), IL, 0, "VOP rows 2^16 (poison)");
        expect(s, VOP({.func = 0x100u, .d = 1, .a = 2, .m = 16, .rows = 1}), IL, 0, "VOP func 256 (poison, not ADD)");

        const op rd = RED({.kind = TSR_RED_KIND_SUM, .to_s = 1, .sd = 5, .a = 300, .m = 100, .rows = 1});
        CHECK(TSR_INSN_GET(rd.w, TSR_RED_TO_S) == 1u && TSR_INSN_GET(rd.w, TSR_RED_SD) == 5u && TSR_INSN_GET(rd.w, TSR_RED_A) == 300u,
              "RED fields");
        expect(s, rd, 0, 0, "RED SUM to s5");
        expect(s, RED({.kind = TSR_RED_KIND_SUM, .to_s = 1, .sd = 5, .a = 300, .m = 100, .rows = 2, .a_stride = 7}), OP, 2,
               "RED TO_S with rows 2");
        expect(s, RED({.kind = TSR_RED_KIND_MAX, .post = TSR_RED_POST_RMS, .d = 1, .a = 300, .m = 100, .rows = 1}), IL, 0,
               "RED MAX with a post-op");
        expect(s, RED({.kind = TSR_RED_KIND_SUM, .sd = 16, .d = 1, .a = 3, .m = 16, .rows = 1}), IL, 0, "RED sd 16 (poison)");

        const op ro = ROPE({.mode = TSR_ROPE_MODE_NEOX, .sm = 1, .sts = 2, .d = 100, .src = 200, .pos = 300, .head_dim = 64,
                            .n_dims = 64, .n_heads = 2, .t = 3});
        CHECK(TSR_INSN_GET(ro.w, TSR_ROPE_HEAD_DIM) == 64u && TSR_INSN_GET(ro.w, TSR_ROPE_N_DIMS) == 64u &&
              TSR_INSN_GET(ro.w, TSR_ROPE_N_HEADS) == 2u && TSR_INSN_GET(ro.w, TSR_ROPE_T) == 3u, "ROPE fields");
        expect(s, ro, 0, 0, "ROPE NEOX legal");
        expect(s, ROPE({.sm = 1, .sts = 2, .d = 100, .src = 200, .pos = 300, .head_dim = 72, .n_dims = 64, .n_heads = 2, .t = 1}),
               OP, 72, "ROPE head_dim 72");
        expect(s, ROPE({.sm = 1, .sts = 2, .d = 100, .src = 200, .pos = 300, .table = 5, .head_dim = 64, .n_dims = 64, .n_heads = 2,
                        .t = 1}), IL, 0, "ROPE table without FF / ATAB");

        const op tk = TOPK({.k = 8, .d = 100, .a = 200, .m = 64, .rows = 3, .a_stride = 4});
        CHECK(TSR_INSN_GET(tk.w, TSR_TOPK_K) == 8u && TSR_INSN_GET(tk.w, TSR_TOPK_ROWS) == 3u, "TOPK fields");
        expect(s, tk, 0, 0, "TOPK legal");
        expect(s, TOPK({.k = 17, .d = 100, .a = 200, .m = 64, .rows = 1}), OP, 17, "TOPK k 17");
        expect(s, TOPK({.k = 32, .d = 100, .a = 200, .m = 64, .rows = 1}), IL, 0, "TOPK k 32 (does not fit: poison)");

        const op dq = DQROW({.fmt = TSR_FMT_Q4_K, .d = 100, .src = 300, .k = 512, .rows = 2});
        CHECK(TSR_INSN_GET(dq.w, TSR_DQROW_FMT) == 1u && dq.w[5] == 512u, "DQROW fields");
        expect(s, dq, 0, 0, "DQROW legal");
        expect(s, DQROW({.fmt = 2, .d = 100, .src = 300, .k = 512, .rows = 1}), BF, 0, "DQROW fmt 2");
        expect(s, DQROW({.fmt = TSR_FMT_Q4_K, .d = 100, .src = 300, .k = 100, .rows = 1}), KR, 0, "DQROW K 100");

        const op sp = SOP({.sub = SOP_SUB_MOVI, .sd = 4, .imm = 0x3F800000u});
        CHECK(TSR_INSN_GET(sp.w, TSR_SOP_SD) == 4u && TSR_INSN_GET(sp.w, TSR_SOP_IMM) == 0x3F800000u, "SOP fields");
        expect(s, sp, 0, 0, "SOP MOVI");
        expect(s, SOP({.sub = SOP_SUB_ADD, .sd = 1, .sa = 2, .sb = 3}), 0, 0, "SOP ADD");
        expect(s, SOP({.sub = SOP_SUB_MOVI, .sd = 4, .sa = 1, .imm = 1}), IL, 0, "SOP MOVI with sa");
        expect(s, SOP({.sub = SOP_SUB_LDS, .sd = 1, .addr = N * 16u}), SR, N * 16u, "SOP LDS beyond the SPM");
        expect(s, SOP({.sub = 15}), IL, 0, "SOP sub 15");

        // VMASK (M9): no handler in this simulator: the issue path refuses it (ILLEGAL_OP).
        const op vm = VMASK({.sub = 0, .x = 100, .win = 5, .k0 = 7, .q0 = 9, .m = 64, .rows = 2, .g = 1});
        CHECK(TSR_INSN_GET(vm.w, TSR_VMASK_X) == 100u && vm.w[2] == 5u && vm.w[3] == 7u && vm.w[4] == 9u &&
              TSR_INSN_GET(vm.w, TSR_VMASK_G) == 1u, "VMASK fields");
        expect(s, vm, IL, 0, "VMASK refused (no CAPS.VMASK)");
    }

    // AT builders (at.hpp): the cases of test_sim_at.c's checker test, built with the builders.
    {
        const op kv = KVST({.kv = TSR_KVST_KV_K, .n_rows_m1 = 2, .src = 100, .base = 1000, .e64 = 2, .lg = 8, .c0 = 5});
        CHECK(TSR_INSN_GET(kv.w, TSR_KVST_N_ROWS_M1) == 2u && TSR_INSN_GET(kv.w, TSR_KVST_E64) == 2u &&
              TSR_INSN_GET(kv.w, TSR_KVST_LG) == 8u && kv.w[4] == 5u, "KVST fields");
        expect(s, kv, 0, 0, "KVST legal");
        expect(s, KVST({.kv = 2, .n_rows_m1 = 2, .src = 100, .base = 1000, .e64 = 2, .lg = 8, .c0 = 5}), IL, 0, "KVST sub 2");
        expect(s, KVST({.kv = TSR_KVST_KV_K, .n_rows_m1 = 2, .src = 100, .base = 1000, .e64 = 0, .lg = 8, .c0 = 5}), KR, 0, "KVST E/64 0");
        expect(s, KVST({.kv = TSR_KVST_KV_K, .n_rows_m1 = 2, .src = 100, .base = 1000, .e64 = 2, .lg = 8, .c0 = 510}), OP, 510,
               "KVST c0 + n_rows > 16 C LG");
        expect(s, KVST({.kv = TSR_KVST_KV_K, .n_rows_m1 = 64, .src = 100, .base = 1000, .e64 = 2, .lg = 8, .c0 = 5}), IL, 0,
               "KVST n_rows_m1 64 (does not fit: poison)");
        expect(s, KVST({.kv = TSR_KVST_KV_V, .n_rows_m1 = 2, .src = 100, .base = LIMIT - 8u * 34u + 1u, .e64 = 2, .lg = 8, .c0 = 5}), DR,
               LIMIT - 8u * 34u + 1u, "KVST plane range");

        const op qk = ATTQK({.maxout = 1, .g_m1 = 3, .hg0 = 0, .dst = 100, .x = 1000, .base = 1000, .e64 = 2, .dcode = 0, .tq_m1 = 1,
                             .gs_m1 = 3, .u0 = 3, .n_groups = 13, .c_lo = 0, .c_hi = 208, .mdst = 2000});
        CHECK(TSR_INSN_GET(qk.w, TSR_ATTQK_G_M1) == 3u && TSR_INSN_GET(qk.w, TSR_ATTQK_TQ_M1) == 1u &&
              TSR_INSN_GET(qk.w, TSR_ATTQK_GS_M1) == 3u && TSR_INSN_GET(qk.w, TSR_ATTQK_N_GROUPS) == 13u &&
              TSR_INSN_GET(qk.w, TSR_ATTQK_U0) == 3u && qk.w[6] == (208u << 16) && qk.w[7] == 2000u, "ATTQK fields");
        expect(s, qk, 0, 0, "ATTQK legal");
        attqk_f q{.maxout = 1, .g_m1 = 3, .dst = 100, .x = 1000, .base = 1000, .e64 = 2, .tq_m1 = 1, .gs_m1 = 3, .u0 = 3,
                  .n_groups = 13, .c_hi = 208, .mdst = 2000};
        q.fmt = 1;
        expect(s, ATTQK(q), BF, 0, "ATTQK fmt 1");
        q.fmt = 0;
        q.hg0 = 2;
        expect(s, ATTQK(q), OP, 6, "ATTQK hg0 + GS > G");
        q.hg0 = 0;
        q.c_hi = 209;
        expect(s, ATTQK(q), OP, 209, "ATTQK c_hi > 16 n_groups");
        q.c_hi = 208;
        q.n_groups = 4096;
        expect(s, ATTQK(q), OP, 4096, "ATTQK n_groups 4096");
        q.n_groups = 0x10000;
        expect(s, ATTQK(q), IL, 0, "ATTQK n_groups 2^16 (does not fit: poison)");

        const op pv = ATTPV({.lsum = 1, .g_m1 = 3, .dst = 3000, .p = 100, .base = 5000, .e64 = 2, .tq_m1 = 1, .gs_m1 = 3, .u0 = 3,
                             .n_groups = 13, .lg = 8});
        CHECK(TSR_INSN_GET(pv.w, TSR_ATTPV_LSUM) == 1u && TSR_INSN_GET(pv.w, TSR_ATTPV_LG) == 8u, "ATTPV fields");
        expect(s, pv, 0, 0, "ATTPV legal");
        expect(s, ATTPV({.lsum = 1, .g_m1 = 3, .dst = 3000, .p = 100, .base = 5000, .e64 = 2, .tq_m1 = 1, .gs_m1 = 3, .u0 = 3,
                         .n_groups = 13, .lg = 3}), OP, 15, "ATTPV last group >= LG");

        const uint32_t sc = 0x3E000000u;
        const op as = ASMAX({.dst = 100, .src = 100, .mvec = 50, .n_lines = 13, .r = 16, .scale = sc});
        CHECK(as.w[3] == 50u && TSR_INSN_GET(as.w, TSR_ASMAX_N_LINES) == 13u && as.w[5] == 16u && as.w[6] == sc, "ASMAX fields");
        expect(s, as, 0, 0, "ASMAX legal");
        expect(s, ASMAX({.dst = 100, .src = 100, .mvec = 50, .n_lines = 13, .r = 16, .scale = 0xBF800000u}), OP, 0xBF800000u,
               "ASMAX negative scale");
        expect(s, ASMAX({.dst = 100, .src = 100, .mvec = 50, .n_lines = 0x10000, .r = 16, .scale = sc}), IL, 0,
               "ASMAX n_lines 2^16 (poison)");
        expect(s, AMASK({.dst = 100, .src = 100, .mask = 400, .n_lines = 13, .r = 16, .rpq = 8, .mstride = 13, .mdst = 50}), 0, 0,
               "AMASK legal");
        expect(s, AMASK({.dst = 100, .src = 100, .mask = 400, .n_lines = 13, .r = 16, .rpq = 0, .mstride = 13, .mdst = 50}), OP, 0,
               "AMASK rpq 0");
        expect(s, AMERGE({.orun = 100, .parts = 200, .mk = 50, .st = 60, .r = 16, .scale = sc}), 0, 0, "AMERGE legal");
        expect(s, AMERGE({.init = 1, .orun = 100, .parts = 200, .mk = 50, .st = 60, .r = 16, .scale = sc}), 0, 0, "AMERGE INIT");
        expect(s, AMERGE({.orun = N - 63, .parts = 200, .mk = 50, .st = 60, .r = 16, .scale = sc}), SR, N - 63, "AMERGE Orun range");
        expect(s, AMERGE({.init = 2, .orun = 100, .parts = 200, .mk = 50, .st = 60, .r = 16, .scale = sc}), IL, 0,
               "AMERGE init 2 (does not fit: poison)");
        expect(s, ANORM({.out = 100, .orun = 200, .l_run = 50, .r = 16, .dcode = 1}), 0, 0, "ANORM legal");
        expect(s, ANORM({.out = 100, .orun = 200, .l_run = 50, .r = 1025, .dcode = 1}), OP, 1025, "ANORM R 1025");
        expect(s, ANORM({.out = 100, .orun = 200, .l_run = 50, .r = 16, .dcode = 4}), IL, 0, "ANORM dcode 4 (poison)");
    }
    tsr_sim_destroy(s);
}

// --- tickets, deps, joins, errors ---------------------------------------------------------------
void test_tickets() {
    tsr_sim * s = mk_sim("fixed", 0, TSR_CAPS_VU);
    host_bind(s);
    recorder rec;
    rec.attach(s);
    rec.begin(pidx_now());

    uint32_t base[TSR_NUM_UNITS];
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) base[u] = issued_seq(u);
    const uint32_t p0 = pidx_now();
    const ticket t1 = issue(LD(spm{0}, dram{0, 1000}, 32));
    CHECK(t1.unit() == UNIT_DMA && t1.seq() == base[UNIT_DMA] && pidx_now() == p0 + 1u, "LD ticket {DMA, ISSUED_SEQ}");
    CHECK(!done(t1), "a fresh LD is not done before time advances (host backend: time moves only in waits)");
    const ticket t2 = issue(QACT(qa::a32, spm{40}, spm{0}, 512, 1), t1);
    CHECK(t2.unit() == UNIT_QACT && t2.seq() == base[UNIT_QACT], "QACT ticket");
    join(t2);
    CHECK(done(t1) && done(t2) && error(t1) == 0u && error(t2) == 0u, "join(t2): t2 and its dep t1 done, no error");
    CHECK(done(ticket{}) && error(ticket{}) == 0u, "none: always done, no error");
    const ticket j = join2(t1, t2);
    CHECK(j.unit() == UNIT_SYNC && pidx_now() == p0 + 3u, "join2: one SYNC ticket");
    join_irq(j);
    CHECK(done(j) && hw::cw_read(CW_WAIT_TICKET) == TICKET_NONE, "join_irq (host: a join), WAIT_TICKET disarmed");
    const uint32_t ps = pidx_now();
    const ticket qc = qclear();
    CHECK(qc.unit() == UNIT_SYNC && pidx_now() == ps + 1u, "qclear: a SYNC ticket and a push index");
    join(qc);
    qhold(true);
    qhold(false);
    qabort();
    trace_mark(0x12345u);
    CHECK(pidx_now() == ps + 1u, "QHOLD / QABORT / TRACE_MARK take no push index");
    join(qclear());   // clears QABORT's error states
    CHECK(err_units() == 0u, "after QABORT + QCLEAR: no unit in error");

    // The push: words of op::used only (a clear bit sends 0), WMASK = used, REC, deps as given.
    op o = LD(spm{1}, dram{0, 2}, 3);
    o.w[5] = 0xDEADu;                     // not in used: must be sent as 0
    // flags: REC, a WMASK bit (ignored: [15:8] comes from op::used) and bit 16 (passed, ignored by
    // the hardware).
    const ticket t3 = issue(o, deps{t1, t2}, PUSH_REC | 0x0000'8000u | 0x0001'0000u);
    join(t3);
    const pushrec & pr = rec.v.back();
    CHECK(pr.w[5] == 0u && pr.w[1] == 1u && pr.w[3] == 3u, "words outside op::used are sent as 0");
    CHECK(pr.push == ((uint32_t(o.used) << 8) | PUSH_REC | 0x0001'0000u), "ISSUE_PUSH = used << 8 | flags: 0x%08x", pr.push);
    CHECK(pr.dep0 == rec.rel(t1.v) && pr.dep1 == rec.rel(t2.v), "deps pushed as given");
    const ticket t4 = issue(ST(spm{1}, dram{1, 50}, 3));
    CHECK(rec.v.back().dep0 == TSR_TICKET_NONE && rec.v.back().dep1 == TSR_TICKET_NONE, "no deps: DEP0 / DEP1 none");
    join(t4);

    // after(): none dropped, same unit folded to the later ticket, distinct units kept.
    static_assert(after().a.none() && after().b.none());
    static_assert(after(make_ticket(0, 5), make_ticket(0, 9)).a.v == make_ticket(0, 9).v && after(make_ticket(0, 5), make_ticket(0, 9)).b.none());
    static_assert(after(make_ticket(0, 0x0FFFFFFFu), make_ticket(0, 1)).a.v == make_ticket(0, 1).v, "wrap-safe later");
    static_assert(after(ticket{}, make_ticket(2, 3)).a.v == make_ticket(2, 3).v && after(ticket{}, make_ticket(2, 3)).b.none());
    static_assert(after(make_ticket(1, 3), make_ticket(2, 3)).b.v == make_ticket(2, 3).v);
    static_assert(handle(from_handle(0x3123'4567u)) == 0x3123'4567u && from_handle(0x3123'4567u).ch == 3u);

    // Refused opcodes go to SYNC (no CAPS.TOPK here): the credit checked is SYNC's, the ticket SYNC's.
    const ticket tt = issue(TOPK({.k = 2, .d = 1, .a = 2, .m = 16, .rows = 1}));
    join(tt);
    CHECK(tt.unit() == UNIT_SYNC && error(tt) == ERR_ILLEGAL_OP, "TOPK without CAPS.TOPK: SYNC, ILLEGAL_OP");
    join(qclear());
    CHECK(host().no_credit_pushes == 0u, "no push without a credit");
    tsr_sim_destroy(s);

    // A VU op on a simulator whose VU is not built: SYNC, ILLEGAL_OP, and no wait on VU's credit 0.
    const uint8_t qd[6] = {8, 4, 4, 0, 0, 4};
    s = mk_sim("fixed", 0, 0, 1, qd);
    host_bind(s);
    const ticket tv = issue(VOP({.func = VOP_FUNC_COPY, .d = 1, .a = 2, .m = 16, .rows = 1}));
    join(tv);
    CHECK(tv.unit() == UNIT_SYNC && error(tv) == ERR_ILLEGAL_OP, "VOP with the VU absent: SYNC, ILLEGAL_OP (no hang)");
    host_bind(nullptr);
    tsr_sim_destroy(s);
}

// --- group --------------------------------------------------------------------------------------
void test_group() {
    tsr_sim * s = mk_sim("fixed", 0, TSR_CAPS_VU);
    host_bind(s);
    auto mk = [&](uint32_t u) -> ticket {
        switch (u) {
        case UNIT_DMA: return issue(LD(spm{0}, dram{0, 0}, 256));
        case UNIT_QACT: return issue(QACT(qa::a32, spm{600}, spm{300}, 2048, 1));
        case UNIT_MVM: return issue(MVM(spm{700}, spm{600}, wref{0, 0}, 256, 256, 1));
        case UNIT_VU: return issue(VOP({.func = VOP_FUNC_COPY, .d = 800, .a = 300, .m = 1024, .rows = 1}));
        default: return issue(NOP(), issue(LD(spm{1000}, dram{1, 0}, 128)));
        }
    };
    struct gcase {
        uint32_t units;      // bit u: a ticket of unit u
        uint32_t nops;       // SYNC NOPs as_deps pushes
        uint32_t ndeps;      // non-none deps it returns
    };
    const uint32_t D = 1u << UNIT_DMA, Q = 1u << UNIT_QACT, M = 1u << UNIT_MVM, V = 1u << UNIT_VU, Y = 1u << UNIT_SYNC;
    const gcase cases[] = {
        {0, 0, 0}, {D, 0, 1}, {D | Q, 0, 2}, {D | Y, 0, 2}, {D | Q | M, 1, 2}, {D | Q | Y, 1, 1},
        {D | Q | M | V, 2, 1}, {D | Q | M | Y, 1, 2}, {D | Q | M | V | Y, 2, 1},
    };
    for (const gcase & c : cases) {
        group g;
        std::vector<ticket> all;
        for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) {
            if (!((c.units >> u) & 1u)) continue;
            const ticket a = mk(u), b = mk(u);       // two of the same unit: the later one is kept
            g.add(b);
            g.add(a);
            all.push_back(a);
            all.push_back(b);
            CHECK(g.t[u] == later(a, b), "units 0x%02x: group keeps the later ticket of unit %u", c.units, u);
        }
        const uint32_t p0 = pidx_now();
        const deps d = g.as_deps();
        const uint32_t nops = pidx_now() - p0;
        const uint32_t nd = (d.a.none() ? 0u : 1u) + (d.b.none() ? 0u : 1u);
        CHECK(nops == c.nops && nd == c.ndeps, "units 0x%02x: %u NOPs %u deps, expected %u / %u", c.units, nops, nd, c.nops,
              c.ndeps);
        const deps d2 = g.as_deps();
        CHECK(pidx_now() - p0 == nops && d2.a == d.a && d2.b == d.b, "units 0x%02x: as_deps again pushes nothing", c.units);
        // The deps imply every ticket of the group (I1): after joining them, all are done.
        join(d.a);
        join(d.b);
        bool ok = true;
        for (const ticket & t : all) ok = ok && done(t);
        CHECK(ok, "units 0x%02x: done(deps) implies every ticket of the group", c.units);
        join_all();
    }
    group g;
    g.add(mk(UNIT_DMA));
    g.add(mk(UNIT_QACT));
    const uint32_t p0 = pidx_now();
    const ticket one = g.as_ticket();
    CHECK(one.unit() == UNIT_SYNC && pidx_now() == p0 + 1u && g.count() == 1u, "as_ticket of two units: one join2");
    CHECK(g.as_ticket() == one && pidx_now() == p0 + 1u, "as_ticket again: the same ticket, no push");
    join_all();
    CHECK(host().no_credit_pushes == 0u, "no push without a credit");
    host_bind(nullptr);
    tsr_sim_destroy(s);
}

// --- spm_ring against a line-level model ---------------------------------------------------------
struct ring_model {
    uint32_t first = 0, end = 0, head = 0;
    std::vector<std::vector<uint32_t>> pend;   // per line: release tickets not yet covered
    std::vector<int> owner;                    // per line: unreleased block id, or -1
    struct blk { uint32_t lo, hi, start; bool released; };
    std::vector<blk> blocks;

    void init(uint32_t f, uint32_t n) {
        first = f;
        end = f + n;
        head = f;
        pend.assign(end, {});
        owner.assign(end, -1);
        blocks.clear();
    }
    static uint32_t pad(uint32_t line, uint32_t parity) { return (parity < 2u && (line & 1u) != parity) ? 1u : 0u; }
    // Predicted placement: start, the owned range [lo, hi), the skipped tail [skip_lo, end) (wrap),
    // or false when the ring must refuse.
    bool predict(uint32_t n, uint32_t parity, uint32_t & s, uint32_t & lo, uint32_t & hi, bool & wrap) const {
        if (n == 0u || parity > 2u || n > end - first) return false;
        s = head + pad(head, parity);
        wrap = false;
        if (s > end || end - s < n) {
            wrap = true;
            s = first + pad(first, parity);
            if (s > end || end - s < n) return false;
        }
        lo = wrap ? first : head;
        hi = s + n;
        for (uint32_t l = lo; l < hi; ++l)
            if (owner[l] >= 0) return false;
        if (wrap)
            for (uint32_t l = head; l < end; ++l)
                if (owner[l] >= 0) return false;
        return true;
    }
};

bool covered(const deps & d, ticket t) {
    for (const ticket & x : {d.a, d.b})
        if (!x.none() && x.unit() == t.unit() && same_or_after(x, t)) return true;
    return false;
}

// One random sequence; returns the must_wait values (relative tickets) for the determinism check.
std::vector<uint32_t> ring_sequence(const char * sched, uint32_t seed, uint32_t first, uint32_t lines, uint32_t steps,
                                    uint32_t maxb, uint32_t * n_fail, uint32_t * n_wrap, uint32_t * n_join) {
    tsr_sim * s = mk_sim(sched, 0, TSR_CAPS_VU);
    host_bind(s);
    uint32_t base[TSR_NUM_UNITS];
    for (uint32_t u = 0; u < TSR_NUM_UNITS; ++u) base[u] = issued_seq(u);
    auto rel = [&](ticket t) { return t.none() ? t.v : make_ticket(t.unit(), t.seq() - base[t.unit()]).v; };
    rng_t r = {seed};
    static spm_ring ring;                      // as the firmware keeps it (static storage)
    ring.init(first, lines);
    ring_model m;
    m.init(first, lines);
    std::vector<uint32_t> held;                // block ids allocated and not released
    std::vector<uint32_t> out;
    ticket last{};
    const uint32_t S0 = 60000u;                // scratch for the ticket-producing ops
    for (uint32_t step = 0; step < steps; ++step) {
        const bool do_alloc = held.empty() || (held.size() < 3u && rng_below(&r, 100) < 60u);
        if (do_alloc) {
            const uint32_t n = rng_below(&r, 50) == 0 ? lines + 1u : 1u + rng_below(&r, maxb);
            const uint32_t parity = rng_below(&r, 50) == 0 ? 3u : rng_below(&r, 3);
            uint32_t ps = 0, plo = 0, phi = 0;
            bool pwrap = false;
            const bool ok = m.predict(n, parity, ps, plo, phi, pwrap);
            deps w;
            const spm b = ring.alloc(n, parity, &w);
            CHECK(valid(b) == ok, "ring %u+%u step %u: alloc(%u, %u) %s, the model says %s", first, lines, step, n, parity,
                  valid(b) ? "succeeded" : "failed", ok ? "success" : "failure");
            if (!valid(b) || !ok) {
                if (!valid(b)) ++*n_fail;
                if (valid(b) != ok) break;
                continue;
            }
            CHECK(b.line == ps, "ring step %u: block at %u, the model says %u", step, b.line, ps);
            CHECK(parity == 2u || (b.line & 1u) == parity, "ring step %u: parity", step);
            CHECK(b.line >= first && b.line + n <= first + lines, "ring step %u: inside the ring", step);
            // Every older release ticket of a reused line: covered by must_wait or done (joined).
            uint32_t unc = 0;
            for (uint32_t l = plo; l < phi; ++l) {
                for (uint32_t tv : m.pend[l])
                    if (!covered(w, ticket{tv}) && !done(ticket{tv})) ++unc;
                m.pend[l].clear();
                m.owner[l] = (int)m.blocks.size();
            }
            if (pwrap) {
                ++*n_wrap;
                for (uint32_t l = m.head; l < m.end; ++l) {
                    for (uint32_t tv : m.pend[l])
                        if (!done(ticket{tv})) ++unc;
                    m.pend[l].clear();
                }
            }
            CHECK(unc == 0u, "ring step %u: %u release tickets of reused lines neither in must_wait nor done", step, unc);
            CHECK(w.a.none() || w.b.none() || w.a.unit() != w.b.unit(), "ring step %u: must_wait folded per unit", step);
            m.head = phi;
            held.push_back((uint32_t)m.blocks.size());
            m.blocks.push_back({plo, phi, ps, false});
            out.push_back(rel(w.a));
            out.push_back(rel(w.b));
        } else {
            // Release a random held block after a fresh op on a random unit (its "last reader").
            const uint32_t k = rng_below(&r, (uint32_t)held.size());
            const uint32_t id = held[k];
            held.erase(held.begin() + k);
            ticket t;
            switch (rng_below(&r, 6)) {
            case 0: t = issue(LD(spm{S0}, dram{0, 0}, 1 + rng_below(&r, 64))); break;
            case 1: t = issue(QACT(qa::a32, spm{S0 + 100}, spm{S0}, 32u * (1u + rng_below(&r, 16)), 1)); break;
            case 2: t = issue(MVM(spm{S0 + 200}, spm{S0 + 100}, wref{0, 0}, 16u * (1u + rng_below(&r, 8)), 32, 1)); break;
            case 3: t = issue(VOP({.func = VOP_FUNC_COPY, .d = S0 + 300, .a = S0, .m = 16u * (1u + rng_below(&r, 32)), .rows = 1})); break;
            case 4: t = issue(NOP(), last); break;
            default: t = ticket{}; break;                                   // released at once
            }
            if (!t.none()) last = t;
            ring.release(spm{m.blocks[id].start}, t);
            m.blocks[id].released = true;
            for (uint32_t l = m.blocks[id].lo; l < m.blocks[id].hi; ++l) {
                if (m.owner[l] == (int)id) {
                    m.owner[l] = -1;
                    if (!t.none()) m.pend[l].push_back(t.v);
                }
            }
            out.push_back(0xFFFF0000u | id);
        }
    }
    *n_join += ring.joins();
    join_all();
    CHECK(host().no_credit_pushes == 0u, "ring: no push without a credit");
    host_bind(nullptr);
    tsr_sim_destroy(s);
    return out;
}

void test_ring() {
    struct rc {
        uint32_t first, lines, steps, maxb;
    };
    const rc cases[] = {{0, 64, 600, 24}, {1001, 257, 800, 40}, {33, 4000, 1500, 4}, {7, 3, 200, 2}, {100, 1, 100, 1},
                        {0, 1000, 800, 300}};
    uint32_t fails = 0, wraps = 0, joins = 0;
    for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const rc & c = cases[i];
        for (uint32_t seed = 1; seed <= 3; ++seed) {
            const std::vector<uint32_t> a =
                ring_sequence("fixed", 1000u * i + seed, c.first, c.lines, c.steps, c.maxb, &fails, &wraps, &joins);
            if (seed == 1) {
                uint32_t f2 = 0, w2 = 0, j2 = 0;
                const std::vector<uint32_t> b =
                    ring_sequence("random:11", 1000u * i + seed, c.first, c.lines, c.steps, c.maxb, &f2, &w2, &j2);
                const std::vector<uint32_t> d =
                    ring_sequence("random:12,0.25,8", 1000u * i + seed, c.first, c.lines, c.steps, c.maxb, &f2, &w2, &j2);
                CHECK(a == b && a == d, "ring case %u: the same alloc / release sequence gives the same deps under 3 schedules", i);
            }
        }
    }
    CHECK(fails > 0u && wraps > 0u && joins > 0u, "ring: failures (%u), wraps (%u) and software joins (%u) exercised", fails,
          wraps, joins);
    printf("ring model: %u refusals, %u wraps, %u software joins\n", fails, wraps, joins);

    // Releasing a block the ring does not hold, or twice, is a firmware bug: tsr::panic (the host
    // build aborts). Checked in child processes.
    for (int mode = 0; mode < 2; ++mode) {
        fflush(stdout);
        fflush(stderr);
        const pid_t pid = fork();
        if (pid == 0) {
            const int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) dup2(devnull, 2);
            static spm_ring r;
            r.init(0, 100);
            if (mode == 0) {
                r.release(spm{3}, ticket{});            // never allocated
            } else {
                deps w;
                const spm b = r.alloc(4, 2, &w);
                r.release(b, ticket{});
                r.release(b, ticket{});                 // twice
            }
            _exit(0);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT, "ring: %s release panics (status 0x%x)",
              mode == 0 ? "a foreign" : "a double", st);
    }
}

// --- the kernels through host_run ----------------------------------------------------------------
struct host_env {
    tsr_sim * sim = nullptr;
    recorder rec;
    uint32_t seq = 0;
    explicit host_env(const char * sched) {
        sim = mk_sim(sched, 0, TSR_CAPS_VU);
        host_bind(sim);
        rec.attach(sim);
    }
    ~host_env() {
        host_bind(nullptr);
        tsr_sim_destroy(sim);
    }
    result run(tsr_kernel_fn fn, const uint32_t a[7], bool stop = false) {
        tsr_args args{};
        memcpy(args.a, a, sizeof args.a);
        args.seq = seq++;
        rec.begin(pidx_now());
        const host_result h = host_run(fn, args, nullptr, 0, stop);
        rec.on = false;
        result r;
        r.ret = h.ret;
        r.err = h.err_code;
        r.err_ticket = h.err_ticket;
        r.err_info = h.err_info;
        r.err_w0 = h.err_w0;
        r.err_rpidx = h.err_code ? h.err_pidx - h.pidx0 : 0u;
        r.stream = rec.v;
        while (!r.stream.empty() && r.stream.back().rpidx >= h.pidx_end - 1u - h.pidx0) r.stream.pop_back();   // the shim's FENCE
        if (err_units() != 0u) join(qclear());     // as the runtime's R9
        return r;
    }
};

void test_kernels_host() {
    host_env e("fixed");
    for (size_t i = 0; i < 3; ++i) {
        stream_case c = k_stream[i];
        stream_setup(e.sim, c, 77u + i);
        uint32_t a[7];
        stream_args(c, a);
        const result r = e.run(hdr_stream, a);
        CHECK(r.ret >= 0 && r.err == 0u, "hdr_stream case %zu: RET %d err %u", i, r.ret, r.err);
        stream_check(e.sim, c, "hdr_stream host");
    }
    {
        const uint32_t a[7] = {h0(CRED_BEAT), 16384, 40, 0, 0, 0, 0};
        const result r = e.run(hdr_credit, a);
        CHECK(r.ret == (int32_t)(6u + 5u * 40u), "hdr_credit: RET %d pushes", r.ret);
    }
    {
        const uint32_t a[7] = {h0(BAD_BEAT), h0(GOOD_BEAT), W_OFF, 0, 0, 0, 0};
        const result r = e.run(hdr_err, a);
        CHECK(r.ret == 0x3FF, "hdr_err: checks 0x%03x (want 0x3FF)", r.ret);
        CHECK(r.err == TSR_ERR_DRAM_RANGE && r.err_info == BAD_BEAT && r.err_rpidx == 3u, "hdr_err: first error (shim R7)");
    }
    {
        const uint32_t a[7] = {h0(BAD_BEAT), h0(GOOD_BEAT), 0, 0, 0, 0, 0};
        const result r = e.run(hdr_qclear, a);
        CHECK(r.ret == 0x1F && r.err == 0u, "hdr_qclear: checks 0x%02x err %u", r.ret, r.err);
    }
    {
        const uint32_t a[7] = {h0(SRC_BEAT), 4096, 0, 0, 0, 0, 0};
        const result r = e.run(hdr_irq, a);
        CHECK(r.ret == 7, "hdr_irq: checks 0x%x", r.ret);
    }
    ring_setup(e.sim);
    for (size_t i = 0; i < 3; ++i) {
        uint32_t a[7];
        ring_args(k_ring[i], a);
        const result r = e.run(hdr_ring, a);
        ring_check(e.sim, k_ring[i], r.ret, "hdr_ring host");
    }
    xring_setup(e.sim);
    for (size_t i = 0; i < 2; ++i) {
        uint32_t a[7] = {};
        xring_args(k_xring[i], a);
        const result r = e.run(hdr_xring, a);
        CHECK(r.ret == 0 && r.err == 0u, "hdr_xring case %zu: RET %d err %u", i, r.ret, r.err);
        xring_check(e.sim, k_xring[i], "hdr_xring host");
    }
    {
        const uint32_t a[7] = {1000000, 0, 0, 0, 0, 0, 0};
        const result r = e.run(hdr_stop, a, true);
        CHECK(r.ret == 1, "hdr_stop: stop_requested() seen (RET %d)", r.ret);
        const uint32_t b[7] = {10, 0, 0, 0, 0, 0, 0};
        CHECK(e.run(hdr_stop, b, false).ret == 0, "hdr_stop without a stop: runs out");
    }
    e.rec.check();
    for (uint32_t u : {UNIT_DMA, UNIT_QACT, UNIT_MVM, UNIT_VU, UNIT_SYNC})
        CHECK(e.rec.maxocc[u] == e.rec.depth[u], "unit %u: the queue filled (max %u of Q_DEPTH %u)", u, e.rec.maxocc[u],
              e.rec.depth[u]);
    CHECK(e.rec.over == 0u, "no push into a full queue (%u)", e.rec.over);
    CHECK(host().no_credit_pushes == 0u && host().pushes > 0u, "no push without a credit (%llu pushes)",
          (unsigned long long)host().pushes);
}

}  // namespace

int main() {
    test_builders();
    test_tickets();
    test_group();
    test_ring();
    test_kernels_host();
    return test_finish("fw_hdr_host");
}
#endif   // TSR_HDR_TEST_HOST

// ===============================================================================================
// RV tests: the fw_hdr image on tsr_rv, against the host build
// ===============================================================================================
#if defined(TSR_HDR_TEST_RV)
namespace {

// The launches of one run, in order. stop: the soft-stop launch (fixed schedule only).
enum { L_STREAM0, L_STREAM1, L_STREAM2, L_CREDIT, L_ERR, L_QCLEAR, L_IRQ, L_RING0, L_RING1, L_RING2, L_XRING0, L_XRING1, L_N };
const char * const k_lname[L_N] = {"hdr_stream", "hdr_stream", "hdr_stream", "hdr_credit", "hdr_err",   "hdr_qclear",
                                   "hdr_irq",    "hdr_ring",   "hdr_ring",   "hdr_ring",   "hdr_xring", "hdr_xring"};

void launch_args(uint32_t l, uint32_t a[7]) {
    memset(a, 0, 7 * sizeof(uint32_t));
    switch (l) {
    case L_STREAM0: case L_STREAM1: case L_STREAM2: stream_args(k_stream[l - L_STREAM0], a); break;
    case L_CREDIT: a[0] = h0(CRED_BEAT); a[1] = 16384; a[2] = 40; break;
    case L_ERR: a[0] = h0(BAD_BEAT); a[1] = h0(GOOD_BEAT); a[2] = W_OFF; break;
    case L_QCLEAR: a[0] = h0(BAD_BEAT); a[1] = h0(GOOD_BEAT); break;
    case L_IRQ: a[0] = h0(SRC_BEAT); a[1] = 4096; break;
    case L_XRING0: case L_XRING1: xring_args(k_xring[l - L_XRING0], a); break;
    default: ring_args(k_ring[l - L_RING0], a); break;
    }
}

tsr_kernel_fn host_fn(uint32_t l) {
    switch (l) {
    case L_STREAM0: case L_STREAM1: case L_STREAM2: return hdr_stream;
    case L_CREDIT: return hdr_credit;
    case L_ERR: return hdr_err;
    case L_QCLEAR: return hdr_qclear;
    case L_IRQ: return hdr_irq;
    case L_XRING0: case L_XRING1: return hdr_xring;
    default: return hdr_ring;
    }
}

// Writes every launch's inputs (the same in the host and RV runs).
void setup_all(tsr_sim * sim, stream_case sc[3]) {
    for (uint32_t i = 0; i < 3; ++i) {
        sc[i] = k_stream[i];
        // Each stream case has its own x / weights: set up just before its launch (setup_stream).
    }
    ring_setup(sim);
    xring_setup(sim);
}

struct rv_env {
    tsr_sim * sim = nullptr;
    recorder rec;
    std::string tag;
    uint32_t seq = 0, lcq_rd = 0;
    std::vector<std::array<uint32_t, 8>> recs;
    uint32_t kidx[16] = {};
    bool ok = false;

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
        // Kernel indices by name (rt.hpp: the table is sorted by (rank, name); tools find kernels by name).
        uint32_t ks = 0, ke = 0;
        int found = 0;
        if (tsr_rv_elf_symbol(&elf, "__tsr_kernels_start", &ks) == 0 && tsr_rv_elf_symbol(&elf, "__tsr_kernels_end", &ke) == 0) {
            for (uint32_t s = 0; s < elf.nseg; ++s) {
                const tsr_rv_elf_seg * g = &elf.seg[s];
                if (ks < g->paddr || ke > g->paddr + g->filesz) continue;
                for (uint32_t a = ks, i = 0; a + sizeof(tsr_kernel_desc) <= ke; a += sizeof(tsr_kernel_desc), ++i) {
                    const char * name = (const char *)g->data + (a - g->paddr) + offsetof(tsr_kernel_desc, name);
                    for (uint32_t l = 0; l < L_N; ++l)
                        if (strncmp(name, k_lname[l], 24) == 0) kidx[l] = i, ++found;
                    if (strncmp(name, "hdr_stop", 24) == 0) kidx[L_N] = i, ++found;
                }
            }
        }
        tsr_rv_elf_free(&elf);
        CHECK(found == (int)L_N + 1, "%s: the hdr_* kernels in the image's .tsr_kernels (%d)", tag.c_str(), found);
        if (found != (int)L_N + 1) return false;
        tsr_sim_csr_write(sim, TSR_CSR_LOG_CFG, TSR_LOG_CFG_ENABLE | TSR_PUT(TSR_LOG_CFG_LEVEL, TSR_LOG_HDR_LEVEL_INFO));
        tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0u);
        tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
        for (int it = 0; it < 2000 && recs.empty(); ++it) {
            tsr_sim_advance(sim, 100000);
            if (!consume()) return false;
        }
        ok = !recs.empty() && TSR_GET(recs[0][0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_HELLO;
        CHECK(ok, "%s: HELLO after boot", tag.c_str());
        recs.clear();
        return ok;
    }
    bool consume() {
        const uint32_t rt = tsr_sim_csr_read(sim, TSR_CSR_RT_STATE);
        if (TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) {
            CHECK(0, "%s: the firmware trapped (RT_INFO 0x%08x, pc 0x%08x)", tag.c_str(), tsr_sim_csr_read(sim, TSR_CSR_RT_INFO),
                  tsr_sim_rv_pc(sim));
            return false;
        }
        const uint32_t wr = tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR);
        for (uint32_t k = lcq_rd; k != wr; ++k) {
            std::array<uint32_t, 8> d;
            for (uint32_t j = 0; j < 8u; ++j) d[j] = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(k % TSR_LCQ_DEPTH, j));
            recs.push_back(d);
        }
        lcq_rd = wr;
        tsr_sim_csr_write(sim, TSR_CSR_LCQ_RD, wr);
        tsr_sim_csr_write(sim, TSR_CSR_LOG_RD, tsr_sim_csr_read(sim, TSR_CSR_LOG_WR));   // never compared (SW §4.7)
        return true;
    }
    // Pushes a launch of kernel `k` (kidx index) and waits for its answer. stop_after_kernel: write
    // LQ_STOP = its sequence once RT_STATE shows the kernel running.
    std::array<uint32_t, 8> launch(uint32_t k, const uint32_t a[7], bool stop_after_kernel = false) {
        std::array<uint32_t, 8> res{};
        if (!ok) return res;
        uint32_t d[8] = {TSR_DESC_TYPE_LAUNCH | TSR_PUT(TSR_DESC_KERNEL, kidx[k]), a[0], a[1], a[2], a[3], a[4], a[5], a[6]};
        for (uint32_t i = 0; i < 8u; ++i) tsr_sim_csr_write(sim, TSR_CSR_LQ_W(i), d[i]);
        tsr_sim_csr_write(sim, TSR_CSR_LQ_PUSH, 1u);
        const uint32_t s = seq++;
        bool stopped = false;
        for (int it = 0; it < 400000; ++it) {
            for (size_t i = 0; i < recs.size(); ++i) {
                const uint32_t ty = TSR_GET(recs[i][0], TSR_LCQ_R0_TYPE);
                if ((ty == TSR_LCQ_R0_TYPE_DONE || ty == TSR_LCQ_R0_TYPE_TRAP) && recs[i][1] == s) {
                    res = recs[i];
                    recs.erase(recs.begin() + (long)i);
                    CHECK(ty == TSR_LCQ_R0_TYPE_DONE, "%s: launch %u answered with a TRAP record", tag.c_str(), s);
                    if (ty != TSR_LCQ_R0_TYPE_DONE) ok = false;
                    return res;
                }
            }
            if (stop_after_kernel && !stopped &&
                TSR_GET(tsr_sim_csr_read(sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_KERNEL) {
                tsr_sim_csr_write(sim, TSR_CSR_LQ_STOP, s);
                stopped = true;
            }
            tsr_sim_advance(sim, stop_after_kernel ? 2000 : 200000);
            if (!consume()) {
                ok = false;
                return res;
            }
        }
        CHECK(0, "%s: launch %u: no DONE record", tag.c_str(), s);
        ok = false;
        return res;
    }
};

struct run_out {
    std::vector<std::array<uint32_t, 7>> done;    // r0..r6 per launch
    std::vector<std::vector<pushrec>> streams;
    uint64_t out_hash = 0;
    uint64_t push_stall = 0;
    obs ob{};                                     // SW §4.7's observables of the whole run (test_sim_obs.h)
};

uint64_t outputs_hash(tsr_sim * sim) {
    std::vector<uint8_t> b(1u << 20);
    uint64_t h = 0;
    tsr_sim_dram_read(sim, 1, (uint64_t)Y_BEAT * 64u, b.data(), 64u * 64u);
    h = h * 31u + fnv64(b.data(), 64u * 64u);
    tsr_sim_dram_read(sim, 1, (uint64_t)DST_BEAT * 64u, b.data(), b.size());
    h = h * 31u + fnv64(b.data(), b.size());
    tsr_sim_dram_read(sim, 0, (uint64_t)GOOD_BEAT * 64u, b.data(), 256u * 64u);
    h = h * 31u + fnv64(b.data(), 256u * 64u);
    tsr_sim_dram_read(sim, 1, (uint64_t)XR_BEAT * 64u, b.data(), 128u * 64u);
    h = h * 31u + fnv64(b.data(), 128u * 64u);
    return h;
}

// One complete RV run under a schedule: every launch, results checked against the references.
// stop_only: instead, only the soft-stop launch (hdr_stop with LQ_STOP written while it runs; its
// NOP count depends on when the host writes LQ_STOP, so it stays out of the comparisons).
bool rv_run(const char * elf, const char * sched, run_out & o, bool stop_only) {
    rv_env e;
    e.sim = mk_sim(sched, 2 /* fatal */, TSR_CAPS_VU);
    e.tag = std::string("fw_hdr_rv ") + tsr_sim_opt_sched(e.sim);
    e.rec.by_marks = true;
    e.rec.attach(e.sim);
    obs_attach(&o.ob, e.sim);
    stream_case sc[3];
    setup_all(e.sim, sc);
    if (!e.boot(elf)) {
        obs_finish(&o.ob, e.sim);
        tsr_sim_destroy(e.sim);
        return false;
    }
    for (uint32_t l = 0; l < L_N && e.ok && !stop_only; ++l) {
        if (l <= L_STREAM2) stream_setup(e.sim, sc[l], 77u + l);
        uint32_t a[7];
        launch_args(l, a);
        const std::array<uint32_t, 8> d = e.launch(l, a);
        std::array<uint32_t, 7> r7;
        memcpy(r7.data(), d.data(), sizeof r7);
        o.done.push_back(r7);
        o.streams.push_back(e.rec.v);
        const uint32_t st = TSR_GET(d[0], TSR_LCQ_R0_STATUS), err = TSR_GET(d[0], TSR_LCQ_R0_ERR_CODE);
        const int32_t ret = (int32_t)d[2];
        char what[160];
        snprintf(what, sizeof what, "%s: launch %u (%s)", e.tag.c_str(), l, k_lname[l]);
        switch (l) {
        case L_STREAM0: case L_STREAM1: case L_STREAM2:
            CHECK(st == TSR_LCQ_R0_STATUS_OK && ret >= 0, "%s: OK (status %u RET %d)", what, st, ret);
            stream_check(e.sim, sc[l], what);
            break;
        case L_CREDIT: CHECK(st == TSR_LCQ_R0_STATUS_OK && ret == (int32_t)(6u + 5u * 40u), "%s: status %u RET %d", what, st, ret); break;
        case L_ERR:
            CHECK(st == TSR_LCQ_R0_STATUS_TILE_ERR && err == TSR_ERR_DRAM_RANGE && ret == 0x3FF && d[4] == BAD_BEAT,
                  "%s: TILE_ERR / DRAM_RANGE, checks 0x%03x, ERR_INFO 0x%08x", what, ret, d[4]);
            break;
        case L_QCLEAR: CHECK(st == TSR_LCQ_R0_STATUS_OK && ret == 0x1F, "%s: status %u checks 0x%x", what, st, ret); break;
        case L_IRQ: CHECK(st == TSR_LCQ_R0_STATUS_OK && ret == 7, "%s: status %u checks 0x%x", what, st, ret); break;
        case L_XRING0: case L_XRING1:
            CHECK(st == TSR_LCQ_R0_STATUS_OK && ret == 0, "%s: status %u RET %d", what, st, ret);
            xring_check(e.sim, k_xring[l - L_XRING0], what);
            break;
        default: {
            CHECK(st == TSR_LCQ_R0_STATUS_OK, "%s: status %u", what, st);
            ring_check(e.sim, k_ring[l - L_RING0], ret, what);
            break;
        }
        }
    }
    if (stop_only && e.ok) {
        const uint32_t a[7] = {100000000u, 0, 0, 0, 0, 0, 0};
        const std::array<uint32_t, 8> d = e.launch(L_N, a, true);
        CHECK(TSR_GET(d[0], TSR_LCQ_R0_STATUS) == TSR_LCQ_R0_STATUS_STOPPED && d[2] == 1u,
              "%s: hdr_stop: DONE STOPPED with RET 1 (status %u RET %u)", e.tag.c_str(), TSR_GET(d[0], TSR_LCQ_R0_STATUS), d[2]);
    }
    const uint32_t C = tsr_sim_config(e.sim)->num_ch;
    o.push_stall = tsr_sim_perf(e.sim, TSR_PERF_PUSH_STALL(C));
    CHECK(o.push_stall == 0u, "%s: PERF PUSH_STALL %llu: a push waited for a credit in hardware", e.tag.c_str(),
          (unsigned long long)o.push_stall);
    // The queues fill under the fixed schedule (op durations of the cycle model); the random ones
    // may shorten the blockers below the time 16 pushes take, so there only "never above".
    e.rec.check();
    if (strcmp(sched, "fixed") == 0 && !stop_only)
        for (uint32_t u : {TSR_UNIT_DMA, TSR_UNIT_QACT, TSR_UNIT_MVM, TSR_UNIT_VU, TSR_UNIT_SYNC})
            CHECK(e.rec.maxocc[u] == e.rec.depth[u], "%s: unit %u queue filled (max %u of %u)", e.tag.c_str(), u,
                  e.rec.maxocc[u], e.rec.depth[u]);
    CHECK(e.rec.over == 0u, "%s: pushes into a full queue: %u", e.tag.c_str(), e.rec.over);
    size_t nrace = 0;
    const tsr_race_report * rr = nullptr;
    nrace = tsr_sim_race_reports(e.sim, &rr);
    CHECK(nrace == 0u, "%s: %zu race reports", e.tag.c_str(), nrace);
    o.out_hash = outputs_hash(e.sim);
    obs_finish(&o.ob, e.sim);
    printf("%s: %llu RV instructions, MTIME %llu\n", e.tag.c_str(), (unsigned long long)tsr_sim_rv_instret(e.sim),
           (unsigned long long)tsr_sim_now(e.sim));
    const bool ok = e.ok;
    tsr_sim_destroy(e.sim);
    return ok;
}

// The same launches on the host build (fixed schedule): the reference push streams.
void host_streams(std::vector<std::vector<pushrec>> & streams, std::vector<int32_t> & rets) {
    tsr_sim * sim = mk_sim("fixed", 0, TSR_CAPS_VU);
    tsr::host_bind(sim);
    recorder rec;
    rec.attach(sim);
    stream_case sc[3];
    setup_all(sim, sc);
    for (uint32_t l = 0; l < L_N; ++l) {
        if (l <= L_STREAM2) stream_setup(sim, sc[l], 77u + l);
        uint32_t a[7];
        launch_args(l, a);
        tsr_args args{};
        memcpy(args.a, a, sizeof args.a);
        args.seq = l;
        rec.begin(tsr::pidx_now());
        const tsr::host_result h = tsr::host_run(host_fn(l), args);
        rec.on = false;
        std::vector<pushrec> v = rec.v;
        while (!v.empty() && v.back().rpidx >= h.pidx_end - 1u - h.pidx0) v.pop_back();   // the shim's FENCE
        streams.push_back(v);
        rets.push_back(h.ret);
        if (tsr::err_units() != 0u) tsr::join(tsr::qclear());   // R9
    }
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <fw_hdr.elf> [--sched <TSR_SIM_SCHED value>]\n", argv[0]);
        return 2;
    }
    const char * elf = argv[1];
    if (access(elf, R_OK) != 0) {
        printf("fw_hdr_rv: %s not built; skipped\n", elf);
        return 77;
    }
    // --mutant: the image built with HDR_MUTANT_NO_MUST_WAIT (hdr_stream ignores spm_ring's
    // must_wait): under the race detector (report mode) its reuse of ring blocks must race.
    if (argc >= 3 && strcmp(argv[2], "--mutant") == 0) {
        rv_env e;
        e.sim = mk_sim("fixed", 1 /* report */, TSR_CAPS_VU);
        e.tag = "fw_hdr_rv mutant";
        e.rec.attach(e.sim);
        stream_case sc[3];
        setup_all(e.sim, sc);
        if (e.boot(elf)) {
            for (uint32_t l = L_XRING0; l <= L_XRING1 && e.ok; ++l) {
                uint32_t a[7];
                launch_args(l, a);
                e.launch(l, a);
            }
        }
        const tsr_race_report * rr = nullptr;
        const size_t n = tsr_sim_race_reports(e.sim, &rr);
        CHECK(n > 0u, "mutant without must_wait: the race detector reports the reuse of ring blocks (%zu reports)", n);
        printf("fw_hdr_rv mutant: %zu race reports (expected > 0)\n", n);
        tsr_sim_destroy(e.sim);
        return test_finish("fw_hdr_rv_mutant");
    }
    // --sched <s>: only that schedule (debugging; no comparisons).
    const char * only = (argc >= 4 && strcmp(argv[2], "--sched") == 0) ? argv[3] : nullptr;
    if (only != nullptr) {
        run_out o;
        rv_run(elf, only, o, false);
        obs_free(&o.ob);
        return test_finish("fw_hdr_rv");
    }
    std::vector<std::vector<pushrec>> hs;
    std::vector<int32_t> hret;
    host_streams(hs, hret);
    const char * scheds[8] = {"fixed", "random:1", "random:2", "random:3,0.25,8", "random:4,0.5,2", "random:5", "random:6,0.1,10",
                              "random:7"};
    run_out ref;
    for (int k = 0; k < 8; ++k) {
        run_out o;
        if (!rv_run(elf, scheds[k], o, false)) break;
        if (k == 0) {
            ref = o;                              // ref owns o.ob's buffers from here
            o.ob = obs{};
            // SW §6.8: the host build pushes the same stream (words, push values, relative deps / tickets).
            for (uint32_t l = 0; l < L_N && l < o.streams.size(); ++l) {
                char what[96];
                snprintf(what, sizeof what, "launch %u (%s): RV vs host push stream", l, k_lname[l]);
                streams_equal(hs[l], o.streams[l], what);
                CHECK(hret[l] == (int32_t)o.done[l][2], "launch %u (%s): RET RV %d host %d", l, k_lname[l], (int32_t)o.done[l][2],
                      hret[l]);
            }
            continue;
        }
        // SW §4.7: every schedule gives the same observables (per-unit op streams, RISC-V SPM
        // stores, final written SPM / DRAM; the records r0-r6 and the push streams per launch).
        obs_equal(&ref.ob, &o.ob, scheds[k]);
        obs_free(&o.ob);
        CHECK(o.done == ref.done, "%s: DONE records r0-r6 differ from the fixed schedule's", scheds[k]);
        CHECK(o.out_hash == ref.out_hash, "%s: outputs differ from the fixed schedule's", scheds[k]);
        for (uint32_t l = 0; l < L_N && l < o.streams.size() && l < ref.streams.size(); ++l) {
            char what[96];
            snprintf(what, sizeof what, "%s: launch %u (%s) push stream", scheds[k], l, k_lname[l]);
            streams_equal(ref.streams[l], o.streams[l], what);
        }
    }
    obs_free(&ref.ob);
    // EX §4.9: a soft stop seen by stop_requested() (the runtime answers STOPPED).
    run_out so;
    rv_run(elf, "fixed", so, true);
    obs_free(&so.ob);
    return test_finish("fw_hdr_rv");
}
#endif   // TSR_HDR_TEST_RV
