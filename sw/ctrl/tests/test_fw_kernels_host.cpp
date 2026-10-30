// test_fw_kernels_host.cpp — fw_kernels_host: the P2.5 kernels compiled for the host (docs/design/
// 06-software-verification.md "SW" §6.8, SW-25: tsr_fw_host under ASan + UBSan), driven through the
// simulator's test hooks (tsr/backend_host.hpp). The reference stays the RV binary on tsr_rv (SW §6.8):
// every host result here is also compared with it or with the C++ references.
//
//   test_fw_kernels_host IMAGE [TRACE]
//   1. kind_mulmat called directly with its own spm_ring and kind_ctx (the executor's contract,
//      tsr/exec.hpp): the shapes and forms of fw_kernels_rv's suite A, y bit-exact against
//      sw/ctrl/ref/mulmat_ref.cpp; the refused records (ACC_DST with an addend, a bad fmt / K, kind
//      flags, a shape the ring cannot hold) set ctx.ret and push nothing.
//   1b. the decode fast path (mmk::mulmat_t1_fast) against the general path on twin simulators: T = 1
//      records with deps on every unit, the same pushes (words, deps, push value), ticket and ring
//      state (SW §7.4: MUL_MAT's push stream is W2a's whichever path runs it).
//   2. replay through host_run: a synthetic stream of three chunks (the chunk LDs interleave with the
//      replayed DMA ops) and, when given, a trace directory (e.g. an error trace of traces/ctrl):
//      expect regions bit-exact and RET = the trace's DONE ret; the argument checks; the RET rule's
//      corners (SW §7.7 step 4): a later chunk LD of the kernel failing first (push.bin ends at the
//      channel's end: RET STREAM) and a DRAM_RANGE record in chunk 2 after two of the kernel's own
//      chunk LDs (RET = 1 + its record index); a QCLEAR record while the SYNC queue is full (NOPs
//      waiting for a long LD): the kernel waits for the SYNC credit (no push without a credit).
//   3. selftest ids 1-10 through host_run and, on a second simulator with the same inputs, through the
//      RV image: RET equal (0) and every output byte equal (data area, id 1's scratch on every
//      channel, id 5's KV planes).
// Exit 0 / 1; 77 (skipped) without the image.
#undef _GNU_SOURCE
#include "test_sim_util.h"

#include <array>
#include <vector>

#include "tsr/exec.hpp"
#include "tsr/rt.hpp"

#include "../kernels/k_mulmat.hpp"

#include "../ref/fw_sim_host.hpp"
#include "../ref/mulmat_ref.hpp"
#include "../ref/selftest_abi.hpp"
#include "tsr_sim_int.h"
#include "tsr_trace.h"

extern "C" int32_t replay(const tsr_args * args);
extern "C" int32_t selftest(const tsr_args * args);

namespace {

namespace mm = tsr::opl::mm;
constexpr uint64_t LINE = 64u;
const char * g_image = nullptr;

tsr_config cfg_u200(uint64_t ch_bytes) {
    tsr_config c = TSR_CONFIG_U200;
    c.t_max = 8;
    c.ch_bytes = ch_bytes;
    c.caps = TSR_SIM_CAPS_IMPLEMENTED;
    return c;
}

tsr_sim * make_sim(const tsr_config & c, const char * sched, uint32_t race) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = race;
    so.threads = 2;
    return tsr_sim_create_opts(&c, &so);
}

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }

// ---------------------------------------------------------------------------
// 1. kind_mulmat
// ---------------------------------------------------------------------------
struct mrec {
    uint32_t a[12];
    int dep = -1;           // op index of a predecessor (its y is this op's x when x_from)
    bool x_from = false;
    std::vector<uint8_t> want;
};

void put_rows(tsr_sim * sim, rng_t * r, uint32_t h, uint32_t rows, uint32_t stride, uint32_t K) {
    std::vector<float> x(K);
    for (uint32_t t = 0; t < rows; ++t) {
        gen_f32(r, x.data(), K);
        put_f32_le(tsr_sim_dram(sim, h >> 28) + (uint64_t(h & 0x0FFFFFFFu) + uint64_t(t) * stride) * LINE, x.data(), K);
    }
}

void test_mulmat(const char * sched) {
    const tsr_config cfg = cfg_u200(1ull << 30);
    tsr_sim * sim = make_sim(cfg, sched, 2);
    CHECK(sim != nullptr, "mulmat: simulator");
    if (!sim) return;
    tsr::host_bind(sim);
    rng_t r = { 0x40571 };
    uint64_t io[4] = { 0x100000u, 0x40000u, 0x40000u, 0x40000u }, wb = 0x800000u;
    auto take = [&](uint32_t ch, uint64_t n) { const uint64_t b = io[ch]; io[ch] += (n + 63u) & ~63ull; return b; };
    std::vector<mrec> recs;
    auto make = [&](uint32_t fmt, uint32_t N, uint32_t K, uint32_t T, uint32_t xch, uint32_t ych, uint32_t xpad,
                    uint32_t ypad, int add, uint32_t apad, bool acc) {
        mrec m;
        const uint32_t xl = K / 16u, yl = (N + 15u) / 16u;
        const size_t rb = size_t(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
        std::vector<uint8_t> blk(rb * N);
        for (uint32_t n = 0; n < N; ++n) gen_row_blocks(&r, fmt, K, blk.data() + rb * n);
        put_weights(sim, fmt, N, K, wb, blk.data());
        m.a[mm::W_OFF] = uint32_t(wb);
        wb += (tsr_tensor_ch_beats(fmt, N, K, cfg.num_ch) + 63u) & ~63ull;
        m.a[mm::FMT] = fmt; m.a[mm::N] = N; m.a[mm::K] = K; m.a[mm::T] = T;
        m.a[mm::X_STRIDE] = xl + xpad;
        m.a[mm::X] = H(xch, take(xch, uint64_t(T) * (xl + xpad)));
        put_rows(sim, &r, m.a[mm::X], T, xl + xpad, K);
        m.a[mm::Y_STRIDE] = yl + ypad;
        m.a[mm::Y] = H(ych, take(ych, uint64_t(T) * (yl + ypad)));
        m.a[mm::ADD] = tsr::opl::HANDLE_NONE;
        m.a[mm::ADD_STRIDE] = 0;
        m.a[mm::FLAGS] = acc ? mm::FLAG_ACC_DST : 0u;
        if (acc) put_rows(sim, &r, m.a[mm::Y], T, yl + ypad, N);
        if (add == 1) {
            m.a[mm::ADD] = H(ych, take(ych, yl));
            put_rows(sim, &r, m.a[mm::ADD], 1, yl, N);
        } else if (add == 2) {
            m.a[mm::ADD_STRIDE] = yl + apad;
            m.a[mm::ADD] = H(ych, take(ych, uint64_t(T) * (yl + apad)));
            put_rows(sim, &r, m.a[mm::ADD], T, yl + apad, N);
        }
        recs.push_back(m);
    };
    const uint32_t Q8 = TSR_FMT_Q8_0, Q4 = TSR_FMT_Q4_K;
    make(Q8, 896, 896, 1, 0, 0, 0, 0, 0, 0, false);
    make(Q8, 128, 896, 8, 0, 1, 0, 0, 0, 0, false);
    make(Q4, 512, 1024, 5, 1, 2, 0, 0, 0, 0, false);
    make(Q8, 300, 256, 3, 0, 0, 0, 0, 1, 0, false);
    make(Q4, 256, 512, 4, 0, 3, 0, 0, 2, 0, false);
    make(Q8, 64, 512, 6, 2, 1, 5, 2, 2, 3, false);
    make(Q8, 896, 896, 2, 0, 0, 0, 0, 0, 0, true);
    make(Q8, 64, 256, 19, 3, 3, 0, 0, 0, 0, false);
    make(Q8, 320, 896, 1, 0, 2, 0, 0, 0, 0, false);              // x = op 0's y
    recs.back().a[mm::X] = recs[0].a[mm::Y];
    recs.back().a[mm::X_STRIDE] = recs[0].a[mm::Y_STRIDE];
    recs.back().dep = 0;
    recs.back().x_from = true;
    // Expected rows (the reference reads the DRAM as filled; op 8's x is op 0's expected y).
    std::vector<uint8_t *> chp(cfg.num_ch);
    for (uint32_t c = 0; c < cfg.num_ch; ++c) chp[c] = tsr_sim_dram(sim, c);
    const tsr::ref::dram_view v{chp.data(), cfg.num_ch, cfg.ch_bytes};
    for (auto & m : recs) {
        const uint32_t N = m.a[mm::N], K = m.a[mm::K], T = m.a[mm::T], yl = (N + 15u) / 16u;
        m.want.assign(size_t(T) * yl * LINE, 0);
        std::vector<float> x(K), add(N), y(N);
        for (uint32_t t = 0; t < T; ++t) {
            const uint8_t * xp = m.x_from ? recs[size_t(m.dep)].want.data() + size_t(t) * ((recs[size_t(m.dep)].a[mm::N] + 15u) / 16u) * LINE
                                          : chp[m.a[mm::X] >> 28] + (uint64_t(m.a[mm::X] & 0x0FFFFFFFu) + uint64_t(t) * m.a[mm::X_STRIDE]) * LINE;
            for (uint32_t k = 0; k < K; ++k) x[k] = u2f(get_u32_le(xp + 4u * k));
            const float * ap = nullptr;
            if (m.a[mm::ADD] != tsr::opl::HANDLE_NONE || (m.a[mm::FLAGS] & mm::FLAG_ACC_DST)) {
                const bool has = m.a[mm::ADD] != tsr::opl::HANDLE_NONE;
                const uint32_t h = has ? m.a[mm::ADD] : m.a[mm::Y], st = has ? m.a[mm::ADD_STRIDE] : m.a[mm::Y_STRIDE];
                const uint8_t * p = chp[h >> 28] + (uint64_t(h & 0x0FFFFFFFu) + uint64_t(t) * st) * LINE;
                for (uint32_t n = 0; n < N; ++n) add[n] = u2f(get_u32_le(p + 4u * n));
                ap = add.data();
            }
            tsr::ref::mulmat_ref_token(v, m.a[mm::FMT], N, K, m.a[mm::W_OFF], x.data(), ap, y.data());
            put_f32_le(m.want.data() + size_t(t) * yl * LINE, y.data(), N);
        }
    }
    // mulmat_ref's whole-record form on a second simulator holding the same DRAM (weights, rows): the
    // records in list order, in place, as the kernel runs them; its y rows must equal the kernel's.
    tsr_sim * sim2 = make_sim(cfg, "fixed", 0);
    for (uint32_t c = 0; c < cfg.num_ch; ++c) {
        const uint64_t lo = c == 0 ? 0x100000u : 0x40000u;
        memcpy(tsr_sim_dram(sim2, c) + lo * LINE, tsr_sim_dram(sim, c) + lo * LINE, size_t(io[c] - lo) * LINE);
        memcpy(tsr_sim_dram(sim2, c) + 0x800000ull * LINE, tsr_sim_dram(sim, c) + 0x800000ull * LINE,
               size_t(wb - 0x800000u) * LINE);
    }
    {
        std::vector<uint8_t *> c2(cfg.num_ch);
        for (uint32_t c = 0; c < cfg.num_ch; ++c) c2[c] = tsr_sim_dram(sim2, c);
        const tsr::ref::dram_view v2{c2.data(), cfg.num_ch, cfg.ch_bytes};
        for (size_t i = 0; i < recs.size(); ++i)
            CHECK(tsr::ref::mulmat_ref(v2, recs[i].a, 2) == tsr::ref::MM_OK, "mulmat_ref: op %zu refused", i);
        uint32_t bad_args[12];
        memcpy(bad_args, recs[3].a, sizeof(bad_args));
        bad_args[mm::FLAGS] = mm::FLAG_ACC_DST;
        CHECK(tsr::ref::mulmat_ref(v2, bad_args) == tsr::ref::MM_BAD_RECORD, "mulmat_ref: ACC_DST with an addend accepted");
    }
    // The executor's contract, by hand: one ring, deps from the predecessors' tickets.
    const uint32_t spm_lines = cfg.spm_bytes / 64u;
    tsr::plan::exec_layout lay{};
    CHECK(tsr::plan::layout(spm_lines, 0, &lay), "plan::layout");
    tsr::spm_ring ring;
    ring.init(lay.ring_line, lay.ring_lines);
    const tsr::plan::dev_cfg dc{spm_lines, cfg.t_max, cfg.k_max};
    tsr::kind_ctx ctx{};
    ctx.ring = &ring;
    ctx.cfg = &dc;
    ctx.mml = tsr::plan::mm_limits_of(dc);
    std::vector<tsr::ticket> tk;
    for (size_t i = 0; i < recs.size(); ++i) {
        ctx.ret = 0;
        ctx.op_index = uint32_t(i);
        const tsr::deps d = recs[i].dep >= 0 ? tsr::deps{tk[size_t(recs[i].dep)]} : tsr::deps{};
        tk.push_back(tsr::kind_mulmat(recs[i].a, 12, 0, d, ctx));
        CHECK(ctx.ret == 0 && !tk.back().none(), "mulmat [%s]: op %zu: ret 0x%08x", sched, i, ctx.ret);
    }
    tsr::join_all();
    CHECK(tsr::err_units() == 0, "mulmat [%s]: a unit in error state", sched);
    for (size_t i = 0; i < recs.size(); ++i) {
        const mrec & m = recs[i];
        const uint32_t yl = (m.a[mm::N] + 15u) / 16u;
        for (uint32_t t = 0; t < m.a[mm::T]; ++t) {
            const uint8_t * got = tsr_sim_dram(sim, m.a[mm::Y] >> 28) + (uint64_t(m.a[mm::Y] & 0x0FFFFFFFu) + uint64_t(t) * m.a[mm::Y_STRIDE]) * LINE;
            CHECK(memcmp(got, m.want.data() + size_t(t) * yl * LINE, size_t(yl) * LINE) == 0, "mulmat [%s]: op %zu token %u: y differs",
                  sched, i, t);
            const uint8_t * ref = tsr_sim_dram(sim2, m.a[mm::Y] >> 28) + (uint64_t(m.a[mm::Y] & 0x0FFFFFFFu) + uint64_t(t) * m.a[mm::Y_STRIDE]) * LINE;
            CHECK(memcmp(got, ref, size_t(yl) * LINE) == 0, "mulmat [%s]: op %zu token %u: y differs from mulmat_ref", sched, i, t);
        }
    }
    // Refused records: ctx.ret set, nothing pushed.
    struct bad { const char * what; uint32_t idx, field, value, kflags, want; };
    const bad bads[] = {
        { "ACC_DST with an addend", 3, mm::FLAGS, mm::FLAG_ACC_DST, 0, tsr::opl::RET_BAD_RECORD },
        { "fmt 2", 0, mm::FMT, 2, 0, tsr::opl::RET_BAD_RECORD },
        { "K 100", 0, mm::K, 100, 0, tsr::opl::RET_BAD_RECORD },
        { "Q4_K K 896", 0, mm::FMT, Q4, 0, tsr::opl::RET_BAD_RECORD },
        { "unknown MUL_MAT flag", 0, mm::FLAGS, 2, 0, tsr::opl::RET_BAD_RECORD },
        { "kind flags", 0, mm::FLAGS, 0, 1, tsr::opl::RET_BAD_RECORD },
        { "T 0", 0, mm::T, 0, 0, tsr::opl::RET_BAD_RECORD },
        { "one token beyond the ring", 0, mm::N, 0x00F00000u, 0, tsr::opl::RET_SCRATCH },
    };
    for (const auto & b : bads) {
        uint32_t a[12];
        memcpy(a, recs[b.idx].a, sizeof(a));
        a[b.field] = b.value;
        const uint32_t p0 = tsr::pidx_now();
        ctx.ret = 0;
        const tsr::ticket t = tsr::kind_mulmat(a, 12, b.kflags, tsr::deps{}, ctx);
        CHECK(ctx.ret == b.want && t.none() && tsr::pidx_now() == p0, "mulmat [%s]: %s: ret 0x%08x (expected 0x%08x), %u pushes",
              sched, b.what, ctx.ret, b.want, tsr::pidx_now() - p0);
    }
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim2);
    tsr_sim_destroy(sim);
}

// ---------------------------------------------------------------------------
// 1b. The decode fast path pushes what the general path pushes (SW §7.4, k_mulmat.hpp; executor
//     optimisation review, 2026-10-30): T = 1 records (Q8_0 and Q4_K; plain, ACC_DST, an addend; with
//     and without the REC push flag) whose deps are tickets of non-DMA units (SYNC, QACT, VU, MVM),
//     of DMA, or none, after the same history (earlier MUL_MATs through kind_mulmat, so that the
//     ring reuses blocks released by STs). mmk::mulmat_t1_fast runs on one simulator and the general
//     path (mmk::mulmat_general = kind_mulmat without the fast path) on a twin: every push (the
//     eight words, both deps, the ISSUE_PUSH value) equal, the same ticket and ring state, the fast
//     path taken, the same y rows; race detector fatal on both.
// ---------------------------------------------------------------------------
struct plog {
    std::vector<std::array<uint32_t, 12>> v;
};
void plog_q(void * ctx, tsr_sim *, const tsr_q_event * ev) {
    if (ev->kind != TSR_QEV_PUSH && ev->kind != TSR_QEV_QCTRL) return;
    std::array<uint32_t, 12> e{};
    for (int k = 0; k < 8; ++k) e[size_t(k)] = ev->w[k];
    e[8] = ev->dep[0];
    e[9] = ev->dep[1];
    e[10] = ev->push;
    e[11] = ev->kind;
    static_cast<plog *>(ctx)->v.push_back(e);
}

struct twin_case {
    const char * what;
    uint32_t fmt, N, K;
    int form;              // 0 plain, 1 ACC_DST, 2 addend
    int d0, d1;            // predecessor unit: -1 none, else 0 SYNC, 1 DMA, 2 QACT, 3 VU, 4 MVM
    bool rec;              // REC on the pushes
};

// One simulator's run of a case: the history, the predecessors, then the record through the fast path
// (fast) or the general path. Returns the record's ticket; *ring_state = {head, blocks, joins}.
uint32_t twin_run(tsr_sim * sim, const twin_case & c, bool fast, plog * log, uint32_t ring_state[3], bool * took_fast) {
    const tsr_config & cfg = *tsr_sim_config(sim);
    tsr::host_bind(sim);
    rng_t r = { 0x7317u };
    uint64_t io = 0x100000u, wb = 0x800000u;
    auto take = [&](uint64_t n) { const uint64_t b = io; io += (n + 63u) & ~63ull; return b; };
    auto weights = [&](uint32_t fmt, uint32_t N, uint32_t K) {
        const size_t rb = size_t(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
        std::vector<uint8_t> blk(rb * N);
        for (uint32_t n = 0; n < N; ++n) gen_row_blocks(&r, fmt, K, blk.data() + rb * n);
        put_weights(sim, fmt, N, K, wb, blk.data());
        const uint32_t w = uint32_t(wb);
        wb += (tsr_tensor_ch_beats(fmt, N, K, cfg.num_ch) + 63u) & ~63ull;
        return w;
    };
    auto record = [&](uint32_t a[12], uint32_t fmt, uint32_t N, uint32_t K, int form) {
        const uint32_t xl = K / 16u, yl = (N + 15u) / 16u;
        a[mm::W_OFF] = weights(fmt, N, K);
        a[mm::FMT] = fmt; a[mm::N] = N; a[mm::K] = K; a[mm::T] = 1;
        a[mm::X_STRIDE] = xl;
        a[mm::X] = H(0, take(xl));
        put_rows(sim, &r, a[mm::X], 1, xl, K);
        a[mm::Y_STRIDE] = yl;
        a[mm::Y] = H(0, take(yl));
        a[mm::ADD] = tsr::opl::HANDLE_NONE;
        a[mm::ADD_STRIDE] = 0;
        a[mm::FLAGS] = form == 1 ? mm::FLAG_ACC_DST : 0u;
        if (form == 1) put_rows(sim, &r, a[mm::Y], 1, yl, N);
        if (form == 2) {
            a[mm::ADD] = H(0, take(yl));
            put_rows(sim, &r, a[mm::ADD], 1, yl, N);
        }
    };
    const uint32_t spm_lines = cfg.spm_bytes / 64u;
    tsr::plan::exec_layout lay{};
    tsr::plan::layout(spm_lines, 0, &lay);
    const tsr::plan::dev_cfg dc{spm_lines, cfg.t_max, cfg.k_max};
    tsr::kind_ctx ctx{};
    ctx.cfg = &dc;
    ctx.mml = tsr::plan::mm_limits_of(dc);
    ctx.push_flags = c.rec ? tsr::PUSH_REC : 0u;
    tsr::plan::mm_tiling tl{};
    CHECK(tsr::plan::mm_plan(ctx.mml, c.fmt, c.N, c.K, 1u, &tl) == tsr::opl::RET_OK, "fast/general %s: plan", c.what);
    // A ring of 2.5 blocks of the record's size: history ops 1 and 2 take [0, b) and [b, 2b), op 3
    // wraps to [0, b) (the general path: a wrap), so the record lands on [b, 2b), reusing op 2's
    // block, released by its ST.
    static tsr::spm_ring ring;
    const uint32_t b = tl.lpt;
    ring.init(lay.ring_line, 2u * b + b / 2u);
    ctx.ring = &ring;
    const int lid = tsr_sim_add_listener(sim, plog_q, log);
    tsr::ticket prev{};
    for (uint32_t i = 0; i < 3u; ++i) {
        uint32_t a[12];
        record(a, c.fmt, c.N, c.K, int(i % 3u));
        ctx.ret = 0;
        prev = tsr::kind_mulmat(a, 12, 0, tsr::deps{prev}, ctx);
        CHECK(ctx.ret == 0, "fast/general %s: history op %u: ret 0x%08x", c.what, i, ctx.ret);
    }
    CHECK(ring.head() == lay.ring_line + b && ring.blocks() == 2u, "fast/general %s: ring head %u, %u blocks after the history",
          c.what, ring.head(), ring.blocks());
    // Predecessors on the units the case names (scratch in the OPL area, outside the ring).
    const uint32_t S0 = lay.opl_line;
    const tsr::ticket ld = tsr::issue(tsr::LD(tsr::spm{S0}, tsr::dram{1, 0x40000u}, 56));
    const tsr::ticket qt = tsr::issue(tsr::QACT(tsr::qa::a32, tsr::spm{S0 + 100}, tsr::spm{S0}, 896, 1), tsr::deps{ld});
    const tsr::ticket vu =
        tsr::issue(tsr::VOP({.func = tsr::VOP_FUNC_COPY, .d = S0 + 300, .a = S0, .m = 16u * 56u, .rows = 1}), tsr::deps{ld});
    const uint32_t wm = weights(TSR_FMT_Q8_0, 64, 896);
    const tsr::ticket mv = tsr::issue(tsr::MVM(tsr::spm{S0 + 200}, tsr::spm{S0 + 100}, tsr::wref{wm, uint8_t(TSR_FMT_Q8_0)}, 64, 896, 1),
                                      tsr::deps{qt});
    const tsr::ticket sy = tsr::issue(tsr::NOP(), tsr::deps{mv});
    auto pred = [&](int u) {
        switch (u) {
        case 0: return sy;
        case 1: return ld;
        case 2: return qt;
        case 3: return vu;
        case 4: return mv;
        default: return tsr::ticket{};
        }
    };
    uint32_t a[12];
    record(a, c.fmt, c.N, c.K, c.form);
    const tsr::ticket t0 = pred(c.d0), t1 = pred(c.d1);
    ctx.ret = 0;
    uint32_t t = 0;
    *took_fast = false;
    if (fast) {
        t = tsr::mmk::mulmat_t1_fast(a, 0, t0, t1, ctx);
        *took_fast = t != tsr::mmk::NOT_FAST;
    } else {
        t = tsr::mmk::mulmat_general(a, 12, 0, tsr::after(t0, t1), ctx).v;
    }
    CHECK(ctx.ret == 0, "fast/general %s: ret 0x%08x (%s path)", c.what, ctx.ret, fast ? "fast" : "general");
    tsr::join_all();
    CHECK(tsr::err_units() == 0, "fast/general %s: a unit in error state (%s path)", c.what, fast ? "fast" : "general");
    tsr_sim_remove_listener(sim, lid);
    ring_state[0] = ring.head();
    ring_state[1] = ring.blocks();
    ring_state[2] = ring.joins();
    tsr::host_bind(nullptr);
    return t;
}

void test_mulmat_fast_vs_general() {
    const uint32_t Q8 = TSR_FMT_Q8_0, Q4 = TSR_FMT_Q4_K;
    const twin_case cases[] = {
        { "Q8_0 plain, deps SYNC", Q8, 896, 896, 0, 0, -1, false },
        { "Q8_0 plain, deps SYNC + DMA", Q8, 896, 896, 0, 0, 1, false },
        { "Q8_0 plain, deps DMA only", Q8, 896, 896, 0, 1, -1, false },
        { "Q8_0 plain, no deps, REC", Q8, 896, 896, 0, -1, -1, true },
        { "Q8_0 ACC_DST, deps QACT + VU", Q8, 896, 896, 1, 2, 3, false },
        { "Q8_0 addend, deps MVM + SYNC, REC", Q8, 128, 896, 2, 4, 0, true },
        { "Q8_0 addend, deps VU + DMA", Q8, 896, 896, 2, 3, 1, false },
        { "Q4_K plain, deps SYNC + MVM", Q4, 512, 1024, 0, 0, 4, false },
        { "Q4_K ACC_DST, deps DMA + QACT, REC", Q4, 256, 512, 1, 1, 2, true },
    };
    tsr_config cfg = cfg_u200(1ull << 30);
    uint32_t n_ok = 0;
    for (const twin_case & c : cases) {
        tsr_sim * sa = make_sim(cfg, "fixed", 2);
        tsr_sim * sb = make_sim(cfg, "fixed", 2);
        CHECK(sa && sb, "fast/general %s: simulators", c.what);
        if (!sa || !sb) return;
        plog la, lb;
        uint32_t ra[3], rb[3];
        bool fa = false, fb = false;
        const uint32_t ta = twin_run(sa, c, true, &la, ra, &fa);
        const uint32_t tb = twin_run(sb, c, false, &lb, rb, &fb);
        size_t first = 0;
        while (first < la.v.size() && first < lb.v.size() && la.v[first] == lb.v[first]) ++first;
        const bool same = la.v.size() == lb.v.size() && first == la.v.size();
        CHECK(fa, "fast/general %s: the fast path declined the record", c.what);
        CHECK(same, "fast/general %s: the pushes differ (%zu / %zu pushes, first difference at %zu: w0 0x%08x / 0x%08x, "
              "deps 0x%08x 0x%08x / 0x%08x 0x%08x, push 0x%08x / 0x%08x)", c.what, la.v.size(), lb.v.size(), first,
              first < la.v.size() ? la.v[first][0] : 0u, first < lb.v.size() ? lb.v[first][0] : 0u,
              first < la.v.size() ? la.v[first][8] : 0u, first < la.v.size() ? la.v[first][9] : 0u,
              first < lb.v.size() ? lb.v[first][8] : 0u, first < lb.v.size() ? lb.v[first][9] : 0u,
              first < la.v.size() ? la.v[first][10] : 0u, first < lb.v.size() ? lb.v[first][10] : 0u);
        CHECK(ta == tb && ra[0] == rb[0] && ra[1] == rb[1] && ra[2] == rb[2],
              "fast/general %s: ticket 0x%08x / 0x%08x, ring head %u / %u, blocks %u / %u, joins %u / %u", c.what, ta, tb,
              ra[0], rb[0], ra[1], rb[1], ra[2], rb[2]);
        bool y_same = true;
        for (uint32_t ch = 0; ch < cfg.num_ch; ++ch)
            y_same = y_same && memcmp(tsr_sim_dram(sa, ch) + 0x100000ull * LINE, tsr_sim_dram(sb, ch) + 0x100000ull * LINE,
                                      0x10000u * LINE) == 0;
        CHECK(y_same, "fast/general %s: DRAM outputs differ", c.what);
        n_ok += fa && same && y_same && ta == tb;
        tsr_sim_destroy(sb);
        tsr_sim_destroy(sa);
    }
    printf("fast/general: %u of %zu T = 1 records pushed the same stream through mulmat_t1_fast and the general path\n",
           n_ok, sizeof(cases) / sizeof(cases[0]));
}

// ---------------------------------------------------------------------------
// 2. replay through host_run
// ---------------------------------------------------------------------------
void load_trace(tsr_sim * sim, const tsr_trace2 * t, uint64_t * load_beat) {
    for (size_t i = 0; i < t->mem.n_init; ++i)
        tsr_sim_dram_write(sim, t->mem.init[i].ch, t->mem.init[i].byte_off, t->mem.init[i].data, t->mem.init[i].size);
    // push.bin far above every region of these traces: ch0 48 MiB (inside the OPL region).
    *load_beat = (48ull << 20) / 64u;
    std::vector<uint8_t> b(t->n_push * TSR_TRACE_PUSH_BYTES + 64u, 0);
    for (size_t i = 0; i < t->n_push; ++i) {
        uint32_t w[12];
        memcpy(w, t->push[i].w, 32);
        w[8] = t->push[i].dep[0]; w[9] = t->push[i].dep[1]; w[10] = t->push[i].push; w[11] = t->push[i].kind;
        for (int k = 0; k < 12; ++k) {
            uint8_t * p = b.data() + i * 48u + 4u * k;
            p[0] = uint8_t(w[k]); p[1] = uint8_t(w[k] >> 8); p[2] = uint8_t(w[k] >> 16); p[3] = uint8_t(w[k] >> 24);
        }
    }
    tsr_sim_dram_write(sim, 0, *load_beat * 64u, b.data(), b.size());
}

// The highest run of `need` SPM lines below the argument range that no op of t touches (the
// checker's extents; as tsr_trace.c's replay route), or UINT32_MAX.
uint32_t free_buffer(tsr_sim * sim, const tsr_trace2 * t, uint32_t need) {
    const uint32_t L = tsr_sim_spm_lines(sim);
    std::vector<uint8_t> used(L, 0);
    for (size_t i = 0; i < t->n_push; ++i) {
        const tsr_trace_push * p = &t->push[i];
        if (p->kind != TSR_TRACE_PUSH_OP || tsr_trace2_push_unit(&t->cfg, p) == TSR_UNIT_SYNC) continue;
        const tsr_op_handler * h = tsr_sim_op_handler(uint8_t(p->w[0] & 0xFFu));
        tsr_extents ext{};
        uint32_t info = 0;
        if (h == nullptr || h->check == nullptr || h->check(sim, p->w, &ext, &info) != 0) continue;
        for (uint32_t k = 0; k < ext.n; ++k)
            if (ext.e[k].kind == TSR_EXT_SPM)
                for (uint64_t l = ext.e[k].lo; l < ext.e[k].hi && l < L; ++l) used[l] = 1;
    }
    uint32_t run = 0;
    for (uint32_t l = L - 16u; l-- > 0;) {
        run = used[l] ? 0u : run + 1u;
        if (run == need) return l;
    }
    return UINT32_MAX;
}

void replay_host(const tsr_trace2 * t, const char * name, uint32_t buf_line, uint32_t chunk) {
    tsr_sim * sim = make_sim(t->cfg.core, "fixed", 2);
    CHECK(sim != nullptr, "%s: simulator", name);
    if (!sim) return;
    if (buf_line == UINT32_MAX) {
        const uint32_t ch = chunk ? chunk : uint32_t((t->n_push + 3u) & ~size_t(3));
        buf_line = free_buffer(sim, t, ch * 48u / 64u);
        CHECK(buf_line != UINT32_MAX, "%s: no free SPM buffer", name);
        if (buf_line == UINT32_MAX) { tsr_sim_destroy(sim); return; }
        chunk = ch;
    }
    uint64_t lb = 0;
    load_trace(sim, t, &lb);
    tsr::host_bind(sim);
    tsr_args a{};
    a.a[0] = uint32_t(lb);
    a.a[1] = uint32_t(t->n_push);
    a.a[2] = TSR_REPLAY_F_BUF;
    a.a[3] = buf_line;
    a.a[4] = chunk;
    const tsr::host_result hr = tsr::host_run(&replay, a);
    CHECK(hr.ret == t->done.ret, "%s: replay RET %d, the trace's %d", name, hr.ret, t->done.ret);
    CHECK(hr.err_code == t->done.err_code, "%s: first error code %u, the trace's %u", name, hr.err_code, t->done.err_code);
    for (size_t i = 0; i < t->mem.n_expect; ++i) {
        const tsr_region * x = &t->mem.expect[i];
        CHECK(memcmp(tsr_sim_dram(sim, x->ch) + x->byte_off, x->data, x->size) == 0, "%s: expect region %zu differs", name, i);
    }
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim);
}

void test_replay(const char * trace_dir) {
    // Synthetic: 3,100 records (chunks of 1,364), deps across chunk boundaries.
    tsr_trace2 * t = tsr_trace2_new();
    tsr_config core = TSR_CONFIG_U200;
    core.ch_bytes = 1ull << 28;
    tsr_trace2_config_from_core(&t->cfg, &core);
    t->cfg.ctl_id = tsr_trace2_ctl_id_full();
    rng_t r = { 0x5A12 };
    std::vector<uint8_t> in(256u * 64u);
    rng_bytes(&r, in.data(), in.size());
    const uint64_t in_beat = TSR_REGION_IO_OFF / 64u, out_beat = in_beat + 1024u;
    tsr_trace2_add_init(t, 0, in_beat * 64u, in.data(), in.size());
    for (uint32_t i = 0; i < 1550u; ++i) {
        const tsr_insn ld = tsr_enc_ld(0, i % 512u, uint32_t(in_beat + (i * 7u) % 256u), 1);
        uint32_t dep = TSR_TICKET_NONE;
        if (i >= 700u && i % 5u == 0u) dep = tsr_trace2_ticket_of(t, t->n_push - 1400u);
        tsr_trace2_add_op(t, ld.w, dep, TSR_TICKET_NONE, 0x0F00u);
        const uint32_t ldt = tsr_trace2_ticket_of(t, t->n_push - 1u);
        const tsr_insn st = tsr_enc_st((i & 1u) ? 1u : 0u, i % 512u, uint32_t(out_beat + i), 1);
        tsr_trace2_add_op(t, st.w, ldt, TSR_TICKET_NONE, 0x0F00u);
    }
    tsr_trace2_replay_opts o;
    memset(&o, 0, sizeof(o));
    o.init_from_reads = 1;
    tsr_trace2_result res;
    CHECK(tsr_trace2_run_ops(t, &o, &res) == 0, "synthetic: run_ops: %s", res.msg);
    replay_host(t, "replay host: synthetic 3100", 60000, TSR_REPLAY_CHUNK_MAX);
    tsr_trace2_free(t);
    {   // Argument checks: nothing is pushed, RET negative (KERNEL_ERR).
        tsr_sim * sim = make_sim(cfg_u200(1ull << 28), "fixed", 2);
        tsr::host_bind(sim);
        struct { uint32_t a[5]; int32_t want; } cases[] = {
            { { 0, 10, 4, 0, 0 }, int32_t(TSR_REPLAY_RET_BAD_ARGS) },                                  // unknown flag
            { { 0, 10, TSR_REPLAY_F_BUF, 0, 6 }, int32_t(TSR_REPLAY_RET_BAD_ARGS) },                   // chunk % 4
            { { 0, 10, TSR_REPLAY_F_BUF, 0, TSR_REPLAY_CHUNK_MAX + 4 }, int32_t(TSR_REPLAY_RET_BAD_ARGS) },
            { { 0, 10, TSR_REPLAY_F_BUF, 65530, 8 }, int32_t(TSR_REPLAY_RET_BAD_ARGS) },               // buffer in the arg range
            { { 0, 4u * TSR_REPLAY_CHUNK_MAX * TSR_REPLAY_CHUNKS_MAX, 0, 0, 0 }, int32_t(TSR_REPLAY_RET_TOO_LONG) },
            { { 0x0FFFFFF0u, 10, 0, 0, 0 }, int32_t(TSR_REPLAY_RET_STREAM) },                          // push.bin out of range
        };
        for (const auto & c : cases) {
            tsr_args a{};
            for (int i = 0; i < 5; ++i) a.a[i] = c.a[i];
            const tsr::host_result hr = tsr::host_run(&replay, a);
            CHECK(hr.ret == c.want, "replay args %u %u %u %u %u: RET 0x%08x, expected 0x%08x", c.a[0], c.a[1], c.a[2], c.a[3],
                  c.a[4], uint32_t(hr.ret), uint32_t(c.want));
            if (hr.err_code != 0) tsr::join(tsr::qclear());
        }
        tsr::host_bind(nullptr);
        tsr_sim_destroy(sim);
    }
    {   // fix W2a: the RET rule's corners and the QCLEAR credit wait (SW §7.7 steps 3-4)
        const uint64_t chb = 1ull << 28, CH = chb / 64u, LB = (48ull << 20) / 64u;
        using rec = std::array<uint32_t, 12>;
        auto op_rec = [](const tsr_insn & o, uint32_t dep0) {
            return rec{ o.w[0], o.w[1], o.w[2], o.w[3], o.w[4], o.w[5], o.w[6], o.w[7], dep0, TSR_TICKET_NONE, 0xFF00u, 0u };
        };
        auto put = [](tsr_sim * sim, uint64_t beat, const std::vector<rec> & v) {
            std::vector<uint8_t> b((v.size() * 48u + 63u) / 64u * 64u, 0);
            for (size_t i = 0; i < v.size(); ++i) memcpy(b.data() + 48u * i, v[i].data(), 48u);   // little-endian host
            tsr_sim_dram_write(sim, 0, beat * 64u, b.data(), b.size());
        };
        auto run = [&](const std::vector<rec> & v, uint64_t beat, uint32_t count, uint32_t chunk, const char * what,
                       int32_t want_ret, uint32_t want_code) {
            tsr_sim * sim = make_sim(cfg_u200(chb), "fixed", 2);
            tsr::host_bind(sim);
            put(sim, beat, v);
            tsr_args a{};
            a.a[0] = uint32_t(beat);
            a.a[1] = count;
            a.a[2] = TSR_REPLAY_F_BUF;
            a.a[3] = 60000;
            a.a[4] = chunk;
            const tsr::host_result hr = tsr::host_run(&replay, a);
            CHECK(hr.ret == want_ret && hr.err_code == want_code, "replay %s: RET 0x%08x first error %u, expected 0x%08x / %u",
                  what, uint32_t(hr.ret), hr.err_code, uint32_t(want_ret), want_code);
            CHECK(tsr::host().no_credit_pushes == 0u, "replay %s: %llu pushes / QCLEARs without a credit", what,
                  (unsigned long long)tsr::host().no_credit_pushes);
            if (hr.err_code != 0) tsr::join(tsr::qclear());
            tsr::host_bind(nullptr);
            tsr_sim_destroy(sim);
        };
        // (a) 12 NOP records in chunks of 4 at the last 6 beats of the channel: chunk 2's LD (pushed by
        // the kernel before chunk 1's records) fails first; no replayed record failed: STREAM.
        std::vector<rec> v(8, op_rec(tsr_enc_nop(), TSR_TICKET_NONE));
        run(v, CH - 6u, 12, 4, "own chunk LD beyond the channel", int32_t(TSR_REPLAY_RET_STREAM), TSR_ERR_DRAM_RANGE);
        // (b) 16 LD records in chunks of 4; record 10 (chunk 2) reads beyond the channel. Before its push
        // the kernel pushed the LDs of chunks 2 and 3: RET = 1 + 10, not counting them.
        v.clear();
        for (uint32_t i = 0; i < 16u; ++i)
            v.push_back(op_rec(i == 10u ? tsr_enc_ld(0, 100u + i, uint32_t(CH - 1u), 2) : tsr_enc_ld(0, 100u + i, 0x2000u + i, 1),
                               TSR_TICKET_NONE));
        run(v, LB, 16, 4, "DRAM_RANGE record in chunk 2", 11, TSR_ERR_DRAM_RANGE);
        // (c) a 1,024-line LD, 4 NOPs on it (the SYNC queue's depth: credit 0), then a QCLEAR record: the
        // kernel must wait for the SYNC credit like qclear() (host backend: no QCLEAR without a credit).
        v.clear();
        v.push_back(op_rec(tsr_enc_ld(0, 0, 0x4000u, 1024), TSR_TICKET_NONE));
        for (uint32_t i = 0; i < 4u; ++i) v.push_back(op_rec(tsr_enc_nop(), TSR_PUT(TSR_TICKET_UNIT, TSR_UNIT_DMA) | 0u));
        v.push_back(rec{ TSR_QCTRL_QCLEAR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1u });
        v.push_back(op_rec(tsr_enc_st(1, 0, 0x4000u, 16), TSR_TICKET_NONE));   // after the LD on the DMA queue
        run(v, LB, uint32_t(v.size()), 8, "QCLEAR record with the SYNC queue full", 0, 0);
    }
    if (trace_dir != nullptr) {
        tsr_trace2 * tr = nullptr;
        char err[512];
        if (tsr_trace2_read(trace_dir, &tr, err, sizeof(err)) == 0) {
            replay_host(tr, trace_dir, UINT32_MAX, 0);   // a buffer outside the trace's SPM operands
            tsr_trace2_free(tr);
        } else {
            printf("fw_kernels_host: %s: %s (not run)\n", trace_dir, err);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. selftest: host build vs RV image, the same inputs
// ---------------------------------------------------------------------------
struct st_case { uint32_t id; uint32_t p[14]; uint32_t np; uint32_t lines; uint32_t seed; };

// Fills every simulator the same way for case c (params, random data, weights, zeroed planes).
void st_fill(tsr_sim * sim, const st_case & c, uint64_t area) {
    rng_t r = { 0xF111u + c.id };
    uint8_t * A = tsr_sim_dram(sim, 0) + area * 64u;
    for (uint32_t l = 1; l < c.lines; ++l) {
        std::vector<float> x(16);
        gen_f32(&r, x.data(), 16);
        put_f32_le(A + size_t(l) * 64u, x.data(), 16);
    }
    memset(A, 0, 64);
    const uint32_t hdr[2] = { TSR_ST_MAGIC, c.id };
    for (uint32_t i = 0; i < 2u + c.np; ++i) {
        const uint32_t v = i < 2u ? hdr[i] : c.p[i - 2u];
        A[4 * i] = uint8_t(v); A[4 * i + 1] = uint8_t(v >> 8); A[4 * i + 2] = uint8_t(v >> 16); A[4 * i + 3] = uint8_t(v >> 24);
    }
    const uint64_t W = TSR_REGION_WEIGHT_OFF / 64u;
    auto weights = [&](uint32_t fmt, uint32_t N, uint32_t K, uint64_t off) {
        const size_t rb = size_t(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
        std::vector<uint8_t> blk(rb * N);
        for (uint32_t n = 0; n < N; ++n) gen_row_blocks(&r, fmt, K, blk.data() + rb * n);
        put_weights(sim, fmt, N, K, off, blk.data());
    };
    if (c.id == 3) { weights(TSR_FMT_Q8_0, c.p[0], c.p[1], c.p[2]); weights(TSR_FMT_Q4_K, c.p[0], c.p[1], c.p[3]); }
    if (c.id == 6) weights(c.p[3], c.p[0], c.p[1], c.p[2]);
    if (c.id == 7 || c.id == 10) weights(TSR_FMT_Q8_0, 16, 32, W);
    if (c.id == 5) {
        for (uint32_t ch = 0; ch < tsr_sim_config(sim)->num_ch; ++ch) {
            memset(tsr_sim_dram(sim, ch) + uint64_t(c.p[3]) * 64u, 0, 17u * 64u);
            memset(tsr_sim_dram(sim, ch) + uint64_t(c.p[4]) * 64u, 0, 17u * 64u);
        }
    }
}

void test_selftest() {
    const uint64_t W = TSR_REGION_WEIGHT_OFF / 64u;
    const tsr_config cfg = cfg_u200(1ull << 30);
    const st_case cases[] = {
        { 1, { 300, 0x400000u }, 2, 301, 5 },
        { 2, { 1024, 8 }, 2, tsr_st2(1024, 8).end, 0 },
        { 3, { 200, 512, uint32_t(W), uint32_t(W + 0x10000u) }, 4, tsr_st3(200, 512, 8).end, 0 },
        { 4, { 40, 5, 0x3F5DF0E3u, 0x3F800000u, 0x3F000000u, 0x3CCCCCCDu, 0x358637BDu }, 7, tsr_st4(40, 5).end, 0 },
        { 5, { 40, 4, 0x3E000000u, 0xC00000u, 0xC00100u }, 5, tsr_st5(40, 4, 4).end, 0 },
        { 6, { 96, 512, uint32_t(W + 0x20000u), 1, 9 }, 5, tsr_st6(96, 512, 9).end, 0 },
        { 7, { 0xFFFFFFF0u, uint32_t(W) }, 2, tsr_st7().end, 0 },
        { 8, { 1000 }, 1, tsr_st8().end, 0x1234567u },
        { 9, { 0 }, 0, 1, 0x9D9C0DEu },
        { 10, { uint32_t(W) }, 1, tsr_st10().end, 0 },
    };
    static uint32_t n_ev;
    static uint32_t ev[8];
    tsr::host_event_sink = [](uint32_t type, uint32_t info, const uint32_t d[6]) {
        ev[0] = type; ev[1] = info;
        for (int i = 0; i < 6; ++i) ev[2 + i] = d[i];
        ++n_ev;
    };
    for (const st_case & c : cases) {
        const uint64_t area = TSR_REGION_IO_OFF / 64u + 0x10000u;
        // Host build.
        tsr_sim * hs = make_sim(cfg, "fixed", 2);
        st_fill(hs, c, area);
        tsr::host_bind(hs);
        tsr::host_spm_lines = cfg.spm_bytes / 64u;
        tsr_args a{};
        a.a[0] = c.id; a.a[1] = uint32_t(area); a.a[2] = c.lines; a.a[3] = c.seed;
        n_ev = 0;
        const tsr::host_result hr = tsr::host_run(&selftest, a);
        tsr::host_bind(nullptr);
        CHECK(hr.ret == 0 && hr.err_code == 0, "selftest %u (host build): RET 0x%08x, first error %u", c.id, uint32_t(hr.ret),
              hr.err_code);
        if (c.id == 9)
            CHECK(n_ev == 1 && ev[0] == TSR_ST9_EVENT && ev[1] == (c.seed & 0xFFFFu) && ev[2] == c.seed && ev[3] == ~c.seed,
                  "selftest 9 (host build): event");
        // RV image, the same inputs.
        tsr_sim * rs = make_sim(cfg, "fixed", 2);
        st_fill(rs, c, area);
        fwh h;
        const bool ok = fwh_open(&h, rs, g_image) == 0;
        CHECK(ok, "selftest %u: boot", c.id);
        const int idx = ok ? fwh_kernel(&h, "selftest") : -1;
        if (ok && idx >= 0) {
            CHECK(fwh_launch(&h, uint32_t(idx), a.a, 1ull << 34) == 0 && fwh_status(&h) == 0 && fwh_ret(&h) == hr.ret,
                  "selftest %u (RV): status %u RET 0x%08x, host build 0x%08x", c.id, fwh_status(&h), uint32_t(fwh_ret(&h)),
                  uint32_t(hr.ret));
            CHECK(memcmp(tsr_sim_dram(hs, 0) + area * 64u, tsr_sim_dram(rs, 0) + area * 64u, size_t(c.lines) * 64u) == 0,
                  "selftest %u: the data area differs between the host build and the RV image", c.id);
            if (c.id == 1)
                for (uint32_t ch = 0; ch < cfg.num_ch; ++ch)
                    CHECK(memcmp(tsr_sim_dram(hs, ch) + 0x400000ull * 64u, tsr_sim_dram(rs, ch) + 0x400000ull * 64u, 2048u * 64u) == 0,
                          "selftest 1: channel %u scratch differs", ch);
            if (c.id == 5)
                for (uint32_t ch = 0; ch < cfg.num_ch; ++ch)
                    CHECK(memcmp(tsr_sim_dram(hs, ch) + uint64_t(c.p[3]) * 64u, tsr_sim_dram(rs, ch) + uint64_t(c.p[3]) * 64u, 0x111u * 64u) == 0,
                          "selftest 5: channel %u planes differ", ch);
        }
        if (ok) fwh_close(&h);
        tsr_sim_destroy(rs);
        tsr_sim_destroy(hs);
    }
    tsr::host_event_sink = nullptr;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s IMAGE [TRACE_DIR]\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    if (FILE * f = fopen(g_image, "rb")) fclose(f);
    else {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    test_mulmat("fixed");
    test_mulmat("random:5");
    test_mulmat_fast_vs_general();
    test_replay(argc > 2 ? argv[2] : nullptr);
    test_selftest();
    return test_finish("fw_kernels_host");
}
