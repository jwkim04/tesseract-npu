// test_fw_gdn_host.cpp — fw_gdn_host: GATED_DELTA_NET's token block pipeline (sw/ctrl/kernels/gdn_kinds.cpp gdn_run_tok;
// docs/design/06-software-verification.md "SW" §7.10, SW-70) compiled for the host (SW §6.8, SW-25: tsr_fw_host
// under ASan + UBSan). The records run on tsr_rv in fw_gdn_rv; this test holds what the host build can reach directly:
//
//   1. the plans (tsr/plan_gdn.hpp): the pipeline and its Gs, R and block on a grid of limits against a recomputation,
//      a device without CAPS_GDN and S_v 16 / 32 plan as before, every record that planned still plans;
//   2. the exact push sequence of a small record with one token (two state tiles) and of a small record with two tokens
//      (resident state tiles): the ops, their dependencies, their order and the SPM lines and DRAM beats they name;
//   3. random records through the kernel (ASan + UBSan, the race detector fatal): every head and token runs one GDN op,
//      every state is loaded once and stored once, the result rows are stored once, a tile is loaded again only after
//      the ST that reads it in the DMA queue, the returned ticket is the last ST's.
// Exit 0 / 1.
#undef _GNU_SOURCE
#include "test_sim_util.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "tsr/exec.hpp"
#include "tsr/plan_gdn.hpp"
#include "tsr/rt.hpp"
#include "tsr_race.h"
#include "tsr_ref_vu.h"
#include "tsr_sim_int.h"

namespace {

namespace opl = tsr::opl;
namespace pl = tsr::plan;
namespace g = tsr::opl::gdn;

uint32_t HND(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------
struct rec_t {
    uint32_t sv, hk, hv, T, ns;
    uint32_t qhs = 0, vhs = 0, qpad = 0, vpad = 0, gpad = 0;   // head strides, token padding (lines); 0: dense
};
// The tensors of the test's DRAM layout (channel, first beat): q ch0 0x100, k ch1 0x100, v ch2 0x100, g ch3 0x100, beta
// ch0 0x200, result ch1 0x400, state out ch2 0x800, state in ch3 0x1000.
constexpr uint32_t Q_BEAT = 0x100, K_BEAT = 0x100, V_BEAT = 0x100, G_BEAT = 0x100, BETA_BEAT = 0x200, DST_BEAT = 0x400,
                   SOUT_BEAT = 0x800, SIN_BEAT = 0x1000, SCALE_BITS = 0x3db504f3u;
std::vector<uint32_t> make_record(const rec_t & r) {
    const uint32_t L = r.sv / 16u, Lg = pl::vu_lines(r.hv);
    const uint32_t qhs = r.qhs != 0u ? r.qhs : L, vhs = r.vhs != 0u ? r.vhs : L;
    const uint32_t qts = r.hk * qhs + r.qpad, vts = r.hv * vhs + r.vpad, gts = Lg + r.gpad;
    std::vector<uint32_t> a(g::NARGS, 0u);
    a[g::DST] = HND(1, DST_BEAT); a[g::STATE_OUT] = HND(2, SOUT_BEAT); a[g::STATE_IN] = HND(3, SIN_BEAT);
    a[g::Q] = HND(0, Q_BEAT); a[g::Q_HEAD_STRIDE] = qhs; a[g::Q_TOKEN_STRIDE] = qts;
    a[g::K] = HND(1, K_BEAT); a[g::K_HEAD_STRIDE] = qhs; a[g::K_TOKEN_STRIDE] = qts;
    a[g::V] = HND(2, V_BEAT); a[g::V_HEAD_STRIDE] = vhs; a[g::V_TOKEN_STRIDE] = vts;
    a[g::G] = HND(3, G_BEAT); a[g::G_TOKEN_STRIDE] = gts;
    a[g::BETA] = HND(0, BETA_BEAT); a[g::BETA_TOKEN_STRIDE] = gts;
    a[g::S_V] = r.sv; a[g::H_K] = r.hk; a[g::H_V] = r.hv; a[g::N_TOKENS] = r.T; a[g::N_SEQS] = r.ns;
    a[g::SCALE] = SCALE_BITS;
    return a;
}

// ---------------------------------------------------------------------------
// 1. Plans
// ---------------------------------------------------------------------------
pl::plan_limits limits(uint32_t spm_lines, uint32_t caps) {
    const pl::dev_cfg c = {spm_lines, 8u, 32768u, 4u, 9344u, 16u, caps};
    return pl::plan_limits_of(c);
}
constexpr uint32_t CAPS_F = pl::CAPS_VU | pl::CAPS_ATT | pl::CAPS_GDN, CAPS_C = pl::CAPS_VU | pl::CAPS_ATT;

// The pipeline's plan, recomputed: R state tiles and two token blocks of 3 Lg + 2 (H_k + Gs) L lines each.
struct want_t {
    bool tok;
    uint32_t R, gs, blk, tb;
};
want_t model(const rec_t & r, uint32_t budget, uint32_t caps) {
    want_t w{false, 0, 0, 0, 0};
    if (!pl::gdn::use_fused(caps, r.sv)) return w;
    const uint32_t L = r.sv / 16u, Lg = pl::vu_lines(r.hv), SL = r.sv * L;
    auto tbl = [&](uint32_t gs) { return uint32_t(3ull * Lg + 2ull * (uint64_t(r.hk) + gs) * L); };
    auto blk = [&](uint32_t R, uint32_t gs) { return uint64_t(R) * SL + 2ull * tbl(gs); };
    if (r.T == 1u) {
        if (blk(2u, r.hv) <= budget) w = {true, 2u, r.hv, uint32_t(blk(2u, r.hv)), tbl(r.hv)};
    } else {
        uint32_t gs = 0;
        for (uint32_t c = 1; c <= r.hv; ++c)
            if (blk(c, c) <= budget) gs = c;
        if (gs != 0u) w = {true, gs, gs, uint32_t(blk(gs, gs)), tbl(gs)};
    }
    return w;
}

void test_plans() {
    const uint32_t spms[] = {65536u, 8192u, 4368u, 2048u};
    long n_tok = 0, n_old = 0, n_none = 0;
    for (uint32_t spm_lines : spms) {
        const pl::plan_limits lf = limits(spm_lines, CAPS_F), lc = limits(spm_lines, CAPS_C);
        for (uint32_t sv : {16u, 32u, 64u, 128u, 256u})
            for (uint32_t hk : {1u, 2u, 4u, 16u, 25u, 64u})
                for (uint32_t mul : {1u, 3u})
                    for (uint32_t T : {1u, 2u, 5u, 64u}) {
                        const rec_t r{sv, hk, hk * mul, T, 1u};
                        const std::vector<uint32_t> a = make_record(r);
                        tsr::kind_plan kf{}, kc{};
                        const bool okf = tsr::plan_gated_delta_net(a.data(), g::NARGS, lf, &kf);
                        const bool okc = tsr::plan_gated_delta_net(a.data(), g::NARGS, lc, &kc);
                        const uint32_t L = sv / 16u, old_blk = 3u * sv * L + 2u * (5u * L + 2u);
                        const bool old_fits = old_blk <= (lf.budget >> 1);
                        CHECK(okc == old_fits, "sv %u hk %u hv %u T %u spm %u: a device without CAPS_GDN plans as before", sv, hk, hk * mul, T, spm_lines);
                        if (okc)
                            CHECK(kc.scratch_lines == old_blk && kc.aux[0] == 2u && kc.aux[1] == 5u * L + 2u && kc.aux[2] == 0u && kc.aux[3] == 0u,
                                  "old plan: block %u aux %u %u %u %u", kc.scratch_lines, kc.aux[0], kc.aux[1], kc.aux[2], kc.aux[3]);
                        const want_t w = model(r, lf.budget, CAPS_F);
                        if (w.tok) {
                            ++n_tok;
                            CHECK(okf && kf.status == opl::RET_OK && kf.scratch_lines == w.blk && kf.aux[0] == 2u && kf.aux[1] == w.tb &&
                                      kf.aux[2] == w.R && kf.aux[3] == w.gs && kf.row_lines == L && kf.tile_rows == sv && kf.n_tiles == hk * mul &&
                                      w.blk <= lf.budget,
                                  "token block plan sv %u hk %u hv %u T %u spm %u: ok %d block %u (want %u) R %u (%u) Gs %u (%u) tb %u (%u)", sv, hk,
                                  hk * mul, T, spm_lines, okf, kf.scratch_lines, w.blk, kf.aux[2], w.R, kf.aux[3], w.gs, kf.aux[1], w.tb);
                        } else {   // the old plan (the head by head pipeline, on the fused path or not), or none
                            CHECK(okf == old_fits && (!okf || (kf.scratch_lines == old_blk && kf.aux[2] == 0u && kf.aux[3] == 0u)),
                                  "sv %u hk %u hv %u T %u spm %u: not on the pipeline: ok %d, the old plan fits %d", sv, hk, hk * mul, T, spm_lines, okf,
                                  old_fits);
                            if (okf) ++n_old;
                            else ++n_none;
                        }
                        CHECK(!okc || okf, "a record that planned without CAPS_GDN plans with it");
                    }
    }
    {   // the release shapes on the U200 (the 27B: S_v 128, H_k 16, H_v 48): one group of all the heads
        const pl::plan_limits l = limits(65536u, CAPS_F);
        tsr::kind_plan k{};
        const std::vector<uint32_t> a27 = make_record({128u, 16u, 48u, 1u, 1u});
        CHECK(tsr::plan_gated_delta_net(a27.data(), g::NARGS, l, &k) && k.aux[2] == 2u && k.aux[3] == 48u && k.aux[1] == 3u * 3u + 2u * 64u * 8u &&
                  k.scratch_lines == 2u * 1024u + 2u * k.aux[1],
              "27B decode: R %u Gs %u block %u", k.aux[2], k.aux[3], k.scratch_lines);
        const std::vector<uint32_t> p27 = make_record({128u, 16u, 48u, 78u, 1u});
        CHECK(tsr::plan_gated_delta_net(p27.data(), g::NARGS, l, &k) && k.aux[2] == 48u && k.aux[3] == 48u &&
                  k.scratch_lines == 48u * 1024u + 2u * k.aux[1],
              "27B prompt: R %u Gs %u block %u", k.aux[2], k.aux[3], k.scratch_lines);
        CHECK(pl::plan_op(pl::dev_cfg{65536u, 8u, 32768u, 4u, 9344u, 16u, CAPS_F}, opl::K_GATED_DELTA_NET, p27.data(), g::NARGS).status == opl::RET_OK,
              "plan_op of the 27B prompt record");
    }
    printf("  gdn plans: %ld on the token block pipeline, %ld head by head, %ld refused\n", n_tok, n_old, n_none);
    CHECK(n_tok > 100 && n_old > 20, "the grid reaches both pipelines");
}

// ---------------------------------------------------------------------------
// Running the handler and capturing its pushes
// ---------------------------------------------------------------------------
struct pev {
    uint32_t unit, ticket, w[8], dep[2];
};
void cap_pushes(void * c, tsr_sim *, const tsr_q_event * ev) {
    if (ev->kind != TSR_QEV_PUSH) return;
    pev e{};
    e.unit = ev->unit;
    e.ticket = ev->ticket;
    for (int k = 0; k < 8; ++k) e.w[k] = ev->w[k];
    e.dep[0] = ev->dep[0];
    e.dep[1] = ev->dep[1];
    static_cast<std::vector<pev> *>(c)->push_back(e);
}

struct run_out {
    std::vector<pev> pushes;
    uint32_t ret = 0, err = 0, ret_ticket = 0, ring_line = 0, plan_status = 0, gs = 0, R = 0;
};

// Runs kind_gated_delta_net on a fresh simulator (host backend) with zero data (the pushes do not depend on it).
void run_record(const rec_t & r, uint32_t spm_bytes, run_out * o) {
    tsr_config cfg = TSR_CONFIG_U200;
    cfg.t_max = 8;
    cfg.spm_bytes = spm_bytes;
    cfg.ch_bytes = 1ull << 30;
    cfg.caps = TSR_SIM_CAPS_IMPLEMENTED;
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.race = 2;
    so.threads = 2;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &so);
    CHECK(sim != nullptr, "gdn: simulator");
    if (!sim) return;
    tsr::host_bind(sim);
    const uint32_t spm_lines = cfg.spm_bytes / 64u;
    tsr::host_spm_lines = spm_lines;
    tsr::plan::exec_layout lay{};
    CHECK(tsr::plan::layout(spm_lines, 0, &lay), "gdn: layout");
    const pl::dev_cfg dc = {spm_lines, cfg.t_max, cfg.k_max, cfg.num_ch, 9344u, 16u, CAPS_F};
    tsr::kind_ctx ctx{};
    static tsr::spm_ring ring;
    ring.init(lay.ring_line, lay.ring_lines);
    ctx.ring = &ring;
    ctx.cfg = &dc;
    ctx.mml = pl::mm_limits_of(dc);
    ctx.pl = pl::plan_limits_of(dc);
    ctx.one = tsr::spm{lay.one_line};
    ctx.one_lines = lay.one_lines;
    o->ring_line = lay.ring_line;
    const std::vector<uint32_t> a = make_record(r);
    tsr::kind_plan kp{};
    tsr::plan_gated_delta_net(a.data(), g::NARGS, ctx.pl, &kp);
    o->plan_status = kp.status;
    o->R = kp.aux[2];
    o->gs = kp.aux[3];
    const int lid = tsr_sim_add_listener(sim, cap_pushes, &o->pushes);
    const tsr::ticket t = tsr::kind_gated_delta_net(a.data(), g::NARGS, 0u, tsr::deps{}, ctx);
    o->ret = ctx.ret;
    o->ret_ticket = t.v;
    tsr_sim_remove_listener(sim, lid);   // before join_all, which pushes a FENCE
    tsr::join_all();
    o->err = tsr::err_units();
    const tsr_race_report * rr = nullptr;
    CHECK(tsr_sim_race_reports(sim, &rr) == 0u, "gdn: race reports");
    tsr::host_bind(nullptr);
    tsr_sim_destroy(sim);
}

// One push as the test compares it: the op, and the words that tell what it moves, the SPM lines relative to the ring
// block B (the state tiles start there).
//   L, S: ch, DRAM beat, SPM line, lines;   V (VOP): function, m, rows, d line, a line;
//   P (SOP): function, sd, address (LDS) or immediate (MOVI);   G (GDN): sa, sb, d, a, k, q, v, o lines, m, rows.
struct xpush {
    char op;
    int d0, d1;
    long k[10];
};
std::vector<long> normalize(const pev & p, uint32_t B) {
    auto rel = [&](uint32_t line) { return long(line) - long(B); };
    switch (p.w[0] & 0xFFu) {
    case tsr::OP_LD: return {long((p.w[0] >> tsr::LD_CH_LSB) & 0xFu), long(p.w[2]), rel(p.w[1]), long(p.w[3])};
    case tsr::OP_ST: return {long((p.w[0] >> tsr::ST_CH_LSB) & 0xFu), long(p.w[2]), rel(p.w[1]), long(p.w[3])};
    case tsr::OP_VOP: return {long((p.w[0] >> 16) & 0xFFu), long(p.w[5]), long(p.w[6] & 0xFFFFu), rel(p.w[1]), rel(p.w[2])};
    case tsr::OP_SOP:
        return {long((p.w[0] >> 8) & 0xFu), long(p.w[1] & 0xFu), (((p.w[0] >> 8) & 0xFu) == tsr::SOP_SUB_MOVI) ? long(p.w[2]) : long(p.w[3])};
    case tsr::OP_GDN:
        return {long((p.w[0] >> 16) & 0xFu), long((p.w[0] >> 20) & 0xFu), rel(p.w[1]), rel(p.w[2]), rel(p.w[3]), rel(p.w[4]),
                rel(p.w[5]), rel(p.w[6]), long(p.w[7] & 0xFFFFu), long(p.w[7] >> 16)};
    default: return {};
    }
}
char opchar(uint32_t w0) {
    switch (w0 & 0xFFu) {
    case tsr::OP_LD: return 'L';
    case tsr::OP_ST: return 'S';
    case tsr::OP_VOP: return 'V';
    case tsr::OP_SOP: return 'P';
    case tsr::OP_GDN: return 'G';
    default: return '?';
    }
}
size_t nk(char op) { return op == 'L' || op == 'S' ? 4 : op == 'V' ? 5 : op == 'P' ? 3 : 10; }

void check_sequence(const char * what, const run_out & o, const std::vector<xpush> & exp) {
    std::map<uint32_t, int> idx;
    for (size_t i = 0; i < o.pushes.size(); ++i) idx[o.pushes[i].ticket] = int(i);
    CHECK(o.pushes.size() == exp.size(), "%s: %zu pushes, expected %zu", what, o.pushes.size(), exp.size());
    for (size_t i = 0; i < o.pushes.size() && i < exp.size(); ++i) {
        const pev & p = o.pushes[i];
        const xpush & x = exp[i];
        auto dep = [&](uint32_t t) {
            if ((t >> 28) == 0xFu) return -1;   // TSR_TICKET_NONE: unit 15
            const auto it = idx.find(t);
            return it == idx.end() ? -2 : it->second;
        };
        int d0 = dep(p.dep[0]), d1 = dep(p.dep[1]);
        if (d0 < d1) std::swap(d0, d1);   // the two deps in either slot
        int e0 = x.d0, e1 = x.d1;
        if (e0 < e1) std::swap(e0, e1);
        const std::vector<long> got = normalize(p, o.ring_line);
        bool same = opchar(p.w[0]) == x.op && d0 == e0 && d1 == e1 && got.size() == nk(x.op);
        for (size_t j = 0; same && j < got.size(); ++j) same = got[j] == x.k[j];
        if (!same) {
            std::string gs, es;
            for (long v : got) gs += " " + std::to_string(v);
            for (size_t j = 0; j < nk(x.op); ++j) es += " " + std::to_string(x.k[j]);
            CHECK(0, "%s: push %zu: %c deps (%d, %d)%s, expected %c deps (%d, %d)%s", what, i, opchar(p.w[0]), d0, d1, gs.c_str(), x.op, e0, e1,
                  es.c_str());
        }
    }
}

// The expected sequence, written as the design says it (SW-70): the helpers only spell the words.
struct seq_t {
    std::vector<xpush> v;
    void ld(int ch, long beat, long spm, long lines, int d0 = -1, int d1 = -1) { v.push_back({'L', d0, d1, {ch, beat, spm, lines}}); }
    void st(int ch, long beat, long spm, long lines, int d0 = -1, int d1 = -1) { v.push_back({'S', d0, d1, {ch, beat, spm, lines}}); }
    void vop(long func, long m, long rows, long d, long a, int d0 = -1) { v.push_back({'V', d0, -1, {func, m, rows, d, a}}); }
    void sop(long sub, long sd, long val, int d0 = -1) { v.push_back({'P', d0, -1, {sub, sd, val}}); }
    void gdn(long d, long k, long q, long vv, long o, int d0 = -1, int d1 = -1) {
        v.push_back({'G', d0, d1, {15, 14, d, d, k, q, vv, o, 64, 64}});
    }
    int next() const { return int(v.size()); }
};
constexpr long S_ALPHA = 15, S_BETA = 14, S_SCALE = 13;

// S_v 64 (L = 4, a state tile of 256 lines), H_k 2, H_v 4 (Lg 1), one sequence, dense rows. T = 1: two state tiles
// (lines 0 and 256), then two token blocks of 3 + 2 (2 + 4) 4 = 51 lines at 512 and 563; the first has g 512,
// beta 513, alpha 514, k 515, q 523, v 531, o 547.
void test_sequence_t1() {
    const rec_t r{64u, 2u, 4u, 1u, 1u};
    run_out o;
    run_record(r, 4u << 20, &o);
    CHECK(o.ret == 0u && o.err == 0u && o.R == 2u && o.gs == 4u, "T = 1: ret 0x%x, error %u, R %u, Gs %u", o.ret, o.err, o.R, o.gs);
    seq_t s;
    s.sop(tsr::SOP_SUB_MOVI, S_SCALE, SCALE_BITS);                               // 0 the scale
    s.ld(3, G_BEAT, 512, 1);                                                      // 1 g
    s.ld(0, BETA_BEAT, 513, 1);                                                   // 2 beta
    s.ld(1, K_BEAT, 515, 8);                                                      // 3 k rows (both key heads)
    s.ld(0, Q_BEAT, 523, 8);                                                      // 4 q rows
    s.ld(2, V_BEAT, 531, 16);                                                     // 5 v rows (the four heads)
    s.ld(3, SIN_BEAT, 0, 256);                                                    // 6 state of head 0
    s.ld(3, SIN_BEAT + 256, 256, 256);                                            // 7 state of head 1
    s.vop(tsr::VOP_FUNC_EXP, 4, 1, 514, 512, 1);                                  // 8 alpha = exp(g)
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 514 + 0);                               // 9 head 0
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 513 + 0, 2);                             // 10
    s.gdn(0, 515, 523, 531, 547, 6);                                              // 11 after the state LD (which follows the rows)
    s.st(2, SOUT_BEAT, 0, 256, 11);                                               // 12 the state
    s.ld(3, SIN_BEAT + 512, 0, 256);                                              // 13 state of head 2, into the tile just stored
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 514 + 1);                               // 14 head 1
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 513 + 1);                                // 15
    s.gdn(256, 519, 527, 535, 551, 7);                                            // 16 k and q row 1, v and o row 1
    s.st(2, SOUT_BEAT + 256, 256, 256, 16);                                       // 17
    s.ld(3, SIN_BEAT + 768, 256, 256);                                            // 18 state of head 3
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 514 + 2);                               // 19 head 2
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 513 + 2);                                // 20
    s.gdn(0, 515, 523, 539, 555, 13);                                             // 21 k and q row 0 again (2 key heads for 4 value heads)
    s.st(2, SOUT_BEAT + 512, 0, 256, 21);                                         // 22
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 514 + 3);                               // 23 head 3
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 513 + 3);                                // 24
    s.gdn(256, 519, 527, 543, 559, 18);                                           // 25
    s.st(2, SOUT_BEAT + 768, 256, 256, 25);                                       // 26
    s.vop(tsr::VOP_FUNC_MULC, 256, 1, 547, 547, -1);                              // 27 the scale over the four rows of o
    s.st(1, DST_BEAT, 547, 16, 27);                                               // 28 the result rows of the token
    check_sequence("T = 1, four heads", o, s.v);
    CHECK(!o.pushes.empty() && o.ret_ticket == o.pushes.back().ticket && opchar(o.pushes.back().w[0]) == 'S',
          "T = 1: the returned ticket is the last ST's");
    printf("  push sequence of a record of four heads and one token: %zu pushes\n", o.pushes.size());
}

// The same shape with T = 2: four resident state tiles (lines 0, 256, 512, 768), then two token blocks of 51 lines at
// 1024 and 1075 (g, beta, alpha, k, q, v, o at +0, +1, +2, +3, +11, +19, +35).
void test_sequence_t2() {
    const rec_t r{64u, 2u, 4u, 2u, 1u};
    run_out o;
    run_record(r, 4u << 20, &o);
    CHECK(o.ret == 0u && o.err == 0u && o.R == 4u && o.gs == 4u, "T = 2: ret 0x%x, error %u, R %u, Gs %u", o.ret, o.err, o.R, o.gs);
    seq_t s;
    s.sop(tsr::SOP_SUB_MOVI, S_SCALE, SCALE_BITS);                               // 0
    s.ld(3, G_BEAT, 1024, 1);                                                     // 1 step 0 (token 0): g
    s.ld(0, BETA_BEAT, 1025, 1);                                                  // 2 beta
    s.ld(1, K_BEAT, 1027, 8);                                                     // 3 k
    s.ld(0, Q_BEAT, 1035, 8);                                                     // 4 q
    s.ld(2, V_BEAT, 1043, 16);                                                    // 5 v
    s.ld(3, SIN_BEAT, 0, 256);                                                    // 6 state of head 0
    s.ld(3, SIN_BEAT + 256, 256, 256);                                            // 7 state of head 1
    s.ld(3, G_BEAT + 1, 1075, 1);                                                 // 8 step 1 (token 1): g
    s.ld(0, BETA_BEAT + 1, 1076, 1);                                              // 9
    s.ld(1, K_BEAT + 8, 1078, 8);                                                 // 10
    s.ld(0, Q_BEAT + 8, 1086, 8);                                                 // 11
    s.ld(2, V_BEAT + 16, 1094, 16);                                               // 12
    s.vop(tsr::VOP_FUNC_EXP, 4, 1, 1026, 1024, 1);                                // 13 alpha of step 0
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 1026 + 0);                              // 14 head 0
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 1025 + 0, 2);                            // 15
    s.gdn(0, 1027, 1035, 1043, 1059, 6);                                          // 16
    s.ld(3, SIN_BEAT + 512, 512, 256);                                            // 17 state of head 2 (head 0 + 2, in its own tile)
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 1026 + 1);                              // 18 head 1
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 1025 + 1);                               // 19
    s.gdn(256, 1031, 1039, 1047, 1063, 7);                                        // 20
    s.ld(3, SIN_BEAT + 768, 768, 256);                                            // 21 state of head 3
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 1026 + 2);                              // 22 head 2
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 1025 + 2);                               // 23
    s.gdn(512, 1027, 1035, 1051, 1067, 17);                                       // 24
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 1026 + 3);                              // 25 head 3
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 1025 + 3);                               // 26
    s.gdn(768, 1031, 1039, 1055, 1071, 21);                                       // 27
    s.vop(tsr::VOP_FUNC_MULC, 256, 1, 1059, 1059);                                // 28
    s.st(1, DST_BEAT, 1059, 16, 28);                                              // 29 result rows of token 0
    s.vop(tsr::VOP_FUNC_EXP, 4, 1, 1077, 1075, 8);                                // 30 alpha of step 1
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 1077 + 0);                              // 31 head 0
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 1076 + 0, 9);                            // 32
    s.gdn(0, 1078, 1086, 1094, 1110, 12);                                         // 33 the last LD of the block's rows
    s.st(2, SOUT_BEAT, 0, 256, 33);                                               // 34 the state: the last token
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 1077 + 1);                              // 35 head 1
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 1076 + 1);                               // 36
    s.gdn(256, 1082, 1090, 1098, 1114);                                           // 37 no dependency: the first head's covers it
    s.st(2, SOUT_BEAT + 256, 256, 256, 37);                                       // 38
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 1077 + 2);                              // 39 head 2
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 1076 + 2);                               // 40
    s.gdn(512, 1078, 1086, 1102, 1118);                                           // 41
    s.st(2, SOUT_BEAT + 512, 512, 256, 41);                                       // 42
    s.sop(tsr::SOP_SUB_LDS, S_ALPHA, 16 * 1077 + 3);                              // 43 head 3
    s.sop(tsr::SOP_SUB_LDS, S_BETA, 16 * 1076 + 3);                               // 44
    s.gdn(768, 1082, 1090, 1106, 1122);                                           // 45
    s.st(2, SOUT_BEAT + 768, 768, 256, 45);                                       // 46
    s.vop(tsr::VOP_FUNC_MULC, 256, 1, 1110, 1110);                                // 47
    s.st(1, DST_BEAT + 16, 1110, 16, 47);                                         // 48 result rows of token 1
    check_sequence("T = 2, four heads", o, s.v);
    CHECK(!o.pushes.empty() && o.ret_ticket == o.pushes.back().ticket && opchar(o.pushes.back().w[0]) == 'S',
          "T = 2: the returned ticket is the last ST's");
    printf("  push sequence of a record of four heads and two tokens: %zu pushes\n", o.pushes.size());
}

// ---------------------------------------------------------------------------
// 3. Random records
// ---------------------------------------------------------------------------
void check_invariants(const char * what, const rec_t & r, const run_out & o) {
    const uint32_t L = r.sv / 16u, SL = r.sv * L, nheads = r.ns * r.hv, nt = r.ns * r.T;
    long n_gdn = 0, n_ld_state = 0, n_st_state = 0;
    std::vector<int> beat_ld(nheads, 0), beat_st(nheads, 0), rows(size_t(nt) * r.hv * L, 0);
    std::map<uint32_t, int> idx;
    for (size_t i = 0; i < o.pushes.size(); ++i) idx[o.pushes[i].ticket] = int(i);
    std::map<uint32_t, std::vector<std::pair<size_t, char>>> tile;   // SPM line -> the pushes that load or store a state there
    for (size_t i = 0; i < o.pushes.size(); ++i) {
        const pev & p = o.pushes[i];
        const uint32_t ch = ((p.w[0] >> tsr::LD_CH_LSB) & 0xFu);
        switch (p.w[0] & 0xFFu) {
        case tsr::OP_GDN: ++n_gdn; break;
        case tsr::OP_LD:
            if (ch == 3u && p.w[2] >= SIN_BEAT) {
                const uint32_t x = (p.w[2] - SIN_BEAT) / SL;
                CHECK((p.w[2] - SIN_BEAT) % SL == 0u && x < nheads && p.w[3] == SL, "%s: state LD at beat 0x%x lines %u", what, p.w[2], p.w[3]);
                if (x < nheads) ++beat_ld[x];
                ++n_ld_state;
                tile[p.w[1]].push_back({i, 'L'});
            }
            break;
        case tsr::OP_ST:
            if (ch == 2u && p.w[2] >= SOUT_BEAT) {
                const uint32_t x = (p.w[2] - SOUT_BEAT) / SL;
                CHECK((p.w[2] - SOUT_BEAT) % SL == 0u && x < nheads && p.w[3] == SL, "%s: state ST at beat 0x%x lines %u", what, p.w[2], p.w[3]);
                if (x < nheads) ++beat_st[x];
                ++n_st_state;
                tile[p.w[1]].push_back({i, 'S'});
            } else if (ch == 1u) {
                for (uint32_t l = 0; l < p.w[3]; ++l) {
                    const size_t at = size_t(p.w[2] - DST_BEAT + l);
                    CHECK(at < rows.size(), "%s: result ST beyond the result", what);
                    if (at < rows.size()) ++rows[at];
                }
            }
            break;
        default: break;
        }
    }
    CHECK(n_gdn == long(nheads) * r.T, "%s: %ld GDN ops, expected %u", what, n_gdn, nheads * r.T);
    CHECK(n_ld_state == long(nheads) && n_st_state == long(nheads), "%s: %ld state LDs and %ld STs, expected %u", what, n_ld_state, n_st_state, nheads);
    CHECK(std::all_of(beat_ld.begin(), beat_ld.end(), [](int c) { return c == 1; }) &&
              std::all_of(beat_st.begin(), beat_st.end(), [](int c) { return c == 1; }),
          "%s: a state is not loaded and stored exactly once", what);
    CHECK(std::all_of(rows.begin(), rows.end(), [](int c) { return c == 1; }), "%s: a result line is not stored exactly once", what);
    // a tile is loaded again only after the ST that reads it: L, S, L, S ... on every state tile
    for (const auto & kv : tile) {
        char want = 'L';
        bool ok = true;
        for (const auto & e : kv.second) {
            ok = ok && e.second == want;
            want = want == 'L' ? 'S' : 'L';
        }
        CHECK(ok && want == 'L', "%s: the pushes on state tile line %u are not load, store, load, store ...", what, kv.first);
    }
    CHECK(o.ret_ticket == o.pushes.back().ticket && opchar(o.pushes.back().w[0]) == 'S', "%s: the returned ticket is not the last ST's", what);
}

void test_random() {
    rng_t rg = {0x6D0A};
    long n_tok = 0, n_old = 0, n_skip = 0;
    for (int it = 0; it < 48; ++it) {
        rec_t r;
        const uint32_t svs[3] = {64u, 128u, 256u};
        r.sv = svs[rng_below(&rg, 3)];
        r.hk = 1u + rng_below(&rg, 5);
        r.hv = r.hk * (1u + rng_below(&rg, 4));
        r.T = it % 3 == 0 ? 1u : 1u + rng_below(&rg, 5);
        r.ns = 1u + rng_below(&rg, 3);
        const uint32_t L = r.sv / 16u;
        if (rng_below(&rg, 2) != 0u) r.qhs = L + rng_below(&rg, 3);
        if (rng_below(&rg, 2) != 0u) r.vhs = L + rng_below(&rg, 3);
        r.qpad = rng_below(&rg, 3);
        r.vpad = rng_below(&rg, 3);
        r.gpad = rng_below(&rg, 2);
        const uint32_t spm = (it & 1) != 0 ? (512u << 10) : (4u << 20);
        const std::vector<uint32_t> a = make_record(r);
        const pl::dev_cfg dc = {spm / 64u, 8u, 32768u, 4u, 9344u, 16u, CAPS_F};
        tsr::kind_plan kp{};
        if (!tsr::plan_gated_delta_net(a.data(), g::NARGS, pl::plan_limits_of(dc), &kp)) {
            ++n_skip;
            continue;
        }
        char what[160];
        snprintf(what, sizeof(what), "random %d (S_v %u, H_k %u, H_v %u, T %u, n_seqs %u, spm %u KiB)", it, r.sv, r.hk, r.hv, r.T, r.ns, spm >> 10);
        run_out o;
        run_record(r, spm, &o);
        CHECK(o.ret == 0u && o.err == 0u, "%s: ret 0x%x, error %u", what, o.ret, o.err);
        if (o.R != 0u) {
            ++n_tok;
            CHECK(o.gs >= 1u && o.gs <= r.hv && (r.T == 1u ? o.R == 2u : o.R == o.gs), "%s: R %u Gs %u", what, o.R, o.gs);
        } else {
            ++n_old;
        }
        if (o.ret == 0u && o.R != 0u) check_invariants(what, r, o);
    }
    printf("  random records: %ld on the token block pipeline, %ld head by head, %ld not planned\n", n_tok, n_old, n_skip);
    CHECK(n_tok >= 24, "the random records reach the token block pipeline (%ld)", n_tok);
}

}  // namespace

int main() {
    test_plans();
    test_sequence_t1();
    test_sequence_t2();
    test_random();
    return test_finish("fw_gdn_host");
}
