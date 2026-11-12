// test_fw_kernels_rv.cpp — fw_kernels_rv*: the MUL_MAT kind (sw/ctrl/kernels/k_mulmat.cpp, SW §7.4)
// in the RISC-V image on tsr_rv, launched through the `graph` executor with op lists
// (docs/design/06-software-verification.md §10.2 "fw_kernels_rv": per kind, 8 schedules, race fatal,
// outputs bit-exact against the references).
//
//   test_fw_kernels_rv IMAGE suite [--scheds N]
//       Two op lists, each launched under the fixed schedule and 7 random ones with TSR_SIM_RACE
//       fatal: suite A (4 MiB SPM): Q8_0 and Q4_K, T = 1 .. T_MAX and T > T_MAX, the addend forms
//       (bias with stride 0, dense and strided per-token rows), ACC_DST, strided x / y rows, rows on
//       several channels, a dependent op (x = an earlier op's y), an op with three predecessors
//       (JOIN records); suite B (512 KiB SPM): the tile boundaries of plan.hpp (lpt 239 / 240:
//       t_tile 8 / 7; lpt 1912 / 1913: double / single buffer, the ring fallback); suite C (the sweep:
//       Q8_0 and Q4_K x T = 1 .. T_MAX x {no addend, bias row, per-token rows, ACC_DST}). Every y row is
//       compared bit for bit with sw/ctrl/ref/mulmat_ref.cpp; DONE must be OK / RET 0; the SW §4.7
//       observables (per-unit op streams, the RISC-V SPM store stream, the final written DRAM and SPM
//       state, DONE r0-r6) must be equal under every schedule; the written DRAM must be exactly the
//       y rows.
//   test_fw_kernels_rv IMAGE split [--scheds N] [--only K]
//       MUL_MAT_BEGIN / MUL_MAT_END (SW §7.4) in the same harness: the suite A and C records lowered as a BEGIN and an END with
//       other records between them (the END after 0 ... 5 records, at the end of the list, in order or reversed, or never),
//       a list with twelve BEGINs before the first END (the table's rule 2), a small SPM whose ring wraps onto a held block
//       (rule 3: the executor closes the open BEGINs and allocates again) and a ring table full of records behind a held
//       one, a list that ends with BEGINs open (rule 4), and refused lists (END without a BEGIN, END twice, END of a
//       MUL_MAT, BEGIN of more than one T-tile, BEGIN with kind flags: BAD_RECORD, and the BEGINs before the refusal
//       still store their rows; a BEGIN closed by a ninth one, whose END then comes twice), and three launches on one simulator
//       (the open table and the closed bits are per launch: the first stops with BEGINs open after a unit error, the second
//       must run bit-exact and leave its y rows alone, the third must refuse an END of a MUL_MAT). 8 schedules, race fatal,
//       y bit-exact, the written DRAM exactly the y rows.
//   test_fw_kernels_rv IMAGE shapes [--scheds N] [--threads N]
//       Model shapes (Qwen2.5-0.5B, Qwen2.5-32B, Gemma 4 12B; the vocab projections 151,936 x 896,
//       152,064 x 5120 and 262,144 x 3840), T = 1 and T_MAX, Q8_0 and Q4_K where the model has it
//       (slow; default 2 schedules).
//   test_fw_kernels_rv IMAGE v1conv DIR... [--sample N]
//       The lowering check (SW §7.4): every converted MUL_MAT trace of traces/model-v1conv
//       (or a sample of N per set) is regrouped into its MUL_MAT ops (consecutive tiles with the same
//       format, shape and weights and contiguous x / y rows), launched as one op list on a fresh
//       simulator loaded with the trace's init regions, and compared: the expect regions bit for
//       bit (the trace's outputs), the written beats = the expect regions, and the per-unit op streams of
//       the kernel against the trace's push stream modulo the operand forms: SPM line numbers (the ring
//       places the tiles; the kernel's dataflow LD -> QACT -> MVM -> ST is checked instead), the
//       deps (the converter's serial chain vs the kernel's tickets) and the push WMASK. Every
//       converted trace has T_MAX 1, so the stream identity holds for single-tile ops only; for
//       T > T_MAX the kernel's prefetch (LD x_{t+1} before ST_t) differs from the trace's per-tile order by
//       design (SW §7.4), and those outputs are checked against mulmat_ref (suite).
// Exit 0 when everything passed, 1 on a failure, 77 (skipped) when the image or a trace set is
// missing.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "../ref/fw_sim_host.hpp"
#include "../ref/mulmat_ref.hpp"
#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_race.h"
#include "tsr_trace.h"

namespace {

namespace mm = tsr::opl::mm;
constexpr uint32_t NONE = tsr::opl::HANDLE_NONE;
constexpr uint64_t LINE = 64u;

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }

const char * g_image = nullptr;
uint32_t g_race = 2;                   // TSR_SIM_RACE fatal; --race report for the mutant images

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
// N x K weights at W_OFF (symmetric layout), generated and repacked row by row.
void put_weights_rows(tsr_sim * sim, rng_t * r, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off) {
    const size_t row_bytes = size_t(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
    std::vector<uint8_t> blk(row_bytes);
    const uint32_t C = tsr_sim_config(sim)->num_ch;
    for (uint32_t n = 0; n < N; ++n) {
        gen_row_blocks(r, fmt, K, blk.data());
        uint32_t ch;
        uint64_t off;
        tsr_row_location(fmt, N, K, C, n, &ch, &off);
        tsr_repack_row(fmt, blk.data(), K, tsr_sim_dram(sim, ch) + (w_off + off) * LINE);
    }
}

// rows x lines of random f32 (K values per row, the rest of each row's lines zero) at handle h.
void put_rows(tsr_sim * sim, rng_t * r, uint32_t h, uint32_t rows, uint32_t stride, uint32_t K) {
    std::vector<float> x(K);
    for (uint32_t t = 0; t < rows; ++t) {
        gen_f32(r, x.data(), K);
        uint8_t * p = tsr_sim_dram(sim, h >> 28) + (uint64_t(h & 0x0FFFFFFFu) + uint64_t(t) * stride) * LINE;
        put_f32_le(p, x.data(), K);
    }
}

tsr::ref::dram_view view_of(tsr_sim * sim, std::vector<uint8_t *> & chp) {
    const tsr_config * c = tsr_sim_config(sim);
    chp.resize(c->num_ch);
    for (uint32_t i = 0; i < c->num_ch; ++i) chp[i] = tsr_sim_dram(sim, i);
    return tsr::ref::dram_view{chp.data(), c->num_ch, c->ch_bytes};
}

// ---------------------------------------------------------------------------
// One op list (a suite), its data and its expected rows
// ---------------------------------------------------------------------------
struct rec {
    uint32_t a[12];
    std::vector<uint32_t> deps;        // op indices (writer.add folds > 2 into JOINs)
    int x_from = -1;                   // the op whose y is this op's x (data dependency)
    int split = -1;                    // -1: a MUL_MAT; else a MUL_MAT_BEGIN whose END follows after that many more records
                                       // (SPLIT_END: at the end of the list, SPLIT_NEVER: none, the executor closes it)
    uint32_t index = 0;                // its op index in the list
    std::vector<uint8_t> want;         // expected y rows, y_lines * 64 bytes per token
};

constexpr int SPLIT_END = 1000, SPLIT_NEVER = -2;

struct suite {
    const char * name;
    tsr_config cfg;
    std::vector<rec> recs;
    struct wspec { uint32_t fmt, N, K; uint64_t w_off; };
    std::vector<wspec> weights;
    struct xspec { uint32_t h, rows, stride, K; };
    std::vector<xspec> inputs;         // x rows, addend rows, old y rows (ACC_DST)
    uint64_t seed;
    std::vector<uint8_t> blob;         // the op list
    uint32_t opl_h = H(0, 0x40000u);   // ch0 16 MiB: inside the OPL region (EX §11)
    bool reverse_end = false;          // the ENDs left for the end of the list come in reverse order
    std::vector<uint8_t> raw;          // a list written by hand (refused lists), else built from recs
    uint32_t want_ret = 0;             // the RET of the launch (a refused list: opl::RET_BAD_RECORD)
};

uint32_t y_lines_of(const rec & r) { return (r.a[mm::N] + 15u) / 16u; }
uint32_t x_lines_of(const rec & r) { return r.a[mm::K] / 16u; }

// Fills the suite's DRAM (deterministic for its seed) and writes the op list.
void fill(tsr_sim * sim, suite & s) {
    rng_t r = { s.seed };
    for (const auto & w : s.weights) put_weights_rows(sim, &r, w.fmt, w.N, w.K, w.w_off);
    for (const auto & x : s.inputs) put_rows(sim, &r, x.h, x.rows, x.stride, x.K);
    tsr_sim_dram_write(sim, s.opl_h >> 28, uint64_t(s.opl_h & 0x0FFFFFFFu) * LINE, s.blob.data(), s.blob.size());
}

// The list of the suite: a MUL_MAT per record, or, for a record with split >= 0 (SPLIT_NEVER), a MUL_MAT_BEGIN
// and later its MUL_MAT_END (with the BEGIN as its dependency). An END comes before any record that depends on its
// record (the consumers then depend on the END), after the record's gap, or at the end of the list.
void build_list(suite & s) {
    if (!s.raw.empty()) {
        s.blob = s.raw;
        return;
    }
    tsr::opl::writer w(tsr::opl::F_STOP_ON_ERROR, 0x5EED);
    std::vector<uint32_t> at(s.recs.size());                 // the op a consumer of record i depends on
    struct open_t { size_t rec; uint32_t begin; int left; };
    std::vector<open_t> open;
    auto end_of = [&](size_t k) {
        const uint32_t b = open[k].begin;
        at[open[k].rec] = w.mul_mat_end(b, &b, 1);
        open.erase(open.begin() + long(k));
    };
    for (size_t i = 0; i < s.recs.size(); ++i) {
        rec & r = s.recs[i];
        std::vector<uint32_t> deps;
        for (uint32_t d : r.deps) deps.push_back(at[d]);
        for (size_t k = 0; k < open.size();) {
            bool uses = r.x_from == int(open[k].rec);
            for (uint32_t d : r.deps) uses = uses || d == open[k].rec;
            if (uses) {
                end_of(k);
                deps.clear();
                for (uint32_t d : r.deps) deps.push_back(at[d]);
            } else {
                ++k;
            }
        }
        if (r.split == -1) {
            r.index = w.mul_mat(r.a[mm::X], r.a[mm::X_STRIDE], r.a[mm::Y], r.a[mm::Y_STRIDE], r.a[mm::W_OFF], r.a[mm::FMT],
                                r.a[mm::N], r.a[mm::K], r.a[mm::T], r.a[mm::ADD], r.a[mm::ADD_STRIDE], r.a[mm::FLAGS],
                                deps.data(), uint32_t(deps.size()));
            at[i] = r.index;
        } else {
            r.index = w.mul_mat_begin(r.a[mm::X], r.a[mm::X_STRIDE], r.a[mm::Y], r.a[mm::Y_STRIDE], r.a[mm::W_OFF],
                                      r.a[mm::FMT], r.a[mm::N], r.a[mm::K], r.a[mm::T], r.a[mm::ADD], r.a[mm::ADD_STRIDE],
                                      r.a[mm::FLAGS], deps.data(), uint32_t(deps.size()));
            at[i] = r.index;
            open.push_back(open_t{i, r.index, r.split});
        }
        for (size_t k = 0; k < open.size();) {               // the records between a BEGIN and its END
            open_t & o = open[k];
            if (o.rec != i && o.left > 0 && o.left < SPLIT_END) --o.left;
            if (o.left == 0) end_of(k);
            else ++k;
        }
    }
    if (s.reverse_end) std::reverse(open.begin(), open.end());
    for (size_t k = 0; k < open.size();) {
        if (open[k].left == SPLIT_NEVER) ++k;
        else end_of(k);
    }
    s.blob = w.finish();
    CHECK(w.ok() && !s.blob.empty(), "%s: OPL writer: %s", s.name, w.error());
}

// Expected y rows of every record (the reference on the filled DRAM; outputs go to a scratch
// copy so the DRAM stays as the launch will find it).
void expect(tsr_sim * sim, suite & s, uint32_t threads) {
    std::vector<uint8_t *> chp;
    const tsr::ref::dram_view v = view_of(sim, chp);
    for (size_t i = 0; i < s.recs.size(); ++i) {
        rec & r = s.recs[i];
        const uint32_t yl = y_lines_of(r), xl = x_lines_of(r), T = r.a[mm::T], K = r.a[mm::K], N = r.a[mm::N];
        r.want.assign(size_t(T) * yl * LINE, 0);
        const bool acc = (r.a[mm::FLAGS] & mm::FLAG_ACC_DST) != 0u, has_add = r.a[mm::ADD] != NONE;
        std::vector<float> x(K), add(N), y(N);
        for (uint32_t t = 0; t < T; ++t) {
            const uint8_t * xp;
            if (r.x_from >= 0) xp = s.recs[size_t(r.x_from)].want.data() + size_t(t) * y_lines_of(s.recs[size_t(r.x_from)]) * LINE;
            else xp = v.ch[r.a[mm::X] >> 28] + (uint64_t(r.a[mm::X] & 0x0FFFFFFFu) + uint64_t(t) * r.a[mm::X_STRIDE]) * LINE;
            (void)xl;
            for (uint32_t k = 0; k < K; ++k) x[k] = u2f(get_u32_le(xp + 4u * k));
            const float * addp = nullptr;
            if (has_add || acc) {
                const uint32_t h = has_add ? r.a[mm::ADD] : r.a[mm::Y];
                const uint32_t st = has_add ? r.a[mm::ADD_STRIDE] : r.a[mm::Y_STRIDE];
                const uint8_t * ap = v.ch[h >> 28] + (uint64_t(h & 0x0FFFFFFFu) + uint64_t(t) * st) * LINE;
                for (uint32_t n = 0; n < N; ++n) add[n] = u2f(get_u32_le(ap + 4u * n));
                addp = add.data();
            }
            tsr::ref::mulmat_ref_token(v, r.a[mm::FMT], N, K, r.a[mm::W_OFF], x.data(), addp, y.data(), threads);
            put_f32_le(r.want.data() + size_t(t) * yl * LINE, y.data(), N);
        }
    }
}

// The DRAM beats the list writes: every op's y rows.
std::vector<ival_t> written_rows(const suite & s) {
    std::vector<ival_t> v;
    for (const auto & r : s.recs) {
        const uint32_t yl = y_lines_of(r);
        for (uint32_t t = 0; t < r.a[mm::T]; ++t) {
            const uint64_t lo = (r.a[mm::Y] & 0x0FFFFFFFu) + uint64_t(t) * r.a[mm::Y_STRIDE];
            v.push_back(ival_t{r.a[mm::Y] >> 28, lo, lo + yl});
        }
    }
    std::sort(v.begin(), v.end(), [](const ival_t & a, const ival_t & b) { return a.ch != b.ch ? a.ch < b.ch : a.lo < b.lo; });
    std::vector<ival_t> m;
    for (const auto & x : v) {
        if (!m.empty() && m.back().ch == x.ch && x.lo <= m.back().hi) m.back().hi = std::max(m.back().hi, x.hi);
        else m.push_back(x);
    }
    return m;
}

struct run_obs {
    obs o;
    uint32_t done[7];
};

// One launch of the suite's list under schedule `sched` on a fresh simulator.
bool run_suite(suite & s, const char * sched, bool first, uint32_t threads, run_obs * out) {
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = g_race;
    so.threads = threads;
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    CHECK(sim != nullptr, "%s: simulator", s.name);
    if (sim == nullptr) return false;
    fill(sim, s);
    if (first) expect(sim, s, threads);
    obs_attach(&out->o, sim);
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    CHECK(ok, "%s [%s]: boot", s.name, sched);
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    CHECK(!ok || gi >= 0, "%s: no graph kernel in the image", s.name);
    if (ok && gi >= 0) {
        const uint32_t a[7] = { s.opl_h, uint32_t(s.blob.size()), tsr::opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
        ok = fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        CHECK(ok, "%s [%s]: launch", s.name, sched);
    }
    if (ok) {
        if (s.want_ret == 0)
            CHECK(TSR_GET(h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK &&
                  fwh_ret(&h) == 0, "%s [%s]: DONE 0x%08x RET 0x%08x (err ticket 0x%08x info 0x%08x w0 0x%08x pidx %u)", s.name,
                  sched, h.done[0], h.done[2], h.done[3], h.done[4], h.done[5], h.done[6]);
        else
            CHECK(TSR_GET(h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && uint32_t(fwh_ret(&h)) == s.want_ret,
                  "%s [%s]: DONE 0x%08x RET 0x%08x, expected RET 0x%08x", s.name, sched, h.done[0], h.done[2], s.want_ret);
        for (int i = 0; i < 7; ++i) out->done[i] = h.done[i];
        for (const auto & r : s.recs) {
            const uint32_t yl = y_lines_of(r);
            for (uint32_t t = 0; t < r.a[mm::T]; ++t) {
                const uint8_t * got = tsr_sim_dram(sim, r.a[mm::Y] >> 28) +
                                      (uint64_t(r.a[mm::Y] & 0x0FFFFFFFu) + uint64_t(t) * r.a[mm::Y_STRIDE]) * LINE;
                const uint8_t * want = r.want.data() + size_t(t) * yl * LINE;
                if (memcmp(got, want, size_t(yl) * LINE) != 0) {
                    size_t k = 0;
                    while (got[k] == want[k]) ++k;
                    CHECK(0, "%s [%s]: op %u (fmt %u N %u K %u T %u) token %u: y differs at byte %zu (0x%08x, expected 0x%08x)",
                          s.name, sched, r.index, r.a[mm::FMT], r.a[mm::N], r.a[mm::K], r.a[mm::T], t, k,
                          get_u32_le(got + (k & ~size_t(3))), get_u32_le(want + (k & ~size_t(3))));
                    break;
                }
            }
        }
        obs_finish(&out->o, sim);
        // Written DRAM = the y rows exactly.
        std::vector<ival_t> want = written_rows(s), got;
        std::vector<ival_t> w(out->o.dram_wr, out->o.dram_wr + out->o.n_dram);
        std::sort(w.begin(), w.end(), [](const ival_t & a, const ival_t & b) { return a.ch != b.ch ? a.ch < b.ch : a.lo < b.lo; });
        for (const auto & x : w) {
            if (!got.empty() && got.back().ch == x.ch && x.lo <= got.back().hi) got.back().hi = std::max(got.back().hi, x.hi);
            else got.push_back(x);
        }
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i].ch == want[i].ch && got[i].lo == want[i].lo && got[i].hi == want[i].hi;
        CHECK(same, "%s [%s]: the written DRAM (%zu ranges) is not exactly the y rows (%zu ranges)", s.name, sched, got.size(),
              want.size());
        size_t reports = 0;
        const tsr_race_report * rr = nullptr;
        reports = tsr_sim_race_reports(sim, &rr);
        CHECK(reports == 0, "%s [%s]: %zu race reports", s.name, sched, reports);
        printf("%s [%s]: %zu ops, %llu core cycles, %llu RV instructions\n", s.name, sched, s.recs.size(),
               (unsigned long long)h.cycles, (unsigned long long)out->o.instret);
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

bool run_scheds(suite & s, uint32_t n_sched, uint32_t threads) {
    build_list(s);
    std::vector<run_obs> runs(n_sched);
    bool ok = true;
    for (uint32_t k = 0; k < n_sched; ++k) {
        char sched[64];
        if (k == 0) snprintf(sched, sizeof(sched), "fixed");
        else snprintf(sched, sizeof(sched), "random:%u", 1000u + k);
        ok = run_suite(s, sched, k == 0, threads, &runs[k]) && ok;
        if (k > 0 && ok) {
            char what[128];
            snprintf(what, sizeof(what), "%s: %s vs fixed", s.name, sched);
            obs_equal(&runs[0].o, &runs[k].o, what);
            CHECK(memcmp(runs[0].done, runs[k].done, sizeof(runs[0].done)) == 0, "%s: DONE r0-r6 differ", what);
        }
    }
    for (auto & r : runs) obs_free(&r.o);
    return ok;
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------
tsr_config base_cfg(uint32_t spm_bytes, uint64_t ch_bytes) {
    tsr_config c = TSR_CONFIG_U200;
    c.t_max = 8;
    c.spm_bytes = spm_bytes;
    c.ch_bytes = ch_bytes;
    return c;
}

// A record with x rows (host data unless x_from), its y rows and optional addend / old y rows.
struct alloc {
    uint64_t io[4] = { 0x4000000u / 64u, 0x1000000u / 64u, 0x1000000u / 64u, 0x1000000u / 64u };   // next free beat per channel
    uint64_t w = 0x20000000u / 64u;                                                                 // next weight beat
    uint64_t take(uint32_t ch, uint64_t beats) { const uint64_t b = io[ch]; io[ch] += (beats + 63u) & ~63ull; return b; }
};

rec make(suite & s, alloc & al, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T, uint32_t xch, uint32_t ych,
         uint32_t x_pad = 0, uint32_t y_pad = 0, int add_mode = 0, uint32_t add_pad = 0, bool acc_dst = false) {
    rec r{};
    const uint32_t xl = K / 16u, yl = (N + 15u) / 16u, C = s.cfg.num_ch;
    const uint64_t wb = tsr_tensor_ch_beats(fmt, N, K, C);
    s.weights.push_back(suite::wspec{fmt, N, K, al.w});
    r.a[mm::W_OFF] = uint32_t(al.w);
    al.w += (wb + 63u) & ~63ull;
    r.a[mm::FMT] = fmt;
    r.a[mm::N] = N;
    r.a[mm::K] = K;
    r.a[mm::T] = T;
    r.a[mm::X_STRIDE] = xl + x_pad;
    r.a[mm::X] = H(xch, al.take(xch, uint64_t(T) * (xl + x_pad)));
    s.inputs.push_back(suite::xspec{r.a[mm::X], T, xl + x_pad, K});
    r.a[mm::Y_STRIDE] = yl + y_pad;
    r.a[mm::Y] = H(ych, al.take(ych, uint64_t(T) * (yl + y_pad)));
    r.a[mm::ADD] = NONE;
    r.a[mm::ADD_STRIDE] = 0;
    r.a[mm::FLAGS] = acc_dst ? mm::FLAG_ACC_DST : 0u;
    if (acc_dst) s.inputs.push_back(suite::xspec{r.a[mm::Y], T, yl + y_pad, N});
    if (add_mode == 1) {                                     // bias: one row, stride 0
        r.a[mm::ADD] = H(ych, al.take(ych, yl));
        s.inputs.push_back(suite::xspec{r.a[mm::ADD], 1, yl, N});
    } else if (add_mode == 2) {                              // per-token rows, stride y_lines + add_pad
        r.a[mm::ADD_STRIDE] = yl + add_pad;
        r.a[mm::ADD] = H(ych, al.take(ych, uint64_t(T) * (yl + add_pad)));
        s.inputs.push_back(suite::xspec{r.a[mm::ADD], T, yl + add_pad, N});
    }
    return r;
}

suite suite_a() {
    suite s{};
    s.name = "suite_a";
    s.cfg = base_cfg(4u << 20, 1ull << 30);
    s.seed = 0xA11CE;
    alloc al;
    const uint32_t Q8 = TSR_FMT_Q8_0, Q4 = TSR_FMT_Q4_K;
    s.recs.push_back(make(s, al, Q8, 896, 896, 1, 0, 0));                         // 0: decode projection
    s.recs.push_back(make(s, al, Q8, 128, 896, 8, 0, 1));                         // 1: T = T_MAX, one tile
    s.recs.push_back(make(s, al, Q4, 512, 1024, 5, 1, 2));                        // 2: Q4_K, rows on ch1 / ch2
    s.recs.push_back(make(s, al, Q8, 300, 256, 3, 0, 0, 0, 0, 1));                // 3: N % 16 != 0, bias (stride 0)
    s.recs.push_back(make(s, al, Q4, 256, 512, 4, 0, 3, 0, 0, 2, 0));             // 4: residual, dense rows
    s.recs.push_back(make(s, al, Q8, 64, 512, 6, 2, 1, 5, 2, 2, 3));              // 5: strided x / y / addend rows
    s.recs.push_back(make(s, al, Q8, 896, 896, 2, 0, 0, 0, 0, 0, 0, true));       // 6: ACC_DST
    s.recs.push_back(make(s, al, Q8, 64, 256, 19, 3, 3));                         // 7: T = 19: tiles 8, 8, 3
    {                                                                              // 8: x = op 0's y (dep)
        rec r = make(s, al, Q8, 320, 896, 1, 0, 2);
        s.inputs.pop_back();                                                       // its x is op 0's output
        r.a[mm::X] = s.recs[0].a[mm::Y];
        r.a[mm::X_STRIDE] = s.recs[0].a[mm::Y_STRIDE];
        r.x_from = 0;
        r.deps = { 0 };
        s.recs.push_back(r);
    }
    {                                                                              // 9: three predecessors (JOIN)
        rec r = make(s, al, Q4, 256, 256, 2, 1, 1, 0, 0, 1);
        r.deps = { 2, 4, 5 };
        s.recs.push_back(r);
    }
    s.recs.push_back(make(s, al, Q4, 128, 256, 8, 0, 0, 0, 0, 0, 0, true));       // 10: Q4_K ACC_DST, T_MAX
    return s;
}

suite suite_b() {
    suite s{};
    s.name = "suite_b";
    s.cfg = base_cfg(512u << 10, 1ull << 30);                                      // 8,192 lines: budget 3,376
    s.seed = 0xB0B;
    alloc al;
    const uint32_t Q8 = TSR_FMT_Q8_0, Q4 = TSR_FMT_Q4_K;
    s.recs.push_back(make(s, al, Q8, 2512, 1024, 8, 0, 0));                       // lpt 239: t_tile 8, one tile
    s.recs.push_back(make(s, al, Q8, 2513, 1024, 8, 0, 1));                       // lpt 240: t_tile 7, tiles 7 + 1
    s.recs.push_back(make(s, al, Q8, 29280, 1024, 2, 0, 2));                      // lpt 1912: double buffer, t_tile 1
    s.recs.push_back(make(s, al, Q8, 29281, 1024, 3, 0, 3, 0, 0, 1));             // lpt 1913: single buffer + bias
    s.recs.push_back(make(s, al, Q4, 2048, 2048, 8, 1, 1, 0, 0, 2, 0));           // Q4_K residual, t_tile 7
    return s;
}

// The sweep of SW §7.4's forms: Q8_0 and Q4_K, T = 1 .. T_MAX, each without an addend, with a bias
// row (stride 0), with per-token addend rows and with ACC_DST; rows spread over the channels.
suite suite_c() {
    suite s{};
    s.name = "suite_c";
    s.cfg = base_cfg(4u << 20, 1ull << 30);
    s.seed = 0xC0C;
    alloc al;
    for (uint32_t fmt : { (uint32_t)TSR_FMT_Q8_0, (uint32_t)TSR_FMT_Q4_K })
        for (uint32_t T = 1; T <= 8u; ++T)
            for (int form = 0; form < 4; ++form) {
                const uint32_t ch = (T + uint32_t(form)) % 4u;
                s.recs.push_back(make(s, al, fmt, 48u + 16u * uint32_t(form), 256u * (1u + (T & 1u)), T, ch, (ch + 1u) % 4u,
                                      0, 0, form == 1 ? 1 : form == 2 ? 2 : 0, 0, form == 3));
            }
    return s;
}

// ---------------------------------------------------------------------------
// MUL_MAT_BEGIN / MUL_MAT_END suites (mode split)
// ---------------------------------------------------------------------------
suite suite_split_a() {
    suite s = suite_a();
    s.name = "split_a";
    // records 0-6 and 9, 10 are one T-tile: BEGINs with the END after 2, 0, 3, 1, at the end, 1, 0 more records,
    // after 2, and never (the executor closes it); 7 (T = 19) is a MUL_MAT; 8 depends on 0 (its END comes first)
    const int gap[] = { 2, 0, 3, 1, SPLIT_END, 1, 0, -1, -1, 2, SPLIT_NEVER };
    for (size_t i = 0; i < s.recs.size(); ++i) s.recs[i].split = gap[i];
    return s;
}

suite suite_split_c() {
    suite s = suite_c();
    s.name = "split_c";
    const int gap[] = { 0, 1, 2, 5, SPLIT_END, 3 };
    for (size_t i = 0; i < s.recs.size(); ++i) s.recs[i].split = gap[i % 6u];
    return s;
}

// Twelve BEGINs before the first END (more than the table's eight): the table closes the oldest as it fills, and
// each END of a closed one returns the ticket kept for it. Every form: Q8_0 / Q4_K, T 1 .. 3, bias / residual / ACC_DST.
suite suite_split_d(bool reverse) {
    suite s{};
    s.name = reverse ? "split_d_reverse" : "split_d";
    s.cfg = base_cfg(4u << 20, 1ull << 30);
    s.seed = 0xD00D;
    s.reverse_end = reverse;
    alloc al;
    for (uint32_t k = 0; k < 12u; ++k) {
        const uint32_t fmt = k % 2u ? (uint32_t)TSR_FMT_Q4_K : (uint32_t)TSR_FMT_Q8_0;
        rec r = make(s, al, fmt, 48u + 16u * k, 256u, 1u + k % 3u, k % 4u, (k + 1u) % 4u, 0, 0, int(k % 3u) == 1 ? 1 : int(k % 3u) == 2 ? 2 : 0,
                     0, k % 5u == 4u);
        if (k % 5u == 4u && r.a[mm::ADD] != NONE) r.a[mm::FLAGS] = 0;
        r.split = SPLIT_END;
        s.recs.push_back(r);
    }
    return s;
}

// Rule 3: a 512 KiB SPM (the ring holds about two of the big blocks). A BEGIN of lpt 1912 holds its block; the next
// MUL_MAT of lpt 1913 has to wrap onto it: the allocation is refused, the executor closes the open BEGINs and
// allocates again. Three rounds, then a small BEGIN that stays open until a MUL_MAT of three tiles needs its lines.
suite suite_split_e() {
    suite s{};
    s.name = "split_e_wrap";
    s.cfg = base_cfg(512u << 10, 1ull << 30);
    s.seed = 0xE0E;
    alloc al;
    const uint32_t Q8 = TSR_FMT_Q8_0;
    for (uint32_t round = 0; round < 3u; ++round) {
        rec a = make(s, al, Q8, 29280, 1024, 1, 0, 0, 0, 0, round == 1 ? 1 : 0);
        a.split = SPLIT_END;
        s.recs.push_back(a);
        s.recs.push_back(make(s, al, Q8, 29281, 1024, 1, 0, 1));
    }
    rec c = make(s, al, Q8, 256, 1024, 3, 1, 2, 0, 0, 2, 0);
    c.split = SPLIT_END;
    s.recs.push_back(c);
    s.recs.push_back(make(s, al, Q8, 29281, 1024, 3, 0, 3));   // three T-tiles: the held block is in the way of tile 1
    return s;
}

// A held block is never forgotten by the ring's table (128 records): a BEGIN, 135 small MUL_MATs, and its END.
suite suite_split_f() {
    suite s{};
    s.name = "split_f_table_full";
    s.cfg = base_cfg(4u << 20, 1ull << 30);
    s.seed = 0xF0F;
    alloc al;
    const uint32_t Q8 = TSR_FMT_Q8_0;
    rec a = make(s, al, Q8, 64, 256, 2, 0, 1);
    a.split = SPLIT_END;
    s.recs.push_back(a);
    for (uint32_t k = 0; k < 135u; ++k) {
        s.recs.push_back(make(s, al, Q8, 32, 64, 1, k % 4u, (k + 1u) % 4u));
        if (k == 60u) {
            rec b = make(s, al, TSR_FMT_Q4_K, 48, 256, 1, 2, 3);
            b.split = SPLIT_END;
            s.recs.push_back(b);
        }
    }
    return s;
}

// Rule 4: the list ends with open BEGINs (and a MUL_MAT after them).
suite suite_split_g() {
    suite s{};
    s.name = "split_g_open_at_end";
    s.cfg = base_cfg(4u << 20, 1ull << 30);
    s.seed = 0x6060;
    alloc al;
    for (uint32_t k = 0; k < 3u; ++k) {
        rec a = make(s, al, k == 1 ? (uint32_t)TSR_FMT_Q4_K : (uint32_t)TSR_FMT_Q8_0, 64, 256, 1 + k, k, 3 - k, 0, 0, int(k));
        a.split = SPLIT_NEVER;
        s.recs.push_back(a);
    }
    s.recs.push_back(make(s, al, TSR_FMT_Q8_0, 96, 256, 1, 3, 2));
    return s;
}

// Refused lists: BAD_RECORD (RET 0x80000004). rec 0 is a BEGIN (T = 2, strided y) whose rows are stored in every case
// (its END comes first, or the executor closes it after the refusal); `bad` is the record that follows.
enum bad_kind { END_ALONE, END_TWICE, END_OF_MUL_MAT, END_OF_ITSELF, BEGIN_TWO_TILES, BEGIN_KFLAGS, BEGIN_ADD_ACC, END_KFLAGS, BEGIN_OPEN_END_BAD,
                END_TWICE_CLOSED };
suite suite_split_bad(bad_kind k) {
    static const char * const names[] = { "bad_end_alone", "bad_end_twice", "bad_end_of_mul_mat", "bad_end_of_itself", "bad_begin_two_tiles",
                                          "bad_begin_kflags", "bad_begin_add_acc", "bad_end_kflags", "bad_open_then_bad_end",
                                          "bad_end_twice_of_closed" };
    suite s{};
    s.name = names[k];
    s.cfg = base_cfg(4u << 20, 1ull << 30);
    s.seed = 0xBAD0 + k;
    s.want_ret = tsr::opl::RET_BAD_RECORD;
    alloc al;
    const uint32_t Q8 = TSR_FMT_Q8_0;
    rec a = make(s, al, Q8, 80, 256, 2, 0, 1, 0, 3, 1);
    rec big = make(s, al, Q8, 64, 256, 19, 1, 2);          // three T-tiles
    tsr::opl::writer w(tsr::opl::F_STOP_ON_ERROR, 0xBAD);
    auto begin = [&](const rec & r, const uint32_t * deps, uint32_t nd, uint32_t kflags = 0, uint32_t flags = ~0u) {
        const uint32_t v[12] = { r.a[mm::X], r.a[mm::X_STRIDE], r.a[mm::Y], r.a[mm::Y_STRIDE], r.a[mm::W_OFF], r.a[mm::FMT],
                                 r.a[mm::N], r.a[mm::K], r.a[mm::T], r.a[mm::ADD], r.a[mm::ADD_STRIDE],
                                 flags == ~0u ? r.a[mm::FLAGS] : flags };
        return w.add(tsr::opl::K_MUL_MAT_BEGIN, v, 12, deps, nd, kflags);
    };
    auto end = [&](uint32_t b, const uint32_t * deps, uint32_t nd, uint32_t kflags = 0) {
        const uint32_t v[1] = { b };
        return w.add(tsr::opl::K_MUL_MAT_END, v, 1, deps, nd, kflags);
    };
    bool stores = true;                                    // rec 0's rows are written
    uint32_t mulmat_rows = 0;                              // rows of `big` a MUL_MAT of the list wrote
    std::vector<rec> more;                                 // the other BEGINs that ran (their rows are stored when the list ends)
    switch (k) {
    case END_ALONE: {
        const uint32_t v[1] = { 0 };
        w.add(tsr::opl::K_MUL_MAT_END, v, 1);
        stores = false;
        break;
    }
    case END_TWICE: {
        const uint32_t b = begin(a, nullptr, 0), e = end(b, &b, 1);
        end(b, &e, 1);
        break;
    }
    case END_OF_MUL_MAT: {
        const uint32_t b = begin(a, nullptr, 0), e = end(b, &b, 1);
        const uint32_t m = w.mul_mat(big.a[mm::X], big.a[mm::X_STRIDE], big.a[mm::Y], big.a[mm::Y_STRIDE], big.a[mm::W_OFF], Q8, 64, 256, 2);
        end(m, &e, 1);
        mulmat_rows = 2;                                   // the MUL_MAT before the refused END ran
        break;
    }
    case END_OF_ITSELF: {
        const uint32_t b = begin(a, nullptr, 0);
        end(b + 1u, &b, 1);                                // its own index
        break;
    }
    case BEGIN_TWO_TILES: {
        begin(big, nullptr, 0);
        stores = false;
        break;
    }
    case BEGIN_KFLAGS: {
        begin(a, nullptr, 0, 1u);
        stores = false;
        break;
    }
    case BEGIN_ADD_ACC: {                                  // an addend and ACC_DST together
        begin(a, nullptr, 0, 0, tsr::opl::mm::FLAG_ACC_DST);
        stores = false;
        break;
    }
    case END_KFLAGS: {
        const uint32_t b = begin(a, nullptr, 0);
        end(b, &b, 1, 1u);
        break;                                             // the executor closes the BEGIN after the refusal
    }
    case BEGIN_OPEN_END_BAD: {
        const uint32_t b = begin(a, nullptr, 0);
        const uint32_t v[1] = { 77 };
        w.add(tsr::opl::K_MUL_MAT_END, v, 1, &b, 1);       // a BEGIN that is not there
        break;
    }
    case END_TWICE_CLOSED: {                               // nine BEGINs: the ninth closes the first (rule 2, its bit set)
        const uint32_t b = begin(a, nullptr, 0);           // its END comes once (it takes the kept ticket and clears the bit), then again
        for (uint32_t j = 0; j < 8u; ++j) {
            rec r = make(s, al, Q8, 48u + 16u * j, 256, 1, j % 4u, (j + 1u) % 4u);
            begin(r, nullptr, 0);
            more.push_back(r);
        }
        const uint32_t e = end(b, &b, 1);
        end(b, &e, 1);                                     // the BEGIN is gone: BAD_RECORD
        break;
    }
    }
    s.raw = w.finish();
    if (stores) s.recs.push_back(a);                       // y rows expected (the rows of a BEGIN that ran)
    for (const rec & r : more) s.recs.push_back(r);
    if (mulmat_rows != 0u) {
        big.a[mm::T] = mulmat_rows;
        s.recs.push_back(big);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Model shapes (slow)
// ---------------------------------------------------------------------------
suite suite_shapes(uint32_t which) {
    suite s{};
    s.cfg = base_cfg(4u << 20, 2ull << 30);
    alloc al;
    const uint32_t Q8 = TSR_FMT_Q8_0, Q4 = TSR_FMT_Q4_K;
    switch (which) {
    case 0:   // Qwen2.5-0.5B (K = 896, d_ff 4864; Q8_0 model, Q4_K for the Q4_K_M ffn_down)
        s.name = "shapes_qwen05b";
        s.seed = 0x05B;
        s.recs.push_back(make(s, al, Q8, 896, 896, 1, 0, 0, 0, 0, 1));            // q / o with bias
        s.recs.push_back(make(s, al, Q8, 128, 896, 8, 0, 1, 0, 0, 1));            // k / v with bias, T_MAX
        s.recs.push_back(make(s, al, Q8, 4864, 896, 8, 0, 2));                    // gate / up
        s.recs.push_back(make(s, al, Q4, 896, 4864, 1, 0, 0, 0, 0, 2));           // down (Q4_K_M), residual
        s.recs.push_back(make(s, al, Q8, 151936, 896, 1, 0, 3));                  // vocab
        s.recs.push_back(make(s, al, Q8, 151936, 896, 8, 1, 2));                  // vocab, prefill tiles
        break;
    case 1:   // Qwen2.5-32B (K = 5120, d_ff 27648)
        s.name = "shapes_qwen32b";
        s.seed = 0x32B;
        s.recs.push_back(make(s, al, Q4, 5120, 5120, 1, 0, 0, 0, 0, 1));
        s.recs.push_back(make(s, al, Q4, 1024, 5120, 8, 0, 1, 0, 0, 1));
        s.recs.push_back(make(s, al, Q4, 5120, 27648, 1, 0, 2, 0, 0, 2));
        s.recs.push_back(make(s, al, Q8, 152064, 5120, 1, 0, 3));                 // vocab (load-time Q8_0)
        s.recs.push_back(make(s, al, Q8, 152064, 5120, 8, 1, 2));
        break;
    default:  // Gemma 4 12B (K = 3840)
        s.name = "shapes_gemma4_12b";
        s.seed = 0x12B;
        s.recs.push_back(make(s, al, Q4, 4096, 3840, 1, 0, 0));
        s.recs.push_back(make(s, al, Q8, 15360, 3840, 8, 0, 1));
        s.recs.push_back(make(s, al, Q4, 3840, 15360, 1, 0, 2, 0, 0, 2));
        s.recs.push_back(make(s, al, Q8, 262144, 3840, 1, 0, 3));                 // vocab
        s.recs.push_back(make(s, al, Q8, 262144, 3840, 8, 1, 2));
        break;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Lowering check against traces/model-v1conv
// ---------------------------------------------------------------------------
// The SPM fields of the tile ops (w1 of LD / ST, w1 / w2 of QACT and MVM) are the operand forms
// that differ; everything else of the per-unit streams must be equal.
void mask_spm(uint32_t w[8]) {
    const uint32_t opc = w[0] & 0xFFu;
    if (opc == TSR_OP_LD || opc == TSR_OP_ST) w[1] = 0;
    if (opc == TSR_OP_QACT || opc == TSR_OP_MVM) w[1] = w[2] = 0;
}

struct cap_push { uint32_t unit; uint32_t w[8]; };
void cap_q(void * ctx, tsr_sim *, const tsr_q_event * ev) {
    if (ev->kind != TSR_QEV_PUSH) return;
    auto * v = static_cast<std::vector<cap_push> *>(ctx);
    cap_push p{ev->unit, {}};
    memcpy(p.w, ev->w, sizeof(p.w));
    v->push_back(p);
}

bool v1conv_one(const char * dir) {
    tsr_trace2 * t = nullptr;
    char err[512];
    if (tsr_trace2_read(dir, &t, err, sizeof(err)) != 0) {
        CHECK(0, "%s: %s", dir, err);
        return false;
    }
    // Regroup the trace's tiles (LD, QACT, MVM, ST each) into MUL_MAT ops.
    struct mmop { uint32_t a[12]; };
    std::vector<mmop> ops;
    bool shape_ok = t->n_push % 4u == 0u;
    for (size_t i = 0; shape_ok && i + 3 < t->n_push; i += 4) {
        const uint32_t * ld = t->push[i].w, * qa = t->push[i + 1].w, * mv = t->push[i + 2].w, * st = t->push[i + 3].w;
        if ((ld[0] & 0xFF) != TSR_OP_LD || (qa[0] & 0xFF) != TSR_OP_QACT || (mv[0] & 0xFF) != TSR_OP_MVM ||
            (st[0] & 0xFF) != TSR_OP_ST) { shape_ok = false; break; }
        const uint32_t fmt = tsr_w0_sub(mv[0]), N = mv[4], K = mv[5] & 0xFFFFu, tt = (mv[5] >> 16) & 0xFu;
        const uint32_t xl = K / 16u, yl = (N + 15u) / 16u;
        mmop * last = ops.empty() ? nullptr : &ops.back();
        if (last && last->a[mm::FMT] == fmt && last->a[mm::N] == N && last->a[mm::K] == K && last->a[mm::W_OFF] == mv[3] &&
            (last->a[mm::X] & 0x0FFFFFFFu) + last->a[mm::T] * xl == ld[2] &&
            (last->a[mm::Y] & 0x0FFFFFFFu) + last->a[mm::T] * yl == st[2]) {
            last->a[mm::T] += tt;
            continue;
        }
        mmop o{};
        o.a[mm::X] = H(tsr_w0_sub(ld[0]), ld[2]);
        o.a[mm::X_STRIDE] = xl;
        o.a[mm::Y] = H(tsr_w0_sub(st[0]), st[2]);
        o.a[mm::Y_STRIDE] = yl;
        o.a[mm::W_OFF] = mv[3];
        o.a[mm::FMT] = fmt;
        o.a[mm::N] = N;
        o.a[mm::K] = K;
        o.a[mm::T] = tt;
        o.a[mm::ADD] = NONE;
        ops.push_back(o);
    }
    CHECK(shape_ok && !ops.empty(), "%s: not a MUL_MAT trace (LD, QACT, MVM, ST tiles)", dir);
    if (!shape_ok || ops.empty()) { tsr_trace2_free(t); return false; }
    // The list (independent ops: the trace ran them in sequence, they share no data within a split).
    tsr::opl::writer w(tsr::opl::F_STOP_ON_ERROR, 0x1C0);
    for (const auto & o : ops)
        w.mul_mat(o.a[mm::X], o.a[mm::X_STRIDE], o.a[mm::Y], o.a[mm::Y_STRIDE], o.a[mm::W_OFF], o.a[mm::FMT], o.a[mm::N],
                  o.a[mm::K], o.a[mm::T]);
    const std::vector<uint8_t> blob = w.finish();
    // A fresh simulator with the trace's config (T_MAX from the trace) and its init.
    tsr_config cfg = t->cfg.core;
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.race = g_race;
    so.threads = 4;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &so);
    CHECK(sim != nullptr, "%s: simulator", dir);
    bool ok = sim != nullptr;
    for (size_t i = 0; ok && i < t->mem.n_init; ++i)
        tsr_sim_dram_write(sim, t->mem.init[i].ch, t->mem.init[i].byte_off, t->mem.init[i].data, t->mem.init[i].size);
    // The list in the OPL region, after the trace's init region (init region at ch0 0).
    const uint32_t opl_h = H(0, 0x100000u / 64u);
    if (ok) tsr_sim_dram_write(sim, 0, uint64_t(opl_h) * LINE, blob.data(), blob.size());
    std::vector<cap_push> pushes;
    obs o;
    fwh h{};
    if (ok) {
        tsr_sim_add_listener(sim, cap_q, &pushes);
        obs_attach(&o, sim);
        ok = fwh_open(&h, sim, g_image) == 0;
        CHECK(ok, "%s: boot", dir);
    }
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    if (ok) {
        pushes.clear();
        const uint32_t a[7] = { opl_h, uint32_t(blob.size()), 0, 0, 0, 0, 0 };
        ok = gi >= 0 && fwh_launch(&h, uint32_t(gi), a, 0) == 0;
        CHECK(ok, "%s: launch", dir);
    }
    if (ok) {
        CHECK(fwh_status(&h) == TSR_LCQ_R0_STATUS_OK && fwh_ret(&h) == 0, "%s: DONE 0x%08x RET 0x%08x", dir, h.done[0], h.done[2]);
        // The trace's outputs, bit for bit.
        for (size_t i = 0; i < t->mem.n_expect; ++i) {
            const tsr_region * x = &t->mem.expect[i];
            const uint8_t * got = tsr_sim_dram(sim, x->ch) + x->byte_off;
            if (memcmp(got, x->data, x->size) != 0) {
                size_t k = 0;
                while (got[k] == x->data[k]) ++k;
                CHECK(0, "%s: expect region ch %u 0x%llx differs at byte %zu", dir, x->ch, (unsigned long long)x->byte_off, k);
            }
        }
        obs_finish(&o, sim);
        uint64_t exp_beats = 0;
        for (size_t i = 0; i < t->mem.n_expect; ++i) exp_beats += t->mem.expect[i].size / 64u;
        CHECK(o.n_written_beats == exp_beats, "%s: %llu DRAM beats written, the expect regions hold %llu", dir,
              (unsigned long long)o.n_written_beats, (unsigned long long)exp_beats);
        // The tile-op streams of DMA, QACT and MVM (the executor's own list LDs read the list).
        const uint64_t opl_lo = opl_h & 0x0FFFFFFFu, opl_hi = opl_lo + (blob.size() + 63u) / 64u;
        for (uint32_t u : { (uint32_t)TSR_UNIT_DMA, (uint32_t)TSR_UNIT_QACT, (uint32_t)TSR_UNIT_MVM }) {
            std::vector<std::array<uint32_t, 8>> want, got;
            for (size_t i = 0; i < t->n_push; ++i) {
                if (tsr_trace2_push_unit(&t->cfg, &t->push[i]) != u) continue;
                std::array<uint32_t, 8> x;
                memcpy(x.data(), t->push[i].w, 32);
                mask_spm(x.data());
                want.push_back(x);
            }
            for (const auto & p : pushes) {
                if (p.unit != u) continue;
                if ((p.w[0] & 0xFF) == TSR_OP_LD && tsr_w0_sub(p.w[0]) == 0 && p.w[2] >= opl_lo && p.w[2] < opl_hi) continue;
                std::array<uint32_t, 8> x;
                memcpy(x.data(), p.w, 32);
                mask_spm(x.data());
                got.push_back(x);
            }
            size_t k = 0;
            while (k < want.size() && k < got.size() && want[k] == got[k]) ++k;
            CHECK(k == want.size() && k == got.size(), "%s: unit %u: op %zu of %zu / %zu differs (w0 0x%08x w2 0x%08x w3 0x%08x, "
                  "expected w0 0x%08x w2 0x%08x w3 0x%08x)", dir, u, k, got.size(), want.size(), k < got.size() ? got[k][0] : 0u,
                  k < got.size() ? got[k][2] : 0u, k < got.size() ? got[k][3] : 0u, k < want.size() ? want[k][0] : 0u,
                  k < want.size() ? want[k][2] : 0u, k < want.size() ? want[k][3] : 0u);
        }
        // The kernel's dataflow inside each tile: QACT reads the LD's lines, MVM the QACT's, ST the MVM's.
        std::vector<const cap_push *> ld, qa, mv, st;
        for (const auto & p : pushes) {
            const uint32_t opc = p.w[0] & 0xFF;
            if (opc == TSR_OP_LD && !(tsr_w0_sub(p.w[0]) == 0 && p.w[2] >= opl_lo && p.w[2] < opl_hi)) ld.push_back(&p);
            else if (opc == TSR_OP_QACT) qa.push_back(&p);
            else if (opc == TSR_OP_MVM) mv.push_back(&p);
            else if (opc == TSR_OP_ST) st.push_back(&p);
        }
        CHECK(ld.size() == qa.size() && qa.size() == mv.size() && mv.size() == st.size(), "%s: tile op counts", dir);
        for (size_t i = 0; i < ld.size() && i < qa.size() && i < mv.size() && i < st.size(); ++i)
            CHECK(qa[i]->w[2] == ld[i]->w[1] && mv[i]->w[2] == qa[i]->w[1] && st[i]->w[1] == mv[i]->w[1],
                  "%s: tile %zu dataflow LD %u -> QACT %u/%u -> MVM %u/%u -> ST %u", dir, i, ld[i]->w[1], qa[i]->w[2],
                  qa[i]->w[1], mv[i]->w[2], mv[i]->w[1], st[i]->w[1]);
        size_t nr = 0;
        const tsr_race_report * rr = nullptr;
        nr = tsr_sim_race_reports(sim, &rr);
        CHECK(nr == 0, "%s: %zu race reports", dir, nr);
    }
    if (sim) {
        obs_free(&o);
        fwh_close(&h);
        tsr_sim_destroy(sim);
    }
    tsr_trace2_free(t);
    return ok;
}

void list_traces(const std::string & d, std::vector<std::string> & out, int depth) {
    struct stat st;
    if (stat((d + "/meta.json").c_str(), &st) == 0) {
        out.push_back(d);
        return;
    }
    if (depth > 4) return;
    DIR * dd = opendir(d.c_str());
    if (!dd) return;
    std::vector<std::string> sub;
    while (struct dirent * e = readdir(dd)) {
        if (e->d_name[0] == '.' || strcmp(e->d_name, "images") == 0) continue;
        sub.push_back(d + "/" + e->d_name);
    }
    closedir(dd);
    std::sort(sub.begin(), sub.end());
    for (const auto & s : sub) list_traces(s, out, depth + 1);
}


// SW-4: the executor's table of open BEGINs and its closed bits are per launch. Three launches on one simulator.
//   1. Nine BEGINs (the ninth closes the first: its bit is set, its END never comes), then a MUL_MAT whose x lies outside
//      the DRAM: a unit error, and with STOP_ON_ERROR the list stops with eight BEGINs open (rule 4 does not run).
//   2. A correct list (a MUL_MAT, a BEGIN and its END, a MUL_MAT): bit for bit against the reference, and the y rows of
//      launch 1's open BEGINs untouched (a stale table would store them again).
//   3. A MUL_MAT and an END that names it: BAD_RECORD (a stale closed bit of launch 1's BEGIN at index 0 would accept it).
bool run_relaunch(const char * sched, uint32_t threads) {
    suite s{};
    s.name = "relaunch";
    s.cfg = base_cfg(4u << 20, 1ull << 30);
    s.seed = 0x4E1A;
    alloc al;
    const uint32_t Q8 = TSR_FMT_Q8_0;
    std::vector<rec> l1;
    for (uint32_t k = 0; k < 9u; ++k) l1.push_back(make(s, al, Q8, 64u + 16u * k, 256, 1, k % 4u, (k + 1u) % 4u));
    rec bad = make(s, al, Q8, 64, 256, 1, 0, 1);
    bad.a[mm::X] = H(0, 0x0F000000u);                       // beyond the channel (1 GiB = 0x1000000 lines)
    s.recs.push_back(make(s, al, Q8, 96, 256, 1, 2, 3));    // launch 2: op 0, a MUL_MAT
    s.recs.push_back(make(s, al, (uint32_t)TSR_FMT_Q4_K, 128, 256, 2, 1, 0, 0, 0, 1));   // op 1, a BEGIN
    s.recs.push_back(make(s, al, Q8, 80, 256, 1, 3, 1));    // op 2, a MUL_MAT
    rec l3 = make(s, al, Q8, 64, 256, 1, 0, 2);
    const uint32_t opl2 = H(0, 0x40000u + 0x1000u), opl3 = H(0, 0x40000u + 0x2000u);
    auto put_mul_mat = [](tsr::opl::writer & w, const rec & r) {
        return w.mul_mat(r.a[mm::X], r.a[mm::X_STRIDE], r.a[mm::Y], r.a[mm::Y_STRIDE], r.a[mm::W_OFF], r.a[mm::FMT], r.a[mm::N],
                         r.a[mm::K], r.a[mm::T], r.a[mm::ADD], r.a[mm::ADD_STRIDE], r.a[mm::FLAGS]);
    };
    auto put_begin = [](tsr::opl::writer & w, const rec & r) {
        return w.mul_mat_begin(r.a[mm::X], r.a[mm::X_STRIDE], r.a[mm::Y], r.a[mm::Y_STRIDE], r.a[mm::W_OFF], r.a[mm::FMT], r.a[mm::N],
                               r.a[mm::K], r.a[mm::T], r.a[mm::ADD], r.a[mm::ADD_STRIDE], r.a[mm::FLAGS]);
    };
    tsr::opl::writer w1(tsr::opl::F_STOP_ON_ERROR | tsr::opl::F_SERIAL, 0x4E1), w2(tsr::opl::F_STOP_ON_ERROR, 0x4E2),
        w3(tsr::opl::F_STOP_ON_ERROR, 0x4E3);
    for (const rec & r : l1) put_begin(w1, r);
    put_mul_mat(w1, bad);
    put_mul_mat(w2, s.recs[0]);
    const uint32_t b2 = put_begin(w2, s.recs[1]);
    put_mul_mat(w2, s.recs[2]);
    const uint32_t e2 = w2.mul_mat_end(b2, &b2, 1);
    (void)e2;
    put_mul_mat(w3, l3);
    w3.mul_mat_end(0, nullptr, 0);
    s.blob = w1.finish();
    const std::vector<uint8_t> blob2 = w2.finish(), blob3 = w3.finish();
    CHECK(w1.ok() && w2.ok() && w3.ok() && !s.blob.empty() && !blob2.empty() && !blob3.empty(), "relaunch: OPL writer");

    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = g_race;
    so.threads = threads;
    tsr_sim * sim = tsr_sim_create_opts(&s.cfg, &so);
    CHECK(sim != nullptr, "relaunch: simulator");
    if (sim == nullptr) return false;
    fill(sim, s);
    expect(sim, s, threads);
    tsr_sim_dram_write(sim, opl2 >> 28, uint64_t(opl2 & 0x0FFFFFFFu) * LINE, blob2.data(), blob2.size());
    tsr_sim_dram_write(sim, opl3 >> 28, uint64_t(opl3 & 0x0FFFFFFFu) * LINE, blob3.data(), blob3.size());
    std::vector<std::vector<uint8_t>> before;                // launch 1's y rows of the BEGINs left open: never written
    for (size_t k = 1; k < l1.size(); ++k) {
        const uint8_t * p = tsr_sim_dram(sim, l1[k].a[mm::Y] >> 28) + uint64_t(l1[k].a[mm::Y] & 0x0FFFFFFFu) * LINE;
        before.emplace_back(p, p + size_t(y_lines_of(l1[k])) * LINE);
    }
    fwh h;
    bool ok = fwh_open(&h, sim, g_image) == 0;
    CHECK(ok, "relaunch [%s]: boot", sched);
    const int gi = ok ? fwh_kernel(&h, "graph") : -1;
    CHECK(!ok || gi >= 0, "relaunch: no graph kernel in the image");
    if (ok && gi >= 0) {
        const uint32_t a1[7] = { s.opl_h, uint32_t(s.blob.size()), tsr::opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
        ok = fwh_launch(&h, uint32_t(gi), a1, 0) == 0;
        CHECK(ok, "relaunch [%s]: launch 1", sched);
        printf("relaunch [%s]: launch 1: DONE 0x%08x RET 0x%08x\n", sched, h.done[0], h.done[2]);
        // The unit error of launch 1 is the runtime's to clear (EX §4.9 R9) before the next launch.
        if (ok) {
            const uint32_t a2[7] = { opl2, uint32_t(blob2.size()), tsr::opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
            ok = fwh_launch(&h, uint32_t(gi), a2, 0) == 0;
            CHECK(ok, "relaunch [%s]: launch 2", sched);
            CHECK(ok && TSR_GET(h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(&h) == TSR_LCQ_R0_STATUS_OK && fwh_ret(&h) == 0,
                  "relaunch [%s]: launch 2: DONE 0x%08x RET 0x%08x", sched, h.done[0], h.done[2]);
        }
        if (ok) {
            for (const auto & r : s.recs) {
                const uint32_t yl = y_lines_of(r);
                for (uint32_t t = 0; t < r.a[mm::T]; ++t) {
                    const uint8_t * got = tsr_sim_dram(sim, r.a[mm::Y] >> 28) +
                                          (uint64_t(r.a[mm::Y] & 0x0FFFFFFFu) + uint64_t(t) * r.a[mm::Y_STRIDE]) * LINE;
                    CHECK(memcmp(got, r.want.data() + size_t(t) * yl * LINE, size_t(yl) * LINE) == 0,
                          "relaunch [%s]: launch 2: y of the list's record (N %u, token %u) differs from the reference", sched, r.a[mm::N], t);
                }
            }
            for (size_t k = 1; k < l1.size(); ++k) {
                const uint8_t * p = tsr_sim_dram(sim, l1[k].a[mm::Y] >> 28) + uint64_t(l1[k].a[mm::Y] & 0x0FFFFFFFu) * LINE;
                CHECK(memcmp(p, before[k - 1].data(), before[k - 1].size()) == 0,
                      "relaunch [%s]: launch 2 wrote the y rows of launch 1's BEGIN %zu (a stale open table)", sched, k);
            }
            if (fwh_status(&h) == TSR_LCQ_R0_STATUS_OK) {
                const uint32_t a3[7] = { opl3, uint32_t(blob3.size()), tsr::opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
                ok = fwh_launch(&h, uint32_t(gi), a3, 0) == 0;
                CHECK(ok, "relaunch [%s]: launch 3", sched);
                CHECK(ok && uint32_t(fwh_ret(&h)) == tsr::opl::RET_BAD_RECORD,
                      "relaunch [%s]: launch 3: DONE 0x%08x RET 0x%08x, expected RET 0x%08x (an END of a MUL_MAT: BAD_RECORD)", sched,
                      h.done[0], h.done[2], tsr::opl::RET_BAD_RECORD);
            }
        }
        const tsr_race_report * rr = nullptr;
        const size_t reports = tsr_sim_race_reports(sim, &rr);
        CHECK(reports == 0, "relaunch [%s]: %zu race reports", sched, reports);
    }
    fwh_close(&h);
    tsr_sim_destroy(sim);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s IMAGE suite|split|shapes|v1conv [DIR...] [--scheds N] [--threads N] [--sample N] [--only K]\n"
                        "       [--race report] [--expect-fail]\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    struct stat st;
    if (stat(g_image, &st) != 0) {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    const std::string mode = argv[2];
    uint32_t n_sched = mode == "suite" || mode == "split" ? 8u : 2u, threads = 4, sample = 0;
    int only = -1;
    bool expect_fail = false;          // a mutant image: the run must find a difference
    std::vector<std::string> dirs;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--scheds") && i + 1 < argc) n_sched = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--sample") && i + 1 < argc) sample = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--race") && i + 1 < argc) g_race = !strcmp(argv[++i], "report") ? 1u : 2u;
        else if (!strcmp(argv[i], "--expect-fail")) expect_fail = true;
        else dirs.push_back(argv[i]);
    }
    if (mode == "suite") {
        suite a = suite_a(), b = suite_b(), c = suite_c();
        if (only < 0 || only == 0) run_scheds(a, n_sched, threads);
        if (only < 0 || only == 1) run_scheds(b, n_sched, threads);
        if (only < 0 || only == 2) run_scheds(c, n_sched, threads);
    } else if (mode == "split") {
        std::vector<suite> all;
        all.push_back(suite_split_a());
        all.push_back(suite_split_c());
        all.push_back(suite_split_d(false));
        all.push_back(suite_split_d(true));
        all.push_back(suite_split_e());
        all.push_back(suite_split_f());
        all.push_back(suite_split_g());
        for (int k = 0; k <= int(END_TWICE_CLOSED); ++k) all.push_back(suite_split_bad(bad_kind(k)));
        for (size_t i = 0; i < all.size(); ++i)
            if (only < 0 || size_t(only) == i) run_scheds(all[i], n_sched, threads);
        if (only < 0 || size_t(only) == all.size())            // the last index: three launches on one simulator
            for (uint32_t k = 0; k < n_sched; ++k) {
                char sched[64];
                if (k == 0) snprintf(sched, sizeof(sched), "fixed");
                else snprintf(sched, sizeof(sched), "random:%u", 1000u + k);
                run_relaunch(sched, threads);
            }
    } else if (mode == "shapes") {
        for (uint32_t k = 0; k < 3; ++k) {
            if (only >= 0 && uint32_t(only) != k) continue;
            suite s = suite_shapes(k);
            run_scheds(s, n_sched, threads);
        }
    } else if (mode == "v1conv") {
        size_t n = 0;
        for (const auto & d : dirs) {
            std::vector<std::string> tr;
            list_traces(d, tr, 0);
            if (tr.empty()) {
                printf("%s: no traces under %s: skipped\n", argv[0], d.c_str());
                return 77;
            }
            const size_t step = sample && tr.size() > sample ? tr.size() / sample : 1;
            for (size_t i = 0; i < tr.size(); i += step) {
                v1conv_one(tr[i].c_str());
                ++n;
            }
        }
        printf("v1conv: %zu traces\n", n);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    if (expect_fail) {
        printf("fw_kernels_rv: mutant image: %d check(s) failed (%s)\n", g_fail, g_fail ? "caught, as required" : "NOT CAUGHT");
        return g_fail ? 0 : 1;
    }
    return test_finish("fw_kernels_rv");
}
