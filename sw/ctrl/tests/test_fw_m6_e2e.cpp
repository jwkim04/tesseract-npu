// test_fw_m6_e2e.cpp — ctests fw_m6_e2e and fw_m6_perf (W2a integration: docs/v2/06-software-
// verification.md "SW" §11 P2.1-P2.5, §10.1 "launch traces", §4.7, §7.8; docs/v2/README.md §10 W2a).
//
//   test_fw_m6_e2e IMAGE SCRATCH [TRACES]
//   A. One session per schedule (fixed + random:1..7), each on a fresh simulator with the race
//      detector fatal and launch capture on (TSR_TRACE directory + in-memory capture), U200 M6
//      configuration (4 channels, 4 MiB SPM, T_MAX 8, CAPS VU | ATT, U200 queue depths):
//        - the production image tsr-fw.elf cold-loaded through the host CSR loader with the receive
//          CRC and VERIFY (EX §3.3), booted; HELLO first with r2 / r3 = the image id that tsr-elf
//          computes (sw/ctrl/tools/tsr-elf), that tsr-elf stamped into .tsr_id and that the
//          simulator's trace library computes (tsr_trace2_image_id), r4 = the SDK ABI of .note.tsr,
//          r5 = 3 kernels; graph / selftest / replay at EX §4.6's indices 0 / 1 / 2; PING -> PONG;
//        - `graph` with a multi-layer MUL_MAT op list: two Qwen2.5-0.5B-shaped layers (q / k / v with
//          bias rows, o with the residual, gate, up, down with the residual; JOIN records for the
//          three-predecessor ops; Q8_0 and, for layer 0's down projection, Q4_K) and an 8192-row
//          output projection with ACC_DST, T = 3 tokens, weights in the device layout made by the
//          tsr_ref layout functions (tsr_row_location + tsr_repack_row over random ggml blocks),
//          VERIFY_CRC and MARK; every y row bit-exact against sw/ctrl/ref/mulmat_ref.cpp applied in
//          op order to a shadow DRAM;
//        - `selftest` ids 1-10 and its misuses (the cases of fw_selftest_sim, fw_selftest_cases.h:
//          every output against tsr_ref / tsr_ref_vu / tsr_ref_at);
//        - `replay` of trace samples (traces/model-v1conv Q8_0 and Q4_K_M with SERIAL, vu/m6,
//          at/tmax8, ctrl/errors (a TILE_ERR stream), ctrl/tmax8; each the longest stream of its set,
//          skipped with a message when the set is not on disk) and of a synthetic 3-chunk stream:
//          DONE status / RET / error code / ERR_INFO / ERR_W0 equal to the trace's, every expect
//          region bit-exact;
//        - `graph` again with the same list, its inputs rewritten by the host first (EX §2.3 H1), after
//          every other kernel ran on this image: the same y rows; its captured trace replays on a
//          fresh simulator in B, so DTCM state that the executor carried from its first launch (or
//          from the other kernels) shows there (V2-SW-4: a first launch after boot cannot show it);
//        - a last PING; the device idle, no unit in error state, no race report.
//      Equal under all 8 schedules: every answer record r0-r6, the SW §4.7 observables (per-unit op
//      streams with relative tickets, the RISC-V SPM store stream, the final written DRAM and SPM
//      state: test_sim_obs.h) and the captured launch traces (push records, DONE, tickets.bin,
//      init / expect regions, image); the schedules change the timing.
//   B. Trace round trip (SW §10.1, V2-SW-4: a DTCM state leak between launches would show here):
//      the fixed session's TSR_TRACE directory holds one trace per captured launch, equal to the
//      in-memory capture; each replays on fresh simulators under fixed and random:11; the random:3
//      session's in-memory captures replay under fixed.
//
//   test_fw_m6_e2e IMAGE SCRATCH --perf
//      fw_m6_perf (fixed schedule, default F_RV / F_CORE): the image sizes (SW §7.8), the launch costs
//      of EX §4.11 on the production image (descriptor push -> kernel entry, empty-launch round trip,
//      DURATION), the RISC-V cycles per graph op (SW §7.8: <= 400) on 0.5B decode lists, and RISC-V
//      versus tile time for one 0.5B decode layer (does the executor stay ahead of the units).
//      Printed; asserted only against regression ceilings.
// Exit 0 / 1; 77 (skipped) without the image.
#include "test_sim_obs.h"

#include "fw_selftest_cases.h"   // session, launch, st_run_ids (+ fw_sim_host.hpp)

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "../ref/mulmat_ref.hpp"
#include "tsr/opl.hpp"
#include "tsr_bus.h"
#include "tsr_elf.h"
#include "tsr_race.h"
#include "tsr_trace.h"

namespace {

namespace opl = tsr::opl;
namespace mm = tsr::opl::mm;

constexpr uint64_t LINE = 64u;
constexpr int N_SCHED = 8;
const char * const k_sched[N_SCHED] = { "fixed", "random:1", "random:2", "random:3",
                                        "random:4", "random:5", "random:6", "random:7" };
const char * const k_replay_sched[2] = { "fixed", "random:11" };
constexpr uint32_t MAX_TR = 64;

const char * g_image = nullptr;
const char * g_traces = nullptr;
uint64_t g_elf_id = 0;                   // tsr-elf's computation
uint32_t g_sdk_abi = 0, g_nkern = 0;

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
uint32_t h_ch(uint32_t h) { return h >> 28; }
uint64_t h_beat(uint32_t h) { return h & 0x0FFFFFFFu; }

// The M6 U200 configuration (EX §11, CC): 4 channels of 16 GiB (lazily mapped), 4 MiB SPM, T_MAX 8,
// the M6 CAPS (VU and the AT ops; TOPK and DQROW come with M8), U200 queue depths (the default).
tsr_config e2e_cfg() {
    tsr_config c = TSR_CONFIG_U200_V1;
    c.t_max = 8;
    c.caps = TSR_CAPS_VU | TSR_CAPS_ATT;
    return c;
}

// =================================================================================================
// The graph case: two 0.5B-shaped layers and an output projection, MUL_MAT records only (M6)
// =================================================================================================
struct region {
    uint32_t ch;
    uint64_t off;                        // bytes
    std::vector<uint8_t> data;
};

struct graph_case {
    std::vector<std::array<uint32_t, 12>> recs;   // MUL_MAT argument words (SW §7.2)
    std::vector<uint32_t> rec_index;               // op index of each MUL_MAT record
    std::vector<region> inputs;                    // what the host writes before the launch
    std::vector<region> want;                      // every y row after the list (the reference)
    std::vector<uint8_t> blob;                     // the op list
    uint32_t opl_h = H(0, 0x40000u);               // ch0 16 MiB, inside the OPL region (EX §11)
    uint32_t n_ops = 0;
    uint64_t weight_bytes = 0, macs = 0;
};

constexpr uint32_t G_T = 3;                                         // tokens
constexpr uint64_t G_W_BEAT = TSR_REGION_WEIGHT_OFF / 64u + 0x100000u;   // 576 MiB: above the selftest weights
constexpr uint64_t G_IO_BEAT = 0x200000u;                           // 128 MiB: above the selftest areas

// Builds the list (decode or prefill shapes) on a shadow simulator's DRAM (no RISC-V runs there):
// the inputs are generated into it and copied out as regions; then mulmat_ref applies every record
// in op order (the list is topologically ordered) and the y rows are read back as the expectation.
// layers / T / with_out select the shape; perf lists use the same builder.
graph_case build_graph(uint32_t layers, uint32_t T, bool with_out, bool want_ref, uint64_t seed) {
    graph_case g;
    const tsr_config cfg = e2e_cfg();
    tsr_sim * sh = tsr_sim_create(&cfg, nullptr);
    CHECK(sh != nullptr, "shadow simulator");
    if (sh == nullptr) return g;
    const uint32_t C = cfg.num_ch;
    rng_t r = { seed };
    uint64_t wcur = G_W_BEAT;
    uint64_t io[4] = { G_IO_BEAT, G_IO_BEAT, G_IO_BEAT, G_IO_BEAT };
    auto keep = [&](uint32_t ch, uint64_t beat, uint64_t lines) {
        region rg{ ch, beat * LINE, std::vector<uint8_t>(size_t(lines * LINE)) };
        memcpy(rg.data.data(), tsr_sim_dram(sh, ch) + beat * LINE, rg.data.size());
        g.inputs.push_back(std::move(rg));
    };
    auto weights = [&](uint32_t fmt, uint32_t N, uint32_t K) -> uint32_t {
        const uint64_t w_off = wcur;
        const size_t rb = size_t(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
        std::vector<uint8_t> blk(rb);
        for (uint32_t n = 0; n < N; ++n) {
            gen_row_blocks(&r, fmt, K, blk.data());
            uint32_t ch;
            uint64_t off;
            tsr_row_location(fmt, N, K, C, n, &ch, &off);
            tsr_repack_row(fmt, blk.data(), K, tsr_sim_dram(sh, ch) + (w_off + off) * LINE);
        }
        const uint64_t beats = tsr_tensor_ch_beats(fmt, N, K, C);
        for (uint32_t c = 0; c < C; ++c) keep(c, w_off, beats);
        g.weight_bytes += beats * LINE * C;
        wcur += (beats + 63u) & ~63ull;
        return uint32_t(w_off);
    };
    // T rows of `stride` lines on channel ch; `vals` random f32 per row (0: an output, not written).
    auto rows = [&](uint32_t ch, uint32_t n_rows, uint32_t stride, uint32_t vals) -> uint32_t {
        const uint64_t beat = io[ch];
        io[ch] += (uint64_t(n_rows) * stride + 63u) & ~63ull;
        if (vals) {
            std::vector<float> x(vals);
            for (uint32_t t = 0; t < n_rows; ++t) {
                gen_f32(&r, x.data(), vals);
                put_f32_le(tsr_sim_dram(sh, ch) + (beat + uint64_t(t) * stride) * LINE, x.data(), vals);
            }
            keep(ch, beat, uint64_t(n_rows) * stride);
        }
        return H(ch, beat);
    };
    opl::writer w(opl::F_STOP_ON_ERROR | opl::F_MARK, 0xE2E0u + layers);
    auto mmul = [&](uint32_t x, uint32_t xs, uint32_t y, uint32_t ys, uint32_t woff, uint32_t fmt, uint32_t N, uint32_t K,
                    uint32_t add, uint32_t add_s, uint32_t flags, std::initializer_list<uint32_t> deps) -> uint32_t {
        std::array<uint32_t, 12> a{};
        a[mm::X] = x; a[mm::X_STRIDE] = xs; a[mm::Y] = y; a[mm::Y_STRIDE] = ys; a[mm::W_OFF] = woff; a[mm::FMT] = fmt;
        a[mm::N] = N; a[mm::K] = K; a[mm::T] = T; a[mm::ADD] = add; a[mm::ADD_STRIDE] = add_s; a[mm::FLAGS] = flags;
        std::vector<uint32_t> d;
        for (uint32_t v : deps)
            if (v != opl::DEP_NONE) d.push_back(v);
        const uint32_t idx = w.mul_mat(x, xs, y, ys, woff, fmt, N, K, T, add, add_s, flags, d.data(), uint32_t(d.size()));
        g.recs.push_back(a);
        g.rec_index.push_back(idx);
        g.macs += uint64_t(N) * K * T;
        return idx;
    };
    const uint32_t Q8 = TSR_FMT_Q8_0, Q4 = TSR_FMT_Q4_K, NONE = opl::HANDLE_NONE;
    const uint32_t D = 896, DKV = 128, DFF = 4864, xs = D / 16u, kvs = DKV / 16u, fs = DFF / 16u;
    uint32_t X = rows(0, T, xs, D);                               // the embeddings of the T tokens
    uint32_t prev = opl::DEP_NONE;
    for (uint32_t l = 0; l < layers; ++l) {
        const uint32_t wq = weights(Q8, D, D), wk = weights(Q8, DKV, D), wv = weights(Q8, DKV, D), wo = weights(Q8, D, D);
        const uint32_t wg = weights(Q8, DFF, D), wu = weights(Q8, DFF, D);
        const uint32_t fd = l == 0 ? Q4 : Q8;                     // Q4_K needs K % 256 == 0: only the down projection
        const uint32_t wd = weights(fd, D, DFF);
        const uint32_t bq = rows(1, 1, xs, D), bk = rows(1, 1, kvs, DKV), bv = rows(1, 1, kvs, DKV);
        const uint32_t Qy = rows(0, T, xs, 0), Ky = rows(1, T, kvs, 0), Vy = rows(2, T, kvs, 0), Hy = rows(3, T, xs, 0);
        const uint32_t Gy = rows(0, T, fs, 0), Uy = rows(1, T, fs, 0), Xn = rows(2, T, xs, 0);
        const uint32_t q = mmul(X, xs, Qy, xs, wq, Q8, D, D, bq, 0, 0, { prev });              // + bias (stride 0)
        const uint32_t k = mmul(X, xs, Ky, kvs, wk, Q8, DKV, D, bk, 0, 0, { prev });
        const uint32_t v = mmul(X, xs, Vy, kvs, wv, Q8, DKV, D, bv, 0, 0, { prev });
        const uint32_t o = mmul(Qy, xs, Hy, xs, wo, Q8, D, D, X, xs, 0, { q, k, v });       // + residual; JOIN
        const uint32_t gg = mmul(Hy, xs, Gy, fs, wg, Q8, DFF, D, NONE, 0, 0, { o });
        const uint32_t u = mmul(Hy, xs, Uy, fs, wu, Q8, DFF, D, NONE, 0, 0, { o });
        prev = mmul(Gy, fs, Xn, xs, wd, fd, D, DFF, Hy, xs, 0, { gg, u, o });                 // + residual; JOIN
        X = Xn;
    }
    if (with_out) {                                               // output projection, y += W x (ACC_DST)
        const uint32_t NV = 8192, vs = NV / 16u;
        const uint32_t wout = weights(Q8, NV, D);
        const uint32_t L = rows(3, T, vs, NV);                    // the old y rows (a logit bias)
        mmul(X, xs, L, vs, wout, Q8, NV, D, NONE, 0, mm::FLAG_ACC_DST, { prev });
    }
    g.blob = w.finish();
    g.n_ops = w.n_ops();
    CHECK(w.ok() && !g.blob.empty(), "OPL writer: %s", w.error());
    {   // the op list itself is a host input (EX §2.3 H1)
        region rg{ h_ch(g.opl_h), h_beat(g.opl_h) * LINE, g.blob };
        rg.data.resize((rg.data.size() + LINE - 1) / LINE * LINE, 0);
        g.inputs.push_back(std::move(rg));
    }
    if (want_ref) {
        std::vector<uint8_t *> chp(C);
        for (uint32_t c = 0; c < C; ++c) chp[c] = tsr_sim_dram(sh, c);
        const tsr::ref::dram_view view{ chp.data(), C, cfg.ch_bytes };
        for (size_t i = 0; i < g.recs.size(); ++i) {
            const int rc = tsr::ref::mulmat_ref(view, g.recs[i].data(), 4);
            CHECK(rc == tsr::ref::MM_OK, "reference of op %u: %d", g.rec_index[i], rc);
        }
        for (const auto & a : g.recs) {
            const uint32_t yl = (a[mm::N] + 15u) / 16u;
            for (uint32_t t = 0; t < a[mm::T]; ++t) {
                const uint64_t beat = h_beat(a[mm::Y]) + uint64_t(t) * a[mm::Y_STRIDE];
                region rg{ h_ch(a[mm::Y]), beat * LINE, std::vector<uint8_t>(size_t(yl) * LINE) };
                memcpy(rg.data.data(), tsr_sim_dram(sh, rg.ch) + rg.off, rg.data.size());
                g.want.push_back(std::move(rg));
            }
        }
    }
    tsr_sim_destroy(sh);
    return g;
}

void put_regions(tsr_sim * sim, const std::vector<region> & v) {
    for (const auto & rg : v) tsr_sim_dram_write(sim, rg.ch, rg.off, rg.data.data(), rg.data.size());
}

// =================================================================================================
// Replay samples (SW §7.7): kind-ops traces launched through the `replay` kernel inside a session
// =================================================================================================
struct sample {
    std::string name;
    tsr_trace2 * t = nullptr;
    uint32_t flags = 0;                  // TSR_REPLAY_F_SERIAL for traces/model-v1conv
};

int name_cmp(const void * a, const void * b) { return strcmp(*(char * const *)a, *(char * const *)b); }

// Every trace directory under dir (depth-first, name order).
void collect_traces(const std::string & dir, std::vector<std::string> & out) {
    struct stat st;
    if (stat((dir + "/meta.json").c_str(), &st) == 0) {
        out.push_back(dir);
        return;
    }
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) return;
    std::vector<std::string> subs;
    for (struct dirent * e; (e = readdir(d)) != nullptr;) {
        if (e->d_name[0] == '.' || !strcmp(e->d_name, "images")) continue;
        const std::string p = dir + "/" + e->d_name;
        if (stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) subs.push_back(p);
    }
    closedir(d);
    std::sort(subs.begin(), subs.end());
    for (const auto & s : subs) collect_traces(s, out);
}

// The sample of a set: its longest push stream (the first in name order among equals), or "".
std::string pick_trace(const std::string & dir) {
    std::vector<std::string> v;
    collect_traces(dir, v);
    std::string best;
    long long best_n = -1;
    for (const auto & t : v) {
        struct stat st;
        const long long n = stat((t + "/ops/push.bin").c_str(), &st) == 0 ? (long long)st.st_size : 0;
        if (n > best_n) {
            best_n = n;
            best = t;
        }
    }
    return best;
}

// A synthetic 3-chunk stream (as fw_replay_sim's): LD / ST pairs with deps across chunk boundaries;
// its expectations from a tile-only run (tsr_trace2_run_ops). Always available.
tsr_trace2 * synthetic_trace() {
    tsr_trace2 * t = tsr_trace2_new();
    const tsr_config core = e2e_cfg();
    tsr_trace2_config_from_core(&t->cfg, &core);
    t->cfg.ctl_id = tsr_trace2_ctl_id_full();
    rng_t r = { 0x5A11E2Eu };
    std::vector<uint8_t> in(256u * 64u);
    rng_bytes(&r, in.data(), in.size());
    const uint64_t in_beat = G_IO_BEAT + 0x100000u, out_beat = in_beat + 1024u;
    tsr_trace2_add_init(t, 1, in_beat * 64u, in.data(), in.size());
    for (uint32_t i = 0; i < 1550u; ++i) {                        // 3,100 records: chunks of 1,364
        const uint32_t line = 4096u + i % 512u, src = uint32_t(in_beat + (i * 7u) % 256u);
        const tsr_insn ld = tsr_enc_ld(1, line, src, 1);
        uint32_t dep = TSR_TICKET_NONE;
        if (i >= 700u && i % 5u == 0u) dep = tsr_trace2_ticket_of(t, t->n_push - 1400u);
        tsr_trace2_add_op(t, ld.w, dep, TSR_TICKET_NONE, 0x0F00u);
        const uint32_t ld_t = tsr_trace2_ticket_of(t, t->n_push - 1u);
        const tsr_insn st = tsr_enc_st((i & 1u) ? 2u : 3u, line, uint32_t(out_beat + i), 1);
        tsr_trace2_add_op(t, st.w, ld_t, TSR_TICKET_NONE, 0x0F00u);
    }
    tsr_trace2_replay_opts o;
    memset(&o, 0, sizeof(o));
    o.init_from_reads = 1;
    o.race = 2;
    tsr_trace2_result res;
    const int rc = tsr_trace2_run_ops(t, &o, &res);
    CHECK(rc == 0, "synthetic stream: run_ops: %s", res.msg);
    t->load_ch = 0;                                               // push.bin at ch0 1 MiB (OPL region)
    t->load_offset = 1u << 20;
    return t;
}

std::vector<sample> load_samples() {
    std::vector<sample> v;
    static const struct { const char * set; uint32_t flags; } sets[] = {
        { "model-v1conv/qwen05b-q8_0", TSR_REPLAY_F_SERIAL },
        { "model-v1conv/qwen05b-q4_k_m-requant", TSR_REPLAY_F_SERIAL },
        { "vu/m6", 0 }, { "at/tmax8", 0 }, { "ctrl/errors", 0 }, { "ctrl/tmax8", 0 },
    };
    const tsr_config c = e2e_cfg();
    for (const auto & s : sets) {
        const std::string dir = g_traces ? pick_trace(std::string(g_traces) + "/" + s.set) : "";
        if (dir.empty()) {
            printf("fw_m6_e2e: no trace under %s/%s (regen.sh): sample not run\n", g_traces ? g_traces : "(none)", s.set);
            continue;
        }
        tsr_trace2 * t = nullptr;
        char err[1024];
        if (tsr_trace2_read(dir.c_str(), &t, err, sizeof(err)) != 0) {
            CHECK(0, "%s: %s", dir.c_str(), err);
            continue;
        }
        const tsr_config & tc = t->cfg.core;
        const bool fits = t->kind == TSR_TRACE2_KIND_OPS && tc.num_ch == c.num_ch && tc.spm_bytes == c.spm_bytes &&
                          tc.ch_bytes == c.ch_bytes && tc.t_max <= c.t_max && (tc.caps & ~c.caps) == 0u &&
                          memcmp(t->cfg.q_depth, "\x08\x04\x04\x10\x00\x04", 6) == 0 && t->load_ch != TSR_TRACE2_LOAD_AUTO;
        CHECK(fits, "%s: not a kind-ops trace of the M6 configuration", dir.c_str());
        if (!fits) {
            tsr_trace2_free(t);
            continue;
        }
        v.push_back(sample{ dir, t, s.flags });
    }
    v.push_back(sample{ "synthetic 3-chunk stream", synthetic_trace(), 0 });
    return v;
}

// Lines [line, line + need) below the runtime argument range that no SPM operand of the trace's
// accepted ops touches, the highest such run (as tsr_trace.c's replay route places the buffer, so
// the kernel's reads of push.bin never race with the replayed ops). 0, or -1.
int spm_buffer(tsr_sim * sim, const tsr_trace2 * t, uint32_t need, uint32_t * line) {
    const uint32_t L = tsr_sim_spm_lines(sim);
    std::vector<uint8_t> used(L, 0);
    for (size_t i = 0; i < t->n_push; ++i) {
        const tsr_trace_push * p = &t->push[i];
        if (p->kind != TSR_TRACE_PUSH_OP) continue;
        const tsr_op_handler * hd = tsr_sim_op_handler(uint8_t(p->w[0] & 0xFFu));
        if (tsr_trace2_push_unit(&t->cfg, p) == TSR_UNIT_SYNC || hd == nullptr || hd->check == nullptr) continue;
        tsr_extents ext;
        memset(&ext, 0, sizeof(ext));
        uint32_t info = 0;
        if (hd->check(sim, p->w, &ext, &info) != 0) continue;
        for (uint32_t k = 0; k < ext.n; ++k)
            if (ext.e[k].kind == TSR_EXT_SPM)
                for (uint64_t l = ext.e[k].lo; l < ext.e[k].hi && l < L; ++l) used[size_t(l)] = 1;
    }
    uint32_t run = 0;
    for (uint32_t l = L - 16u; l-- > 0;) {
        run = used[l] ? 0u : run + 1u;
        if (run == need) {
            *line = l;
            return 0;
        }
    }
    return -1;
}

// =================================================================================================
// One session (A)
// =================================================================================================
using ans = std::array<uint32_t, 7>;

struct e2e_run {
    const char * sched = nullptr;
    std::string trace_dir;
    std::vector<ans> answers;            // HELLO, PONG, every DONE, PONG (r0-r6)
    obs ob{};
    std::vector<tsr_trace2> tr;          // captured launches, in answer order
    tsr_trace_capture_stats cs{};
    size_t n_race = 0;
    uint64_t mtime = 0, instret = 0;
    uint64_t graph_cycles = 0;
    int kidx[3] = { -1, -1, -1 };
};

void keep_answer(e2e_run & R, const uint32_t * r) {
    ans a;
    memcpy(a.data(), r, sizeof(a));
    R.answers.push_back(a);
}

void session_graph(session & s, e2e_run & R, const graph_case & g) {
    const uint32_t a[7] = { g.opl_h, uint32_t(g.blob.size()), opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
    if (fwh_launch(&s.h, uint32_t(R.kidx[0]), a, 0) != 0) {
        CHECK(0, "[%s] graph: no answer", s.sched);
        return;
    }
    keep_answer(R, s.h.done);
    if (R.graph_cycles == 0) R.graph_cycles = s.h.cycles;
    CHECK(TSR_GET(s.h.done[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(&s.h) == TSR_LCQ_R0_STATUS_OK &&
              fwh_ret(&s.h) == 0, "[%s] graph: DONE 0x%08x RET 0x%08x (first error ticket 0x%08x info 0x%08x w0 0x%08x pidx %u)",
          s.sched, s.h.done[0], s.h.done[2], s.h.done[3], s.h.done[4], s.h.done[5], s.h.done[6]);
    size_t bad = 0;
    for (const auto & rg : g.want) {
        const uint8_t * got = tsr_sim_dram(s.sim, rg.ch) + rg.off;
        if (memcmp(got, rg.data.data(), rg.data.size()) != 0 && bad++ < 3) {
            size_t k = 0;
            while (got[k] == rg.data[k]) ++k;
            CHECK(0, "[%s] graph: y row at ch %u beat 0x%llx differs at byte %zu (0x%08x, reference 0x%08x)", s.sched, rg.ch,
                  (unsigned long long)(rg.off / LINE), k, get_u32_le(got + (k & ~size_t(3))),
                  get_u32_le(rg.data.data() + (k & ~size_t(3))));
        }
    }
    CHECK(bad == 0, "[%s] graph: %zu of %zu y rows differ from the reference", s.sched, bad, g.want.size());
}

void session_replay(session & s, e2e_run & R, const sample & sp) {
    const tsr_trace2 * t = sp.t;
    for (size_t i = 0; i < t->mem.n_init; ++i)
        tsr_sim_dram_write(s.sim, t->mem.init[i].ch, t->mem.init[i].byte_off, t->mem.init[i].data, t->mem.init[i].size);
    const uint64_t bytes = uint64_t(t->n_push) * TSR_TRACE_PUSH_BYTES, lines = (bytes + 63u) / 64u;
    std::vector<uint8_t> buf(size_t(lines * 64u), 0);
    for (size_t i = 0; i < t->n_push; ++i) {
        uint8_t * q = buf.data() + i * TSR_TRACE_PUSH_BYTES;
        const tsr_trace_push * p = &t->push[i];
        uint32_t v[12];
        memcpy(v, p->w, 32);
        v[8] = p->dep[0]; v[9] = p->dep[1]; v[10] = p->push; v[11] = p->kind;
        for (int k = 0; k < 12; ++k) wr(q, uint32_t(k), v[k]);
    }
    tsr_sim_dram_write(s.sim, t->load_ch, t->load_offset, buf.data(), buf.size());
    const uint32_t chunk = t->n_push <= TSR_REPLAY_CHUNK_MAX ? uint32_t((t->n_push + 3u) & ~size_t(3)) : TSR_REPLAY_CHUNK_MAX;
    const uint32_t chunk_lines = chunk * TSR_TRACE_PUSH_BYTES / 64u, nbuf = t->n_push > chunk ? 2u : 1u;
    uint32_t bl = 0;
    if (spm_buffer(s.sim, t, nbuf * chunk_lines, &bl) != 0) {
        CHECK(0, "[%s] replay %s: no SPM buffer of %u lines outside the trace's operands", s.sched, sp.name.c_str(),
              nbuf * chunk_lines);
        return;
    }
    const uint32_t a[7] = { H(t->load_ch, t->load_offset / 64u), uint32_t(t->n_push), sp.flags | TSR_REPLAY_F_BUF, bl,
                            chunk ? chunk : 4u, 0, 0 };
    if (fwh_launch(&s.h, uint32_t(R.kidx[2]), a, 0) != 0) {
        CHECK(0, "[%s] replay %s: no answer", s.sched, sp.name.c_str());
        return;
    }
    keep_answer(R, s.h.done);
    const uint32_t * d = s.h.done;
    const tsr_trace_done & w = t->done;
    const uint32_t st = fwh_status(&s.h);
    CHECK(TSR_GET(d[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && st == w.status && fwh_ret(&s.h) == w.ret &&
              TSR_GET(d[0], TSR_LCQ_R0_ERR_CODE) == w.err_code &&
              (st != TSR_LCQ_R0_STATUS_TILE_ERR || (d[4] == w.err_info && d[5] == w.err_w0)),
          "[%s] replay %s: DONE status %u RET %d code %u info 0x%08x w0 0x%08x, trace: %u %d %u 0x%08x 0x%08x", s.sched,
          sp.name.c_str(), st, fwh_ret(&s.h), TSR_GET(d[0], TSR_LCQ_R0_ERR_CODE), d[4], d[5], w.status, w.ret, w.err_code,
          w.err_info, w.err_w0);
    for (size_t i = 0; i < t->mem.n_expect; ++i) {
        const tsr_region & rg = t->mem.expect[i];
        CHECK(memcmp(tsr_sim_dram(s.sim, rg.ch) + rg.byte_off, rg.data, size_t(rg.size)) == 0,
              "[%s] replay %s: expect region %zu (ch %u 0x%llx, %llu B) differs", s.sched, sp.name.c_str(), i, rg.ch,
              (unsigned long long)rg.byte_off, (unsigned long long)rg.size);
    }
}

void run_session(const char * sched, e2e_run & R, const graph_case & g, const std::vector<sample> & samples) {
    R.sched = sched;
    const tsr_config cfg = e2e_cfg();
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = 2;                         // fatal: a race aborts the test
    so.threads = 2;
    so.trace_dir = R.trace_dir.c_str();  // TSR_TRACE: one trace per launch
    session s;
    memset(&s, 0, sizeof(s));
    s.sched = sched;
    s.sim = tsr_sim_create_opts(&cfg, &so);
    CHECK(s.sim != nullptr, "[%s] simulator", sched);
    if (s.sim == nullptr) return;
    CHECK(tsr_sim_opt_race(s.sim) == 2u && *tsr_sim_model_slot(s.sim, TSR_MODEL_RACE) != nullptr &&
              tsr_sim_opt_trace_dir(s.sim) != nullptr && *tsr_sim_model_slot(s.sim, TSR_MODEL_TRACE) != nullptr,
          "[%s] the race detector (fatal) and the TRACE model (TSR_TRACE) are attached", sched);
    obs_attach(&R.ob, s.sim);
    tsr_sim_capture_launch(s.sim, 1);
    put_regions(s.sim, g.inputs);        // before boot: H1 holds trivially
    if (fwh_open(&s.h, s.sim, g_image) != 0) {   // loader with receive CRC + VERIFY, HELLO id = tsr_trace2_image_id
        CHECK(0, "[%s] boot", sched);
        tsr_sim_destroy(s.sim);
        return;
    }
    {
        const uint32_t * h = s.h.hello;
        keep_answer(R, h);
        const uint64_t id = (uint64_t(h[3]) << 32) | h[2];
        CHECK(id == g_elf_id && id == s.h.image_id, "[%s] HELLO id %016llx, tsr-elf %016llx, trace library %016llx", sched,
              (unsigned long long)id, (unsigned long long)g_elf_id, (unsigned long long)s.h.image_id);
        CHECK((h[4] & 0xFFFFu) == g_sdk_abi && h[5] == g_nkern && h[5] == 3u, "[%s] HELLO r4 0x%08x (SDK ABI %u) r5 %u (%u kernels)",
              sched, h[4], g_sdk_abi, h[5], g_nkern);
    }
    static const char * const names[3] = { "graph", "selftest", "replay" };
    for (int k = 0; k < 3; ++k) {
        R.kidx[k] = fwh_kernel(&s.h, names[k]);
        CHECK(R.kidx[k] == k, "[%s] kernel %s at index %d (EX §4.6 base kernels: graph 0, selftest 1, replay 2)", sched,
              names[k], R.kidx[k]);
    }
    if (R.kidx[0] == 0 && R.kidx[1] == 1 && R.kidx[2] == 2) {
        CHECK(fwh_ping(&s.h, 0xE2E0E2E0u) == 0, "[%s] PING -> PONG with the ARG0 echo", sched);
        keep_answer(R, s.h.done);
        session_graph(s, R, g);
        s.idx = R.kidx[1];
        st_run_ids(&s);
        for (uint32_t i = 0; i < s.n_done; ++i) {
            ans a;
            memcpy(a.data(), s.done[i], sizeof(a));
            R.answers.push_back(a);
        }
        for (const auto & sp : samples) session_replay(s, R, sp);
        // fix W2a: graph a second time, after selftest and replay ran on this image (the inputs, incl.
        // the ACC_DST rows, rewritten while the device is idle): same rows; its trace replays fresh (B).
        put_regions(s.sim, g.inputs);
        session_graph(s, R, g);
        CHECK(fwh_ping(&s.h, 0xE2E0FFFFu) == 0, "[%s] last PING", sched);
        keep_answer(R, s.h.done);
        tsr_sim_advance(s.sim, 100000);
        const uint32_t st = tsr_sim_csr_read(s.sim, TSR_CSR_STATUS);
        CHECK((st & TSR_STATUS_QIDLE) && !(st & TSR_STATUS_QERR) &&
                  !(st & (TSR_STATUS_LQ_OVERFLOW | TSR_STATUS_RING_RD_BAD | TSR_STATUS_LOG_DROP)),
              "[%s] STATUS 0x%08x at the end (idle, no unit in error, no sticky error)", sched, st);
        CHECK(TSR_GET(tsr_sim_csr_read(s.sim, TSR_CSR_RT_STATE), TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_IDLE,
              "[%s] RT_STATE IDLE at the end", sched);
    }
    obs_finish(&R.ob, s.sim);
    for (uint32_t i = 0; i < MAX_TR; ++i) {
        tsr_trace2 t;
        tsr_trace2_init(&t);
        if (tsr_sim_capture_take(s.sim, &t) != 0) {
            tsr_trace2_clear(&t);
            break;
        }
        R.tr.push_back(t);
    }
    tsr_sim_capture_stats(s.sim, &R.cs);
    const tsr_race_report * rr = nullptr;
    R.n_race = tsr_sim_race_reports(s.sim, &rr);
    R.mtime = tsr_sim_now(s.sim);
    R.instret = tsr_sim_rv_instret(s.sim);
    fwh_close(&s.h);
    tsr_sim_destroy(s.sim);
}

void run_free(e2e_run & R) {
    obs_free(&R.ob);
    for (auto & t : R.tr) tsr_trace2_clear(&t);
    R.tr.clear();
}

// Everything SW §5.4 compares and a trace carries (not sched, rv_ratio, perf), as sim_w1b_e2e.
bool regions_equal(const tsr_region * a, size_t na, const tsr_region * b, size_t nb) {
    if (na != nb) return false;
    for (size_t i = 0; i < na; ++i) {
        if (a[i].ch != b[i].ch || a[i].byte_off != b[i].byte_off || a[i].size != b[i].size) return false;
        if (a[i].size != 0 && memcmp(a[i].data, b[i].data, size_t(a[i].size)) != 0) return false;
    }
    return true;
}

// Field by field: tsr_config has padding before ch_bytes.
bool core_equal(const tsr_config & a, const tsr_config & b) {
    return a.num_ch == b.num_ch && a.beat_bits == b.beat_bits && a.lanes == b.lanes && a.t_max == b.t_max &&
           a.k_max == b.k_max && a.spm_bytes == b.spm_bytes && a.vl == b.vl && a.nsreg == b.nsreg &&
           a.rd_outstanding == b.rd_outstanding && a.ch_bytes == b.ch_bytes && a.addr_beat_w == b.addr_beat_w &&
           a.caps == b.caps;
}

bool trace_equal(const tsr_trace2 * a, const tsr_trace2 * b, const char * what) {
    const char * diff = nullptr;
    if (a->kind != b->kind) diff = "kind";
    else if (a->image_id != b->image_id || a->image_size != b->image_size ||
             (a->image_size && memcmp(a->image, b->image, a->image_size) != 0)) diff = "image";
    else if (a->kernel_index != b->kernel_index || a->prof != b->prof || memcmp(a->args, b->args, sizeof(a->args)) != 0)
        diff = "launch descriptor";
    else if (a->n_push != b->n_push || (a->n_push && memcmp(a->push, b->push, a->n_push * sizeof(tsr_trace_push)) != 0))
        diff = "push records";
    else if (memcmp(&a->done, &b->done, sizeof(a->done)) != 0) diff = "DONE";
    else if (a->n_units != b->n_units || memcmp(a->unit, b->unit, a->n_units * sizeof(tsr_trace_unit)) != 0) diff = "tickets";
    else if (!regions_equal(a->mem.init, a->mem.n_init, b->mem.init, b->mem.n_init)) diff = "init regions";
    else if (!regions_equal(a->mem.expect, a->mem.n_expect, b->mem.expect, b->mem.n_expect)) diff = "expect regions";
    else if (!core_equal(a->cfg.core, b->cfg.core) || a->cfg.ctl_id != b->cfg.ctl_id ||
             a->cfg.cfg3 != b->cfg.cfg3 || a->cfg.itcm_bytes != b->cfg.itcm_bytes || a->cfg.dtcm_bytes != b->cfg.dtcm_bytes ||
             memcmp(a->cfg.q_depth, b->cfg.q_depth, sizeof(a->cfg.q_depth)) != 0) diff = "config";
    CHECK(diff == nullptr, "%s: captured traces differ (%s)", what, diff ? diff : "");
    return diff == nullptr;
}

void run_equal(const e2e_run & a, const e2e_run & b, const char * what) {
    CHECK(a.answers.size() == b.answers.size() && std::equal(a.answers.begin(), a.answers.end(), b.answers.begin()),
          "%s: answer records r0-r6 differ (%zu vs %zu records)", what, a.answers.size(), b.answers.size());
    obs_equal(&a.ob, &b.ob, what);
    CHECK(a.tr.size() == b.tr.size(), "%s: %zu captured traces, reference %zu", what, b.tr.size(), a.tr.size());
    for (size_t i = 0; i < a.tr.size() && i < b.tr.size(); ++i) {
        char w[160];
        snprintf(w, sizeof(w), "%s, trace %zu", what, i);
        trace_equal(&a.tr[i], &b.tr[i], w);
    }
}

std::vector<std::string> list_traces(const std::string & dir) {
    std::vector<std::string> v;
    DIR * d = opendir(dir.c_str());
    if (d == nullptr) return v;
    for (struct dirent * e; (e = readdir(d)) != nullptr;) {
        if (e->d_name[0] == '.' || !strcmp(e->d_name, "images")) continue;
        struct stat st;
        const std::string p = dir + "/" + e->d_name;
        if (stat((p + "/meta.json").c_str(), &st) == 0) v.push_back(p);
    }
    closedir(d);
    std::sort(v.begin(), v.end());
    return v;
}

int e2e_main(const std::string & base) {
    const graph_case g = build_graph(2, G_T, true, true, 0xE2E05Bu);
    printf("fw_m6_e2e: graph list: %u records (%zu MUL_MAT, the rest JOIN), T = %u, %.1f MiB of weights, %.1f M MACs, "
           "%zu B op list\n", g.n_ops, g.recs.size(), G_T, double(g.weight_bytes) / (1 << 20), double(g.macs) / 1e6, g.blob.size());
    {   // the reference is not trivially zero: nearly every y word is a nonzero f32
        uint64_t words = 0, nz = 0;
        for (const auto & rg : g.want)
            for (size_t i = 0; i + 4 <= rg.data.size(); i += 4) {
                ++words;
                nz += get_u32_le(rg.data.data() + i) != 0u;
            }
        CHECK(words > 0 && nz * 100 >= words * 95, "graph reference: %llu of %llu y words nonzero", (unsigned long long)nz,
              (unsigned long long)words);
    }
    std::vector<sample> samples = load_samples();
    for (const auto & sp : samples)
        printf("fw_m6_e2e: replay sample %s: %zu records%s, DONE status %u RET %d\n", sp.name.c_str(), sp.t->n_push,
               sp.flags & TSR_REPLAY_F_SERIAL ? " (SERIAL)" : "", sp.t->done.status, sp.t->done.ret);
    static e2e_run R[N_SCHED];
    for (int k = 0; k < N_SCHED; ++k) {
        char d[64];
        snprintf(d, sizeof(d), "/run%d", k);
        R[k].trace_dir = base + d;
        run_session(k_sched[k], R[k], g, samples);
        CHECK(R[k].n_race == 0u, "[%s] %zu race reports", k_sched[k], R[k].n_race);
        // Every LAUNCH answered by DONE is captured: all answers but HELLO and the two PONGs (graph
        // twice, the selftest launches, the replays); the PINGs are counted as skipped.
        const uint32_t n_launch = R[k].answers.size() >= 3u ? uint32_t(R[k].answers.size() - 3u) : 0u;
        CHECK(n_launch == 2u + 16u + samples.size(), "[%s] %u launches answered, expected 2 graph + 16 selftest + %zu replays",
              k_sched[k], n_launch, samples.size());
        CHECK(R[k].cs.captured == n_launch && R[k].cs.written == n_launch && R[k].cs.skipped == 2u && R[k].cs.failed == 0u &&
                  R[k].tr.size() == n_launch,
              "[%s] captures %u written %u skipped %u failed %u taken %zu (expected %u launches, 2 PINGs)", k_sched[k],
              R[k].cs.captured, R[k].cs.written, R[k].cs.skipped, R[k].cs.failed, R[k].tr.size(), n_launch);
        if (k > 0) {
            char what[64];
            snprintf(what, sizeof(what), "A, %s vs fixed", k_sched[k]);
            run_equal(R[0], R[k], what);
        }
        printf("fw_m6_e2e A %-9s: %zu answers, %zu traces, graph %llu core cycles, MTIME %llu, %llu RV instructions\n",
               k_sched[k], R[k].answers.size(), R[k].tr.size(), (unsigned long long)R[k].graph_cycles,
               (unsigned long long)R[k].mtime, (unsigned long long)R[k].instret);
        if (k != 0 && k != 3 && !g_fail) rm_rf(R[k].trace_dir.c_str());
        if (k != 0 && k != 3) run_free(R[k]);
    }
    int distinct = 0;
    for (int k = 0; k < N_SCHED; ++k) {
        bool same = false;
        for (int j = 0; j < k; ++j) same |= R[j].mtime == R[k].mtime;
        distinct += !same;
    }
    CHECK(distinct >= 6, "A: the schedules change the timing (%d distinct final MTIMEs of %d)", distinct, N_SCHED);

    // ---- B. the fixed session's traces on disk, replayed on fresh simulators ----
    const std::vector<std::string> dirs = list_traces(R[0].trace_dir);
    CHECK(dirs.size() == R[0].tr.size(), "B: %zu trace directories under %s, %zu captured", dirs.size(),
          R[0].trace_dir.c_str(), R[0].tr.size());
    uint32_t n_replay = 0, n_ok = 0;
    for (size_t i = 0; i < dirs.size(); ++i) {
        tsr_trace2 * t = nullptr;
        char err[1024];
        if (tsr_trace2_read(dirs[i].c_str(), &t, err, sizeof(err)) != 0) {
            CHECK(0, "B: read %s: %s", dirs[i].c_str(), err);
            continue;
        }
        CHECK(t->kind == TSR_TRACE2_KIND_LAUNCH, "B: %s is a launch trace", dirs[i].c_str());
        if (i < R[0].tr.size()) {
            std::string what = "B, " + dirs[i] + " vs its in-memory capture";
            trace_equal(&R[0].tr[i], t, what.c_str());
        }
        for (const char * sc : k_replay_sched) {
            tsr_trace2_replay_opts o;
            memset(&o, 0, sizeof(o));
            o.sched = sc;
            o.race = 2;
            o.threads = 2;
            tsr_trace2_result res;
            const int rc = tsr_trace2_replay_sim_ex(t, &o, &res);
            ++n_replay;
            n_ok += rc == 0 && res.ok;
            CHECK(rc == 0 && res.ok, "B: replay %s (kernel %s) under %s: %s", dirs[i].c_str(), t->kernel, sc, res.msg);
        }
        tsr_trace2_free(t);
    }
    for (size_t i = 0; i < R[3].tr.size(); ++i) {                 // captured under random:3, replayed under fixed
        tsr_trace2_replay_opts o;
        memset(&o, 0, sizeof(o));
        o.sched = "fixed";
        o.race = 2;
        o.threads = 2;
        tsr_trace2_result res;
        const int rc = tsr_trace2_replay_sim_ex(&R[3].tr[i], &o, &res);
        ++n_replay;
        n_ok += rc == 0 && res.ok;
        CHECK(rc == 0 && res.ok, "B: random:3 trace %zu (kernel %s) replayed under fixed: %s", i, R[3].tr[i].kernel, res.msg);
    }
    printf("fw_m6_e2e B: %u of %u replays of captured launches on fresh simulators passed\n", n_ok, n_replay);
    run_free(R[0]);
    run_free(R[3]);
    for (auto & sp : samples) tsr_trace2_free(sp.t);
    return 0;
}

// =================================================================================================
// fw_m6_perf
// =================================================================================================
constexpr double F_CORE = 276.3, F_RV = 166.7;   // MHz (U200: D13 core clock; CC's F_RV)

struct perf_ctx {
    tsr_rv * rv = nullptr;
    uint64_t last = 0, busy = 0, insns = 0;
    // queue events of the current measurement
    std::vector<uint64_t> mark_time;
    uint64_t last_push = 0, first_push = 0;
    uint32_t pushes = 0;
    struct mvm_ev { uint64_t enq, start, done; };
    std::vector<mvm_ev> mvm;
    uint64_t last_done = 0;
};

int perf_retire(void * ctx, const tsr_rv_retire * r) {
    (void)r;
    perf_ctx * p = static_cast<perf_ctx *>(ctx);
    const uint64_t now = tsr_rv_cycles(p->rv);
    const uint64_t d = now - p->last;
    p->last = now;
    if (d <= 256u) p->busy += d;         // larger steps are sleeps / fast-forwarded waits (as fw_exec_perf)
    p->insns++;
    return 0;
}

void perf_q(void * ctx, tsr_sim *, const tsr_q_event * ev) {
    perf_ctx * p = static_cast<perf_ctx *>(ctx);
    if (ev->kind == TSR_QEV_MARK_PUSH) p->mark_time.push_back(ev->time);
    if (ev->kind == TSR_QEV_PUSH) {
        if (p->pushes++ == 0) p->first_push = ev->time;
        p->last_push = ev->time;
    }
    if (ev->unit == TSR_UNIT_MVM) {
        if (ev->kind == TSR_QEV_ENQ) p->mvm.push_back({ ev->time, 0, 0 });
        else if (ev->kind == TSR_QEV_START) {
            for (auto & m : p->mvm)
                if (m.start == 0) { m.start = ev->time; break; }
        } else if (ev->kind == TSR_QEV_DONE) {
            for (auto & m : p->mvm)
                if (m.done == 0) { m.done = ev->time; break; }
        }
    }
    if (ev->kind == TSR_QEV_DONE) p->last_done = ev->time;
}

void perf_reset(perf_ctx & p) {
    p.mark_time.clear();
    p.mvm.clear();
    p.pushes = 0;
    p.first_push = p.last_push = p.last_done = 0;
}

tsr_sim * perf_boot(fwh & h, perf_ctx & p) {
    const tsr_config cfg = e2e_cfg();
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = "fixed";
    so.threads = 4;
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &so);
    if (sim == nullptr) return nullptr;
    if (fwh_open(&h, sim, g_image) != 0) {
        CHECK(0, "perf: boot");
        tsr_sim_destroy(sim);
        return nullptr;
    }
    p.rv = tsr_rv_agent_rv(tsr_sim_rv_agent(sim));
    p.last = tsr_rv_cycles(p.rv);
    tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), perf_retire, &p);
    tsr_sim_add_listener(sim, perf_q, &p);
    return sim;
}

uint64_t perf_x(tsr_sim * sim, uint32_t xoff) {
    return tsr_sim_perf(sim, TSR_PERF_CH_BASE + 2u * tsr_sim_config(sim)->num_ch + xoff);
}

int perf_main() {
    // ---- image sizes (SW §7.8) ----
    {
        char err[256];
        tsr_elf * e = nullptr;
        if (tsr_elf_read(g_image, &e, err, sizeof(err)) == 0) {
            char rep[4096];
            tsr_elf_size_report(e, nullptr, rep, sizeof(rep));
            printf("fw_m6_perf: image %s\n%s", g_image, rep);
            tsr_elf_free(e);
        }
    }
    const int sel = 1;                    // selftest: an unknown id returns at once (an empty launch)
    // ---- launch costs (EX §4.11) ----
    {
        fwh h;
        perf_ctx p;
        tsr_sim * sim = perf_boot(h, p);
        if (sim == nullptr) return 1;
        constexpr int N = 16;
        uint64_t entry = 0, rtt = 0, dur = 0, emin = ~0ull, rmin = ~0ull, busy = 0;
        tsr_sim_advance(sim, 100000);
        for (int k = 0; k < N; ++k) {
            perf_reset(p);
            const uint64_t b0 = p.busy;
            const uint64_t t0 = tsr_sim_now(sim);
            const uint32_t a[7] = { 0xFFFFu, 0, 0, 0, 0, 0, 0 };
            if (fwh_launch(&h, uint32_t(sel), a, 1ull << 30) != 0 || p.mark_time.empty()) {
                CHECK(0, "perf: empty launch %d", k);
                break;
            }
            CHECK(fwh_status(&h) == TSR_LCQ_R0_STATUS_KERNEL_ERR && fwh_ret(&h) == int32_t(TSR_ST_RET_BAD_ID),
                  "perf: selftest with an unknown id: status %u RET 0x%08x", fwh_status(&h), uint32_t(fwh_ret(&h)));
            const uint64_t e = p.mark_time[0] - t0, d = h.cycles;
            entry += e; rtt += d; dur += h.done[7]; busy += p.busy - b0;
            emin = std::min(emin, e); rmin = std::min(rmin, d);
            tsr_sim_advance(sim, 100000);
        }
        // 8 outstanding
        const uint64_t t0 = tsr_sim_now(sim);
        const uint32_t seq0 = tsr_sim_csr_read(sim, TSR_CSR_LQ_SEQ);
        for (int k = 0; k < 8; ++k) {
            tsr_sim_csr_write(sim, TSR_CSR_LQ_W(0), TSR_PUT(TSR_DESC_TYPE, TSR_DESC_TYPE_LAUNCH) | TSR_PUT(TSR_DESC_KERNEL, sel));
            tsr_sim_csr_write(sim, TSR_CSR_LQ_W(1), 0xFFFFu);
            for (uint32_t i = 2; i < 8u; ++i) tsr_sim_csr_write(sim, TSR_CSR_LQ_W(i), 0u);
            tsr_sim_csr_write(sim, TSR_CSR_LQ_PUSH, 1u);
        }
        fwh_run(&h, 1, seq0 + 7u, 1ull << 30, "8 outstanding");
        const uint64_t t8 = tsr_sim_now(sim) - t0;
        const double rr = tsr_sim_rv_ratio(sim);
        printf("fw_m6_perf: launch costs on tsr-fw.elf (selftest with an unknown id = an empty kernel; fixed schedule, "
               "F_RV/F_CORE %.4f = %.1f / %.1f MHz)\n", rr, F_RV, F_CORE);
        printf("  descriptor push -> kernel entry (LAUNCH_BEGIN): %6.0f core cycles (min %llu) = %5.0f RV cycles = %.2f us "
               "[EX §4.11 <= 5 us]\n", double(entry) / N, (unsigned long long)emin, double(entry) / N * rr, double(entry) / N / F_CORE);
        printf("  empty launch, push -> DONE in the LCQ:         %6.0f core cycles (min %llu) = %5.0f RV cycles = %.2f us "
               "[device part of the <= 10 us round trip]\n", double(rtt) / N, (unsigned long long)rmin, double(rtt) / N * rr,
               double(rtt) / N / F_CORE);
        printf("  DONE r7 DURATION (pop -> post):                %6.0f core cycles = %.2f us; RISC-V busy %.0f RV cycles per launch\n",
               double(dur) / N, double(dur) / N / F_CORE, double(busy) / N);
        printf("  8 outstanding empty launches:                  %6.0f core cycles per launch = %.2f us (%.0f K launches/s)\n",
               double(t8) / 8.0, double(t8) / 8.0 / F_CORE, 8.0 * F_CORE * 1e3 / double(t8));
        CHECK(double(entry) / N / F_CORE <= 5.0, "perf: descriptor push -> kernel entry <= 5 us (EX §4.11)");
        tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), nullptr, nullptr);
        fwh_close(&h);
        tsr_sim_destroy(sim);
    }
    // ---- RV cycles per graph op (SW §7.8) and RV vs tile time per 0.5B decode layer ----
    {
        const graph_case g1 = build_graph(1, 1, false, false, 0x9E1u);
        const graph_case g3 = build_graph(3, 1, false, false, 0x9E3u);
        double cyc[2] = { 0, 0 }, ins[2] = { 0, 0 };
        uint32_t nops[2] = { g1.n_ops, g3.n_ops }, nmm[2] = { uint32_t(g1.recs.size()), uint32_t(g3.recs.size()) };
        for (int k = 0; k < 2; ++k) {
            const graph_case & g = k ? g3 : g1;
            fwh h;
            perf_ctx p;
            tsr_sim * sim = perf_boot(h, p);
            if (sim == nullptr) return 1;
            put_regions(sim, g.inputs);
            tsr_sim_advance(sim, 100000);
            perf_reset(p);
            const uint64_t b0 = p.busy, i0 = p.insns;
            uint64_t pf[6];
            const uint32_t X = TSR_PERF_CH_BASE + 2u * tsr_sim_config(sim)->num_ch;
            const uint32_t slot[6] = { TSR_PERF_MVM_CYCLES, X + TSR_PERF_XOFF_DMA_CYCLES, X + TSR_PERF_XOFF_QACT_CYCLES,
                                       X + TSR_PERF_XOFF_MVM_WAIT, X + TSR_PERF_XOFF_MVM_EMPTY, X + TSR_PERF_XOFF_PUSH_STALL };
            for (int i = 0; i < 6; ++i) pf[i] = tsr_sim_perf(sim, slot[i]);
            const uint64_t t0 = tsr_sim_now(sim);
            const uint32_t a[7] = { g.opl_h, uint32_t(g.blob.size()), 0, 0, 0, 0, 0 };
            if (fwh_launch(&h, 0, a, 0) != 0 || fwh_status(&h) != TSR_LCQ_R0_STATUS_OK) {
                CHECK(0, "perf: graph launch (%u layers): status %u RET 0x%08x", k ? 3u : 1u, fwh_status(&h), h.done[2]);
                tsr_sim_destroy(sim);
                return 1;
            }
            cyc[k] = double(p.busy - b0);
            ins[k] = double(p.insns - i0);
            for (int i = 0; i < 6; ++i) pf[i] = tsr_sim_perf(sim, slot[i]) - pf[i];
            if (k == 0) {
                // One 0.5B decode layer: who sets the pace? The MVM unit idles "issue-bound" when an MVM
                // reaches its queue only after the previous MVM finished.
                uint64_t issue_gap = 0, mvm_busy = 0;
                for (size_t i = 0; i < p.mvm.size(); ++i) {
                    mvm_busy += p.mvm[i].done - p.mvm[i].start;
                    if (i > 0 && p.mvm[i].enq > p.mvm[i - 1].done) issue_gap += p.mvm[i].enq - p.mvm[i - 1].done;
                }
                const uint64_t span = p.mvm.empty() ? 0 : p.mvm.back().done - p.mvm.front().start;
                const double rv_us = cyc[0] / F_RV;
                printf("fw_m6_perf: one Qwen2.5-0.5B decode layer (T = 1; q, k, v, o, gate, up, down + 2 JOIN = %u records; "
                       "%.1f MiB of weights), fixed schedule:\n", g1.n_ops, double(g1.weight_bytes) / (1 << 20));
                printf("  launch push -> DONE %llu core cycles = %.1f us; RISC-V busy %.0f RV cycles = %.1f us (%.0f instructions); "
                       "pushes from %.1f to %.1f us after the push\n", (unsigned long long)h.cycles, double(h.cycles) / F_CORE,
                       cyc[0], rv_us, ins[0], double(p.first_push - t0) / F_CORE, double(p.last_push - t0) / F_CORE);
                printf("  units: MVM busy %llu (%.1f us; %zu MVM ops, first start -> last done %.1f us), DMA busy %llu (%.1f us), "
                       "QACT busy %llu; MVM idle with an empty queue (PERF MVM_EMPTY) %llu, waiting on deps (MVM_WAIT) %llu, "
                       "push stall %llu RV cycles\n", (unsigned long long)pf[0], double(pf[0]) / F_CORE, p.mvm.size(),
                       double(span) / F_CORE, (unsigned long long)pf[1], double(pf[1]) / F_CORE, (unsigned long long)pf[2],
                       (unsigned long long)pf[4], (unsigned long long)pf[3], (unsigned long long)pf[5]);
                printf("  executor ahead of the MVM unit: issue-bound MVM gaps %llu core cycles (%.2f %% of the MVM span); "
                       "RISC-V work %.1f us vs MVM work %.1f us per layer\n", (unsigned long long)issue_gap,
                       span ? 100.0 * double(issue_gap) / double(span) : 0.0, rv_us, double(mvm_busy) / F_CORE);
            }
            tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), nullptr, nullptr);
            fwh_close(&h);
            tsr_sim_destroy(sim);
        }
        const double dr = double(nops[1] - nops[0]), dm = double(nmm[1] - nmm[0]);
        const double per_rec = (cyc[1] - cyc[0]) / dr, per_mm = (cyc[1] - cyc[0]) / dm;
        printf("fw_m6_perf: RISC-V cycles per graph op (0.5B decode lists, 3 vs 1 layers, difference / ops): %.0f per record "
               "(%.0f instructions), %.0f per MUL_MAT (JOINs folded in); SW §7.8 target <= 400: %s\n", per_rec,
               (ins[1] - ins[0]) / dr, per_mm, per_rec <= 400.0 ? "met" : "NOT met");
        printf("  at %.1f MHz: %.2f us per record; a 0.5B decode layer (%u records) costs the RISC-V %.1f us\n", F_RV,
               per_rec / F_RV, g1.n_ops, per_rec * g1.n_ops / F_RV);
        CHECK(per_rec <= 4000.0, "perf: regression ceiling 4000 RV cycles per graph record (%.0f)", per_rec);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s IMAGE SCRATCH [TRACES] | IMAGE SCRATCH --perf\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    const bool perf = argc > 3 && !strcmp(argv[3], "--perf");
    if (argc > 3 && !perf) g_traces = argv[3];
    struct stat st;
    if (stat(g_image, &st) != 0) {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    unsetenv("TSR_SIM_SCHED");
    unsetenv("TSR_SIM_LOG");
    unsetenv("TSR_SIM_RV_RATIO");
    unsetenv("TSR_TRACE");
    setenv("TSR_SIM_RACE", "fatal", 1);
    {   // tsr-elf's view of the image: id (computed and stamped), kernels, .note.tsr
        char err[256];
        tsr_elf * e = nullptr;
        if (tsr_elf_read(g_image, &e, err, sizeof(err)) != 0) {
            CHECK(0, "tsr-elf: %s", err);
            return test_finish("fw_m6_e2e");
        }
        uint64_t stored = 0;
        g_elf_id = tsr_elf_id(e);
        CHECK(tsr_elf_stored_id(e, &stored) == 0 && stored == g_elf_id, "tsr-elf: stamped id %016llx, computed %016llx",
              (unsigned long long)stored, (unsigned long long)g_elf_id);
        tsr_elf_note note;
        CHECK(tsr_elf_get_note(e, &note) == 0, "tsr-elf: .note.tsr");
        g_sdk_abi = note.sdk_abi;
        g_nkern = tsr_elf_n_kernels(e);
        CHECK(g_nkern == note.n_kernels, "tsr-elf: %u kernels, .note.tsr says %u", g_nkern, note.n_kernels);
        tsr_elf_free(e);
    }
    if (perf) {
        perf_main();
        return test_finish("fw_m6_perf");
    }
    char * base = make_scratch(argv[2], "fw_m6_e2e");
    e2e_main(base);
    if (!g_fail) rm_rf(base);
    else fprintf(stderr, "fw_m6_e2e: scratch kept in %s\n", base);
    free(base);
    return test_finish("fw_m6_e2e");
}
