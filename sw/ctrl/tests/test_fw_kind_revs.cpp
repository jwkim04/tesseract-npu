// test_fw_kind_revs.cpp — ctest fw_kind_revs (docs/design/06-software-verification.md "SW" §6.9, SW-54):
// the guard that makes (kind, numerics revision) a reproducibility key. The other bit-exact tests
// compare each kind with a reference that changes together with it, so they cannot see a numerics
// change; this one compares each kind's outputs on a fixed corpus with a recorded digest.
//
//   test_fw_kind_revs IMAGE TABLE [--update] [--digests-only] [--paths]
//       Boots IMAGE (the production tsr-fw.elf) on tsr_sim (U200: C = 4, T_MAX 8, CAPS of the
//       simulator, fixed schedule) and runs a fixed corpus, one-list launches of `graph` per kind:
//         NOP, JOIN     a list of NOPs whose last one has three predecessors (the writer's JOIN)
//         MUL_MAT       Q8_0 and Q4_K, T 1 and 8, per-token addend rows, a bias row, ACC_DST (T 3 and 1)
//         ADD, MUL      b broadcast, a row period, one scalar, per-row scalars; dense rows with every
//                       operand-b form of the decode fast entry
//         SCALE         without and with a bias; dense lists of records that repeat or change s / b
//         RMS_NORM      with a weight (period 1, period 2) and without; lists of norms that repeat or
//                       change eps / inv_n (the hot entry), several rows
//         GLU           SWIGLU, GEGLU16, GEGLU, strided and dense up rows; a tile that needs a ring block
//         ROPE          NORM mode, NEOX, NEOX with the A table (ATAB); lists of one-token records with
//                       and without a table or freq_scale
//         GET_ROWS      repeated and reversed ids, rows of one and of three lines; one id
//         SET_ROWS_KV   K and V rows into a plane pair (AT §4); one row
//         FLASH_ATTN    MSUM (preload) with GQA: decode, a prefill tile with FA_NEW cells, a GENERAL
//                       (finite) mask with its f32 copy
//         CPY           dense and strided rows; one row
//         UNARY         TANH, GELU16, SILU, EXP, GELU
//         MOE_ROUTE     128 and 64 experts: one token, three, tiles of 16 and 4, argsort-shaped and dense ids rows,
//                       ties and a NaN row
//         MUL_MAT_ID    the gate-up form (x shared by the slots, Q4_K) and the down form (x per slot, Q8_0, scale table)
//         SOFTPLUS, SSM_CONV, GATED_DELTA_NET   small fixed records (SW §7.10)
//       GATED_DELTA_NET has two paths (SW-78), chosen from CAPS, so its table has two rows. The default row is the run
//       on the simulator's CAPS (bit 8, the fused GDN op: S_v 64 and above take the fused path); the row marked
//       "composed" is a second run on a simulator whose CAPS lacks bit 8, which takes the composition for every S_v.
//       The composed row's corpus is the first two records, the corpus of revision 1, so its digest is revision 1's:
//       the proof that revision 2 leaves the composed path's bits alone. The default row adds records at S_v 128
//       and 256 (the production shape and the largest m of the op).
//       Inputs come from an integer PRNG (splitmix64); every f32 input is an integer times a power of
//       two and every f32 parameter a fixed bit pattern, so no libm call enters a digest. A kind's
//       digest is FNV-1a 64 over, per launch in corpus order, the DONE status, ERR_CODE and RET, then
//       every output region (the result rows, both KV planes) as the device left it (each region is
//       filled with 0xA5 before its launch).
//       Paths: the decode fast entries are the code production decode runs, so the corpus must take
//       them. A retire hook counts the entries into the image's general-path and fast-path functions
//       (by name), and each launch states which it must (not) enter: a launch of fast records enters
//       no general function and the kind's fast entry (RMS_NORM / ROPE: the hot entry, which only a
//       list's later records take, enters rms_fast_any / rope_fast_any once, for the first); a launch
//       of general records enters the general function. A record that stops taking its fast entry
//       fails ("path"), so a numerics change confined to a fast entry cannot hide behind a general-path
//       digest. --paths prints the entries per launch.
//       Checks against TABLE (rows "0x10 MUL_MAT 1 <16 hex digits>", '#' comments; a kind with a second path
//       has a second row with a fifth word "composed"): every kind with a
//       revision (tsr/opl.hpp kind_rev) has a row; a digest that moved at the same revision fails
//       ("bump the revision"), a revision that moved with none of the kind's digests changed fails
//       ("revert the bump");
//       every corpus launch answers DONE OK / RET 0 (a kind whose handler is missing answers
//       BAD_RECORD); the image's .tsr_kinds declares exactly the kinds with a revision, each at
//       kind_rev (a kind it runs but does not declare, or declares without a handler, fails).
//       --update rewrites TABLE from this run after a deliberate revision bump (never from ctest). It
//       applies the same rules first: a digest that moved at an unchanged revision, a bump without a
//       change and a lowered revision are refused, so the update records only the kinds whose revision
//       went up (and new kinds), never a change nobody bumped. A new corpus (this file's launches
//       changed) has new digests for the kinds it touches: write the table to a new path, after
//       checking with --digests-only against the previous image's table (written the same way from the
//       image built before) that no output bit moved.
//       --digests-only prints the digests and compares them with TABLE without the .tsr_kinds and
//       revision checks (an image built before the kind table: the invariance proof of a change); with
//       --update it writes TABLE from this run without the table rules (a base for that comparison).
// Exit 0 when every check passed, 1 on a failure, 77 (skipped) when the image is missing.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ref/fw_sim_host.hpp"
#include "fw_prof.h"
#include "tsr/opl.hpp"
#include "tsr/plan_attn.hpp"
#include "tsr/plan_gdn.hpp"
#include "tsr_elf.h"
#include "tsr_ref_at.h"
#include "tsr_rv.h"
#include "tsr_sim_int.h"

namespace {

namespace opl = tsr::opl;
constexpr uint64_t LINE = 64u;
constexpr uint32_t NONE = opl::HANDLE_NONE;
constexpr uint32_t OPL_BEAT = 0x40000u;          // ch0: the list of each launch
constexpr uint64_t W_BEAT = 0x100000u;            // weights (every channel, symmetric)
constexpr uint64_t KV_BEAT = 0x200000u;           // KV planes (every channel)

uint32_t H(uint32_t ch, uint64_t beat) { return (ch << 28) | uint32_t(beat & 0x0FFFFFFFu); }
uint32_t L16(uint32_t n) { return (n + 15u) / 16u; }

// ---------------------------------------------------------------------------
// libm-free data
// ---------------------------------------------------------------------------
// An f32 in [-2, 2) with 2^-10 steps, times mul (a power of two in the corpus).
float val(rng_t * r, float mul) { return float(int32_t(rng_u32(r) >> 20) - 2048) * (1.0f / 1024.0f) * mul; }
void put_f32_rows(tsr_sim * sim, rng_t * r, uint32_t h, uint32_t rows, uint32_t stride, uint32_t n, float mul) {
    std::vector<uint32_t> w(size_t(stride) * 16u);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < w.size(); ++j) w[j] = j < n ? f2u(val(r, mul)) : rng_u32(r);   // garbage after n
        tsr_sim_dram_write(sim, h >> 28, (uint64_t(h & 0x0FFFFFFFu) + uint64_t(i) * stride) * LINE, w.data(), w.size() * 4u);
    }
}

// ---------------------------------------------------------------------------
// Paths: which functions of the image a launch entered
// ---------------------------------------------------------------------------
// The functions of the handlers' general paths and of their fast entries, by (a part of) their
// demangled name. A retire hook counts the instructions fetched at the first address of each: the
// entries, as calls and tail calls make them (a function entered past its first instruction would be
// a different function).
const char * const WATCH[] = {
    // general paths
    "addmul_general(", "scale_general(", "unary_general(", "glu_general(", "rms_general(", "rope_general(",
    "get_rows_general(", "cpy_general(", "set_rows_any(", "mulmat_any(", "route_general(", "mmid_general(",
    // the Gated DeltaNet kinds (one general function each: they have no fast entry)
    "softplus_run(", "sconv_run(", "gdn_run(",
    // GATED_DELTA_NET's token block pipeline (the fused path, SW-70) and the two token steps of the head by head
    // pipeline (SW-78): the composition and the fused GDN op
    "gdn_run_tok(", "gdn_step_composed(", "gdn_step_fused(",
    // fast entries
    "kind_add_fast(", "kind_mul_fast(", "kind_scale_fast(", "kind_unary_fast(", "kind_glu_fast(", "kind_rms_norm_fast(",
    "kind_rope_fast(", "kind_get_rows_fast(", "kind_cpy_fast(", "kind_set_rows_kv_fast(", "mulmat_t1_fast(",
    // the second tier of the two with a hot entry
    "rms_fast_any(", "rope_fast_any(",
};
constexpr size_t N_WATCH = sizeof(WATCH) / sizeof(WATCH[0]);

struct watcher {
    fwprof::prof prof;
    std::unordered_map<uint32_t, uint32_t> entry;   // first address of a watched function -> its WATCH index
    uint64_t hits[N_WATCH] = {};
    bool ready = false;
    bool load(const char * image) {
        if (!prof.load_symbols(image)) return false;
        for (size_t i = 0; i < N_WATCH; ++i) {
            bool found = false;
            for (const fwprof::sym & s : prof.symbols())
                if (s.name.find(WATCH[i]) != std::string::npos) {
                    entry[s.lo] = uint32_t(i);
                    found = true;
                }
            CHECK(found, "%s: no function named *%s* (renamed or inlined: update WATCH)", image, WATCH[i]);
        }
        ready = true;
        return true;
    }
};
watcher g_watch;
bool g_print_paths = false;

int path_retire(void * ctx, const tsr_rv_retire * r) {
    watcher * w = static_cast<watcher *>(ctx);
    auto it = w->entry.find(r->pc);
    if (it != w->entry.end()) ++w->hits[it->second];
    return 0;
}

// What a launch must (not) enter: names from WATCH.
struct path_exp {
    std::vector<const char *> some;   // entered at least once
    std::vector<const char *> zero;   // never entered
    std::vector<std::pair<const char *, uint64_t>> exact;   // entered exactly that many times
};

size_t watch_index(const char * name) {
    for (size_t i = 0; i < N_WATCH; ++i)
        if (!strcmp(WATCH[i], name)) return i;
    return N_WATCH;
}

// The paths of a launch: fast (the kind's fast entry takes every record: the general function is not
// entered) or general (it is).
path_exp fast_path(const char * fast, const char * general) { return path_exp{ { fast }, { general }, {} }; }
path_exp general_path(const char * general) { return path_exp{ { general }, {}, {} }; }

// ---------------------------------------------------------------------------
// One launch: a list, its preload sources, the regions it writes
// ---------------------------------------------------------------------------
struct region {
    uint32_t ch;
    uint64_t beat, beats;
};

struct launch_ctx {
    tsr_sim * sim;
    fwh * h;
    int graph;
    rng_t r;
    uint64_t next[4] = { 0x80000u, 0x10000u, 0x10000u, 0x10000u };   // next free beat per channel
    uint32_t rr = 0;                                                    // channel rotation
    uint32_t take(uint64_t beats, uint32_t ch) {
        const uint64_t b = next[ch];
        next[ch] += (beats + 63u) & ~63ull;
        return H(ch, b);
    }
    uint32_t take(uint64_t beats) { return take(beats, rr++ & 3u); }
};

struct kind_digest {
    uint64_t h = 1469598103934665603ull;
    uint32_t launches = 0, ok = 0;
    void add(const void * p, size_t n) {
        const uint8_t * b = static_cast<const uint8_t *>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    }
    void add32(uint32_t v) { add(&v, 4); }
};

std::map<uint32_t, kind_digest> g_dig;
std::map<uint32_t, kind_digest> g_dig_composed;      // GATED_DELTA_NET on a device without CAPS bit 8 (SW-78)
std::map<uint32_t, kind_digest> * g_dig_cur = &g_dig;   // where run() adds

// Rows written by a record: rows x lines, `stride` lines apart, from handle h.
void rows_region(std::vector<region> & out, uint32_t h, uint32_t rows, uint32_t stride, uint32_t lines) {
    for (uint32_t i = 0; i < rows; ++i) out.push_back(region{ h >> 28, (h & 0x0FFFFFFFu) + uint64_t(i) * stride, lines });
}

void check_paths(const char * what, const path_exp & px) {
    if (g_print_paths) {
        printf("fw_kind_revs: paths %-58s", what);
        for (size_t i = 0; i < N_WATCH; ++i)
            if (g_watch.hits[i] != 0) printf(" %.*s:%llu", int(strlen(WATCH[i]) - 1u), WATCH[i], (unsigned long long)g_watch.hits[i]);
        printf("\n");
    }
    for (const char * n : px.some) {
        const size_t i = watch_index(n);
        CHECK(i < N_WATCH && g_watch.hits[i] != 0, "%s: path: %s was not entered (the record no longer takes that path)", what, n);
    }
    for (const auto & e : px.exact) {
        const size_t i = watch_index(e.first);
        CHECK(i < N_WATCH && g_watch.hits[i] == e.second, "%s: path: %s was entered %llu times, expected %llu (the hot entry takes the "
              "records after a list's first)", what, e.first, (unsigned long long)(i < N_WATCH ? g_watch.hits[i] : 0u),
              (unsigned long long)e.second);
    }
    for (const char * n : px.zero) {
        const size_t i = watch_index(n);
        CHECK(i < N_WATCH && g_watch.hits[i] == 0, "%s: path: %s was entered %llu times (a record of the fast entry went elsewhere)",
              what, n, (unsigned long long)(i < N_WATCH ? g_watch.hits[i] : 0u));
    }
}

// Writes the list, fills every output region with 0xA5 (unless it is an input too: ACC_DST rows, KV
// planes), launches `graph` (fixed schedule) and adds DONE and the regions to the digest of `kind`;
// px: the paths the launch must (not) take.
void run(launch_ctx & c, uint32_t kind, const char * what, const opl::writer & w, const std::vector<region> & out,
         bool prefill = true, const path_exp & px = path_exp{}) {
    const std::vector<uint8_t> blob = w.finish();
    CHECK(w.ok() && !blob.empty(), "%s: OPL writer: %s", what, w.error());
    if (blob.empty()) return;
    for (const region & g : out) {
        if (!prefill) break;
        const std::vector<uint8_t> fill(size_t(g.beats) * LINE, 0xA5u);
        tsr_sim_dram_write(c.sim, g.ch, g.beat * LINE, fill.data(), fill.size());
    }
    tsr_sim_dram_write(c.sim, 0, uint64_t(OPL_BEAT) * LINE, blob.data(), blob.size());
    const uint32_t a[7] = { H(0, OPL_BEAT), uint32_t(blob.size()), opl::LF_VERIFY_CRC, 0, 0, 0, 0 };
    kind_digest & d = (*g_dig_cur)[kind];
    d.launches++;
    memset(g_watch.hits, 0, sizeof(g_watch.hits));
    if (fwh_launch(c.h, uint32_t(c.graph), a, 0) != 0) {
        CHECK(0, "%s: launch failed (trap or timeout)", what);
        return;
    }
    check_paths(what, px);
    const uint32_t r0 = c.h->done[0];
    const bool ok = TSR_GET(r0, TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_DONE && fwh_status(c.h) == TSR_LCQ_R0_STATUS_OK &&
                    fwh_ret(c.h) == 0;
    CHECK(ok, "%s (%s): DONE 0x%08x RET 0x%08x: a kind whose handler is not in the image answers BAD_RECORD", what,
          opl::kind_name(kind), r0, c.h->done[2]);
    d.ok += ok;
    d.add32(TSR_GET(r0, TSR_LCQ_R0_TYPE) | TSR_GET(r0, TSR_LCQ_R0_STATUS) << 8 | TSR_GET(r0, TSR_LCQ_R0_ERR_CODE) << 16);
    d.add32(c.h->done[2]);
    for (const region & g : out) {
        const uint8_t * p = tsr_sim_dram(c.sim, g.ch) + g.beat * LINE;
        if (prefill) {                                        // a result row the record wrote
            size_t fill = 0;
            while (fill < LINE && p[fill] == 0xA5u) ++fill;
            CHECK(fill < LINE, "%s: ch%u beat 0x%llx still holds the fill pattern", what, g.ch, (unsigned long long)g.beat);
        }
        d.add(p, size_t(g.beats) * LINE);
    }
}

// ---------------------------------------------------------------------------
// The corpus, kind by kind
// ---------------------------------------------------------------------------
void corpus_nop_join(launch_ctx & c) {
    opl::writer w(opl::F_STOP_ON_ERROR, 0x4B52u);
    const uint32_t a = w.add(opl::K_NOP, nullptr, 0), b = w.add(opl::K_NOP, nullptr, 0), d = w.add(opl::K_NOP, nullptr, 0);
    const uint32_t deps[3] = { a, b, d };
    w.add(opl::K_NOP, nullptr, 0, deps, 3);               // a JOIN of a and b first
    run(c, opl::K_NOP, "NOP x 4", w, {});
    run(c, opl::K_JOIN, "JOIN (three predecessors)", w, {});
}

void corpus_mulmat(launch_ctx & c) {
    struct mm_case { const char * what; uint32_t fmt, N, K, T; int add; bool acc; };
    static const mm_case cases[] = {
        { "MUL_MAT Q8_0 96 x 256, T 1", TSR_FMT_Q8_0, 96, 256, 1, 0, false },
        { "MUL_MAT Q4_K 64 x 512, T 8, per-token addend", TSR_FMT_Q4_K, 64, 512, 8, 2, false },
        { "MUL_MAT Q8_0 48 x 128, T 3, ACC_DST", TSR_FMT_Q8_0, 48, 128, 3, 0, true },
        { "MUL_MAT Q4_K 40 x 256, T 1, bias row", TSR_FMT_Q4_K, 40, 256, 1, 1, false },
        { "MUL_MAT Q8_0 64 x 128, T 1, ACC_DST", TSR_FMT_Q8_0, 64, 128, 1, 0, true },
    };
    uint64_t wbeat = W_BEAT;
    for (const mm_case & m : cases) {
        const uint32_t C = tsr_sim_config(c.sim)->num_ch;
        const size_t row_bytes = size_t(m.K / tsr_fmt_block_size(m.fmt)) * tsr_fmt_ggml_block_bytes(m.fmt);
        std::vector<uint8_t> blk(row_bytes);
        for (uint32_t n = 0; n < m.N; ++n) {                     // gen_row_blocks: integer bytes and fp16 scales
            gen_row_blocks(&c.r, m.fmt, m.K, blk.data());
            uint32_t ch;
            uint64_t off;
            tsr_row_location(m.fmt, m.N, m.K, C, n, &ch, &off);
            tsr_repack_row(m.fmt, blk.data(), m.K, tsr_sim_dram(c.sim, ch) + (wbeat + off) * LINE);
        }
        const uint32_t xl = m.K / 16u, yl = L16(m.N);
        const uint32_t x = c.take(uint64_t(m.T) * (xl + 1u)), y = c.take(uint64_t(m.T) * yl);
        put_f32_rows(c.sim, &c.r, x, m.T, xl + 1u, m.K, 1.0f);
        uint32_t add = NONE, add_stride = 0;
        if (m.add == 1) {
            add = c.take(yl);
            put_f32_rows(c.sim, &c.r, add, 1, yl, m.N, 4.0f);
        } else if (m.add == 2) {
            add_stride = yl + 2u;
            add = c.take(uint64_t(m.T) * add_stride);
            put_f32_rows(c.sim, &c.r, add, m.T, add_stride, m.N, 4.0f);
        }
        std::vector<region> out;
        rows_region(out, y, m.T, yl, yl);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x4D4Du);
        w.mul_mat(x, xl + 1u, y, yl, uint32_t(wbeat), m.fmt, m.N, m.K, m.T, add, add_stride, m.acc ? opl::mm::FLAG_ACC_DST : 0u);
        if (m.acc) put_f32_rows(c.sim, &c.r, y, m.T, yl, m.N, 8.0f);   // y += W x: the old y rows are an input
        run(c, opl::K_MUL_MAT, m.what, w, out, !m.acc,
            m.T == 1u ? fast_path("mulmat_t1_fast(", "mulmat_any(") : general_path("mulmat_any("));
        wbeat += (tsr_tensor_ch_beats(m.fmt, m.N, m.K, C) + 63u) & ~63ull;
    }
}

void corpus_mulmat_epilogue(launch_ctx & c, uint32_t kind) {
    constexpr uint32_t N = 69u, K = 256u, T = 3u, yl = (N + 15u) / 16u;
    for (const uint32_t fmt : {TSR_FMT_Q8_0, TSR_FMT_Q4_K}) {
        const size_t row_bytes = size_t(K / tsr_fmt_block_size(fmt)) * tsr_fmt_ggml_block_bytes(fmt);
        std::vector<uint8_t> blk(row_bytes);
        for (uint32_t n = 0; n < N; ++n) {
            gen_row_blocks(&c.r, fmt, K, blk.data());
            uint32_t ch;
            uint64_t off;
            tsr_row_location(fmt, N, K, tsr_sim_num_ch(c.sim), n, &ch, &off);
            tsr_repack_row(fmt, blk.data(), K, tsr_sim_dram(c.sim, ch) + (W_BEAT + off) * LINE);
        }
        const uint32_t x = c.take(T * (K / 16u)), y = c.take(T * yl);
        put_f32_rows(c.sim, &c.r, x, T, K / 16u, K, 1.0f);
        std::vector<region> out;
        rows_region(out, y, T, yl, yl);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x5343u);
        const uint32_t a[] = {x, K / 16u, y, yl, uint32_t(W_BEAT), fmt, N, K, T,
                              opl::HANDLE_NONE, 0u, 0u, 0xBE800000u, 0x40400000u};   // -0.25f and +3.0f
        w.add(kind, a, opl::kind_nargs(kind));
        run(c, kind, opl::kind_name(kind), w, out);
    }
}

// ADD / MUL: a rows x n, b by period p (scalar: max(p, 1) f32 words; else min(p, rows) rows, one for p = 1).
// dense: a and dst rows at stride L (a decode record the fast entry takes; else a stride L + 1 for rows > 1: the general path).
void ew_case(launch_ctx & c, uint32_t kind, const char * what, uint32_t n, uint32_t rows, uint32_t p, bool scalar, bool dense) {
    const uint32_t l = L16(n), as = dense ? l : l + 1u;
    const uint32_t a = c.take(uint64_t(rows) * as), d = c.take(uint64_t(rows) * l);
    put_f32_rows(c.sim, &c.r, a, rows, as, n, 2.0f);
    uint32_t b, bs;
    if (scalar) {
        b = c.take(L16(std::max(p, 1u)));
        put_f32_rows(c.sim, &c.r, b, 1, L16(std::max(p, 1u)), std::max(p, 1u), 1.0f);
        bs = 0;
    } else {
        const uint32_t brows = p == 1u ? 1u : std::min(p, rows);
        bs = l;
        b = c.take(uint64_t(brows) * l);
        put_f32_rows(c.sim, &c.r, b, brows, l, n, 1.0f);
    }
    opl::writer w(opl::F_STOP_ON_ERROR, 0x4557u);
    const opl::ew_args args{ d, l, a, as, b, bs, n, rows, p, scalar ? opl::ew::FLAG_B_SCALAR : 0u };
    if (kind == opl::K_ADD) w.add_add(args); else w.add_mul(args);
    std::vector<region> out;
    rows_region(out, d, rows, l, l);
    // The fast entry takes one row, or dense rows with b as one row / per-row rows / scalars (k_elem.hpp ew2_fast).
    const bool fast = rows == 1u || dense;
    run(c, kind, what, w, out, true,
        fast ? fast_path(kind == opl::K_ADD ? "kind_add_fast(" : "kind_mul_fast(", "addmul_general(") : general_path("addmul_general("));
}

void corpus_ew(launch_ctx & c) {
    ew_case(c, opl::K_ADD, "ADD 896, b broadcast", 896, 1, 1, false, false);
    ew_case(c, opl::K_ADD, "ADD 300 x 3, one scalar", 300, 3, 1, true, false);
    ew_case(c, opl::K_ADD, "ADD 64 x 6, row period 3", 64, 6, 3, false, false);
    ew_case(c, opl::K_MUL, "MUL 256 x 4, row period 2", 256, 4, 2, false, false);
    ew_case(c, opl::K_MUL, "MUL 128 x 5, per-row scalars", 128, 5, 5, true, false);
    ew_case(c, opl::K_MUL, "MUL 512 x 2, b broadcast", 512, 2, 1, false, false);
    // The decode fast entry's operand-b forms: broadcast, per-row rows, one scalar, per-row scalars.
    for (uint32_t kind : { uint32_t(opl::K_ADD), uint32_t(opl::K_MUL) }) {
        const char * nm = kind == opl::K_ADD ? "ADD" : "MUL";
        char w[4][64];
        snprintf(w[0], 64, "%s dense 128 x 3, b broadcast", nm);
        snprintf(w[1], 64, "%s dense 64 x 4, b rows", nm);
        snprintf(w[2], 64, "%s 200, one scalar", nm);
        snprintf(w[3], 64, "%s dense 300 x 3, per-row scalars", nm);
        ew_case(c, kind, w[0], 128, 3, 1, false, true);
        ew_case(c, kind, w[1], 64, 4, 4, false, true);
        ew_case(c, kind, w[2], 200, 1, 1, true, true);
        ew_case(c, kind, w[3], 300, 3, 3, true, true);
    }
}

void corpus_scale(launch_ctx & c) {
    struct sc { const char * what; uint32_t n, rows, s, b; };
    static const sc cases[] = {
        { "SCALE 512 x 2 by 0.125", 512, 2, 0x3E000000u, 0u },
        { "SCALE 200 x 3 by 3.5 + 0.25", 200, 3, 0x40600000u, 0x3E800000u },
    };
    for (const sc & s : cases) {
        const uint32_t l = L16(s.n), src = c.take(uint64_t(s.rows) * l), dst = c.take(uint64_t(s.rows) * (l + 1u));
        put_f32_rows(c.sim, &c.r, src, s.rows, l, s.n, 4.0f);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x5343u);
        w.add_scale(opl::scale_args{ dst, l + 1u, src, l, s.n, s.rows, s.s, s.b });
        std::vector<region> out;
        rows_region(out, dst, s.rows, l + 1u, l);
        run(c, opl::K_SCALE, s.what, w, out, true, general_path("scale_general("));
    }
    // The decode fast entry: one row or dense rows, in lists whose records repeat or change s and b (the
    // first record of a list pushes the SOP.MOVIs; one that finds s / b in the s-register shadow pushes none).
    static const sc list[] = {
        { "", 512, 1, 0x3E000000u, 0u },            // MOVI s
        { "", 300, 1, 0x3E000000u, 0u },            // the same s: no MOVI
        { "", 200, 3, 0x40600000u, 0x3E800000u },   // new s and a bias
        { "", 200, 3, 0x40600000u, 0x3E800000u },   // the same: no MOVI, with the bias
        { "", 128, 2, 0x40600000u, 0u },            // the same s, no bias
        { "", 128, 2, 0xBE800000u, 0x3F000000u },   // both new
    };
    opl::writer w(opl::F_STOP_ON_ERROR, 0x5343u);
    std::vector<region> out;
    for (const sc & s : list) {
        const uint32_t l = L16(s.n), src = c.take(uint64_t(s.rows) * l), dst = c.take(uint64_t(s.rows) * l);
        put_f32_rows(c.sim, &c.r, src, s.rows, l, s.n, 4.0f);
        w.add_scale(opl::scale_args{ dst, l, src, l, s.n, s.rows, s.s, s.b });
        rows_region(out, dst, s.rows, l, l);
    }
    run(c, opl::K_SCALE, "SCALE list of six dense records (s and b repeated and changed)", w, out, true,
        fast_path("kind_scale_fast(", "scale_general("));
}

struct rc { const char * what; uint32_t n, rows, p, eps, inv_n; float mul = 2.0f; };   // mul: the inputs' magnitude
constexpr uint32_t EPS6 = 0x358637BDu /* 1e-6 */, EPS5 = 0x3727C5ACu /* 1e-5 */;

// Adds one RMS_NORM of q to w with its operands in fresh buffers; weights are shared by the records of
// a list that use the same weight rows (wh, NONE: allocated here).
void rms_record(launch_ctx & c, opl::writer & w, std::vector<region> & out, const rc & q, uint32_t wh_shared = NONE) {
    CHECK(q.inv_n == f2u(tsr_fdiv(1.0f, float(q.n))), "%s: INV_N is tsr_fdiv(1, n) (SW §7.2)", q.what);
    const uint32_t l = L16(q.n), src = c.take(uint64_t(q.rows) * l), dst = c.take(uint64_t(q.rows) * l);
    put_f32_rows(c.sim, &c.r, src, q.rows, l, q.n, q.mul);
    uint32_t wh = NONE;
    if (q.p != 0) {
        wh = wh_shared;
        if (wh == NONE) {
            const uint32_t wrows = q.p == 1u ? 1u : std::min(q.p, q.rows);
            wh = c.take(uint64_t(wrows) * l);
            put_f32_rows(c.sim, &c.r, wh, wrows, l, q.n, 1.0f);
        }
    }
    w.add_rms_norm(opl::rms_norm_args{ dst, l, src, l, q.n, q.rows, q.eps, q.inv_n, wh, q.p });
    rows_region(out, dst, q.rows, l, l);
}

void corpus_rms(launch_ctx & c) {
    static const rc cases[] = {
        { "RMS_NORM 896 with a weight", 896, 1, 1, EPS6, 0x3A924925u /* fdiv(1, 896) */ },
        { "RMS_NORM 128 x 6, weight period 2", 128, 6, 2, EPS5, 0x3C000000u /* 1/128 */ },
        { "RMS_NORM 256 x 3 without a weight", 256, 3, 0, EPS6, 0x3B800000u /* 1/256 */ },
    };
    for (const rc & q : cases) {
        opl::writer w(opl::F_STOP_ON_ERROR, 0x524Eu);
        std::vector<region> out;
        rms_record(c, w, out, q);
        run(c, opl::K_RMS_NORM, q.what, w, out, true, q.p == 2u ? general_path("rms_general(") : fast_path("rms_fast_any(", "rms_general("));
    }
    // The hot entry: one row with a weight of period 1 while the launch's s-register shadow holds 1/n and
    // eps, i.e. every norm of a list after the first (a decode layer: the attention and ffn norms); the
    // first, and a norm that changes eps or 1/n, take rms_fast_any (the SOP.MOVIs).
    {
        opl::writer w(opl::F_STOP_ON_ERROR, 0x524Eu);
        std::vector<region> out;
        const rc q{ "", 896, 1, 1, EPS6, 0x3A924925u };
        const uint32_t wh = c.take(L16(q.n));
        put_f32_rows(c.sim, &c.r, wh, 1, L16(q.n), q.n, 1.0f);
        rc small = q;
        small.mul = 1.0f / 1024.0f;   // x^2 about eps: the sum is sensitive to eps
        for (int i = 0; i < 4; ++i) rms_record(c, w, out, i & 1 ? small : q, wh);
        path_exp px = fast_path("kind_rms_norm_fast(", "rms_general(");
        px.exact.push_back({ "rms_fast_any(", 1u });
        run(c, opl::K_RMS_NORM, "RMS_NORM list of four 896 norms with a weight (hot entry after the first)", w, out, true, px);
    }
    {
        opl::writer w(opl::F_STOP_ON_ERROR, 0x524Eu);
        std::vector<region> out;
        const float sm = 1.0f / 1024.0f;
        const rc a{ "", 128, 1, 1, EPS6, 0x3C000000u, sm }, b{ "", 128, 1, 1, EPS5, 0x3C000000u, sm },    // eps changes
            d{ "", 256, 1, 1, EPS5, 0x3B800000u, sm }, e{ "", 512, 1, 1, EPS5, 0x3B000000u };             // 1/n changes
        const uint32_t w128 = c.take(L16(128)), w256 = c.take(L16(256)), w512 = c.take(L16(512));
        put_f32_rows(c.sim, &c.r, w128, 1, L16(128), 128, 1.0f);
        put_f32_rows(c.sim, &c.r, w256, 1, L16(256), 256, 1.0f);
        put_f32_rows(c.sim, &c.r, w512, 1, L16(512), 512, 1.0f);
        rms_record(c, w, out, a, w128);   // MOVI 1/n, eps
        rms_record(c, w, out, a, w128);   // hot
        rms_record(c, w, out, b, w128);   // MOVI eps
        rms_record(c, w, out, b, w128);   // hot
        rms_record(c, w, out, d, w256);   // MOVI 1/n
        rms_record(c, w, out, d, w256);   // hot
        rms_record(c, w, out, e, w512);   // MOVI 1/n
        path_exp px = fast_path("kind_rms_norm_fast(", "rms_general(");
        px.exact.push_back({ "rms_fast_any(", 4u });
        run(c, opl::K_RMS_NORM, "RMS_NORM list changing eps and 1/n", w, out, true, px);
    }
    {
        // eps and 1/n that are each other's bits (eps 2^-8 and 2^-7, n 128 and 256): a record that finds the
        // s-register shadow holding the other register's value must still push both SOP.MOVIs.
        opl::writer w(opl::F_STOP_ON_ERROR, 0x524Eu);
        std::vector<region> out;
        const rc a{ "", 128, 1, 1, 0x3B800000u, 0x3C000000u, 0.25f }, b{ "", 256, 1, 1, 0x3C000000u, 0x3B800000u, 0.25f };
        const uint32_t w128 = c.take(L16(128)), w256 = c.take(L16(256));
        put_f32_rows(c.sim, &c.r, w128, 1, L16(128), 128, 1.0f);
        put_f32_rows(c.sim, &c.r, w256, 1, L16(256), 256, 1.0f);
        rms_record(c, w, out, a, w128);
        rms_record(c, w, out, b, w256);
        rms_record(c, w, out, b, w256);   // hot
        path_exp px = fast_path("kind_rms_norm_fast(", "rms_general(");
        px.exact.push_back({ "rms_fast_any(", 2u });
        run(c, opl::K_RMS_NORM, "RMS_NORM list swapping eps and 1/n", w, out, true, px);
    }
    {
        // The second tier for several rows and a weight, one row without one, and a hot-entry norm after them.
        opl::writer w(opl::F_STOP_ON_ERROR, 0x524Eu);
        std::vector<region> out;
        rms_record(c, w, out, rc{ "", 64, 8, 1, EPS5, 0x3C800000u });
        rms_record(c, w, out, rc{ "", 48, 1, 0, EPS5, 0x3CAAAAABu });
        rms_record(c, w, out, rc{ "", 300, 5, 0, EPS6, 0x3B5A740Eu /* fdiv(1, 300) */ });
        run(c, opl::K_RMS_NORM, "RMS_NORM 64 x 8 with a weight, 48 and 300 x 5 without", w, out, true,
            fast_path("rms_fast_any(", "rms_general("));
    }
}

void corpus_glu(launch_ctx & c) {
    // up_pad: lines the up rows are apart beyond n / 16. The fast entry takes one row, or dense rows (up
    // included); a tile larger than the one-tile block (1024 lines) takes a ring block.
    struct gc { const char * what; uint32_t variant, n, rows, up_pad; bool fast; };
    static const gc cases[] = {
        { "GLU SWIGLU 512 x 2", opl::glu::SWIGLU, 512, 2, 2, false },
        { "GLU GEGLU16 320", opl::glu::GEGLU16, 320, 1, 2, true },
        { "GLU GEGLU 128 x 3", opl::glu::GEGLU, 128, 3, 2, false },
        { "GLU GEGLU16 200 x 2", opl::glu::GEGLU16, 200, 2, 2, false },
        { "GLU SWIGLU 512 x 2, dense", opl::glu::SWIGLU, 512, 2, 0, true },
        { "GLU GEGLU16 200 x 4, dense", opl::glu::GEGLU16, 200, 4, 0, true },
        { "GLU GEGLU 128 x 3, dense", opl::glu::GEGLU, 128, 3, 0, true },
        { "GLU SWIGLU 8960 (a ring block)", opl::glu::SWIGLU, 8960, 1, 0, true },
        { "GLU GEGLU 8960 x 2 (a ring block)", opl::glu::GEGLU, 8960, 2, 0, true },
    };
    for (const gc & g : cases) {
        const uint32_t l = L16(g.n), us = l + g.up_pad;
        const uint32_t gate = c.take(uint64_t(g.rows) * l), up = c.take(uint64_t(g.rows) * us), dst = c.take(uint64_t(g.rows) * l);
        put_f32_rows(c.sim, &c.r, gate, g.rows, l, g.n, 4.0f);
        put_f32_rows(c.sim, &c.r, up, g.rows, us, g.n, 1.0f);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x474Cu);
        w.add_glu(opl::glu_args{ dst, l, gate, l, up, us, g.n, g.rows, g.variant });
        std::vector<region> out;
        rows_region(out, dst, g.rows, l, l);
        run(c, opl::K_GLU, g.what, w, out, true, g.fast ? fast_path("kind_glu_fast(", "glu_general(") : general_path("glu_general("));
    }
}

// One ROPE record of q into w (T tokens at random positions below 32768): the positions as a preload of
// its own, the A / frequency-factor table when q.table.
struct rp { const char * what; uint32_t hd, nd, nh, T, mode, ts, m; bool table; uint32_t fs; };
constexpr uint32_t F32_1 = 0x3F800000u;
void rope_record(launch_ctx & c, opl::writer & w, std::vector<region> & out, const rp & q) {
    const uint32_t hl = q.nh * q.hd / 16u;
    const uint32_t src = c.take(uint64_t(q.T) * hl), dst = c.take(uint64_t(q.T) * (hl + 1u));
    put_f32_rows(c.sim, &c.r, src, q.T, hl, hl * 16u, 2.0f);
    uint32_t table = NONE;
    if (q.table) {                                         // A_i = (65 - i) / 256: exact (ATAB); factors 1 + i / 64 (FF)
        std::vector<uint32_t> t(size_t(L16(q.nd / 2u)) * 16u, 0u);
        for (uint32_t i = 0; i < q.nd / 2u; ++i)
            t[i] = f2u((q.mode & opl::rope::MODE_ATAB) ? float(65u - i) * (1.0f / 256.0f) : 1.0f + float(i) * (1.0f / 64.0f));
        table = c.take(L16(q.nd / 2u));
        tsr_sim_dram_write(c.sim, table >> 28, uint64_t(table & 0x0FFFFFFFu) * LINE, t.data(), t.size() * 4u);
    }
    std::vector<int32_t> pos(size_t(L16(q.T)) * 16u, 0);
    for (uint32_t t = 0; t < q.T; ++t) pos[t] = int32_t(rng_below(&c.r, 32768u));
    const uint32_t pos_h = c.take(L16(q.T), 0);
    tsr_sim_dram_write(c.sim, 0, uint64_t(pos_h & 0x0FFFFFFFu) * LINE, pos.data(), pos.size() * 4u);
    const uint32_t pre = w.add_preload(pos_h, 4u * q.T);
    w.add_rope(opl::rope_args{ dst, hl + 1u, src, hl, pre, table, q.ts, q.m, q.fs, q.hd, q.nd, q.nh, q.T, q.mode });
    rows_region(out, dst, q.T, hl + 1u, hl);
}

void corpus_rope(launch_ctx & c) {
    // ts = powf(base, -2 / n_dims) as ggml-cpu computes it (fixed bits: base 1e6 / 64 dims, 1e4 / 128).
    constexpr uint32_t TS64 = 0x3F263DE0u, TS128 = 0x3F5DAFD7u;
    static const rp cases[] = {
        { "ROPE NORM 4 x 64, T 2", 64, 64, 4, 2, 0u, TS64, F32_1, false, F32_1 },
        { "ROPE NEOX 2 x 128, T 3", 128, 128, 2, 3, opl::rope::MODE_NEOX, TS128, F32_1, false, F32_1 },
        { "ROPE NEOX ATAB 2 x 64, T 1", 64, 64, 2, 1, opl::rope::MODE_NEOX | opl::rope::MODE_ATAB, TS64,
          0x3F8CCCCDu /* 1.1 */, true, F32_1 },
        { "ROPE NEOX ATAB 2 x 64, T 3", 64, 64, 2, 3, opl::rope::MODE_NEOX | opl::rope::MODE_ATAB, TS64,
          0x3F8CCCCDu, true, F32_1 },
    };
    for (const rp & q : cases) {
        opl::writer w(opl::F_STOP_ON_ERROR, 0x524Fu);
        std::vector<region> out;
        rope_record(c, w, out, q);
        run(c, opl::K_ROPE, q.what, w, out, true, q.T == 1u ? fast_path("rope_fast_any(", "rope_general(") : general_path("rope_general("));
    }
    // The decode fast entries: one-token records. The hot entry takes a record without a table and with
    // freq_scale 1 while the launch's s-register shadow holds TS and M (every ROPE of a list after the
    // first with these); rope_fast_any the rest (the first record, new TS / M, a table, freq_scale).
    {
        const rp a = { "", 64, 64, 4, 1, 0u, TS64, F32_1, false, F32_1 };
        rp b = a, d = a, e = { "", 128, 128, 2, 1, opl::rope::MODE_NEOX, TS128, F32_1, false, F32_1 };
        b.mode = opl::rope::MODE_NEOX;
        opl::writer w(opl::F_STOP_ON_ERROR, 0x524Fu);
        std::vector<region> out;
        rope_record(c, w, out, a);   // MOVI TS, M
        rope_record(c, w, out, b);   // hot
        rope_record(c, w, out, b);   // hot
        rope_record(c, w, out, d);   // hot
        rope_record(c, w, out, e);   // MOVI TS, new head size
        rope_record(c, w, out, e);   // hot
        rp g = a, h = a;             // TS and M that are each other's bits: both SOP.MOVIs, whatever the shadow holds
        g.ts = F32_1;
        g.m = TS64;
        h.ts = TS64;
        h.m = F32_1;
        rope_record(c, w, out, h);
        rope_record(c, w, out, g);
        path_exp px = fast_path("kind_rope_fast(", "rope_general(");
        px.exact.push_back({ "rope_fast_any(", 4u });
        run(c, opl::K_ROPE, "ROPE list of eight one-token records (hot entry after the first)", w, out, true, px);
    }
    {
        const uint32_t half = 0x3F000000u /* 0.5 */;
        rp fs = { "", 64, 64, 4, 1, opl::rope::MODE_NEOX, TS64, F32_1, false, half };
        rp fs_norm = fs, ff = { "", 64, 64, 4, 1, opl::rope::MODE_NEOX, TS64, F32_1, true, F32_1 };
        fs_norm.mode = 0u;
        rp ff_fs = ff, ff_norm = ff;
        ff_fs.fs = half;
        ff_norm.mode = 0u;
        rp fs_ts = fs;
        fs_ts.fs = TS64;                   // freq_scale equal to TS's bits (TS comes from the record before)
        opl::writer w(opl::F_STOP_ON_ERROR, 0x524Fu);
        std::vector<region> out;
        rope_record(c, w, out, rp{ "", 64, 64, 4, 1, opl::rope::MODE_NEOX, TS64, F32_1, false, F32_1 });
        rope_record(c, w, out, fs_ts);
        rope_record(c, w, out, fs);        // freq_scale
        rp ts_fs = fs;                     // TS equal to the freq_scale of the record before
        ts_fs.ts = fs.fs;
        ts_fs.fs = F32_1;
        rope_record(c, w, out, ts_fs);
        rope_record(c, w, out, fs_norm);
        rope_record(c, w, out, ff);        // frequency factors
        rope_record(c, w, out, ff_norm);
        rope_record(c, w, out, ff_fs);     // both
        path_exp px = fast_path("rope_fast_any(", "rope_general(");
        px.exact.push_back({ "rope_fast_any(", 8u });
        run(c, opl::K_ROPE, "ROPE list of eight records with freq_scale and frequency factors", w, out, true, px);
    }
}

void corpus_rows(launch_ctx & c) {
    // The fast entry takes one id (a decode graph's inp_out_ids).
    struct gr { const char * what; uint32_t rl, src_rows; std::vector<int32_t> ids; };
    const gr cases[] = {
        { "GET_ROWS 4 of 16 rows of 3 lines (repeated id)", 3, 16, { 5, 0, 15, 5 } },
        { "GET_ROWS 6 of 9 rows of 1 line (reversed)", 1, 9, { 8, 7, 6, 2, 1, 0 } },
        { "GET_ROWS 1 of 16 rows of 3 lines", 3, 16, { 11 } },
        { "GET_ROWS 1 of 9 rows of 1 line", 1, 9, { 8 } },
    };
    for (const gr & g : cases) {
        const uint32_t n = uint32_t(g.ids.size());
        const uint32_t src = c.take(uint64_t(g.src_rows) * (g.rl + 1u)), dst = c.take(uint64_t(n) * g.rl);
        put_f32_rows(c.sim, &c.r, src, g.src_rows, g.rl + 1u, 16u * g.rl, 1.0f);
        std::vector<int32_t> ids(size_t(L16(n)) * 16u, 0);
        std::copy(g.ids.begin(), g.ids.end(), ids.begin());
        const uint32_t ids_h = c.take(L16(n), 0);
        tsr_sim_dram_write(c.sim, 0, uint64_t(ids_h & 0x0FFFFFFFu) * LINE, ids.data(), ids.size() * 4u);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x4752u);
        const uint32_t pre = w.add_preload(ids_h, 4u * n);
        w.add_get_rows(opl::get_rows_args{ dst, g.rl, src, g.rl + 1u, pre, n, g.rl });
        std::vector<region> out;
        rows_region(out, dst, n, g.rl, g.rl);
        run(c, opl::K_GET_ROWS, g.what, w, out, true,
            n == 1u ? fast_path("kind_get_rows_fast(", "get_rows_general(") : general_path("get_rows_general("));
    }
    // CPY: the fast entry takes one row, or rows dense on both sides.
    struct cp { const char * what; uint32_t rl, rows, s_pad, d_pad; };
    static const cp copies[] = {
        { "CPY 4 lines x 3 rows", 4, 3, 0, 0 },
        { "CPY 1 line x 5 rows, strided", 1, 5, 2, 1 },
        { "CPY 3 lines x 1 row, strided", 3, 1, 2, 1 },
    };
    for (const cp & q : copies) {
        const uint32_t src = c.take(uint64_t(q.rows) * (q.rl + q.s_pad)), dst = c.take(uint64_t(q.rows) * (q.rl + q.d_pad));
        put_f32_rows(c.sim, &c.r, src, q.rows, q.rl + q.s_pad, 16u * q.rl, 1.0f);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x4350u);
        w.add_cpy(opl::cpy_args{ dst, q.rl + q.d_pad, src, q.rl + q.s_pad, q.rl, q.rows });
        std::vector<region> out;
        rows_region(out, dst, q.rows, q.rl + q.d_pad, q.rl);
        const bool fast = q.rows == 1u || (q.s_pad == 0u && q.d_pad == 0u);
        run(c, opl::K_CPY, q.what, w, out, true, fast ? fast_path("kind_cpy_fast(", "cpy_general(") : general_path("cpy_general("));
    }
}

void corpus_unary(launch_ctx & c) {
    static const char * const names[] = { "UNARY TANH", "UNARY GELU16", "UNARY SILU", "UNARY EXP", "UNARY GELU" };
    static const char * const names_s[] = { "UNARY TANH, strided", "UNARY GELU16, strided", "UNARY SILU, strided",
                                            "UNARY EXP, strided", "UNARY GELU, strided" };
    for (int strided = 0; strided < 2; ++strided) {      // dense rows: the fast entry; a dst stride L + 1: the general path
        for (uint32_t f = 0; f <= opl::unary::GELU; ++f) {   // SIGMOID (5) was added at revision 1: not part of the digest
            const uint32_t n = 200, rows = 2, l = L16(n), ds = l + uint32_t(strided);
            const uint32_t src = c.take(uint64_t(rows) * l), dst = c.take(uint64_t(rows) * ds);
            put_f32_rows(c.sim, &c.r, src, rows, l, n, 4.0f);
            opl::writer w(opl::F_STOP_ON_ERROR, 0x554Eu);
            w.add_unary(opl::unary_args{ dst, ds, src, l, n, rows, f });
            std::vector<region> out;
            rows_region(out, dst, rows, ds, l);
            run(c, opl::K_UNARY, strided ? names_s[f] : names[f], w, out, true,
                strided ? general_path("unary_general(") : fast_path("kind_unary_fast(", "unary_general("));
        }
    }
}

// MOE_ROUTE and MUL_MAT_ID (SW §7.5): the routing of 128 / 64 experts (tiles of 16 tokens, argsort-shaped and
// dense ids rows, ties, a NaN row) and the two projection forms (the x row shared by the slots, one x row per
// slot with the scale table). Both kinds have one path (their general function).
void corpus_moe(launch_ctx & c) {
    struct rt { const char * what; uint32_t n, k, T, istride; int mode; };
    static const rt routes[] = {
        { "MOE_ROUTE 128 experts, 8 used, T 1", 128, 8, 1, 8, 0 },
        { "MOE_ROUTE 128 experts, 8 used, T 3, argsort-shaped ids rows", 128, 8, 3, 8, 0 },
        { "MOE_ROUTE 64 experts, 4 used, T 20 (tiles of 16 and 4), dense ids rows", 64, 4, 20, 1, 0 },
        { "MOE_ROUTE 128 experts, 8 used, T 4, ties and a NaN row", 128, 8, 4, 8, 1 },
    };
    for (const rt & q : routes) {
        const uint32_t ln = L16(q.n);
        const uint32_t logits = c.take(uint64_t(q.T) * ln), ids = c.take(uint64_t(q.T) * q.istride + 1u),
                       wout = c.take(L16(q.T * q.k) + 1u);
        put_f32_rows(c.sim, &c.r, logits, q.T, ln, q.n, 2.0f);
        if (q.mode == 1) {                                  // equal logits across the boundary; the last row has a NaN
            for (uint32_t t = 0; t < q.T; ++t) {
                std::vector<uint32_t> row(size_t(ln) * 16u, 0u);
                for (uint32_t i = 0; i < q.n; ++i) row[i] = f2u(-2.0f + 0.5f * float(rng_below(&c.r, 5)));
                if (t + 1u == q.T) row[37] = 0x7FC00000u;
                tsr_sim_dram_write(c.sim, logits >> 28, (uint64_t(logits & 0x0FFFFFFFu) + uint64_t(t) * ln) * LINE, row.data(),
                                   row.size() * 4u);
            }
        }
        opl::writer w(opl::F_STOP_ON_ERROR, 0x4D52u);
        w.add_moe_route(opl::moe_route_args{ logits, ln, ids, q.istride, wout, q.n, q.k, q.T, 0u });
        std::vector<region> out;
        for (uint32_t t = 0; t < q.T; ++t) out.push_back(region{ ids >> 28, (ids & 0x0FFFFFFFu) + uint64_t(t) * q.istride, 1u });
        out.push_back(region{ wout >> 28, wout & 0x0FFFFFFFu, L16(q.T * q.k) });
        run(c, opl::K_MOE_ROUTE, q.what, w, out, true, general_path("route_general("));
    }
    struct mm { const char * what; uint32_t fmt, N, K, T, nu, ne; bool shared, scale; };
    static const mm mms[] = {
        { "MUL_MAT_ID Q4_K 64 x 256, 8 experts, 4 used, T 2, x shared by the slots", TSR_FMT_Q4_K, 64, 256, 2, 4, 8, true, false },
        { "MUL_MAT_ID Q8_0 48 x 64, 8 experts, 4 used, T 2, x per slot, scale table", TSR_FMT_Q8_0, 48, 64, 2, 4, 8, false, true },
    };
    uint64_t ebase = W_BEAT + 0x40000u;
    for (const mm & q : mms) {
        const uint32_t C = tsr_sim_config(c.sim)->num_ch;
        const uint64_t stride = tsr_tensor_ch_beats(q.fmt, q.N, q.K, C);
        const size_t row_bytes = size_t(q.K / tsr_fmt_block_size(q.fmt)) * tsr_fmt_ggml_block_bytes(q.fmt);
        std::vector<uint8_t> blk(row_bytes);
        for (uint32_t e = 0; e < q.ne; ++e)
            for (uint32_t n = 0; n < q.N; ++n) {
                gen_row_blocks(&c.r, q.fmt, q.K, blk.data());
                uint32_t ch;
                uint64_t off;
                tsr_row_location(q.fmt, q.N, q.K, C, n, &ch, &off);
                tsr_repack_row(q.fmt, blk.data(), q.K, tsr_sim_dram(c.sim, ch) + (ebase + e * stride + off) * LINE);
            }
        const uint32_t xl = q.K / 16u, yl = L16(q.N);
        const uint32_t xslot = q.shared ? 0u : xl, xtok = q.shared ? xl : q.nu * xl;
        const uint32_t x = c.take(uint64_t(q.T) * xtok), dst = c.take(uint64_t(q.T) * q.nu * yl);
        put_f32_rows(c.sim, &c.r, x, q.T * (q.shared ? 1u : q.nu), xl, q.K, 1.0f);
        const uint32_t idh = c.take(L16(q.T * q.nu) + 1u, 0);
        std::vector<uint32_t> idw(size_t(L16(q.T * q.nu) + 1u) * 16u, 0u);
        for (uint32_t i = 0; i < q.T * q.nu; ++i) idw[i] = rng_below(&c.r, q.ne);
        tsr_sim_dram_write(c.sim, 0, uint64_t(idh & 0x0FFFFFFFu) * LINE, idw.data(), idw.size() * 4u);
        uint32_t sc = NONE;
        if (q.scale) {
            sc = c.take(L16(q.ne));
            put_f32_rows(c.sim, &c.r, sc, 1, L16(q.ne), q.ne, 2.0f);
        }
        opl::writer w(opl::F_STOP_ON_ERROR, 0x4D49u);
        w.add_mul_mat_id(opl::mul_mat_id_args{ dst, yl, x, xslot, xtok, idh, 4u * q.nu, uint32_t(ebase), uint32_t(stride), q.fmt, q.N,
                                               q.K, q.T, q.nu, q.ne, sc, 0u });
        std::vector<region> out;
        rows_region(out, dst, q.T * q.nu, yl, yl);
        run(c, opl::K_MUL_MAT_ID, q.what, w, out, true, general_path("mmid_general("));
        ebase += (stride * q.ne + 63u) & ~63ull;
    }
}

// The Gated DeltaNet kinds (SW §7.10): small fixed records; inputs are integers times powers of two, so
// the digests do not depend on libm either (the kinds use the VU's own functions).
void corpus_gdn(launch_ctx & c, bool fused_dev, bool extras) {
    // SOFTPLUS: one row of 48 (a decode gate vector) and 5 rows of 100 with a strided destination.
    struct spc { const char * what; uint32_t n, rows, pad; };
    static const spc sps[] = { { "SOFTPLUS 1 x 48", 48, 1, 0 }, { "SOFTPLUS 5 x 100, strided dst", 100, 5, 2 } };
    for (const spc & q : sps) {
        const uint32_t l = L16(q.n), ds = l + q.pad;
        const uint32_t src = c.take(uint64_t(q.rows) * l), dst = c.take(uint64_t(q.rows) * ds);
        put_f32_rows(c.sim, &c.r, src, q.rows, l, q.n, 8.0f);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x5350u);
        const uint32_t a[opl::softplus::NARGS] = { dst, ds, src, l, q.n, q.rows };
        w.add(opl::K_SOFTPLUS, a, opl::softplus::NARGS);
        std::vector<region> out;
        rows_region(out, dst, q.rows, ds, l);
        run(c, opl::K_SOFTPLUS, q.what, w, out, true, general_path("softplus_run("));
    }
    // SSM_CONV: C 192, d_conv 4: a decode token and three tokens of two sequences, with SILU and without.
    struct scc { const char * what; uint32_t C, dc, T, ns, flags; };
    static const scc scs[] = { { "SSM_CONV C 192 d_conv 4, T 1, SILU", 192, 4, 1, 1, opl::sconv::FLAG_SILU },
                               { "SSM_CONV C 96 d_conv 3, T 3 x 2 sequences", 96, 3, 3, 2, 0u } };
    for (const scc & q : scs) {
        const uint32_t lc = q.C / 16u;
        const uint32_t wt = c.take(uint64_t(q.dc) * lc), si = c.take(uint64_t(q.ns) * (q.dc - 1u) * lc),
                       so = c.take(uint64_t(q.ns) * (q.dc - 1u) * lc), x = c.take(uint64_t(q.ns) * q.T * lc),
                       dst = c.take(uint64_t(q.ns) * q.T * lc);
        put_f32_rows(c.sim, &c.r, wt, q.dc, lc, q.C, 0.5f);
        put_f32_rows(c.sim, &c.r, si, q.ns * (q.dc - 1u), lc, q.C, 1.0f);
        put_f32_rows(c.sim, &c.r, x, q.ns * q.T, lc, q.C, 1.0f);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x5343u);
        const uint32_t a[opl::sconv::NARGS] = { dst, lc, x, lc, si, so, wt, q.C, q.dc, q.T, q.ns, q.flags };
        w.add(opl::K_SSM_CONV, a, opl::sconv::NARGS);
        std::vector<region> out;
        rows_region(out, dst, q.ns * q.T, lc, lc);
        rows_region(out, so, q.ns * (q.dc - 1u), lc, lc);
        run(c, opl::K_SSM_CONV, q.what, w, out, true, general_path("sconv_run("));
    }
    // GATED_DELTA_NET: S_v 64 with 2 key heads for 4 value heads, one token; S_v 32 with 3 tokens of 2 sequences
    // (the corpus of revision 1, SW-78: its digest is the composed row of the table). `fused_dev`: the simulator has
    // CAPS bit 8, so a record with S_v >= 64 takes the fused path and S_v 32 the composition. `extras` (the default
    // row only) adds the production shape S_v 128 and the largest m of the GDN op, S_v 256.
    struct gdc { const char * what; uint32_t sv, hk, hv, T, ns; };
    static const gdc gds[] = { { "GATED_DELTA_NET S_v 64, 4 heads of 2, T 1", 64, 2, 4, 1, 1 },
                               { "GATED_DELTA_NET S_v 32, 2 heads, T 3 x 2 sequences", 32, 2, 2, 3, 2 },
                               { "GATED_DELTA_NET S_v 128, 4 heads of 2, T 3", 128, 2, 4, 3, 1 },
                               { "GATED_DELTA_NET S_v 256, 2 heads, T 1 x 2 sequences", 256, 1, 2, 1, 2 } };
    for (const gdc & q : gds) {
        if (q.sv > 64u && !extras) continue;
        namespace g = opl::gdn;
        const uint32_t L = q.sv / 16u, nt = q.ns * q.T, Lg = L16(q.hv);
        const uint32_t hq = c.take(uint64_t(nt) * q.hk * L), hk = c.take(uint64_t(nt) * q.hk * L),
                       hv = c.take(uint64_t(nt) * q.hv * L), hg = c.take(uint64_t(nt) * Lg), hb = c.take(uint64_t(nt) * Lg),
                       si = c.take(uint64_t(q.ns) * q.hv * q.sv * L), so = c.take(uint64_t(q.ns) * q.hv * q.sv * L),
                       dst = c.take(uint64_t(nt) * q.hv * L);
        put_f32_rows(c.sim, &c.r, hq, nt, q.hk * L, q.hk * L * 16u, 0.125f);
        put_f32_rows(c.sim, &c.r, hk, nt, q.hk * L, q.hk * L * 16u, 0.125f);
        put_f32_rows(c.sim, &c.r, hv, nt, q.hv * L, q.hv * L * 16u, 1.0f);
        put_f32_rows(c.sim, &c.r, si, q.ns * q.hv * q.sv, L, q.sv, 0.25f);
        // g in [-2, 0): minus an integer times 2^-5 (up to 2); beta in [0, 1): an integer times 2^-5.
        std::vector<uint32_t> gv(size_t(Lg) * 16u), bv(size_t(Lg) * 16u);
        for (uint32_t n = 0; n < nt; ++n) {
            for (uint32_t h = 0; h < q.hv; ++h) {
                gv[h] = f2u(-float(1u + rng_below(&c.r, 63u)) * (1.0f / 32.0f));
                bv[h] = f2u(float(rng_below(&c.r, 32u)) * (1.0f / 32.0f));
            }
            tsr_sim_dram_write(c.sim, hg >> 28, (uint64_t(hg & 0x0FFFFFFFu) + uint64_t(n) * Lg) * LINE, gv.data(), gv.size() * 4u);
            tsr_sim_dram_write(c.sim, hb >> 28, (uint64_t(hb & 0x0FFFFFFFu) + uint64_t(n) * Lg) * LINE, bv.data(), bv.size() * 4u);
        }
        opl::writer w(opl::F_STOP_ON_ERROR, 0x4744u);
        uint32_t a[g::NARGS] = {};
        a[g::DST] = dst; a[g::STATE_OUT] = so; a[g::STATE_IN] = si;
        a[g::Q] = hq; a[g::Q_HEAD_STRIDE] = L; a[g::Q_TOKEN_STRIDE] = q.hk * L;
        a[g::K] = hk; a[g::K_HEAD_STRIDE] = L; a[g::K_TOKEN_STRIDE] = q.hk * L;
        a[g::V] = hv; a[g::V_HEAD_STRIDE] = L; a[g::V_TOKEN_STRIDE] = q.hv * L;
        a[g::G] = hg; a[g::G_TOKEN_STRIDE] = Lg; a[g::BETA] = hb; a[g::BETA_TOKEN_STRIDE] = Lg;
        a[g::S_V] = q.sv; a[g::H_K] = q.hk; a[g::H_V] = q.hv; a[g::N_TOKENS] = q.T; a[g::N_SEQS] = q.ns;
        a[g::SCALE] = q.sv == 64u ? 0x3E000000u /* 1/8 */ : q.sv == 128u ? 0x3DB504F3u /* 1/sqrt(128) */
                      : q.sv == 256u ? 0x3D800000u /* 1/16 */ : 0x3E3504F3u /* 1/sqrt(32) */;
        w.add(opl::K_GATED_DELTA_NET, a, g::NARGS);
        std::vector<region> out;
        rows_region(out, dst, nt * q.hv, L, L);
        rows_region(out, so, q.ns * q.hv * q.sv, L, L);
        const bool fused = fused_dev && tsr::plan::gdn::use_fused(tsr_sim_config(c.sim)->caps, q.sv);
        run(c, opl::K_GATED_DELTA_NET, q.what, w, out, true,
            fused ? path_exp{ { "gdn_run_tok(" }, { "gdn_run(", "gdn_step_composed(", "gdn_step_fused(" }, {} }
                  : path_exp{ { "gdn_run(", "gdn_step_composed(" }, { "gdn_step_fused(" }, {} });
    }
}

// A KV plane pair (AT §4) at KV_BEAT on every channel: the geometry, every cell below n_fill holding a
// row stored the device's way (QACT A32 + KVST of the reference: tile ops, not a kind).
struct planes {
    uint32_t E = 0, LG = 0, kB = 0, vB = 0;
    uint64_t beats = 0;
};
planes make_planes(launch_ctx & c, uint32_t E, uint32_t n_kv, uint32_t n_fill) {
    planes p;
    const uint32_t C = tsr_sim_config(c.sim)->num_ch;
    p.E = E;
    p.LG = tsr_kv_lg_min(n_kv, C);
    const tsr_kv_geom g = { E, C, p.LG };
    p.beats = tsr_kv_plane_beats(&g);
    p.kB = uint32_t(KV_BEAT);
    p.vB = uint32_t(KV_BEAT + ((p.beats + 63u) & ~63ull));
    std::vector<std::vector<uint8_t>> kp(C, std::vector<uint8_t>(size_t(p.beats) * LINE, 0)), vp = kp;
    std::vector<uint8_t *> kpp(C), vpp(C);
    for (uint32_t i = 0; i < C; ++i) { kpp[i] = kp[i].data(); vpp[i] = vp[i].data(); }
    std::vector<uint8_t> qa(size_t(tsr_qa_lines(TSR_QA_A32, E)) * LINE);
    std::vector<float> x(E);
    for (uint32_t j = 0; j < n_fill; ++j) {
        for (int kv = 0; kv < 2; ++kv) {
            for (float & v : x) v = val(&c.r, 1.0f);
            tsr_qact(TSR_QA_A32, x.data(), E, qa.data());
            tsr_ref_kvst(uint32_t(kv), &g, kv ? vpp.data() : kpp.data(), j, 1u, qa.data(), 0u, nullptr);
        }
    }
    for (uint32_t i = 0; i < C; ++i) {
        tsr_sim_dram_write(c.sim, i, uint64_t(p.kB) * LINE, kp[i].data(), kp[i].size());
        tsr_sim_dram_write(c.sim, i, uint64_t(p.vB) * LINE, vp[i].data(), vp[i].size());
    }
    return p;
}
void plane_regions(launch_ctx & c, const planes & p, std::vector<region> & out) {
    for (uint32_t i = 0; i < tsr_sim_config(c.sim)->num_ch; ++i) {
        out.push_back(region{ i, p.kB, p.beats });
        out.push_back(region{ i, p.vB, p.beats });
    }
}

void corpus_set_rows(launch_ctx & c) {
    struct sr { const char * what; uint32_t D, nkvh, n_kv; std::vector<uint32_t> cells; };
    const sr cases[] = {
        { "SET_ROWS_KV K and V, D 64 x 2 heads, 3 cells", 64, 2, 48, { 40, 41, 7 } },
        { "SET_ROWS_KV K and V, D 128, a run of 5 cells and a repeat", 128, 1, 64, { 10, 11, 12, 13, 14, 12 } },
        { "SET_ROWS_KV K and V, D 64 x 2 heads, one cell", 64, 2, 48, { 41 } },        // the fast entry: one row
        { "SET_ROWS_KV K and V, D 128, one cell", 128, 1, 64, { 13 } },
    };
    for (const sr & q : cases) {
        const uint32_t E = q.D * q.nkvh, n = uint32_t(q.cells.size()), sl = E / 16u;
        planes p = make_planes(c, E, q.n_kv, q.n_kv / 2u);
        const uint32_t srck = c.take(uint64_t(n) * (sl + 1u)), srcv = c.take(uint64_t(n) * (sl + 1u));
        put_f32_rows(c.sim, &c.r, srck, n, sl + 1u, E, 1.0f);
        put_f32_rows(c.sim, &c.r, srcv, n, sl + 1u, E, 1.0f);
        std::vector<uint8_t> idx(size_t(L16(2u * n)) * LINE, 0);
        for (uint32_t t = 0; t < n; ++t) memcpy(idx.data() + 8u * t, &q.cells[t], 4);
        const uint32_t idx_h = c.take(L16(2u * n), 0);
        tsr_sim_dram_write(c.sim, 0, uint64_t(idx_h & 0x0FFFFFFFu) * LINE, idx.data(), idx.size());
        const uint32_t geom = opl::geom_word(E / 64u, p.LG);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x5352u);
        const uint32_t pre = w.add_preload(idx_h, 8u * n);
        const uint32_t k = w.add_set_rows_kv(opl::set_rows_kv_args{ H(0, p.kB), 0u, srck, sl + 1u, pre, n, E, opl::setrows::WHICH_K,
                                                                     p.kB, geom, 0u, 0u });
        w.add_set_rows_kv(opl::set_rows_kv_args{ H(0, p.vB), 0u, srcv, sl + 1u, pre, n, E, opl::setrows::WHICH_V, p.vB, geom, 0u, 0u },
                          { k });
        std::vector<region> out;
        plane_regions(c, p, out);
        run(c, opl::K_SET_ROWS_KV, q.what, w, out, false,       // the planes are inputs too
            n == 1u ? fast_path("kind_set_rows_kv_fast(", "set_rows_any(") : general_path("set_rows_any("));
    }
}

void corpus_fa(launch_ctx & c) {
    struct fc {
        const char * what;
        uint32_t D, nkvh, G, T, n_kv, pos0, scale;
        bool finite;                    // a GENERAL mask: finite values on the visible cells
        uint32_t n_new;                 // FA_NEW cells (the last n_new positions), 0: none
    };
    static const fc cases[] = {
        { "FLASH_ATTN decode D 64, 2 KV heads, G 7, 48 cells", 64, 2, 7, 1, 48, 40, 0x3E000000u /* 0.125 */, false, 0 },
        { "FLASH_ATTN prefill T 3, D 128, G 4, 64 cells, FA_NEW", 128, 1, 4, 3, 64, 30, 0x3DB504F3u /* 1/sqrt 128 */, false, 3 },
        { "FLASH_ATTN decode D 64, G 2, finite (GENERAL) mask", 64, 2, 2, 1, 32, 25, 0x3E000000u, true, 0 },
    };
    const tsr_config * cfg = tsr_sim_config(c.sim);
    const tsr_att_cfg acfg{ cfg->num_ch, cfg->t_max, tsr_qa_lines(TSR_QA_A256, cfg->k_max) * 16u, cfg->k_max };
    for (const fc & q : cases) {
        const uint32_t E = q.D * q.nkvh, n_head = q.G * q.nkvh, K = n_head * q.D, ql = K / 16u;
        planes p = make_planes(c, E, q.n_kv, q.pos0 + q.T);
        const uint32_t qh = c.take(uint64_t(q.T) * (ql + 1u)), dst = c.take(uint64_t(q.T) * ql);
        put_f32_rows(c.sim, &c.r, qh, q.T, ql + 1u, K, 1.0f);
        // The F16 mask [T][n_kv] (causal; GENERAL: -k/4 on the visible cells) and its MSUM.
        std::vector<uint16_t> mask(size_t(q.T) * q.n_kv);
        for (uint32_t t = 0; t < q.T; ++t)
            for (uint32_t j = 0; j < q.n_kv; ++j) {
                const bool vis = j <= q.pos0 + t;
                uint16_t h16 = vis ? 0x0000u : 0xFC00u;
                if (vis && q.finite) h16 = tsr_f32_to_fp16(-0.25f * float((q.pos0 + t - j) % 13u));
                mask[size_t(t) * q.n_kv + j] = h16;
            }
        const uint32_t mask_h = c.take(uint64_t(q.T) * L16(q.n_kv), 1);
        std::vector<uint32_t> msum(tsr_msum_words(q.T));
        const uint32_t mflags = tsr_msum_build_f16(mask.data(), q.n_kv, q.n_kv, q.T, mask_h, msum.data());
        bool need_f32 = (mflags & TSR_MSUM_GENERAL) != 0u;
        if (!need_f32) {                                        // a tile the planner sends GENERAL
            tsr_msum ms;
            tsr_msum_view(msum.data(), msum.size(), &ms);
            uint32_t GS = 0, Tq = 0;
            tsr_fa_pass_rule(&acfg, q.G, &GS, &Tq);
            for (uint32_t q0 = 0; q0 < q.T && !need_f32; q0 += Tq) {
                std::vector<tsr_fa_chunk> ch(4096);
                const long n = tsr_fa_plan(&acfg, E, q.D, &ms, q0, std::min(Tq, q.T - q0), nullptr, 0, ch.data(), ch.size());
                for (long i = 0; i < n && size_t(i) < ch.size(); ++i) need_f32 = need_f32 || ch[size_t(i)].general;
            }
        }
        if (need_f32) {
            std::vector<float> mf(size_t(q.T) * 16u * L16(q.n_kv));
            tsr_msum_mask_f32(mask.data(), q.n_kv, q.n_kv, q.T, u2f(q.scale), mf.data());
            tsr_sim_dram_write(c.sim, 1, uint64_t(mask_h & 0x0FFFFFFFu) * LINE, mf.data(), mf.size() * 4u);
        }
        const uint32_t msum_h = c.take(L16(uint32_t(msum.size())), 0);
        tsr_sim_dram_write(c.sim, 0, uint64_t(msum_h & 0x0FFFFFFFu) * LINE, msum.data(), msum.size() * 4u);
        opl::writer w(opl::F_STOP_ON_ERROR, 0x4641u);
        const uint32_t pre_msum = w.add_preload(msum_h, uint32_t(msum.size() * 4u));
        uint32_t flags = 0;
        if (q.n_new) {                                          // the new tokens' cells, as a SET_ROWS_KV's I64 indices
            std::vector<uint8_t> idx(size_t(L16(2u * q.n_new)) * LINE, 0);
            for (uint32_t j = 0; j < q.n_new; ++j) {
                const uint32_t cell = q.pos0 + q.T - q.n_new + j;
                memcpy(idx.data() + 8u * j, &cell, 4);
            }
            const uint32_t idx_h = c.take(L16(2u * q.n_new), 0);
            tsr_sim_dram_write(c.sim, 0, uint64_t(idx_h & 0x0FFFFFFFu) * LINE, idx.data(), idx.size());
            flags = tsr::plan::attn::fa_flags(w.add_preload(idx_h, 8u * q.n_new), q.n_new);
        }
        const uint32_t geom = opl::geom_word(E / 64u, p.LG);
        w.add_flash_attn(opl::flash_attn_args{ dst, ql, qh, q.D / 16u, ql + 1u, H(0, p.kB), H(0, p.vB), q.n_kv, q.D, n_head,
                                               q.nkvh, q.T, pre_msum, need_f32 ? mask_h : NONE, q.scale, 0u, p.kB, p.vB, geom,
                                               q.D, q.G, flags });
        std::vector<region> out;
        rows_region(out, dst, q.T, ql, ql);
        run(c, opl::K_FLASH_ATTN, q.what, w, out);
    }
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------
struct row {
    uint32_t rev;
    uint64_t digest;
};

// The table: rows of the default run, and the rows marked "composed" (kinds with a second path, SW-78).
struct table_t {
    std::map<uint32_t, row> def, composed;
};

// Kinds with a composed row, and the digest of a kind's variant (0 default, 1 composed).
bool has_composed(uint32_t kind) { return kind == opl::K_GATED_DELTA_NET; }
const kind_digest & digest_of(uint32_t kind, int variant) { return variant ? g_dig_composed[kind] : g_dig[kind]; }
const char * variant_name(int variant) { return variant ? " (composed path)" : ""; }

bool read_table(const char * path, table_t & t) {
    FILE * f = fopen(path, "r");
    if (f == nullptr) return false;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        unsigned kind = 0, rev = 0;
        unsigned long long dg = 0;
        char name[32], var[32] = "";
        const int n = sscanf(line, "%x %31s %u %llx %31s", &kind, name, &rev, &dg, var);
        if (n == 4 || (n == 5 && !strcmp(var, "composed"))) (n == 5 ? t.composed : t.def)[kind] = row{ rev, dg };
        else CHECK(0, "%s: malformed row: %s", path, line);
    }
    fclose(f);
    return true;
}

bool write_table(const char * path) {
    FILE * f = fopen(path, "w");
    if (f == nullptr) return false;
    fprintf(f, "# sw/ctrl/tests/kind_revs.txt — the numerics revision of every op list kind and the digest of its\n"
               "# outputs on test_fw_kind_revs's fixed corpus (ctest fw_kind_revs; docs/design/06-software-\n"
               "# verification.md §6.9, SW-54). Written by `test_fw_kind_revs tsr-fw.elf kind_revs.txt --update` after\n"
               "# a deliberate revision bump in tsr/opl.hpp kind_rev, together with a CHANGELOG \"Unreleased\" line.\n"
               "# A kind with a second path has a second row marked \"composed\": GATED_DELTA_NET on a device without CAPS\n"
               "# bit 8 (SW-78); the first row is the run on the simulator's CAPS.\n"
               "# kind  name         rev  digest (FNV-1a 64: DONE status / RET and every output region, per launch)\n");
    for (uint32_t k = 0; k < opl::N_KINDS; ++k) {
        if (opl::kind_rev(k) == 0) continue;
        fprintf(f, "0x%02x  %-12s %-4u %016llx\n", k, opl::kind_name(k), opl::kind_rev(k), (unsigned long long)g_dig[k].h);
        if (has_composed(k))
            fprintf(f, "0x%02x  %-12s %-4u %016llx composed\n", k, opl::kind_name(k), opl::kind_rev(k),
                    (unsigned long long)g_dig_composed[k].h);
    }
    return fclose(f) == 0;
}

// The rules of the table for this run's digests (rows of kinds without a revision are errors, a kind
// with one needs its row, or its rows when it has two paths). updating: --update, which records the kinds
// whose revision went up and any new kind, and refuses what a plain run refuses otherwise: a digest that moved
// at an unchanged revision, a revision that went up with none of the kind's digests changed, a revision that went
// down. A kind with two paths needs only one of its digests to move for a bump: the path that did not change
// keeps its digest (that is the point of its row).
void check_table(const char * table, const table_t & t, bool digests_only, bool updating) {
    for (uint32_t k = 0; k < opl::N_KINDS; ++k) {
        const uint32_t want = opl::kind_rev(k);
        if (want == 0) {
            CHECK(t.def.find(k) == t.def.end() && t.composed.find(k) == t.composed.end(),
                  "%s: row for kind 0x%02x, which has no revision", table, k);
            continue;
        }
        bool bumped = false, moved = false;
        for (int v = 0; v < (has_composed(k) ? 2 : 1); ++v) {
            const auto & m = v ? t.composed : t.def;
            auto it = m.find(k);
            if (it == m.end()) {
                CHECK(updating, "%s: no row for %s%s (a new kind: run --update and add a CHANGELOG line)", table, opl::kind_name(k),
                      variant_name(v));
                continue;
            }
            const row & r = it->second;
            const uint64_t got = digest_of(k, v).h;
            if (digests_only) {
                CHECK(r.digest == got, "%s%s: digest %016llx, recorded %016llx", opl::kind_name(k), variant_name(v),
                      (unsigned long long)got, (unsigned long long)r.digest);
            } else if (r.rev == want) {
                CHECK(r.digest == got, "%s%s r%u: the outputs changed (digest %016llx, recorded %016llx): bump the revision in "
                                       "tsr/opl.hpp kind_rev, run --update, add a CHANGELOG line", opl::kind_name(k), variant_name(v),
                      want, (unsigned long long)got, (unsigned long long)r.digest);
            } else if (want < r.rev) {
                CHECK(0, "%s%s: revision %u -> %u: a revision never goes down (a released revision stays reproducible)",
                      opl::kind_name(k), variant_name(v), r.rev, want);
            } else {
                bumped = true;
                moved = moved || r.digest != got;
                if (!updating)
                    CHECK(r.digest == got, "%s%s: revision %u -> %u with new outputs: run --update to record digest %016llx",
                          opl::kind_name(k), variant_name(v), r.rev, want, (unsigned long long)got);
            }
        }
        if (bumped)
            CHECK(moved, "%s: revision bumped but none of its digests changed: revert the bump", opl::kind_name(k));
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: test_fw_kind_revs IMAGE TABLE [--update] [--digests-only] [--paths]\n");
        return 2;
    }
    const char * image = argv[1];
    const char * table = argv[2];
    bool update = false, digests_only = false;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--update")) update = true;
        else if (!strcmp(argv[i], "--digests-only")) digests_only = true;
        else if (!strcmp(argv[i], "--paths")) g_print_paths = true;
        else { fprintf(stderr, "test_fw_kind_revs: unknown option %s\n", argv[i]); return 2; }
    }
    struct stat st;
    if (stat(image, &st) != 0) {
        printf("fw_kind_revs: %s missing: skipped\n", image);
        return 77;
    }
    // The image's kind table.
    uint8_t rev[TSR_ELF_N_KINDS] = {};
    if (!digests_only) {
        tsr_elf * e = nullptr;
        char err[256];
        CHECK(tsr_elf_read(image, &e, err, sizeof(err)) == 0, "%s", err);
        if (e != nullptr) {
            char why[96];
            CHECK(tsr_elf_kinds(e, rev, why, sizeof(why)) >= 0, "%s: malformed .tsr_kinds (%s)", image, why);
            tsr_elf_free(e);
        }
    }

    tsr_config cfg = TSR_CONFIG_U200;
    cfg.t_max = 8;
    cfg.ch_bytes = 256ull << 20;
    cfg.caps = TSR_SIM_CAPS_IMPLEMENTED;
    CHECK((cfg.caps & TSR_CAPS_GDN) != 0u, "the simulator does not implement CAPS bit 8 (the fused GDN op)");
    bool watch_loaded = false;
    // One simulator with its own boot of the image: `body` runs the launches. The retire hook counts the entries into
    // the image's functions (g_watch).
    auto with_sim = [&](const tsr_config & c0, const std::function<void(launch_ctx &)> & body) {
        tsr_sim_opts so;
        memset(&so, 0, sizeof(so));
        so.sched = "fixed";
        so.threads = 2;
        tsr_config c1 = c0;
        tsr_sim * sim = tsr_sim_create_opts(&c1, &so);
        CHECK(sim != nullptr, "simulator");
        if (sim == nullptr) return;
        fwh h;
        if (fwh_open(&h, sim, image) != 0) {
            CHECK(0, "%s: boot", image);
            tsr_sim_destroy(sim);
            return;
        }
        if (!watch_loaded) {
            CHECK(g_watch.load(image), "%s: no function symbols (a stripped image: the path checks need them)", image);
            watch_loaded = true;
        }
        tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), path_retire, &g_watch);
        launch_ctx c{ sim, &h, fwh_kernel(&h, "graph"), rng_t{ 0x6B696E64u } };
        CHECK(c.graph >= 0, "%s: no graph kernel", image);
        if (c.graph >= 0) body(c);
        tsr_rv_agent_set_retire_hook(tsr_sim_rv_agent(sim), nullptr, nullptr);
        fwh_close(&h);
        tsr_sim_destroy(sim);
    };
    rng_t corpus_base{};
    with_sim(cfg, [&](launch_ctx & c) {
        corpus_nop_join(c);
        corpus_mulmat(c);
        corpus_ew(c);
        corpus_scale(c);
        corpus_rms(c);
        corpus_glu(c);
        corpus_rope(c);
        corpus_rows(c);
        corpus_set_rows(c);
        corpus_fa(c);
        corpus_unary(c);
        corpus_mulmat_epilogue(c, opl::K_MUL_MAT_SCALE);
        corpus_mulmat_epilogue(c, opl::K_MUL_MAT_SOFTCAP);
        // The launches share one PRNG, so a corpus that runs earlier would move the digests of the later ones.
        // The MoE and the Gated DeltaNet corpora therefore run last and each starts from the same PRNG state:
        // the digests of their kinds stay as they are, and adding the other corpus
        // does not move them.
        corpus_base = c.r;
        corpus_moe(c);
        c.r = corpus_base;
        corpus_gdn(c, true, true);
    });
    // The composed path of GATED_DELTA_NET (SW-78): a simulator without CAPS bit 8, the same PRNG state, the corpus of
    // revision 1. Its softplus / ssm_conv digests are the default run's and are not recorded.
    {
        tsr_config cfg2 = cfg;
        cfg2.caps &= ~uint32_t(TSR_CAPS_GDN);
        g_dig_cur = &g_dig_composed;
        with_sim(cfg2, [&](launch_ctx & c) {
            c.r = corpus_base;
            corpus_gdn(c, false, false);
        });
        g_dig_cur = &g_dig;
    }

    // Every kind with a revision ran its corpus, answered OK and is declared at kind_rev; nothing else
    // is declared.
    for (uint32_t k = 0; k < opl::N_KINDS; ++k) {
        const uint32_t want = opl::kind_rev(k);
        if (want == 0) {
            CHECK(rev[k] == 0, "the image declares kind 0x%02x (revision %u), which tsr/opl.hpp does not define", k, rev[k]);
            continue;
        }
        for (int v = 0; v < (has_composed(k) ? 2 : 1); ++v) {
            const kind_digest & d = digest_of(k, v);
            CHECK(d.launches > 0 && d.ok == d.launches, "%s%s: %u of %u corpus launches answered OK", opl::kind_name(k),
                  variant_name(v), d.ok, d.launches);
        }
        if (!digests_only)
            CHECK(rev[k] == want, "%s: the image's .tsr_kinds says revision %u, tsr/opl.hpp kind_rev %u%s", opl::kind_name(k),
                  rev[k], want, rev[k] == 0 ? " (kind missing from the image's table: TSR_KIND beside its handler)" : "");
    }
    for (uint32_t k = 0; k < opl::N_KINDS; ++k)
        if (opl::kind_rev(k)) {
            printf("fw_kind_revs: 0x%02x %-12s rev %u digest %016llx (%u launches)\n", k, opl::kind_name(k), opl::kind_rev(k),
                   (unsigned long long)g_dig[k].h, g_dig[k].launches);
            if (has_composed(k))
                printf("fw_kind_revs: 0x%02x %-12s rev %u digest %016llx (%u launches) composed path\n", k, opl::kind_name(k),
                       opl::kind_rev(k), (unsigned long long)g_dig_composed[k].h, g_dig_composed[k].launches);
        }

    table_t t;
    const bool have = read_table(table, t);
    if (update) {
        // --digests-only: a base to compare another image with; else the table rules decide what may be recorded.
        if (!digests_only && have) check_table(table, t, false, true);
        CHECK(g_fail == 0, "--update: not written after a failure");
        if (g_fail == 0) {
            CHECK(write_table(table), "cannot write %s", table);
            printf("fw_kind_revs: wrote %s\n", table);
        }
        return test_finish("fw_kind_revs");
    }
    CHECK(have, "cannot read %s", table);
    check_table(table, t, digests_only, false);
    return test_finish("fw_kind_revs");
}
