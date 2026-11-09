// k_fattn.cpp — the FLASH_ATTN kind of the graph executor (op list kind 0x19, two slots; docs/design/
// 06-software-verification.md "SW" §7.2, §7.5; docs/design/03-attention-kv.md "AT" §5.2-§5.7, §6.6
// (normative), §7.1-§7.5, §12).
//
// ggml's FLASH_ATTN_EXT with K and V in the device KV planes (AT §4) and the mask as its summary
// MSUM in the preload area (AT §7.3). The tile-op sequence is the numerical contract (SW §7.5, AT
// §6.6) and is exactly tsr_ref_fa's (sw/sim/tsr_ref_at.c), which sw/ctrl/ref/fattn_ref.cpp calls with
// this record's inputs:
//   per query tile of T_q = floor(T_MAX / GS) tokens (GS = min(G, T_MAX); the last tile shorter):
//     1. LD the tile's Q rows (n_head·D f32 each, heads contiguous) into QS, QACT A32 of them
//        (K = n_head·D, the scale not applied: ASMAX / AMERGE apply it) into X;
//     2. the tile's chunk list from the MSUM rows (sw/sim/include/tsr_fa_plan.h, the planner
//        tsr_ref_fa uses: same source), and per chunk k in that order:
//          - per head subset hg0 = 0, GS, ... (< G, GS' = min(GS, G - hg0)): ATTQK into S[k mod 3]
//            (FAST: the chunk's bounds and CAUSAL / SLIDE, MAXOUT into MK[k mod 4]; GENERAL: no
//            bounds, no MAXOUT);
//          - GENERAL: LD the tile's rows of the f32 mask copy (word 13; row q0 + q, lines u0 ..
//            u0 + n of rows ceil(n_kv/16) lines long, AT §7.3) into M[k mod 2], AMASK (in place,
//            rpq = n_head, row maxima into MK[k mod 4]);
//          - ASMAX in place (MK, scale);
//          - one chunk behind (AT §7.1): ATTPV (LSUM) of chunk k - 1 per head subset into P[(k-1)
//            mod 2], then AMERGE of chunk k - 1 (INIT for chunk 0) into Orun / the running state;
//        then ATTPV and AMERGE of the last chunk;
//     3. ANORM into QS (Q's rows are dead after QACT: the output rows [token][head][D]), ST to the
//        tile's dst rows.
// tsr_ref_fa runs the same ops per chunk in the same order (all ATTQK of the chunk, AMASK, ASMAX,
// all ATTPV, AMERGE); pushing ATTPV / AMERGE one chunk later moves them only across ops of the next
// chunk that use other buffers, so every op sees the same inputs (AT §7.1's overlap).
//
// Scratch: one ring block per op (plan_attn.hpp fa_plan: QS, X, Orun, state, MK[4], S[3], P[2],
// M[2]), reused by every tile. Deps and queue orders (AT §7.1; every op <= 2 deps):
//   LD Q (tile 0: the op's deps and the block's ring deps without DMA tickets; later tiles: none,
//   the DMA queue puts them after the previous tile's ST, which follows every op of that tile) ->
//   QACT -> ATTQK of chunk 0's first subset (the later ATTQK / ATTPV follow on the MVM queue);
//   AMASK <- {its chunk's last ATTQK, its last mask LD}; a mask LD <- the AMASK that last read its
//   M buffer (chunk k - 2); ASMAX <- {its chunk's last ATTQK} (FAST; GENERAL: the VU queue after
//   AMASK); ATTPV (first subset) <- {its chunk's ASMAX}; AMERGE <- {its chunk's last ATTPV}; ANORM
//   follows the last AMERGE on the VU queue; the first ST <- {ANORM}.
//   Queue orders: MVM ATTQK0, ATTQK1, ATTPV0, ATTQK2, ATTPV1, ...; VU [AMASK0] ASMAX0, [AMASK1]
//   ASMAX1, AMERGE0, ...; DMA LD Q, [mask LDs], ST.
//   Buffer reuse by queue order: ATTQK k writes S[k mod 3] after ATTPV k - 3 read it (MVM queue),
//   and ATTPV k - 3 followed ASMAX / AMASK k - 3; ATTPV k writes P[k mod 2] after its ASMAX k, which
//   follows AMERGE k - 2 on the VU queue; ATTQK k (MAXOUT) writes MK[k mod 4] after ATTPV k - 2 (MVM
//   queue), whose ASMAX k - 2 follows AMERGE k - 4 (and ASMAX k - 4) on the VU queue: four MK
//   buffers, because with three (AT §7.1's sketch) ATTQK k would race with AMERGE k - 3 (nothing
//   orders them); AMASK k writes MK on the VU queue after them. Within a tile every op's inputs are
//   written by its deps or earlier ops of its queue; across tiles the next tile's LD Q follows this
//   tile's ST (DMA queue), which follows every op of the tile (ST <- ANORM <- VU queue <- last AMERGE
//   <- last ATTPV <- MVM queue <- every ATTQK <- QACT <- LD Q).
//   The returned ticket is the last tile's ST: it implies every op of the op (I1).
//
// Records (plan_attn.hpp has the argument rules, the flags word and the layout): the MSUM at word
// 12 (preload, 64-aligned, magic, n_q = n_tokens, n_kv = word 7); the f32 mask copy (word 13) is
// needed by every tile whose chunks take the GENERAL path (a GENERAL MSUM, or a multi-row tile whose
// rows are not one run of the CAUSAL / SLIDE form, e.g. a prefill tile of a wrapped SWA ring);
// FA_NEW names the cells written in this launch (rule 7, newest last). Tile 0 is planned and its
// chunk list verified before the first push (refused: BAD_RECORD, nothing pushed); a later tile is
// planned after the previous tile's pushes, while the units still work on them, and a refusal there
// (a malformed MSUM row, a GENERAL chunk without the mask copy, the planner's rule 6) ends the
// record with RET BAD_RECORD after the earlier tiles' ops (exec.hpp: they stay joinable).
//
// Cost (SW §7.8; AT §7.4, §9.3): per tile one planner run and one chunk list (tsr_fap_list: every
// chunk generated once, kept in a 32-entry array; a longer list is walked with tsr_fap_next, a dry
// run first), 2 + 4·NC + 2·(passes - 1)·NC pushes; the ops of a tile are built once (the generated
// builders) and patched per chunk; the first chunk is planned before LD Q, so LD Q, QACT and the
// first ATTQK leave without a planner step between them. FA is not a short kind: the requirement is
// that the RISC-V stays ahead of the units, which fw_kvfa_perf measures (Qwen2.5-32B decode at 4K:
// ~3.4 K RV cycles = 20 us of RISC-V work against ~140 us on the units, the units' span equal to
// the one with a near-ideal issuer).
#include "k_fast.hpp"
#include "k_mulmat.hpp"
#include "tsr/plan_attn.hpp"
#include "../../sim/include/tsr_local_tile.h"

// The op list kinds this source handles (SW §6.9: .tsr_kinds entries at tsr/opl.hpp kind_rev).
TSR_KIND(FLASH_ATTN);

namespace tsr {

namespace {

namespace fa = opl::fa;
namespace at = plan::attn;

// The cells written in this launch (FA_NEW), as group runs [glo, ghi): a chunk [u0, u0 + n) holds
// one iff some run overlaps it (the indices' cell runs overlap it iff their group runs do: chunks
// are whole groups). More runs than fit: the indices are scanned per chunk.
constexpr uint32_t NEW_RUNS_MAX = 8u;
struct new_cells {
    uint32_t n_runs;
    uint32_t glo[NEW_RUNS_MAX], ghi[NEW_RUNS_MAX];
    bool scan;
    const volatile uint32_t * idx;   // I64 indices (preload area)
    uint32_t n_idx;
};

TSR_HOT uint32_t idx_group(const volatile uint32_t * p, uint32_t i) {
    const uint32_t lo = p[2u * i], hi = p[2u * i + 1u];
    return hi == 0u ? lo >> 4 : 0xFFFFFFFFu;            // >= 2^32: in no chunk
}

void new_cells_of(new_cells & nc, const volatile uint32_t * idx, uint32_t n) {
    nc.n_runs = 0;
    nc.scan = false;
    nc.idx = idx;
    nc.n_idx = n;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t g = idx_group(idx, i);
        if (g == 0xFFFFFFFFu) continue;
        if (nc.n_runs > 0u) {
            const uint32_t k = nc.n_runs - 1u;
            if (g >= nc.glo[k] && g <= nc.ghi[k]) {      // inside or just after the last run
                if (g == nc.ghi[k]) nc.ghi[k] = g + 1u;
                continue;
            }
            if (g + 1u == nc.glo[k]) {                   // just before it
                nc.glo[k] = g;
                continue;
            }
        }
        if (nc.n_runs == NEW_RUNS_MAX) {
            nc.scan = true;
            return;
        }
        nc.glo[nc.n_runs] = g;
        nc.ghi[nc.n_runs] = g + 1u;
        ++nc.n_runs;
    }
}

// Rule 7's callback (tsr_fa_plan.h): does the chunk [u0, u0 + n) hold a new cell.
int has_new(void * ctx, uint32_t u0, uint32_t n) {
    const new_cells & nc = *static_cast<const new_cells *>(ctx);
    const uint32_t end = u0 + n;
    if (!nc.scan) {
        for (uint32_t k = 0; k < nc.n_runs; ++k)
            if (nc.glo[k] < end && nc.ghi[k] > u0) return 1;
        return 0;
    }
    for (uint32_t i = 0; i < nc.n_idx; ++i) {
        const uint32_t g = idx_group(nc.idx, i);
        if (g >= u0 && g < end) return 1;
    }
    return 0;
}

struct fa_op {
    at::fa_layout f;
    uint32_t w[fa::NARGS];
    uint32_t C, T, n_kv, msum_flags, mask, mstride;
    uint64_t pc;                      // the planner's per-op NCG term (tsr_fap_pc)
    const volatile uint32_t * msum;   // the MSUM image (preload area)
    new_cells nc;
    bool use_new, local;
    spm blk;
    uint32_t pf;
};

// One tile's plan: the planner state and its chunk list in execution order (tsr_fap_list: every
// chunk generated once), or, for a list longer than LIST_CAP, only the state (the pushes then walk
// it with tsr_fap_next). LIST_CAP chunks cover a tile of 32·NCG groups (64 Ki cells at NCG 128).
constexpr uint32_t LIST_CAP = 32u;
struct tile_plan_t {
    tsr_fap_tile t;
    long n;                           // chunks (> LIST_CAP: not stored)
    const tsr_fap_chunk * cached;     // points into the bounded record preflight cache
    tsr_fap_chunk list[LIST_CAP];
};

TSR_HOT uint32_t tile_tokens(const fa_op & o, uint32_t q0) { return o.T - q0 < o.f.Tq ? o.T - q0 : o.f.Tq; }

// A dry run of a tile's chunk list to the end: the planner verifies every chunk's encoding
// (tsr_fa_plan.h rule 6), and a GENERAL chunk needs the f32 mask copy.
bool walk_ok(const fa_op & o, const tsr_fap_tile & t) {
    tsr_fap_iter it;
    tsr_fap_iter_init(&it);
    for (;;) {
        tsr_fap_chunk c;
        const int r = tsr_fap_next(&t, &it, o.use_new ? has_new : nullptr, const_cast<new_cells *>(&o.nc), &c);
        if (r < 0) return false;
        if (r == 0) return true;
        if (c.general && o.mask == opl::HANDLE_NONE) return false;
    }
}

// Tile [q0, q0 + Tq) planned (rules 1-5; its MSUM rows copied from the preload area, a GENERAL
// MSUM's rows not read) and its chunk list made and verified (rules 6-8; a GENERAL chunk needs the
// f32 mask copy). False: the tile is refused.
[[gnu::noinline]] bool plan_tile(const fa_op & o, uint32_t q0, tile_plan_t & p) {
    const uint32_t Tq = tile_tokens(o, q0);
    uint32_t rows[4u * TSR_FAP_T_MAX];
    if (!(o.msum_flags & TSR_FAP_GENERAL)) {
        const volatile uint32_t * r = o.msum + opl::MSUM_HDR_WORDS + opl::MSUM_ROW_WORDS * q0;
        for (uint32_t i = 0; i < opl::MSUM_ROW_WORDS * Tq; ++i) rows[i] = r[i];
    }
    if (tsr_fap_tile_init_pc(&p.t, o.C, o.pc, o.n_kv, o.msum_flags, rows, Tq) != 0) return false;
    p.cached = nullptr;
    p.n = tsr_fap_list(&p.t, o.use_new ? has_new : nullptr, const_cast<new_cells *>(&o.nc), p.list, LIST_CAP);
    if (p.n < 0) return false;
    if (p.n > long(LIST_CAP)) return walk_ok(o, p.t);
    for (long i = 0; i < p.n; ++i)
        if (p.list[i].general && o.mask == opl::HANDLE_NONE) return false;
    return true;
}

// The chunks of a planned tile in execution order: its list, or a walk.
struct chunk_src {
    const tile_plan_t & p;
    const fa_op & o;
    tsr_fap_iter it;
    long i = 0;
    chunk_src(const tile_plan_t & pp, const fa_op & oo) : p(pp), o(oo) { tsr_fap_iter_init(&it); }
    TSR_HOT bool next(tsr_fap_chunk & c) {
        if (p.n <= long(LIST_CAP)) {
            if (i >= p.n) return false;
            c = (p.cached ? p.cached : p.list)[i++];
            return true;
        }
        return tsr_fap_next(&p.t, &it, o.use_new ? has_new : nullptr, const_cast<new_cells *>(&o.nc), &c) > 0;
    }
};

// Reuse the checked chunk lists for short prefills. A long record can exceed
// this stack budget and falls back to its original deterministic planner walk.
struct preflight_cache {
    static constexpr uint32_t TILES = 8u;
    uint32_t tiles = 0u;
    uint8_t end[TILES + 1u] = {};
    tsr_fap_chunk list[LIST_CAP];

    void remember(uint32_t tile, const tile_plan_t & p) {
        if (tile != tiles || tiles == TILES || p.n > long(LIST_CAP - end[tiles])) return;
        const uint32_t start = end[tiles];
        for (uint32_t i = 0; i < uint32_t(p.n); ++i) list[start + i] = p.list[i];
        end[++tiles] = uint8_t(start + p.n);
    }

    bool get(uint32_t tile, tile_plan_t & p) const {
        if (tile >= tiles) return false;
        p.cached = list + end[tile];
        p.n = end[tile + 1u] - end[tile];
        return true;
    }
};

// Select the path for the whole record before allocating its ring block or issuing LD.
static bool select_local_plan(fa_op & o, const tile_plan_t & first, preflight_cache & cache) {
    o.local = o.C == TSR_LOCAL_AT_OWNERS && (o.msum_flags & TSR_FAP_GENERAL) == 0u;
    if (!o.local) return true;
    // The existing first-tile validation suffices for a FAST decode record
    // if even its largest possible planner chunk fits.
    if (o.T == 1u && tsr_local_at_fits(o.C, o.f.R, o.f.ncg)) return true;
    if (first.n <= long(LIST_CAP)) {
        uint32_t groups = 0u;
        for (long i = 0; i < first.n; ++i) {
            if (first.list[i].general) {
                o.local = false;
                return true;
            }
            groups += first.list[i].n_groups;
        }
        // A short multi-tile prefill gets little overlap after moving its
        // later-tile planning before the first issue. Keep the existing path
        // when the first tile spans at most one minimum planner chunk.
        if (o.T > o.f.Tq && groups <= 4u * o.C) {
            o.local = false;
            return true;
        }
    }
    tile_plan_t p;
    for (uint32_t q0 = 0u, tile = 0u; q0 < o.T; q0 += o.f.Tq, ++tile) {
        if (q0 != 0u) {
            if (!plan_tile(o, q0, p)) return false;
            cache.remember(tile - 1u, p);
        }
        if (!o.local) continue;   // still validate every tile for the legacy path
        chunk_src src(q0 == 0u ? first : p, o);
        tsr_fap_chunk c;
        const uint32_t rows = tile_tokens(o, q0) * o.f.n_head;
        while (src.next(c)) {
            if (c.general || !tsr_local_at_fits(o.C, rows, c.n_groups)) {
                o.local = false;
                break;
            }
        }
    }
    return true;
}

// The ops of a tile, built once per tile with the generated builders (at.hpp) and patched per
// chunk / head subset in the words that change (the same encodings: the fields are the builders',
// placed with at.hpp's field constants; every value is inside its field, fa_plan and the planner
// checked them: u0 < 2^16, n_groups <= 128, c_hi <= 2048, hg0 < 64, GS <= 8).
struct tile_ops {
    op qk, pv, sm, mg;
    uint32_t Tq, R;
};

TSR_HOT op qk_op(const tile_ops & to, uint32_t hg0, uint32_t gs, const tsr_fap_chunk & c, spm S, spm MK, uint32_t slot) {
    op o = to.qk;
    if (!c.general)
        o.w[0] |= ATTQK_MAXOUT | ((c.flags & TSR_FAP_CAUSAL) ? ATTQK_CAUSAL : 0u) | ((c.flags & TSR_FAP_SLIDE) ? ATTQK_SLIDE : 0u);
    o.w[0] |= hg0 << ATTQK_HG0_LSB;
    o.w[4] = (o.w[4] & ~ATTQK_GS_M1) | (gs - 1u) << ATTQK_GS_M1_LSB;
    if (slot != 0u) {
        o.w[0] |= slot << ATTQK_LOCAL_SLOT_LSB;
        o.w[4] |= tsr_local_at_stride(c.n_groups) << ATTQK_RSTRIDE_LSB;
    }
    o.w[1] = S.line;
    o.w[5] = c.u0 | c.n_groups << ATTQK_N_GROUPS_LSB;
    o.w[6] = c.c_lo | c.c_hi << ATTQK_C_HI_LSB;
    o.w[7] = c.general ? 0u : MK.line;
    return o;
}
TSR_HOT op sm_op(const tile_ops & to, spm S, spm MK, const tsr_fap_chunk & c, uint32_t slot) {
    op o = to.sm;
    o.w[1] = S.line;
    o.w[2] = S.line;
    o.w[3] = MK.line;
    o.w[4] = c.n_groups;
    if (slot != 0u) {
        o.w[4] |= tsr_local_at_stride(c.n_groups) << ASMAX_STRIDE_LSB;
        o.w[7] = slot | (c.u0 << ASMAX_LOCAL_U0_LSB);
        o.used |= uint8_t(1u << ASMAX_LOCAL_SLOT_WORD);
    }
    return o;
}

// The ops of one chunk that run one chunk behind (AT §7.1): ATTPV (LSUM) per head subset, AMERGE.
struct chunk_ref {
    uint32_t u0, n, k, slot;
    ticket sm;                        // its ASMAX
};

ticket push_pv(const fa_op & o, const tile_ops & to, const chunk_ref & c) {
    const at::fa_layout & f = o.f;
    const spm P = o.blk + f.p + (c.k & 1u) * f.p_lines;
    const spm S = o.blk + f.s + (c.k % 3u) * f.s_lines;
    const spm MK = o.blk + f.mk + (c.k & 3u) * f.rl;
    ticket pv{};
    for (uint32_t hg0 = 0; hg0 < f.G; hg0 += f.GS) {
        const uint32_t gs = f.G - hg0 < f.GS ? f.G - hg0 : f.GS;
        op x = to.pv;
        x.w[0] |= hg0 << ATTPV_HG0_LSB;
        x.w[4] = (x.w[4] & ~ATTPV_GS_M1) | (gs - 1u) << ATTPV_GS_M1_LSB;
        if (c.slot != 0u) {
            x.w[0] |= c.slot << ATTPV_LOCAL_SLOT_LSB;
            x.w[4] |= tsr_local_at_stride(c.n) << ATTPV_PSTRIDE_LSB;
        }
        x.w[1] = P.line;
        x.w[2] = S.line;
        x.w[5] = c.u0 | c.n << ATTPV_N_GROUPS_LSB;
        pv = issue(x, hg0 == 0u ? deps{c.sm} : deps{}, o.pf);
    }
    op m = to.mg;
    if (c.k == 0u) m.w[0] |= AMERGE_INIT;
    m.w[2] = P.line;
    m.w[3] = MK.line;
    return issue(m, deps{pv}, o.pf);
}

// One query tile [q0, q0 + Tq) (header comment, steps 1-3), planned (p). first: the first LD's deps.
// Returns the tile's last ST.
ticket push_tile(fa_op & o, uint32_t q0, const tile_plan_t & p, deps first) {
    const at::fa_layout & f = o.f;
    const uint32_t Tq = tile_tokens(o, q0), R = Tq * f.n_head, Kl = f.K >> 4, pf = o.pf;
    const spm QS = o.blk + f.qs, X = o.blk + f.x;
    // The tile's first chunk before the first push: LD Q, QACT and the first ATTQK then follow each
    // other without a planner step between them (the head the units wait for).
    chunk_src src(p, o);
    tsr_fap_chunk c;
    bool more = src.next(c);          // true: every tile has a chunk
    tile_ops to;
    to.Tq = Tq;
    to.R = R;
    to.qk = ATTQK({.fmt = at::KV_FMT_Q8_0, .g_m1 = f.G - 1u, .x = X.line, .base = o.w[fa::K_PLANE], .e64 = f.E >> 6,
                   .dcode = f.dcode, .tq_m1 = Tq - 1u});
    to.pv = ATTPV({.fmt = at::KV_FMT_Q8_0, .lsum = 1u, .g_m1 = f.G - 1u, .base = o.w[fa::V_PLANE], .e64 = f.E >> 6,
                   .dcode = f.dcode, .tq_m1 = Tq - 1u, .lg = f.LG});
    to.sm = ASMAX({.r = R, .scale = o.w[fa::SCALE]});
    to.mg = AMERGE({.orun = (o.blk + f.orun).line, .st = (o.blk + f.st).line, .r = R, .dcode = f.dcode, .scale = o.w[fa::SCALE]});
    // 1. Q rows, QACT.
    const ticket ld = kf::load_rows(kf::chk_t{}, QS, from_handle(o.w[fa::Q]), o.w[fa::Q_TOKEN_STRIDE], Kl, q0, Tq, first, pf);
    const ticket qa = issue(QACT(qa::a32, X, QS, f.K, Tq, Kl), deps{ld}, pf);
    // 2. Chunks.
    ticket am[2]{};                   // the AMASK that last read M[0] / M[1] in this tile
    chunk_ref prev{};
    uint32_t k = 0;
    for (; more; more = src.next(c)) {
        const spm S = o.blk + f.s + (k % 3u) * f.s_lines;
        const spm MK = o.blk + f.mk + (k & 3u) * f.rl;
        const uint32_t slot = o.local ? 1u + k % TSR_LOCAL_AT_SLOTS : 0u;
        ticket qk{};
        for (uint32_t hg0 = 0; hg0 < f.G; hg0 += f.GS) {
            const uint32_t gs = f.G - hg0 < f.GS ? f.G - hg0 : f.GS;
            qk = issue(qk_op(to, hg0, gs, c, S, MK, slot), (k == 0u && hg0 == 0u) ? deps{qa} : deps{}, pf);
        }
        ticket sm{};
        if (c.general) {
            // The tile's mask rows of the chunk's groups into M[k mod 2] (after the AMASK of chunk
            // k - 2 read it), then AMASK; ASMAX follows it on the VU queue.
            const spm M = o.blk + f.m + (k & 1u) * f.m_lines;
            const dram mk0 = from_handle(o.mask) + (q0 * o.mstride + c.u0);
            const ticket ml = kf::load_rows(kf::chk_t{}, M, mk0, o.mstride, c.n_groups, 0u, Tq, deps{am[k & 1u]}, pf);
            am[k & 1u] = issue(AMASK({.dst = S.line, .src = S.line, .mask = M.line, .n_lines = c.n_groups, .stride = 0u,
                                      .r = R, .rpq = f.n_head, .mstride = c.n_groups, .mdst = MK.line}),
                               deps{qk, ml}, pf);
            sm = issue(sm_op(to, S, MK, c, slot), deps{}, pf);
        } else {
            sm = issue(sm_op(to, S, MK, c, slot), deps{qk}, pf);
        }
        if (k != 0u) push_pv(o, to, prev);
        prev = chunk_ref{c.u0, c.n_groups, k, slot, sm};
        ++k;
    }
    push_pv(o, to, prev);             // the last chunk (every tile has one)
    // 3. ANORM into QS (after the last AMERGE: VU queue), ST.
    const ticket an = issue(ANORM({.out = QS.line, .orun = (o.blk + f.orun).line, .l_run = (o.blk + f.st + ((R + 15u) >> 4)).line,
                                   .r = R, .dcode = f.dcode}),
                            deps{}, pf);
    return kf::store_rows(kf::chk_t{}, QS, from_handle(o.w[fa::DST]), o.w[fa::DST_STRIDE], Kl, q0, Tq, deps{an}, pf);
}

[[gnu::noinline]] ticket flash_attn(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    if (n_args != fa::NARGS || kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    fa_op o;
    for (uint32_t i = 0; i < fa::NARGS; ++i) o.w[i] = a[i];
    const plan_limits l = plan::plan_limits_of(*ctx.cfg);
    if (const uint32_t st = at::fa_plan(o.w, l, &o.f)) return ctx.fail(st);
    o.C = l.num_ch;
    o.pc = tsr_fap_pc(l.num_ch, l.pcap, o.f.E, o.f.D);
    o.T = o.w[fa::N_TOKENS];
    o.n_kv = o.w[fa::N_KV];
    o.mask = o.w[fa::MASK];
    o.mstride = (o.n_kv + 15u) >> 4;
    o.pf = ctx.push_flags;
    // The MSUM (data: SW §7.2): inside the preload area, its header matching the record.
    const uint32_t ms_off = o.w[fa::MSUM];
    if (!ctx.pre_ok(ms_off, opl::MSUM_HDR_WORDS + opl::MSUM_ROW_WORDS * o.T, 2u, opl::PRE_MSUM_ALIGN - 1u))
        return ctx.fail(opl::RET_BAD_RECORD);
    o.msum = ctx.pre_lb(ms_off);
    if (o.msum[0] != opl::MSUM_MAGIC || o.msum[1] != o.T || o.msum[2] != o.n_kv) return ctx.fail(opl::RET_BAD_RECORD);
    o.msum_flags = o.msum[3];
    // The new cells (rule 7).
    o.use_new = (o.w[fa::FLAGS] & at::FA_NEW) != 0u;
    o.nc = new_cells{};
    if (o.use_new) {
        const uint32_t off = at::fa_new_pre_byte(o.w[fa::FLAGS]), n = at::fa_n_new(o.w[fa::FLAGS]);
        if (!ctx.pre_ok(off, n, 3u, opl::PRE_KVIDX_ALIGN - 1u)) return ctx.fail(opl::RET_BAD_RECORD);
        new_cells_of(o.nc, ctx.pre_lb(off), n);
    }
    // Tile 0 planned before the first push (a refused tile 0: nothing pushed); a later tile is
    // planned after the previous tile's pushes (the RISC-V runs ahead of the units by then), and a
    // refusal there ends the record with the earlier tiles pushed (exec.hpp: they stay joinable).
    tile_plan_t p;
    if (!plan_tile(o, 0u, p)) return ctx.fail(opl::RET_BAD_RECORD);
    o.local = false;
    preflight_cache cache;
    if ((l.caps & CAPS_ATT_LOCAL) != 0u && !select_local_plan(o, p, cache))
        return ctx.fail(opl::RET_BAD_RECORD);
    // The block, then the tiles one after another in it.
    deps rd;
    o.blk = ctx.ring->alloc(o.f.total, 2u, &rd);
    if (!valid(o.blk)) return ctx.fail(opl::RET_SCRATCH);
    ticket last = push_tile(o, 0u, p, mmk::first_deps(d, rd));
    for (uint32_t q0 = o.f.Tq, tile = 0u; q0 < o.T; q0 += o.f.Tq, ++tile) {
        if (!cache.get(tile, p) && !plan_tile(o, q0, p)) {
            ctx.ret = opl::RET_BAD_RECORD;
            break;
        }
        last = push_tile(o, q0, p, deps{});
    }
    ctx.ring->release(o.blk, last);
    return last;
}

}  // namespace

ticket kind_flash_attn(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    return flash_attn(a, n_args, kflags, d, ctx);
}

}  // namespace tsr
