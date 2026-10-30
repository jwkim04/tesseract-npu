// k_rows.cpp — the row-copy kinds GET_ROWS (0x17) and CPY (0x1A) of the graph executor (docs/v2/
// 06-software-verification.md "SW" §7.2, §7.5; handler interface tsr/exec.hpp, SW §7.3; fast path
// SW §7.3 "Hot path", k_elem.hpp / k_fast.hpp).
//
// Both copy f32 rows bit for bit through the ring (no VU op; DMA only, so the tile-op sequence has no
// numerics): per tile (tsr/plan_elem.hpp, kind_plan: t rows of rl lines, or one chunk of a row that
// does not fit half the budget), one block (the one-tile block when it fits: copies are DMA only,
// V2-SW-36; else the ring), then
//   1. LD the tile's source rows into the block, packed (rl lines apart): CPY rows r0 .. r0 + t - 1
//      (one LD for dense rows; a CPY dense on both sides, or of one row, is one run of rows x rl
//      lines, CP_MERGED); GET_ROWS rows ids[r0] ..., the ids read by the RISC-V from the preload area
//      (I32; the executor joined the preload before op 0), one LD per run of consecutive ids when the
//      source rows are dense (src stride = rl), else one per id;
//   2. ST the block to the destination rows r0 .. (one ST for dense rows, else one per row).
// The ST needs no dep: the DMA queue runs in order, so it follows the tile's LDs (queue order is a
// happens-before edge, SW §4.6). The op's deps and the block's ring deps (without DMA tickets) go on
// the op's first LD; the returned ticket is the last ST; each block is released after its ST.
// A GET_ROWS id that is negative or puts the row beyond 2^32 beats gets the LD beat 0xFFFF_FFFF: the
// DMA unit reports DRAM_RANGE (ERR_INFO 0xFFFF_FFFF) and the launch ends with TILE_ERR on this op
// (a data error, SW §6.6: a code, never a panic); an id beyond the channel gives DRAM_RANGE as it is.
// The record's kind flags must be 0 (BAD_RECORD otherwise), as must every argument plan_elem.hpp
// refuses; GET_ROWS ids outside the list's preload area are BAD_RECORD (kind_ctx::pre_ok).
// Decode fast paths (k_elem.hpp's form): GET_ROWS of one id (a decode graph's inp_out_ids) and CPY of
// one row or a dense run, within half the budget: LD, ST (2 pushes).
#include "k_elem.hpp"

namespace tsr {

namespace ek {
namespace {

namespace gr = opl::getrows;
namespace cp = opl::cpy;

// The source row of GET_ROWS id `id` (src stride ss): beat 0xFFFF_FFFF (DRAM_RANGE) for a negative id
// or one beyond 2^32 beats.
TSR_HOT dram id_row(dram src, uint32_t ss, uint32_t id) {
    const uint64_t b = uint64_t(src.beat) + uint64_t(id) * ss;
    return (id & 0x8000'0000u) != 0u || b > 0xFFFF'FFFFu ? dram{src.ch, 0xFFFF'FFFFu} : dram{src.ch, uint32_t(b)};
}

struct cp_op {
    dram dst, src;
    uint32_t ds, ss;
    kind_plan kp;          // plan_cpy / plan_get_rows
    const volatile uint32_t * ids;   // GET_ROWS: the ids (preload area, exact SPM window: they are read
                                     // between pushes, which invalidate the line buffer); CPY: nullptr
    uint32_t pf;
};

// The source row of the op's row r (GET_ROWS: the row of ids[r]).
TSR_HOT dram src_row(const cp_op & c, uint32_t r) { return c.ids != nullptr ? id_row(c.src, c.ss, c.ids[r]) : c.src + r * c.ss; }

// The general path: tiles of t rows (or chunks of one row), each: alloc, its LDs (the op's first
// LD carries the op's deps and the block's ring deps, later ones the ring deps, all without DMA
// tickets), its STs (no deps: the DMA queue order), release after the last ST. A copy has no VU work
// to overlap, so the tiles are single-buffered.
[[gnu::noinline, gnu::cold]] ticket cp_run(const cp_op & c, deps d, kind_ctx & ctx) {
    spm_ring & ring = *ctx.ring;
    const uint32_t rl = c.kp.row_lines, rows = c.kp.aux[0];
    const bool chunked = ((c.kp.flags >> plan::KP_KIND_LSB) & plan::CP_CHUNKED) != 0u;
    const uint32_t t = chunked ? 1u : c.kp.tile_rows, chunk = chunked ? c.kp.aux[1] : rl;
    ticket last{};
    bool first_op = true;
    for (uint32_t r0 = 0; r0 < rows; r0 += t) {
        const uint32_t nr = rows - r0 < t ? rows - r0 : t;
        for (uint32_t off = 0; off < rl; off += chunk) {    // one pass unless chunked
            const uint32_t n = rl - off < chunk ? rl - off : chunk;
            // A copy tile is DMA only (LDs then STs): it takes the one-tile block whenever it fits
            // (kind_ctx::one; W2b integration): the DMA queue serialises the tiles whatever their lines.
            const bool one = nr * n <= ctx.one_lines;
            deps rd{};
            const spm blk = one ? ctx.one : alloc_g(ring, nr * n, &rd);
            if (!valid(blk)) return ctx.fail(opl::RET_SCRATCH);
            deps f = first_op ? first_deps(d, rd) : kf::drop_dma(rd);
            first_op = false;
            // LDs: packed rows (GET_ROWS: runs of consecutive ids when the source rows are dense).
            for (uint32_t i = 0; i < nr;) {
                const dram s = src_row(c, r0 + i);
                uint32_t run = 1u;
                if (!chunked && c.ss == rl) {
                    if (c.ids == nullptr) run = nr - i;
                    else
                        while (i + run < nr && c.ids[r0 + i + run] == c.ids[r0 + i] + run && c.ids[r0 + i] + run > c.ids[r0 + i]) ++run;
                }
                // A bad id's beat stays 0xFFFF_FFFF (DRAM_RANGE): no wrap into the channel.
                const dram so = s.beat == 0xFFFF'FFFFu ? s : dram{s.ch, s.beat + off};
                push_g(LD(blk + i * n, so, run * n), f, c.pf);
                f = deps{};
                i += run;
            }
            // STs: one for dense rows (or one row), else one per row.
            if (nr == 1u || c.ds == n) {
                last = push_g(ST(blk, c.dst + r0 * c.ds + off, nr * n), deps{}, c.pf);
            } else {
                for (uint32_t i = 0; i < nr; ++i) last = push_g(ST(blk + i * n, c.dst + (r0 + i) * c.ds + off, n), deps{}, c.pf);
            }
            if (!one) ring.release(blk, last);
        }
    }
    return last;
}

// LD, ST: the one-block pushes of both fast paths.
TSR_HOT ticket push_ld_st(bool chk, spm blk, dram src, dram dst, uint32_t lines, deps first, uint32_t pf) {
    kpush<UNIT_DMA>(chk, LD(blk, src, lines), first, pf);
    return kpush<UNIT_DMA>(chk, ST(blk, dst, lines), deps{}, pf);
}
// One block of `lines` lines: LD src, ST dst (both fast paths; inline in each: a shared copy measured
// ~22 RV cycles more per record).
TSR_HOT uint32_t copy_fast(dram src, dram dst, uint32_t lines, deps d, kind_ctx & ctx) {
    kf::block blk;
    if (!kf::alloc_fast(ctx, lines, &blk)) [[unlikely]] return NOT_FAST;
    const deps first = kf::first_deps(d);
    const uint32_t pf = ctx.push_flags;
    const ticket st = push_ld_st(need_chk(2u, 0u), blk.at, src, dst, lines, first, pf);
    kf::release(blk, st);
    return st.v;
}

[[gnu::noinline, gnu::cold]] ticket get_rows_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                     kind_ctx & ctx) {
    uint32_t w[gr::NARGS];
    for (uint32_t i = 0; i < gr::NARGS; ++i) w[i] = a[i];
    cp_op c{};
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    if (!plan_get_rows(w, n_args, ctx.pl, &c.kp)) return ctx.fail(c.kp.status);
    if (ctx.preload == nullptr || !ctx.pre_ok(w[gr::IDS], w[gr::N_IDS], 2u, opl::PRE_ID_ALIGN - 1u))
        return ctx.fail(opl::RET_BAD_RECORD);
    c.dst = from_handle(w[gr::DST]);
    c.ds = w[gr::DST_STRIDE];
    c.src = from_handle(w[gr::SRC]);
    c.ss = w[gr::SRC_STRIDE];
    c.ids = spm_ptr<uint32_t>(ctx.pre, w[gr::IDS]);   // exact window: ids are read across pushes
    c.pf = ctx.push_flags;
    return cp_run(c, d, ctx);
}

[[gnu::noinline, gnu::cold]] ticket cpy_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                kind_ctx & ctx) {
    uint32_t w[cp::NARGS];
    for (uint32_t i = 0; i < cp::NARGS; ++i) w[i] = a[i];
    cp_op c{};
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    if (!plan_cpy(w, n_args, ctx.pl, &c.kp)) return ctx.fail(c.kp.status);
    const bool merged = ((c.kp.flags >> plan::KP_KIND_LSB) & plan::CP_MERGED) != 0u;
    c.dst = from_handle(w[cp::DST]);
    c.ds = merged ? c.kp.row_lines : w[cp::DST_STRIDE];
    c.src = from_handle(w[cp::SRC]);
    c.ss = merged ? c.kp.row_lines : w[cp::SRC_STRIDE];
    c.ids = nullptr;
    c.pf = ctx.push_flags;
    return cp_run(c, d, ctx);
}

}  // namespace
}  // namespace ek

namespace ek {
namespace {
// GET_ROWS: the fast path takes one id whose row fits half the budget (a decode graph's
// inp_out_ids): LD the row, ST it (2 pushes).
TSR_HOT uint32_t get_rows_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t hd = a[gr::DST], hs = a[gr::SRC], ss = a[gr::SRC_STRIDE], ids = a[gr::IDS], n = a[gr::N_IDS],
                   rl = a[gr::ROW_LINES];
    if (kflags != 0u || n != 1u || rl - 1u >= (ctx.pl.budget >> 1) || ctx.preload == nullptr ||
        !ctx.pre_ok(ids, 1u, 2u, opl::PRE_ID_ALIGN - 1u)) [[unlikely]]
        return NOT_FAST;
    const uint32_t id = ctx.pre_lb(ids)[0];
    return copy_fast(id_row(from_handle(hs), ss, id), from_handle(hd), rl, d, ctx);
}
// CPY: the fast path takes one row, or rows dense on both sides, of fewer than 2^16 rows of fewer
// than 2^16 lines that fit half the budget (CP_MERGED, one tile): LD, ST (2 pushes).
TSR_HOT uint32_t cpy_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t hd = a[cp::DST], ds = a[cp::DST_STRIDE], hs = a[cp::SRC], ss = a[cp::SRC_STRIDE],
                   rl = a[cp::ROW_LINES], rows = a[cp::ROWS];
    if (kflags != 0u || rl - 1u >= 0xFFFFu || rows - 1u >= 0xFFFFu || (rows != 1u && (ss != rl || ds != rl))) [[unlikely]]
        return NOT_FAST;
    const uint32_t tot = rows * rl;
    if (tot > (ctx.pl.budget >> 1)) [[unlikely]] return NOT_FAST;
    return copy_fast(from_handle(hs), from_handle(hd), tot, d, ctx);
}
}  // namespace

#if defined(TSR_FW_HOST)
// Host builds only (fw_kernels_elem_host, k_elem.hpp).
ticket host_general_rows(uint32_t kind, const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                         kind_ctx & ctx) {
    return kind == opl::K_CPY ? cpy_general(a, n_args, kflags, d, ctx) : get_rows_general(a, n_args, kflags, d, ctx);
}
uint32_t host_fast_rows(uint32_t kind, const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    return kind == opl::K_CPY ? cpy_fast(a, kflags, d, ctx) : get_rows_fast(a, kflags, d, ctx);
}
#endif
}  // namespace ek

// The fast entries (exec.hpp kind_<name>_fast) and the handlers (the fast entry, else the general path).
[[gnu::noinline]] uint32_t kind_get_rows_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    return ek::get_rows_fast(a, kflags, d, ctx);
}
[[gnu::noinline]] uint32_t kind_cpy_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    return ek::cpy_fast(a, kflags, d, ctx);
}

ticket kind_get_rows(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t t = kind_get_rows_fast(a, kflags, d, ctx);
    if (t != ek::NOT_FAST) return ticket{t};
    return ek::get_rows_general(a, n_args, kflags, d, ctx);
}

ticket kind_cpy(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t t = kind_cpy_fast(a, kflags, d, ctx);
    if (t != ek::NOT_FAST) return ticket{t};
    return ek::cpy_general(a, n_args, kflags, d, ctx);
}

}  // namespace tsr
