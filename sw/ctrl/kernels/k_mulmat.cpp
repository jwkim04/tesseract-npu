// k_mulmat.cpp — the MUL_MAT kind of the graph executor (op list kind 0x10; docs/design/
// 06-software-verification.md "SW" §7.2, §7.4; handler interface tsr/exec.hpp, SW §7.3).
//
// y = W x for T tokens, optionally y = addend + W x (a fused bias or residual, ggml's MUL_MAT + ADD)
// or y += W x (ACC_DST). The tile-op sequence is the spec's lowering (docs/spec.md:502-512),
// so the numerics match the converted traces bit for bit
// (traces/model-v1conv checks it, sw/ctrl/tests/test_fw_kernels*); per T tile (plan.hpp mm_plan:
// t_tile tokens, the last tile shorter):
//   1. LD x: the tile's x rows (K f32 each) into the tile's x block (x_lines apart; one LD when the
//      rows are dense in DRAM, x stride = x_lines, else one LD per token);
//   2. with an addend: LD the addend rows into the y block (one LD for dense rows, else one per
//      token; addend stride 0 repeats row 0 for every token); with ACC_DST: LD the old y rows;
//   3. QACT (A32 for Q8_0, A256 for Q4_K) of the tile's tokens, src stride x_lines;
//   4. MVM fmt, N, K, t_tile tokens, dst stride y_lines, ACC when step 2 loaded a row;
//   5. ST y (one ST for dense rows, y stride = y_lines, else one per token).
// Numerics: y = tsr_fadd(old, tsr_dot_row(W row, QACT(x))) with an addend or ACC_DST (one f32
// addition, commutative: bit-identical to ggml's separate ADD on the device), else the dot product;
// sw/ctrl/ref/mulmat_ref.cpp composes the same tsr_ref calls. A record with both an addend and
// ACC_DST is refused (BAD_RECORD: the y block holds one old row; decision P2.5), as are MUL_MAT
// kind flags (none is defined).
//
// Scratch (spm_ring, SW §6.7): one contiguous block per tile, [x | QA | y] (the spec's partition order),
// t_tile x lpt lines; plan.hpp sizes t_tile so that two tiles fit the planning budget (double
// buffering), else 1. The block's ring deps (releases of the older blocks it overlaps) go on the
// tile's first LD; every other write to the block follows it (DMA queue order, the QACT's and the
// MVM's deps), so it is ordered after them too. The block is released after the tile's ST, which
// implies every reader (QACT read x, MVM read QA, both done before the ST by its deps).
//
// Push order (docs/ctrl-riscv.md §8.7's matvec_batch pattern): LD x_0 [LD a_0], then per tile t:
// QACT_t, MVM_t, the prefetch of tile t+1 (alloc, LD x_{t+1} [LD a_{t+1}]), ST_t. So on the DMA
// queue the LD of tile t+1 precedes the ST of tile t and runs while MVM_t streams its weights; it
// waits only for its block's release (tile t-1's ST, earlier on the same queue): the double
// buffering of SW §7.4. When the ring cannot place tile t+1 while tile t's block is live (the ring
// is fragmented by earlier ops, or one tile takes more than half the ring), ST_t is pushed and its
// block released first, then tile t+1 is allocated (deterministic: it depends only on the
// allocation sequence, SW §4.7).
//
// Deps: the op's predecessors d go on the first DMA op (the first LD x); everything else follows
// it on the DMA queue or through the chain LD -> QACT -> MVM -> ST, so the whole op is after d
// (I1). Deps on the DMA unit are dropped from DMA ops (queue order implies them). The returned
// ticket is the last ST: it implies every tile op of the op (I1: the last ST follows the last MVM,
// which follows every earlier MVM, QACT and LD by queue order and deps).
//
// Cost (SW §7.3 "Hot path", §7.8; executor optimisation): a T = 1 record (a decode token) runs
// mmk::mulmat_t1_fast (k_mulmat.hpp), which the executor calls before kind_mulmat. Otherwise the
// argument words are read in place from the OPL area (exec.hpp) into locals before the first
// push; the plan is plan.hpp's mm_plan (shifts only, no divide when the T-tile is 1); an op of one
// T-tile (a prefill op with T <= the T-tile) takes a straight-line path, the multi-tile loop keeps
// the tile records; both are [[gnu::cold]] with their pushes out of line (code size). All paths
// push the same ops in the same order with the same words and deps as the W2a lowering (for one
// tile the loop is: alloc, load_tile, compute_tile, store_tile, release).
#include "k_mulmat.hpp"

namespace tsr {

namespace {

namespace mm = opl::mm;

struct mm_op {
    // arguments
    dram x, y, add;
    uint32_t x_stride, y_stride, add_stride;
    wref w;
    uint32_t N, K, T;
    bool has_add, acc_dst;
    // plan
    plan::mm_tiling tl;
    uint32_t push_flags;
};

// One tile's scratch block and tickets.
struct tile {
    uint32_t t0, tt;                 // first token, tokens
    spm blk;                         // [x | QA | y]
    ticket ld_x, ld_add, mvm;
};

using mmk::drop_dma;
using mmk::first_deps;

// The general path's pushes and ring allocations, out of line (code size: issue() and
// spm_ring::alloc are forced inline for the decode path, which this path is not). Same calls, so
// the same pushes.
using kf::alloc_g;   // the general path's out-of-line push and allocation (k_fast.hpp: one copy per image)
using kf::push_g;
[[gnu::noinline]] ticket qact_g(qa kind, spm q, spm x, uint32_t K, uint32_t tt, uint32_t x_lines, ticket ld_x, uint32_t pf) {
    return mmk::push_qact<true>(kind, q, x, K, tt, x_lines, ld_x, pf);
}

// Rows [t0, t0 + tt) of a row tensor (stride lines apart in DRAM; stride 0: row 0 repeated) into
// dst, `lines` per row: one LD when the rows are dense, else one per token. The first LD carries
// `first`; returns the last LD's ticket. (For tt = 1 both forms are the same LD.)
ticket load_rows(spm dst, dram src, uint32_t stride, uint32_t lines, uint32_t t0, uint32_t tt, deps first,
                         uint32_t pf) {
    if (tt == 1u || stride == lines) return push_g(LD(dst, src + t0 * stride, tt * lines), first, pf);
    ticket last{};
    for (uint32_t i = 0; i < tt; ++i)
        last = push_g(LD(dst + i * lines, src + (t0 + i) * stride, lines), i == 0u ? first : deps{}, pf);
    return last;
}
// y rows [t0, t0 + tt) from src (y_lines per token) to DRAM; the first ST carries `first`.
ticket store_rows(spm src, dram dst, uint32_t stride, uint32_t lines, uint32_t t0, uint32_t tt, deps first,
                          uint32_t pf) {
    if (tt == 1u || stride == lines) return push_g(ST(src, dst + t0 * stride, tt * lines), first, pf);
    ticket last{};
    for (uint32_t i = 0; i < tt; ++i)
        last = push_g(ST(src + i * lines, dst + (t0 + i) * stride, lines), i == 0u ? first : deps{}, pf);
    return last;
}

spm x_of(const mm_op &, const tile & t) { return t.blk; }
spm q_of(const mm_op & m, const tile & t) { return t.blk + t.tt * m.tl.x_lines; }
spm y_of(const mm_op & m, const tile & t) { return t.blk + t.tt * (m.tl.x_lines + m.tl.qa_lines); }

// Steps 1-2 of a tile whose block is allocated: the x rows, then the addend / old y rows.
[[gnu::cold]] void load_tile(const mm_op & m, tile & t, deps first) {
    t.ld_x = load_rows(x_of(m, t), m.x, m.x_stride, m.tl.x_lines, t.t0, t.tt, first, m.push_flags);
    t.ld_add = ticket{};
    if (m.has_add) t.ld_add = load_rows(y_of(m, t), m.add, m.add_stride, m.tl.y_lines, t.t0, t.tt, deps{}, m.push_flags);
    else if (m.acc_dst) t.ld_add = load_rows(y_of(m, t), m.y, m.y_stride, m.tl.y_lines, t.t0, t.tt, deps{}, m.push_flags);
}

// Steps 3-4.
[[gnu::cold]] void compute_tile(const mm_op & m, tile & t) {
    const ticket q = qact_g(qa_for(m.w), q_of(m, t), x_of(m, t), m.K, t.tt, m.tl.x_lines, t.ld_x,
                                          m.push_flags);
    const bool acc = m.has_add || m.acc_dst;
    t.mvm = push_g(MVM(y_of(m, t), q_of(m, t), m.w, m.N, m.K, t.tt, m.tl.y_lines, acc), after(q, t.ld_add),
                  m.push_flags);
}

// Step 5; returns the last ST.
[[gnu::cold]] ticket store_tile(const mm_op & m, const tile & t) {
    const uint32_t yl = m.tl.y_lines;
    if (m.y_stride == yl) return push_g(ST(y_of(m, t), m.y + t.t0 * yl, t.tt * yl), deps{t.mvm}, m.push_flags);
    ticket last{};
    for (uint32_t i = 0; i < t.tt; ++i)
        last = push_g(ST(y_of(m, t) + i * yl, m.y + (t.t0 + i) * m.y_stride, yl), i == 0u ? deps{t.mvm} : deps{},
                     m.push_flags);
    return last;
}

// More than one T-tile (prefill with T > t_tile): the double-buffered tile loop. This path and
// mulmat_any are [[gnu::cold]] (optimised for size): they run once per prefill op, whose tiles
// carry T_MAX tokens of MVM work each, not once per decode token.
[[gnu::noinline, gnu::cold]] ticket mulmat_tiles(const mm_op & m, deps d, kind_ctx & ctx) {
    spm_ring & ring = *ctx.ring;
    // Tile 0: its block, the op's deps and the block's ring deps on the first LD (DMA tickets are
    // implied by queue order; in the common case nothing else is left and no SYNC NOP is needed).
    tile tl[2];
    tile * cur = &tl[0], * nxt = &tl[1];
    cur->t0 = 0u;
    cur->tt = m.tl.t_tile < m.T ? m.tl.t_tile : m.T;
    deps rd;
    cur->blk = alloc_g(ring, cur->tt * m.tl.lpt, &rd);
    if (!valid(cur->blk)) {
        ctx.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    load_tile(m, *cur, first_deps(d, rd));

    ticket last{};
    for (;;) {
        compute_tile(m, *cur);
        const uint32_t next_t0 = cur->t0 + cur->tt;
        if (next_t0 >= m.T) {
            last = store_tile(m, *cur);
            ring.release(cur->blk, last);
            break;
        }
        nxt->t0 = next_t0;
        nxt->tt = m.T - next_t0 < m.tl.t_tile ? m.T - next_t0 : m.tl.t_tile;
        nxt->blk = alloc_g(ring, nxt->tt * m.tl.lpt, &rd);
        bool stored = false;
        if (!valid(nxt->blk)) {
            // No room beside the live block: store and release it first, then allocate.
            last = store_tile(m, *cur);
            ring.release(cur->blk, last);
            stored = true;
            nxt->blk = alloc_g(ring, nxt->tt * m.tl.lpt, &rd);
            if (!valid(nxt->blk)) {                   // cannot happen: one tile fits the ring
                ctx.ret = opl::RET_SCRATCH;
                return last;
            }
        }
        load_tile(m, *nxt, drop_dma(rd));
        if (!stored) {
            last = store_tile(m, *cur);
            ring.release(cur->blk, last);
        }
        tile * t = cur;
        cur = nxt;
        nxt = t;
    }
    return last;
}

// Any T (kind_mulmat's T != 1 case).
[[gnu::noinline, gnu::cold]] ticket mulmat_any(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                    kind_ctx & ctx) {
    // Every argument word first (one line-buffer fill; any push would invalidate the buffer).
    const uint32_t ax = a[mm::X], axs = a[mm::X_STRIDE], ay = a[mm::Y], ays = a[mm::Y_STRIDE], woff = a[mm::W_OFF];
    const uint32_t fmt = a[mm::FMT], N = a[mm::N], K = a[mm::K], T = a[mm::T];
    const uint32_t aadd = a[mm::ADD], aadds = a[mm::ADD_STRIDE], fl = a[mm::FLAGS];
    const bool has_add = aadd != opl::HANDLE_NONE, acc_dst = (fl & mm::FLAG_ACC_DST) != 0u;
    if (n_args != mm::NARGS || kflags != 0u || (fl & ~mm::FLAG_ACC_DST) != 0u || (has_add && acc_dst)) {
        ctx.ret = opl::RET_BAD_RECORD;
        return ticket{};
    }
    plan::mm_tiling tl;
    const uint32_t pr = plan::mm_plan(*ctx.cfg, fmt, N, K, T, &tl);
    if (pr != opl::RET_OK) {
        ctx.ret = pr;
        return ticket{};
    }
    const uint32_t pf = ctx.push_flags;
    const wref w{woff, uint8_t(fmt)};
    if (tl.n_tiles != 1u) {
        mm_op m;
        m.x = from_handle(ax);
        m.x_stride = axs;
        m.y = from_handle(ay);
        m.y_stride = ays;
        m.w = w;
        m.N = N;
        m.K = K;
        m.T = T;
        m.has_add = has_add;
        m.add = from_handle(aadd);
        m.add_stride = aadds;
        m.acc_dst = acc_dst;
        m.tl = tl;
        m.push_flags = pf;
        return mulmat_tiles(m, d, ctx);
    }

    // One tile of T tokens: [x | QA | y] in one block.
    spm_ring & ring = *ctx.ring;
    deps rd;
    const spm blk = alloc_g(ring, T * tl.lpt, &rd);
    if (!valid(blk)) {
        ctx.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    const spm q = blk + T * tl.x_lines;
    const spm yb = q + T * tl.qa_lines;
    const dram y = from_handle(ay);
    const ticket ld_x = load_rows(blk, from_handle(ax), axs, tl.x_lines, 0u, T, first_deps(d, rd), pf);
    ticket ld_add{};
    if (has_add) ld_add = load_rows(yb, from_handle(aadd), aadds, tl.y_lines, 0u, T, deps{}, pf);
    else if (acc_dst) ld_add = load_rows(yb, y, ays, tl.y_lines, 0u, T, deps{}, pf);
    const ticket qt = qact_g(qa_for(w), q, blk, K, T, tl.x_lines, ld_x, pf);
    const ticket mv = push_g(MVM(yb, q, w, N, K, T, tl.y_lines, has_add || acc_dst), after(qt, ld_add), pf);
    const ticket st = store_rows(yb, y, ays, tl.y_lines, 0u, T, deps{mv}, pf);
    ring.release(blk, st);
    return st;
}

}  // namespace

ticket kind_mulmat(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    if (n_args == mm::NARGS) {
        const uint32_t t = mmk::mulmat_t1_fast(a, kflags, d.a, d.b, ctx);   // the executor calls it first itself
        if (t != mmk::NOT_FAST) return ticket{t};
    }
    return mulmat_any(a, n_args, kflags, d, ctx);
}

#if defined(TSR_FW_HOST)
namespace mmk {
// Host builds only (fw_kernels_host): the general path alone, which a test compares push for push
// with mulmat_t1_fast on the same record and state.
ticket mulmat_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    return mulmat_any(a, n_args, kflags, d, ctx);
}
}  // namespace mmk
#endif

namespace mmk {
// Both the op's deps and ring deps (without DMA tickets): fold to <= 2 with group (SYNC NOPs).
deps first_deps_slow(deps dd, deps rr) { return kf::fold_deps(dd, rr); }
}  // namespace mmk

}  // namespace tsr
