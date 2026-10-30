// k_mulmat.cpp — the MUL_MAT kind of the graph executor (OPL v1 kind 0x10; docs/v2/
// 06-software-verification.md "SW" §7.2, §7.4; handler interface tsr/exec.hpp, SW §7.3).
//
// y = W x for T tokens, optionally y = addend + W x (a fused bias or residual, ggml's MUL_MAT + ADD)
// or y += W x (ACC_DST). The tile-op sequence is v1's lowering (docs/spec.md:502-512,
// v1-m4:sw/ggml-tesseract/tsr-lower.cpp tsr_mm_emit), so the numerics are v1's bit for bit
// (traces/model-v1conv checks it, sw/ctrl/tests/test_fw_kernels*); per T tile (plan.hpp mm_plan:
// t_tile tokens, the last tile shorter):
//   1. LD x: the tile's x rows (K f32 each) into the tile's x block (x_lines apart; one LD when the
//      rows are dense in DRAM, x stride = x_lines, else one LD per token);
//   2. with an addend: LD the addend rows into the y block (one LD for dense rows, else one per
//      token; addend stride 0 repeats row 0 for every token); with ACC_DST: LD the old y rows;
//   3. QACT (A32 for Q8_0, A256 for Q4_K) of the tile's tokens, src stride x_lines (as v1);
//   4. MVM fmt, N, K, t_tile tokens, dst stride y_lines (as v1), ACC when step 2 loaded a row;
//   5. ST y (one ST for dense rows, y stride = y_lines, else one per token).
// Numerics: y = tsr_fadd(old, tsr_dot_row(W row, QACT(x))) with an addend or ACC_DST (one f32
// addition, commutative: bit-identical to ggml's separate ADD on the device), else the dot product;
// sw/ctrl/ref/mulmat_ref.cpp composes the same tsr_ref calls. A record with both an addend and
// ACC_DST is refused (BAD_RECORD: the y block holds one old row; decision P2.5), as are MUL_MAT
// kind flags (none is defined).
//
// Scratch (spm_ring, SW §6.7): one contiguous block per tile, [x | QA | y] (v1's partition order),
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
#include "tsr/exec.hpp"

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

// d without its DMA tickets (for a DMA op: queue order implies them).
deps drop_dma(deps d) {
    const ticket a = d.a.unit() == UNIT_DMA ? ticket{} : d.a;
    const ticket b = d.b.unit() == UNIT_DMA ? ticket{} : d.b;
    return after(a, b);
}

spm x_of(const mm_op &, const tile & t) { return t.blk; }
spm q_of(const mm_op & m, const tile & t) { return t.blk + t.tt * m.tl.x_lines; }
spm y_of(const mm_op & m, const tile & t) { return t.blk + t.tt * (m.tl.x_lines + m.tl.qa_lines); }

// Rows [t0, t0 + tt) of a row tensor (stride lines apart in DRAM; stride 0: row 0 repeated) into
// dst, `lines` per row: one LD when the rows are dense, else one per token. The first LD carries
// `first`; returns the last LD's ticket.
ticket load_rows(const mm_op & m, spm dst, dram src, uint32_t stride, uint32_t lines, uint32_t t0, uint32_t tt,
                 deps first) {
    if (stride == lines) return issue(LD(dst, src + t0 * stride, tt * lines), first, m.push_flags);
    ticket last{};
    for (uint32_t i = 0; i < tt; ++i) {
        last = issue(LD(dst + i * lines, src + (t0 + i) * stride, lines), i == 0u ? first : deps{}, m.push_flags);
    }
    return last;
}

// Steps 1-2 of a tile whose block is allocated: the x rows, then the addend / old y rows.
void load_tile(const mm_op & m, tile & t, deps first) {
    t.ld_x = load_rows(m, x_of(m, t), m.x, m.x_stride, m.tl.x_lines, t.t0, t.tt, first);
    t.ld_add = ticket{};
    if (m.has_add) t.ld_add = load_rows(m, y_of(m, t), m.add, m.add_stride, m.tl.y_lines, t.t0, t.tt, deps{});
    else if (m.acc_dst) t.ld_add = load_rows(m, y_of(m, t), m.y, m.y_stride, m.tl.y_lines, t.t0, t.tt, deps{});
}

// Steps 3-4.
void compute_tile(const mm_op & m, tile & t) {
#if defined(TSR_KMM_MUTANT) && TSR_KMM_MUTANT == 1
    // P2.5 negative test image fw_kmm_mut_nodep: the QACT without its dep on the x LD (the race
    // detector must report it, sw/ctrl/tests/kernels.cmake).
    const ticket q = issue(QACT(qa_for(m.w), q_of(m, t), x_of(m, t), m.K, t.tt, m.tl.x_lines), deps{}, m.push_flags);
#elif defined(TSR_KMM_MUTANT) && TSR_KMM_MUTANT == 2
    // P2.5 negative test image fw_kmm_mut_stride: src stride 0 (the same lines, but not v1's encoding;
    // the v1 stream comparison must report it).
    const ticket q = issue(QACT(qa_for(m.w), q_of(m, t), x_of(m, t), m.K, t.tt, 0u), deps{t.ld_x}, m.push_flags);
#else
    const ticket q = issue(QACT(qa_for(m.w), q_of(m, t), x_of(m, t), m.K, t.tt, m.tl.x_lines), deps{t.ld_x},
                           m.push_flags);
#endif
    const bool acc = m.has_add || m.acc_dst;
    t.mvm = issue(MVM(y_of(m, t), q_of(m, t), m.w, m.N, m.K, t.tt, m.tl.y_lines, acc), after(q, t.ld_add),
                  m.push_flags);
}

// Step 5; returns the last ST.
ticket store_tile(const mm_op & m, const tile & t) {
    const uint32_t yl = m.tl.y_lines;
    if (m.y_stride == yl) return issue(ST(y_of(m, t), m.y + t.t0 * yl, t.tt * yl), deps{t.mvm}, m.push_flags);
    ticket last{};
    for (uint32_t i = 0; i < t.tt; ++i)
        last = issue(ST(y_of(m, t) + i * yl, m.y + (t.t0 + i) * m.y_stride, yl), i == 0u ? deps{t.mvm} : deps{},
                     m.push_flags);
    return last;
}

}  // namespace

ticket kind_mulmat(const uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    const bool has_add = a[mm::ADD] != opl::HANDLE_NONE, acc_dst = (a[mm::FLAGS] & mm::FLAG_ACC_DST) != 0u;
    if (n_args != mm::NARGS || kflags != 0u || (a[mm::FLAGS] & ~mm::FLAG_ACC_DST) != 0u || (has_add && acc_dst)) {
        ctx.ret = opl::RET_BAD_RECORD;
        return ticket{};
    }
    mm_op m;                                           // every field set below (no zero-fill: SW §7.8)
    const uint32_t pr = plan::mm_plan(*ctx.cfg, a[mm::FMT], a[mm::N], a[mm::K], a[mm::T], &m.tl);
    if (pr != opl::RET_OK) {
        ctx.ret = pr;
        return ticket{};
    }
    m.x = from_handle(a[mm::X]);
    m.x_stride = a[mm::X_STRIDE];
    m.y = from_handle(a[mm::Y]);
    m.y_stride = a[mm::Y_STRIDE];
    m.w = wref{a[mm::W_OFF], uint8_t(a[mm::FMT])};
    m.N = a[mm::N];
    m.K = a[mm::K];
    m.T = a[mm::T];
    m.has_add = has_add;
    m.add = from_handle(a[mm::ADD]);
    m.add_stride = a[mm::ADD_STRIDE];
    m.acc_dst = acc_dst;
    m.push_flags = ctx.push_flags;
    spm_ring & ring = *ctx.ring;

    // Tile 0: its block, the op's deps and the block's ring deps on the first LD (DMA tickets are
    // implied by queue order; in the common case nothing else is left and no SYNC NOP is needed).
    tile tl[2];
    tile * cur = &tl[0], * nxt = &tl[1];
    cur->t0 = 0u;
    cur->tt = m.tl.t_tile < m.T ? m.tl.t_tile : m.T;
    deps rd;
    cur->blk = ring.alloc(cur->tt * m.tl.lpt, 2u, &rd);
    if (!valid(cur->blk)) {
        ctx.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    const deps dd = drop_dma(d), rr = drop_dma(rd);
    if (rr.a.none()) {
        load_tile(m, *cur, dd);
    } else if (dd.a.none()) {
        load_tile(m, *cur, rr);
    } else {
        group g;
        g.add(dd);
        g.add(rr);
        load_tile(m, *cur, g.as_deps());
    }

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
        nxt->blk = ring.alloc(nxt->tt * m.tl.lpt, 2u, &rd);
        bool stored = false;
        if (!valid(nxt->blk)) {
            // No room beside the live block: store and release it first, then allocate.
            last = store_tile(m, *cur);
            ring.release(cur->blk, last);
            stored = true;
            nxt->blk = ring.alloc(nxt->tt * m.tl.lpt, 2u, &rd);
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

}  // namespace tsr
