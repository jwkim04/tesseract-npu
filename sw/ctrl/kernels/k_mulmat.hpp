// k_mulmat.hpp — shared pieces of the MUL_MAT kind and its decode fast path (docs/v2/
// 06-software-verification.md "SW" §7.4, §7.8); kernel-private, included by k_mulmat.cpp and by the
// graph executor (graph.cpp).
//
// mulmat_t1_fast: a MUL_MAT record with T = 1 (a decode token) in the common case, as a leaf
// function (no call: at -Os / -O2 a called handler with calls inside pays a callee-saved prologue
// and epilogue, ~50 RV cycles, plus spills). It pushes exactly what kind_mulmat pushes for the
// record (k_mulmat.cpp: alloc [x | QA | y] of lpt lines, LD x with the op's non-DMA deps, [LD
// addend / old y], QACT (dep LD x), MVM (deps QACT, LD addend), ST (dep MVM), release after the
// ST; for T = 1 the strides do not matter, row 0 only). It returns NOT_FAST, having changed
// nothing, when the record is not that case: T != 1, a record kind_mulmat refuses (BAD_RECORD,
// SCRATCH), or a block whose ring deps are not DMA-only (spm_ring::alloc_dma_first); the executor
// then calls kind_mulmat, which handles it (and gives the same result in the fast case too).
// The pushes skip the per-push credit read when one ISSUE_CREDIT read covers all of them
// (kernel.hpp credits_cover: per-unit fields and the issue-FIFO bound; the same pushes either way).
#pragma once

#include "tsr/exec.hpp"

namespace tsr {
namespace mmk {

namespace mm = opl::mm;

constexpr uint32_t NOT_FAST = 0xE000'0001u;   // unit 14: never a ticket (CC §7.4)

// d without its DMA tickets (for a DMA op: queue order implies them).
TSR_HOT deps drop_dma(deps d) {
    const ticket a = d.a.unit() == UNIT_DMA ? ticket{} : d.a;
    const ticket b = d.b.unit() == UNIT_DMA ? ticket{} : d.b;
    return after(a, b);
}

// Both present (rare: the ring's release tickets are STs, i.e. DMA): group folds them (k_mulmat.cpp).
deps first_deps_slow(deps dd, deps rr);

#if defined(TSR_FW_HOST)
// Host builds only (k_mulmat.cpp): kind_mulmat without the decode fast path (fw_kernels_host compares
// the two paths' pushes for T = 1 records).
ticket mulmat_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
#endif

// The deps of a tile's first LD: the op's deps and the block's ring deps, without DMA tickets.
TSR_HOT deps first_deps(deps d, deps rd) {
    const deps dd = drop_dma(d);
    if (rd.a.none()) return dd;
    const deps rr = drop_dma(rd);
    if (rr.a.none()) return dd;
    if (dd.a.none()) return rr;
    return first_deps_slow(dd, rr);
}

// The QACT of a tile (step 3); the P2.5 mutant images change it.
template <bool CHK>
TSR_HOT ticket push_qact(qa kind, spm q, spm x, uint32_t K, uint32_t tt, uint32_t x_lines, ticket ld_x, uint32_t pf) {
#if defined(TSR_KMM_MUTANT) && TSR_KMM_MUTANT == 1
    // P2.5 negative test image fw_kmm_mut_nodep: the QACT without its dep on the x LD (the race
    // detector must report it, sw/ctrl/tests/kernels.cmake).
    (void)ld_x;
    return issue_c<CHK>(QACT(kind, q, x, K, tt, x_lines), deps{}, pf);
#elif defined(TSR_KMM_MUTANT) && TSR_KMM_MUTANT == 2
    // P2.5 negative test image fw_kmm_mut_stride: src stride 0 (the same lines, but not v1's encoding;
    // the v1 stream comparison must report it).
    (void)x_lines;
    return issue_c<CHK>(QACT(kind, q, x, K, tt, 0u), deps{ld_x}, pf);
#else
    return issue_c<CHK>(QACT(kind, q, x, K, tt, x_lines), deps{ld_x}, pf);
#endif
}

// The pushes of a T = 1 op whose block is blk (CHK: issue() with its credit wait, else issue_nc).
template <bool CHK>
TSR_HOT ticket t1_push(spm blk, uint32_t xl, uint32_t ql, uint32_t yl, uint32_t ax, uint32_t ay, uint32_t aadd,
                       bool has_add, bool acc, wref w, uint32_t N, uint32_t K, deps first, uint32_t pf) {
    const spm q = blk + xl;
    const spm yb = q + ql;
    const ticket ld_x = issue_c<CHK>(LD(blk, from_handle(ax), xl), first, pf);
    ticket ld_add{};
    if (acc) ld_add = issue_c<CHK>(LD(yb, from_handle(has_add ? aadd : ay), yl), deps{}, pf);
    const ticket qt = push_qact<CHK>(qa_for(w), q, blk, K, 1u, xl, ld_x, pf);
    // after(qt, ld_add) = {qt, ld_add}: qt is a QACT ticket, ld_add none or a DMA one.
    const ticket mv = issue_c<CHK>(MVM(yb, q, w, N, K, 1u, yl, acc), deps{qt, ld_add}, pf);
    return issue_c<CHK>(ST(yb, from_handle(ay), yl), deps{mv}, pf);
}

// The decode fast path (header comment). kflags = word0 [23:16]; the executor has checked n_args.
// t0, t1: the predecessors' tickets (the op's deps are after(t0, t1)); the first LD's deps are
// drop_dma(after(t0, t1)) = after(t0 without DMA, t1 without DMA): both fold same-unit pairs to
// the later ticket and drop none, and removing DMA tickets before or after the fold leaves the same
// non-DMA ones (a DMA pair folds to a DMA ticket, which goes either way).
[[gnu::noinline]] inline uint32_t mulmat_t1_fast(const volatile uint32_t * a, uint32_t kflags, ticket t0, ticket t1,
                                                 kind_ctx & ctx) {
    if (a[mm::T] != 1u) return NOT_FAST;
    const uint32_t ax = a[mm::X], ay = a[mm::Y], woff = a[mm::W_OFF], fmt = a[mm::FMT], N = a[mm::N], K = a[mm::K];
    const uint32_t aadd = a[mm::ADD], fl = a[mm::FLAGS];
    const bool has_add = aadd != opl::HANDLE_NONE, acc_dst = (fl & mm::FLAG_ACC_DST) != 0u;
    if (kflags != 0u || (fl & ~mm::FLAG_ACC_DST) != 0u || (has_add && acc_dst)) return NOT_FAST;
    plan::mm_tiling tl;
    if (plan::mm_plan(ctx.mml, fmt, N, K, 1u, &tl) != opl::RET_OK) return NOT_FAST;
    uint32_t * rel = nullptr;
    const spm blk = ctx.ring->alloc_dma_first(tl.lpt, &rel);
    if (!valid(blk)) return NOT_FAST;
    // Committed from here on. The ring deps are DMA tickets only: implied for the first LD.
    const bool acc = has_add || acc_dst;
    const wref w{woff, uint8_t(fmt)};
    const deps first = after(t0.unit() == UNIT_DMA ? ticket{} : t0, t1.unit() == UNIT_DMA ? ticket{} : t1);
    const uint32_t pf = ctx.push_flags;
    // One ISSUE_CREDIT read for the op's 4 or 5 pushes when it covers them (kernel.hpp credits_cover),
    // else issue()'s read per push.
    const uint32_t c = credits(), n_dma = acc ? 3u : 2u;
    const ticket st = has_credits(c, UNIT_DMA, n_dma) && has_credits(c, UNIT_QACT, 1u) && has_credits(c, UNIT_MVM, 1u) &&
                              credits_cover(c, n_dma + 2u)
                          ? t1_push<false>(blk, tl.x_lines, tl.qa_lines, tl.y_lines, ax, ay, aadd, has_add, acc, w, N, K, first, pf)
                          : t1_push<true>(blk, tl.x_lines, tl.qa_lines, tl.y_lines, ax, ay, aadd, has_add, acc, w, N, K, first, pf);
    spm_ring::release_at(rel, st);
    return st.v;
}

}  // namespace mmk
}  // namespace tsr
