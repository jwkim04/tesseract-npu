// k_fast.hpp — the fast-path toolkit of the short M7 kinds (docs/design/06-software-verification.md
// "SW" §7.3 "Hot path" (SW-35), §7.5, §7.8). Kernel-private: included by the kind
// sources in sw/ctrl/kernels (and by k_mulmat.hpp, whose decode fast path these pieces come from).
//
// The form (MUL_MAT's mulmat_t1_fast, 387 RV cycles of work per decode record; the requirement
// for every short kind: <= 350 at decode shapes, measured with fw_exec_perf's "short record"
// scenarios (fw_exec_perf_m7 --strict) and fw_prof.h):
//   1. read the argument words in place (args: the record in the OPL area through the line-buffered
//      SPM alias, exec.hpp) into locals, all of them before the first push (a push invalidates the
//      line buffer: one line fill per record);
//   2. decide "fast" with shifts and compares only (no divide: CC §3.2 / SW §4.2, 38 cycles each):
//      a single-tile op whose plan is the kind's (plan.hpp: the same rules as plan_<kind>, on the
//      locals, without the args copy); otherwise return NOT_FAST having changed nothing, and the
//      general path (out of line, [[gnu::cold]] / noinline) handles the record;
//   3. the op's block from alloc_fast: the executor's one-tile block (kind_ctx::one, plan.hpp
//      one_block_lines, SW-36), which a DMA-bracketed tile reuses without deps,
//      ring record or release (the DMA queue orders every reuse); too large -> NOT_FAST (GLU's rows
//      of the larger models take a ring block instead, k_elem.hpp ew2_fast RING);
//   4. one ISSUE_CREDIT read for the record's pushes (credits_for: per-unit fields and the
//      issue-FIFO bound, kernel.hpp credits_cover), else issue()'s read per push: push_batch runs
//      the push sequence (a generic lambda taking a chk_t / nochk_t tag) in one of the two forms,
//      which push the same words, deps and flags (SW §4.7);
//   5. the first push (an LD) carries first_deps(d): the op's deps without DMA tickets;
//   6. nothing to release (kf::release is empty: kept for the call sites' symmetry);
//   7. no calls in the common path: everything here is forced inline (TSR_HOT), the kind's fast
//      entry (exec.hpp kind_<name>_fast, called by the executor first; SW-37) is a leaf, and its
//      refusal tests are [[unlikely]] (else GCC lays them out as the fall-through);
//   8. VU words without run-time range checks for planned fields (vop_fast, red_fast, sop_movi: the
//      generated builders spend ~3 instructions per field on checks), and no SOP.MOVI for an s
//      register that already holds the value (sreg_movi, the launch's s-register shadow in
//      kind_ctx: RMS_NORM's 1/n and eps, ROPE's TS / M repeat across records);
//   9. no ISSUE_W store for a word the previous push of the record staged with the same
//      value, and no ISSUE_TICKET read for a ticket nothing uses (push_s below; CC §5.1); a kind whose
//      common record differs from its general fast form keeps a straight-line hot entry for it and
//      tail-calls the general fast form (k_norm.cpp, k_rope.cpp: one entry for every shape measured
//      about 60 RV cycles more, register pressure).
// The kind sources are compiled -O2 (SW §6.1 TSR_FW_HOT_GLOBS: kernels/k_*.cpp).
//
// Determinism (SW §4.7, SW-30): ISSUE_CREDIT only chooses between the two push forms; the ring
// state, the record words and the tickets decide everything pushed.
#pragma once

#include "tsr/exec.hpp"

namespace tsr {
namespace kf {

// Returned by a fast path that does not take the record (unit 14: never a ticket, CC §7.4). The
// same value as mmk::NOT_FAST.
constexpr uint32_t NOT_FAST = KIND_NOT_FAST;

// d without its DMA tickets (for a DMA op: queue order implies them).
TSR_HOT deps drop_dma(deps d) {
    const ticket a = d.a.unit() == UNIT_DMA ? ticket{} : d.a;
    const ticket b = d.b.unit() == UNIT_DMA ? ticket{} : d.b;
    return after(a, b);
}
// The deps of a fast op's first push when it is a DMA op (LD) on the block from alloc_fast (the
// one-tile block: its earlier users' last ops are DMA ops, implied): the op's deps without DMA tickets. d is the executor's
// d = after(t0, t1) (exec.hpp kind_fn): already folded (a none implies b none; a and b on different
// units), so removing the DMA ticket needs no second fold (= drop_dma(d) for every folded d).
TSR_HOT deps first_deps(deps d) {
    ticket a = d.a, b = d.b;
    if (a.unit() == UNIT_DMA) {
        a = b;
        b = ticket{};
    } else if (b.unit() == UNIT_DMA) {
        b = ticket{};
    }
    return deps{a, b};
}

// The op's block (step 3).
struct block {
    spm at;
};
// The one-tile block (plan.hpp one_block_lines, kind_ctx::one) for a tile of
// `lines` lines that is DMA-bracketed (plan.hpp layout comment: its first write an LD, its last read
// a DMA op, every other op between them by deps); false (nothing changed) when it does not fit (the
// caller's general path then uses spm_ring::alloc). No ring record, no deps: the DMA queue order
// orders every reuse of these lines.
TSR_HOT bool alloc_fast(kind_ctx & ctx, uint32_t lines, block * out) {
    out->at = ctx.one;
    return lines <= ctx.one_lines;
}
// Nothing to release: the next DMA-bracketed tile's first LD is ordered after this tile's last DMA
// op by the DMA queue (kept for the call sites' symmetry).
TSR_HOT void release(const block &, ticket) {}

// Step 4: one ISSUE_CREDIT read c covers pushes of n_dma DMA ops, n_vu VU ops, n_mvm MVM ops,
// n_qact QACT ops and n_sync SYNC ops when every unit's field holds its count and the issue FIFO
// holds them all (kernel.hpp credits_cover). Counts that are 0 at compile time fold away.
TSR_HOT bool credits_for(uint32_t c, uint32_t n_dma, uint32_t n_vu, uint32_t n_mvm = 0u, uint32_t n_qact = 0u,
                         uint32_t n_sync = 0u) {
    return (n_dma == 0u || has_credits(c, UNIT_DMA, n_dma)) && (n_vu == 0u || has_credits(c, UNIT_VU, n_vu)) &&
           (n_mvm == 0u || has_credits(c, UNIT_MVM, n_mvm)) && (n_qact == 0u || has_credits(c, UNIT_QACT, n_qact)) &&
           (n_sync == 0u || has_credits(c, UNIT_SYNC, n_sync)) &&
           credits_cover(c, n_dma + n_vu + n_mvm + n_qact + n_sync);
}

// The two push forms: issue_c<chk_t::v> with issue()'s credit wait, issue_c<nochk_t::v> without.
struct chk_t { static constexpr bool v = true; };
struct nochk_t { static constexpr bool v = false; };
template <class C> TSR_HOT ticket push(C, const op & o, deps d, uint32_t pf) { return issue_c<C::v>(o, d, pf); }

// Staged-word reuse (CC §5.1: ISSUE_W0-W7 keep their values after a push, and a push sends
// word i, for WMASK bit i set, from ISSUE_Wi). push_s<SKIP, TICKET> is issue_nc (the nochk_t push: the
// caller's one credit read covered it) without the ISSUE_W stores of the words in SKIP, and without
// the ISSUE_TICKET read unless TICKET (CC §5.1: reading it is optional; the ticket is then not
// returned). SKIP: words the previous push_s of the same push sequence staged with the same value,
// by the sequence's construction (the same field of two consecutive ops, e.g. the a line and m of a
// VOP after a RED on the same rows; the caller states why at the call). WMASK stays op::used, so the
// words pushed are exactly the op's. Host builds (TSR_FW_HOST) store every word (the simulator hook)
// and check the claim: each SKIP word was staged by a push_s since the sequence's stage_reset() and
// holds the op's value, else panic(PANIC_STAGED): the host twin tests run every fast path's sequence.
// One control-window store (3 RV cycles) per skipped word.
constexpr uint32_t W1 = 1u << 1, W2 = 1u << 2, W3 = 1u << 3, W4 = 1u << 4, W5 = 1u << 5, W6 = 1u << 6, W7 = 1u << 7;
constexpr uint32_t PANIC_STAGED = 0x5354'0001u;   // a push_s SKIP word is not the staged one (host check)
#if defined(TSR_FW_HOST)
namespace detail {
inline uint32_t g_staged[8] = {};
inline uint32_t g_staged_known = 0u;   // words a push_s staged since the last stage_reset()
}  // namespace detail
// The start of a push sequence (its first push stages everything it sends; host check only).
TSR_HOT void stage_reset() { detail::g_staged_known = 0u; }
#else
TSR_HOT void stage_reset() {}
#endif
template <uint32_t SKIP = 0u, bool TICKET = true> TSR_HOT ticket push_s(const op & o, deps d, uint32_t pf) {
    static_assert((SKIP & 1u) == 0u && SKIP < 0x100u, "push_s: w0 is always staged; eight words");
    const uint32_t used = o.used;
#if defined(TSR_FW_HOST)
    for (uint32_t i = 1; i < 8u; ++i) {
        const uint32_t m = 1u << i;
        if ((SKIP & used & m) != 0u && ((detail::g_staged_known & m) == 0u || detail::g_staged[i] != o.w[i]))
            panic(PANIC_STAGED, i, o.w[i], detail::g_staged[i]);
    }
    for (uint32_t i = 0; i < 8u; ++i)
        if (i == 0u || ((used >> i) & 1u) != 0u) detail::g_staged[i] = o.w[i];
    detail::g_staged_known |= used | 1u;
    const ticket t = issue_nc(o, d, pf);
    return TICKET ? t : ticket{};
#else
    volatile uint32_t * w = hw::cw();
    const uint32_t st = used & ~SKIP;
    w[CW_ISSUE_W(0) / 4u] = o.w[0];
    if (st & 0x02u) w[CW_ISSUE_W(1) / 4u] = o.w[1];
    if (st & 0x04u) w[CW_ISSUE_W(2) / 4u] = o.w[2];
    if (st & 0x08u) w[CW_ISSUE_W(3) / 4u] = o.w[3];
    if (st & 0x10u) w[CW_ISSUE_W(4) / 4u] = o.w[4];
    if (st & 0x20u) w[CW_ISSUE_W(5) / 4u] = o.w[5];
    if (st & 0x40u) w[CW_ISSUE_W(6) / 4u] = o.w[6];
    if (st & 0x80u) w[CW_ISSUE_W(7) / 4u] = o.w[7];
    if ((d.a.v >> 28) != UNIT_NONE) w[CW_ISSUE_DEP0 / 4u] = d.a.v;
    if ((d.b.v >> 28) != UNIT_NONE) w[CW_ISSUE_DEP1 / 4u] = d.b.v;
    w[CW_ISSUE_PUSH / 4u] = (used << ISSUE_PUSH_WMASK_LSB) | (pf & ~ISSUE_PUSH_WMASK);
    if constexpr (TICKET) return tsr::detail::pushed(w[CW_ISSUE_TICKET / 4u]);
    else return ticket{};
#endif
}

// Runs f(tag) once, tag = nochk_t{} when one credit read covers the counts, else chk_t{}; f pushes
// the record's ops with push(tag, ...) and returns the last ticket. Both forms are instantiated
// (the push code appears twice, as mulmat_t1_fast's t1_push<false / true>).
template <class F>
TSR_HOT ticket push_batch(uint32_t n_dma, uint32_t n_vu, uint32_t n_mvm, uint32_t n_qact, uint32_t n_sync, F && f) {
    return credits_for(credits(), n_dma, n_vu, n_mvm, n_qact, n_sync) ? f(nochk_t{}) : f(chk_t{});
}

// A VOP of one row-block (VU §5.2) with the per-op fields filled; the caller sets func / cmode / si
// / c. rows <= 1: strides unused (VU §4.2), passed as given.
TSR_HOT constexpr vop_f vop_rows(uint32_t func, spm d, spm a, spm b, uint32_t m, uint32_t rows, uint32_t d_stride,
                       uint32_t a_stride, uint32_t b_stride) {
    vop_f f{};
    f.func = func;
    f.d = d.line;
    f.a = a.line;
    f.b = b.line;
    f.m = m;
    f.rows = rows;
    f.d_stride = d_stride;
    f.a_stride = a_stride;
    f.b_stride = b_stride;
    return f;
}

// VOP / RED / SOP words without the generated builders' run-time range checks (vu.hpp: about three
// instructions per field, 36 per VOP), for fields the kind's plan has already bounded (lines and m
// < 2^24, rows and strides < 2^16, func / cmode / si / kind / sd / sk / se their enumerations): then
// the words are exactly vu.hpp's (static_asserts below), so the pushes are the same. A field out of
// range here would be masked into another value: never use them on unplanned values.
TSR_HOT constexpr op vop_fast(const vop_f & f) {
    op o{};
    o.w[0] = OP_VOP | (f.cmode << 8) | (f.func << 16) | (f.si << 24);
    o.w[1] = f.d;
    o.w[2] = f.a;
    o.w[3] = f.b;
    o.w[4] = f.c;
    o.w[5] = f.m;
    o.w[6] = f.rows | (f.d_stride << 16);
    o.w[7] = f.a_stride | (f.b_stride << 16);
    o.used = uint8_t(VOP_WORDS);
    return o;
}
TSR_HOT constexpr op red_fast(const red_f & f) {
    op o{};
    o.w[0] = OP_RED | (f.kind << RED_KIND_LSB) | (f.to_s << RED_TO_S_LSB) | (f.sd << RED_SD_LSB) |
             (f.post << RED_POST_LSB) | (f.sk << RED_SK_LSB) | (f.se << RED_SE_LSB);
    o.w[1] = f.d;
    o.w[2] = f.a;
    o.w[5] = f.m;
    o.w[6] = f.rows;
    o.w[7] = f.a_stride;
    o.used = uint8_t(RED_WORDS);
    return o;
}
// SOP.MOVI s[sd] = imm (VU §5.8), the words of vu.hpp's SOP for sd < 16 (static_assert below).
TSR_HOT constexpr op sop_movi(uint32_t sd, uint32_t imm) {
    op o{};
    o.w[0] = OP_SOP | (SOP_SUB_MOVI << 8);
    o.w[1] = sd;
    o.w[2] = imm;
    o.used = uint8_t(SOP_WORDS);
    return o;
}

// The s-register shadow (exec.hpp kind_ctx::sreg_valid; VU §4.4). sreg_holds: s[i] already holds
// `bits` for every VU op pushed from now on (no MOVI needed); sreg_need: the MOVI count (0 / 1) for
// a credit count; sreg_movi: s[i] = bits for the VU ops pushed after it, pushing SOP.MOVI only when
// the shadow does not already hold it (no deps: VU queue order orders it before the record's VU
// ops); sreg_clobber: a push of this record writes s[i] otherwise (RED TO_S, SOP).
TSR_HOT bool sreg_holds(const kind_ctx & ctx, uint32_t i, uint32_t bits) {
    return ((ctx.sreg_valid >> i) & 1u) != 0u && ctx.sreg_val[i] == bits;
}
TSR_HOT uint32_t sreg_need(const kind_ctx & ctx, uint32_t i, uint32_t bits) { return sreg_holds(ctx, i, bits) ? 0u : 1u; }
template <class C> TSR_HOT void sreg_movi(C tag, kind_ctx & ctx, uint32_t i, uint32_t bits, uint32_t pf) {
    if (sreg_holds(ctx, i, bits)) return;
    push(tag, sop_movi(i, bits), deps{}, pf);
    ctx.sreg_val[i] = bits;
    ctx.sreg_valid |= 1u << i;
}
TSR_HOT void sreg_clobber(kind_ctx & ctx, uint32_t i) { ctx.sreg_valid &= ~(1u << i); }

namespace detail {
constexpr bool same_op(const op & x, const op & y) {
    if (x.used != y.used) return false;
    for (uint32_t i = 0; i < 8u; ++i)
        if (x.w[i] != y.w[i]) return false;
    return true;
}
constexpr vop_f vop_sample(uint32_t cmode, uint32_t func, uint32_t si) {
    vop_f f{};
    f.cmode = cmode;
    f.func = func;
    f.si = si;
    f.d = 0xFFFFFFu;
    f.a = 0x123456u;
    f.b = 7u;
    f.c = cmode == 2u ? 0xABCDEFu : 0u;
    f.m = 0xFFFFFFu;
    f.rows = 0xFFFFu;
    f.d_stride = 0xFFFFu;
    f.a_stride = 0x8001u;
    f.b_stride = 0u;
    return f;
}
constexpr red_f red_sample(uint32_t kind, uint32_t to_s, uint32_t post) {
    red_f f{};
    f.kind = kind;
    f.to_s = to_s;
    f.sd = to_s ? 15u : 0u;
    f.post = post;
    f.sk = post == RED_POST_RMS ? 2u : 0u;
    f.se = post == RED_POST_RMS ? 15u : 0u;
    f.d = to_s ? 0u : 0xFFFFFFu;
    f.a = 0x10u;
    f.m = 896u;
    f.rows = to_s ? 1u : 0xFFFFu;
    f.a_stride = 56u;
    return f;
}
}  // namespace detail
static_assert(detail::same_op(vop_fast(detail::vop_sample(0, VOP_FUNC_ADD, 0)), VOP(detail::vop_sample(0, VOP_FUNC_ADD, 0))) &&
                  detail::same_op(vop_fast(detail::vop_sample(1, VOP_FUNC_MULC, 15)), VOP(detail::vop_sample(1, VOP_FUNC_MULC, 15))) &&
                  detail::same_op(vop_fast(detail::vop_sample(2, VOP_FUNC_MULCB, 0)), VOP(detail::vop_sample(2, VOP_FUNC_MULCB, 0))) &&
                  detail::same_op(vop_fast(detail::vop_sample(0, VOP_FUNC_GEGLU16, 0)), VOP(detail::vop_sample(0, VOP_FUNC_GEGLU16, 0))),
              "k_fast.hpp: vop_fast = vu.hpp VOP for in-range fields");
static_assert(detail::same_op(red_fast(detail::red_sample(RED_KIND_SUMSQ, 1, RED_POST_RMS)),
                              RED(detail::red_sample(RED_KIND_SUMSQ, 1, RED_POST_RMS))) &&
                  detail::same_op(red_fast(detail::red_sample(RED_KIND_SUM, 0, RED_POST_NONE)),
                                  RED(detail::red_sample(RED_KIND_SUM, 0, RED_POST_NONE))) &&
                  detail::same_op(red_fast(detail::red_sample(RED_KIND_MAX, 0, RED_POST_NONE)),
                                  RED(detail::red_sample(RED_KIND_MAX, 0, RED_POST_NONE))),
              "k_fast.hpp: red_fast = vu.hpp RED for in-range fields");
static_assert(detail::same_op(sop_movi(15u, 0x3F800000u), SOP(sop_f{SOP_SUB_MOVI, 15u, 0u, 0u, 0x3F800000u, 0u})) &&
                  detail::same_op(sop_movi(0u, 0xFFFFFFFFu), SOP(sop_f{SOP_SUB_MOVI, 0u, 0u, 0u, 0xFFFFFFFFu, 0u})),
              "k_fast.hpp: sop_movi = vu.hpp SOP MOVI");

// The general paths' out-of-line helpers, shared by every kernel source (inline
// functions are emitted once per image (COMDAT), so the image holds one copy instead of one per
// source, -3 KiB of ITCM): issue() with its credit wait, spm_ring::alloc (any parity), and two dep
// pairs (both without DMA tickets) folded to <= 2 with group (SYNC NOPs when more units remain).
[[gnu::noinline]] inline ticket push_g(const op & o, deps d, uint32_t pf) { return issue(o, d, pf); }
[[gnu::noinline]] inline spm alloc_g(spm_ring & ring, uint32_t lines, deps * rd) { return ring.alloc(lines, 2u, rd); }
[[gnu::noinline]] inline deps fold_deps(deps dd, deps rr) {
    group g;
    g.add(dd);
    g.add(rr);
    return g.as_deps();
}

// Rows [r0, r0 + n) of a DRAM row tensor (stride lines apart; stride == lines: dense) into dst,
// `lines` lines per row, packed: one LD when dense or n = 1, else one LD per row (the first carries
// `first`). Returns the last LD's ticket. Out of the fast path's common case only when n > 1 and
// the rows are not dense.
template <class C>
TSR_HOT ticket load_rows(C tag, spm dst, dram src, uint32_t stride, uint32_t lines, uint32_t r0, uint32_t n, deps first,
                         uint32_t pf) {
    if (n == 1u || stride == lines) return push(tag, LD(dst, src + r0 * stride, n * lines), first, pf);
    ticket last{};
    for (uint32_t i = 0; i < n; ++i)
        last = push(tag, LD(dst + i * lines, src + (r0 + i) * stride, lines), i == 0u ? first : deps{}, pf);
    return last;
}
// The mirror image: n packed rows of `lines` lines from src to DRAM rows [r0, r0 + n).
template <class C>
TSR_HOT ticket store_rows(C tag, spm src, dram dst, uint32_t stride, uint32_t lines, uint32_t r0, uint32_t n, deps first,
                          uint32_t pf) {
    if (n == 1u || stride == lines) return push(tag, ST(src, dst + r0 * stride, n * lines), first, pf);
    ticket last{};
    for (uint32_t i = 0; i < n; ++i)
        last = push(tag, ST(src + i * lines, dst + (r0 + i) * stride, lines), i == 0u ? first : deps{}, pf);
    return last;
}

}  // namespace kf
}  // namespace tsr
