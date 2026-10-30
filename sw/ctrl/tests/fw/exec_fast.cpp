// exec_fast.cpp — the executor tests' fast-path reference kind (W2b infra; docs/v2/06-software-
// verification.md "SW" §7.3 "Hot path", §7.8): test kind 0x39 FASTVU, the k_fast.hpp form of a
// short VU record, linked into the fw_exec test image and the host driver with exec_kinds.cpp
// (which dispatches 0x39 here). fw_exec_perf's "short VU record" scenarios measure it beside the
// M7 kinds of the image at decode shapes: it is the floor a short kind reaches with the toolkit
// (plus the test-kind lookup, tsr_exec_test_kind, which production kinds do not pay).
//
//   0x39 FASTVU a0 src handle, a1 dst handle, a2 m (f32 elements, 1..65,535 x 16), a3 0
//               LD x (L = ceil(m / 16) lines) into a ring block [x | y], VOP ADD x + x -> y (one
//               row), ST y: 3 pushes, each after the previous; ticket: the ST. The k_fast.hpp form:
//               words read in place, the plan by shifts, the block from alloc_fast, one credit read
//               (push_batch), vop_fast (the VOP's words without the builder's range checks: the plan
//               bounded them), the block released after the ST; a block the fast allocation refuses
//               (a ring wrap, non-DMA release tickets) takes spm_ring::alloc and its deps (the
//               general path, out of line).
//   0x3A SREG   a0 src, a1 dst, a2 m, a3 s index i (0..15), a4 value v (f32 bits), a5 flags (bit0
//               CLOBBER)  s[i] = v through the launch's s-register shadow (k_fast.hpp sreg_movi:
//               SOP.MOVI only when the shadow does not hold v), LD x, VOP MULC y = x * s[i] (cmode
//               1), ST y; with CLOBBER then RED.SUM TO_S s[i] = sum(x) (after the ST) and
//               sreg_clobber(i): the next record wanting v in s[i] must push its MOVI again. Ticket:
//               the ST, or the RED (which depends on it). fw_exec's sc_m7 checks y = x * v and
//               counts the SOP pushes (W2b infra, the executor-level s-register rule).
//
// Compiled with -O2 like the kind sources (SW §6.1 TSR_FW_HOT_GLOBS covers sw/ctrl/kernels/k_*.cpp
// only; a test image's extra sources get -Os): the pragma below.
#if defined(__GNUC__) && !defined(__clang__) && !defined(TSR_FW_HOST)
#pragma GCC optimize("O2")
#endif
#include "tsr/exec.hpp"
#include "tsr/kernel.hpp"
#include "../../kernels/k_fast.hpp"

namespace {

using namespace tsr;

// The pushes (tag: kf::chk_t / kf::nochk_t).
template <class C>
TSR_HOT ticket fastvu_push(C tag, spm x, uint32_t L, uint32_t src, uint32_t dst, uint32_t m, deps first, uint32_t pf) {
    const spm y = x + L;
    const ticket ld = kf::push(tag, LD(x, from_handle(src), L), first, pf);
    const ticket v = kf::push(tag, kf::vop_fast(kf::vop_rows(VOP_FUNC_ADD, y, x, x, m, 1u, L, L, L)), deps{ld}, pf);
    return kf::push(tag, ST(y, from_handle(dst), L), deps{v}, pf);
}

// The general path: spm_ring::alloc with its deps folded into the LD's (as exec_kinds.cpp's VU).
[[gnu::noinline, gnu::cold]] ticket fastvu_general(uint32_t src, uint32_t dst, uint32_t m, uint32_t L, deps d,
                                                   kind_ctx & ctx) {
    deps w{};
    const spm x = ctx.ring->alloc(2u * L, 2u, &w);
    if (!valid(x)) return ctx.fail(opl::RET_SCRATCH);
    group g{};
    g.add(d);
    g.add(w);
    const ticket t = fastvu_push(kf::chk_t{}, x, L, src, dst, m, g.as_deps(), ctx.push_flags);
    ctx.ring->release(x, t);
    return t;
}

}  // namespace

namespace tsr {
ticket exec_test_fastvu(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & ctx) {
    const uint32_t src = a[0], dst = a[1], m = a[2], z = a[3];   // every word before the first push
    if (m - 1u >= 16u * 0xFFFFu || z != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    const uint32_t L = plan::vu_lines(m);
    if (2u * L > ctx.pl.budget) return ctx.fail(opl::RET_SCRATCH);
    kf::block b;
    if (!kf::alloc_fast(ctx, 2u * L, &b)) return fastvu_general(src, dst, m, L, d, ctx);
    const deps first = kf::first_deps(d);
    const uint32_t pf = ctx.push_flags;
    const ticket t = kf::push_batch(2u, 1u, 0u, 0u, 0u, [&](auto tag) { return fastvu_push(tag, b.at, L, src, dst, m, first, pf); });
    kf::release(b, t);
    return t;
}
ticket exec_test_sreg(const volatile uint32_t * a, uint32_t, uint32_t, deps d, kind_ctx & ctx) {
    const uint32_t src = a[0], dst = a[1], m = a[2], si = a[3], v = a[4], fl = a[5];
    if (m - 1u >= 16u * 0xFFFFu || si > 15u || fl > 1u) return ctx.fail(opl::RET_BAD_RECORD);
    const uint32_t L = plan::vu_lines(m);
    deps w{};
    const spm x = ctx.ring->alloc(2u * L + 1u, 2u, &w);
    if (!valid(x)) return ctx.fail(opl::RET_SCRATCH);
    const spm y = x + L;
    group g{};
    g.add(d);
    g.add(w);
    const uint32_t pf = ctx.push_flags;
    kf::sreg_movi(kf::chk_t{}, ctx, si, v, pf);
    const ticket ld = issue(LD(x, from_handle(src), L), g.as_deps(), pf);
    vop_f f = kf::vop_rows(VOP_FUNC_MULC, y, x, spm{0}, m, 1u, L, L, 0u);
    f.cmode = 1u;
    f.si = si;
    const ticket vo = issue(kf::vop_fast(f), deps{ld}, pf);
    ticket t = issue(ST(y, from_handle(dst), L), deps{vo}, pf);
    if (fl & 1u) {
        red_f r{};
        r.kind = RED_KIND_SUM;
        r.to_s = 1u;
        r.sd = si;
        r.a = x.line;
        r.m = m;
        r.rows = 1u;
        t = issue(kf::red_fast(r), deps{t}, pf);
        kf::sreg_clobber(ctx, si);
    }
    ctx.ring->release(x, t);
    return t;
}
}  // namespace tsr
