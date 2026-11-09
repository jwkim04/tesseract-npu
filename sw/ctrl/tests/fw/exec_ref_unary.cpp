// exec_ref_unary.cpp — the k_fast.hpp reference short kind on the production dispatch path
// (docs/design/06-software-verification.md "SW" §7.3 "Hot path", §7.8). Linked only into the test
// image fw_exec_ref (sw/ctrl/tests/exec.cmake: graph.cpp, k_mulmat.cpp, zz_kind_stubs.cpp, the
// runtime kernels and this file), where it is tsr::kind_unary: the executor
// reaches it through its kind table exactly as a production M7 kind (fw_exec's test kinds pay the
// test-kind lookup on top). fw_exec_perf_ref measures its decode record with the "short VU record"
// harness: the floor of a 3-push short kind written with k_fast.hpp.
//
// Not the UNARY kind (k_elem.cpp's is): one row only (rows = 1, any n of the VOP m
// field), LD src -> ring block [x | y], VOP func (TANH, GELU16, SILU, EXP, GELU) y = f(x), ST y;
// other records are refused (BAD_RECORD). A block the fast allocation refuses takes spm_ring::alloc
// and folds its deps (out of line).
#if defined(__GNUC__) && !defined(__clang__) && !defined(TSR_FW_HOST)
#pragma GCC optimize("O2")   // as the kind sources (SW §6.1 hot flags; a test image's extra sources get -Os)
#endif
#include "tsr/exec.hpp"
#include "tsr/kernel.hpp"
#include "../../kernels/k_fast.hpp"

namespace {

using namespace tsr;
namespace un = opl::unary;

// UNARY func -> VOP function (SW §7.2 UNARY: 0 TANH, 1 GELU16, 2 SILU, 3 EXP, 4 GELU).
constexpr uint8_t VFUNC[5] = {uint8_t(VOP_FUNC_TANH), uint8_t(VOP_FUNC_GELU16), uint8_t(VOP_FUNC_SILU),
                              uint8_t(VOP_FUNC_EXP), uint8_t(VOP_FUNC_GELU)};

template <class C>
TSR_HOT ticket ref_push(C tag, spm x, uint32_t L, uint32_t src, uint32_t dst, uint32_t m, uint32_t vf, deps first,
                        uint32_t pf) {
    const spm y = x + L;
    const ticket ld = kf::push(tag, LD(x, from_handle(src), L), first, pf);
    const ticket v = kf::push(tag, kf::vop_fast(kf::vop_rows(vf, y, x, spm{0}, m, 1u, L, L, 0u)), deps{ld}, pf);
    return kf::push(tag, ST(y, from_handle(dst), L), deps{v}, pf);
}

[[gnu::noinline, gnu::cold]] ticket ref_general(uint32_t src, uint32_t dst, uint32_t m, uint32_t L, uint32_t vf, deps d,
                                                kind_ctx & ctx) {
    deps w{};
    const spm x = ctx.ring->alloc(2u * L, 2u, &w);
    if (!valid(x)) return ctx.fail(opl::RET_SCRATCH);
    group g{};
    g.add(d);
    g.add(w);
    const ticket t = ref_push(kf::chk_t{}, x, L, src, dst, m, vf, g.as_deps(), ctx.push_flags);
    ctx.ring->release(x, t);
    return t;
}

}  // namespace

namespace tsr {
ticket kind_unary(const volatile uint32_t * a, uint32_t, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t dst = a[un::DST], src = a[un::SRC], m = a[un::N], rows = a[un::ROWS], func = a[un::FUNC];
    if (rows != 1u || func > un::FUNC_MAX || kflags != 0u || m - 1u >= plan::VU_M_MAX) return ctx.fail(opl::RET_BAD_RECORD);
    const uint32_t L = plan::vu_lines(m), vf = VFUNC[func];
    kf::block b;
    if (!kf::alloc_fast(ctx, 2u * L, &b)) return ref_general(src, dst, m, L, vf, d, ctx);
    const deps first = kf::first_deps(d);
    const uint32_t pf = ctx.push_flags;
    const ticket t = kf::push_batch(2u, 1u, 0u, 0u, 0u, [&](auto tag) { return ref_push(tag, b.at, L, src, dst, m, vf, first, pf); });
    kf::release(b, t);
    return t;
}
}  // namespace tsr
