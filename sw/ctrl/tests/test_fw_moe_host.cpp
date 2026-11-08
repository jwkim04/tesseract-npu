// test_fw_moe_host.cpp — fw_kernels_moe_host: the MoE kinds (sw/ctrl/kernels/k_moe.cpp) compiled for the host
// (docs/design/06-software-verification.md "SW" §6.8, SW-25: tsr_fw_host under ASan + UBSan). The records
// themselves run on tsr_rv in fw_kernels_moe_rv; this test holds what the host build can reach directly:
//
//   1. the planners (tsr/plan_moe.hpp): the refusals of their header comment, the tiling of MOE_ROUTE
//      (tiles of 64 tokens, ring block lines) and of MUL_MAT_ID (token block, head block) on a grid of
//      shapes, RET_SCRATCH exactly when the block exceeds the budget, plan_kind / plan_op dispatch,
//      opl::check_args;
//   2. the RISC-V selection (the kernel's integer top-k) against tsr_ref_topk on random words, on words from
//      the special set (NaN of every payload, ±inf, ±0, subnormals), on rows of few distinct values (ties)
//      and on every n 1 .. 70 with every k 1 .. min(n, 16): the expert numbers and the DAZ'd values equal.
// Exit 0 / 1.
#undef _GNU_SOURCE
#include "test_sim_util.h"

#include <vector>

#include "tsr/exec.hpp"
#include "tsr/plan_moe.hpp"
#include "tsr_ref_vu.h"

namespace tsr {
void host_moe_select_topk(const uint32_t * p, uint32_t n, uint32_t k, uint32_t * ids, uint32_t * vals);
}

namespace {

namespace opl = tsr::opl;
namespace pl = tsr::plan;
namespace mr = tsr::opl::moer;
namespace mi = tsr::opl::mmid;

void test_route_plans() {
    uint32_t a[mr::NARGS] = {0x10000000u, 8u, 0x20000000u, 8u, 0x30000000u, 128u, 8u, 1u, 0u};
    tsr::kind_plan kp{};
    CHECK(tsr::plan_moe_route(a, mr::NARGS, pl::LIMITS_U200, &kp) && kp.status == opl::RET_OK, "route plans");
    CHECK(kp.scratch_lines == 8u + 1u + 2u && kp.n_tiles == 1u && kp.tile_rows == 1u && kp.row_lines == 8u, "route T=1 block");
    for (uint32_t T : { 2u, 15u, 16u, 17u, 63u, 64u, 65u, 129u, 512u, 4096u }) {
        a[mr::T] = T;
        a[mr::LOGITS_STRIDE] = 8u;
        a[mr::IDS_STRIDE] = 8u;
        CHECK(tsr::plan_moe_route(a, mr::NARGS, pl::LIMITS_U200, &kp), "route T=%u plans", T);
        const uint32_t tr = T < 64u ? T : 64u;
        CHECK(kp.tile_rows == tr && kp.n_tiles == (T + 63u) / 64u, "route T=%u: %u tiles of %u", T, kp.n_tiles, kp.tile_rows);
        CHECK(kp.scratch_lines == tr * 8u + pl::vu_lines(tr) + 2u * tr + pl::vu_lines(tr * 8u), "route T=%u: block %u", T, kp.scratch_lines);
        CHECK(tsr::plan::plan_op(pl::U200_K4, opl::K_MOE_ROUTE, a, mr::NARGS).status == opl::RET_OK, "plan_op route");
    }
    // refusals
    auto refuse = [&](uint32_t word, uint32_t v, uint32_t status, const char * what) {
        uint32_t b[mr::NARGS];
        for (uint32_t i = 0; i < mr::NARGS; ++i) b[i] = a[i];
        b[word] = v;
        tsr::kind_plan k2{};
        const bool ok = tsr::plan_moe_route(b, mr::NARGS, pl::LIMITS_U200, &k2);
        CHECK(!ok && k2.status == status && k2.scratch_lines == pl::SCRATCH_UNSUPPORTED, "route %s: ok %d status 0x%x", what, ok,
              k2.status);
    };
    a[mr::T] = 4u;
    refuse(mr::FLAGS, 1u, opl::RET_BAD_RECORD, "flags");
    refuse(mr::N_USED, 0u, opl::RET_BAD_RECORD, "k = 0");
    refuse(mr::N_USED, 17u, opl::RET_BAD_RECORD, "k > 16");
    refuse(mr::N_USED, 129u, opl::RET_BAD_RECORD, "k > n");
    refuse(mr::N_EXPERT, 513u, opl::RET_BAD_RECORD, "n > 512");
    refuse(mr::T, 0u, opl::RET_BAD_RECORD, "T = 0");
    refuse(mr::LOGITS_STRIDE, 7u, opl::RET_BAD_RECORD, "overlapping logits rows");
    refuse(mr::IDS_STRIDE, 0u, opl::RET_BAD_RECORD, "overlapping ids rows");
    {   // no VU, and a budget that cannot hold one tile
        tsr::kind_plan k2{};
        pl::plan_limits l = pl::LIMITS_U200;
        l.caps = 0;
        CHECK(!tsr::plan_moe_route(a, mr::NARGS, l, &k2) && k2.status == opl::RET_BAD_RECORD, "route without VU");
        l = pl::LIMITS_U200;
        l.budget = 10u;
        CHECK(!tsr::plan_moe_route(a, mr::NARGS, l, &k2) && k2.status == opl::RET_SCRATCH, "route over the budget");
    }
    CHECK(opl::check_args(opl::K_MOE_ROUTE, a, mr::NARGS, 0u) == opl::RET_OK, "check_args route");
    a[mr::FLAGS] = 2u;
    CHECK(opl::check_args(opl::K_MOE_ROUTE, a, mr::NARGS, 0u) == opl::RET_BAD_RECORD, "check_args route flags");
}

void test_mmid_plans() {
    auto rec = [](uint32_t fmt, uint32_t N, uint32_t K, uint32_t T, uint32_t nu, uint32_t slot_stride, bool scale) {
        std::vector<uint32_t> a(mi::NARGS, 0u);
        const uint32_t xl = K / 16u;
        a[mi::DST] = 0x10000000u; a[mi::DST_STRIDE] = (N + 15u) / 16u; a[mi::X] = 0x20000000u; a[mi::X_SLOT_STRIDE] = slot_stride;
        a[mi::X_TOK_STRIDE] = slot_stride == 0u ? xl : nu * slot_stride; a[mi::IDS] = 0x30000000u; a[mi::IDS_STRIDE] = 512u;
        a[mi::W_OFF] = 0u; a[mi::EXP_STRIDE] = 4096u; a[mi::FMT] = fmt; a[mi::N] = N; a[mi::K] = K; a[mi::T] = T;
        a[mi::N_USED] = nu; a[mi::N_EXPERT] = 128u; a[mi::SCALE] = scale ? 0x40000000u : opl::HANDLE_NONE; a[mi::FLAGS] = 0u;
        return a;
    };
    tsr::kind_plan kp{};
    // Gemma 4 26B-A4B: gate-up (Q4_K 1408 x 2816, shared x) and down (Q8_0 2816 x 704, x per slot, scale)
    {
        auto a = rec(pl::FMT_Q4_K, 1408u, 2816u, 1u, 8u, 0u, false);
        CHECK(tsr::plan_mul_mat_id(a.data(), mi::NARGS, pl::LIMITS_U200, &kp), "gate-up plans");
        const uint32_t xl = 176u, ql = pl::qa_lines(pl::QA_A256, 2816u), yl = 88u;
        CHECK(kp.aux[0] == xl + ql + 8u * yl && kp.aux[1] == pl::ID_CHUNK_LINES && kp.aux[2] == 1u && kp.n_tiles == 1u,
              "gate-up block %u head %u xs %u", kp.aux[0], kp.aux[1], kp.aux[2]);
        CHECK(kp.scratch_lines == kp.aux[0] + kp.aux[1], "gate-up scratch");
    }
    {
        auto a = rec(pl::FMT_Q8_0, 2816u, 704u, 512u, 8u, 44u, true);
        CHECK(tsr::plan_mul_mat_id(a.data(), mi::NARGS, pl::LIMITS_U200, &kp), "down plans");
        const uint32_t xl = 44u, ql = pl::qa_lines(pl::QA_A32, 704u), yl = 176u;
        CHECK(kp.aux[0] == 8u * (xl + ql) + 8u * yl && kp.aux[1] == pl::ID_CHUNK_LINES + 8u && kp.aux[2] == 8u && kp.n_tiles == 512u,
              "down block %u head %u xs %u tiles %u", kp.aux[0], kp.aux[1], kp.aux[2], kp.n_tiles);
        CHECK(tsr::plan::plan_op(pl::U200_K4, opl::K_MUL_MAT_ID, a.data(), mi::NARGS).status == opl::RET_OK, "plan_op mmid");
    }
    // refusals
    auto bad = [&](std::vector<uint32_t> a, uint32_t word, uint32_t v, uint32_t status, const char * what) {
        a[word] = v;
        tsr::kind_plan k2{};
        const bool ok = tsr::plan_mul_mat_id(a.data(), mi::NARGS, pl::LIMITS_U200, &k2);
        CHECK(!ok && k2.status == status, "mmid %s: ok %d status 0x%x", what, ok, k2.status);
    };
    const auto base = rec(pl::FMT_Q8_0, 64u, 64u, 2u, 2u, 4u, true);
    bad(base, mi::FLAGS, 1u, opl::RET_BAD_RECORD, "flags");
    bad(base, mi::FMT, 2u, opl::RET_BAD_RECORD, "format");
    bad(base, mi::K, 48u, opl::RET_BAD_RECORD, "K % 32 != 0");
    bad(base, mi::N_USED, 0u, opl::RET_BAD_RECORD, "k = 0");
    bad(base, mi::N_USED, 17u, opl::RET_BAD_RECORD, "k > 16");
    bad(base, mi::N_EXPERT, 0u, opl::RET_BAD_RECORD, "no experts");
    bad(base, mi::N_EXPERT, 4097u, opl::RET_BAD_RECORD, "too many experts");
    bad(base, mi::T, 0u, opl::RET_BAD_RECORD, "T = 0");
    bad(base, mi::EXP_STRIDE, 0u, opl::RET_BAD_RECORD, "no expert stride");
    bad(base, mi::IDS_STRIDE, 6u, opl::RET_BAD_RECORD, "ids stride");
    bad(base, mi::IDS_STRIDE, 4u, opl::RET_BAD_RECORD, "ids rows overlap");
    bad(base, mi::X_SLOT_STRIDE, 3u, opl::RET_BAD_RECORD, "slot stride below x_lines");
    bad(base, mi::DST_STRIDE, 1u, opl::RET_BAD_RECORD, "dst rows overlap");
    {
        tsr::kind_plan k2{};
        pl::plan_limits l = pl::LIMITS_U200;
        auto a = base;
        l.budget = 100u;
        CHECK(!tsr::plan_mul_mat_id(a.data(), mi::NARGS, l, &k2) && k2.status == opl::RET_SCRATCH, "mmid over the budget");
        l = pl::LIMITS_U200;
        l.caps = 0;
        CHECK(!tsr::plan_mul_mat_id(a.data(), mi::NARGS, l, &k2) && k2.status == opl::RET_BAD_RECORD, "mmid scale without VU");
        a[mi::SCALE] = opl::HANDLE_NONE;
        CHECK(tsr::plan_mul_mat_id(a.data(), mi::NARGS, l, &k2), "mmid without scale needs no VU");
    }
    CHECK(opl::check_args(opl::K_MUL_MAT_ID, base.data(), mi::NARGS, 0u) == opl::RET_OK, "check_args mmid");
}

// ---------------------------------------------------------------------------
void test_select() {
    rng_t r = { 0x5E1Eull };
    const uint32_t specials[] = { 0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                                  0xFFC00001u, 0x7F800001u, 0xFFFFFFFFu, 0x3F800000u, 0xBF800000u, 0x00800000u, 0x80800000u,
                                  0x41200000u, 0xC1200000u };
    long rows = 0;
    for (uint32_t n = 1; n <= 70; ++n)
        for (uint32_t k = 1; k <= (n < 16u ? n : 16u); ++k)
            for (int mode = 0; mode < 3; ++mode) {
                std::vector<uint32_t> x(n);
                for (uint32_t i = 0; i < n; ++i)
                    x[i] = mode == 0 ? rng_u32(&r) : mode == 1 ? specials[rng_below(&r, 16)] : f2u(float(rng_below(&r, 4)) * 0.25f);
                uint32_t ids[16], vals[16], d[32];
                tsr::host_moe_select_topk(x.data(), n, k, ids, vals);
                tsr_ref_topk(k, 1, n, d, 0, x.data(), 0);
                for (uint32_t j = 0; j < k; ++j) {
                    CHECK(ids[j] == d[2 * j] && vals[j] == d[2 * j + 1], "select n %u k %u mode %d slot %u: (%u, 0x%08x) vs (%u, 0x%08x)",
                          n, k, mode, j, ids[j], vals[j], d[2 * j], d[2 * j + 1]);
                }
                ++rows;
            }
    for (int it = 0; it < 4000; ++it) {                 // the production size: 128 experts, 8 used, and 512 / 16
        const uint32_t n = it % 2 ? 128u : 512u, k = it % 2 ? 8u : 16u;
        std::vector<uint32_t> x(n);
        for (uint32_t i = 0; i < n; ++i) x[i] = (it % 5 == 0) ? specials[rng_below(&r, 16)] : (it % 5 == 1) ? f2u(float(rng_below(&r, 3))) : rng_u32(&r);
        uint32_t ids[16], vals[16], d[32];
        tsr::host_moe_select_topk(x.data(), n, k, ids, vals);
        tsr_ref_topk(k, 1, n, d, 0, x.data(), 0);
        for (uint32_t j = 0; j < k; ++j) CHECK(ids[j] == d[2 * j] && vals[j] == d[2 * j + 1], "select n %u it %d slot %u", n, it, j);
        ++rows;
    }
    printf("  select: %ld rows equal to tsr_ref_topk\n", rows);
}

}  // namespace

int main() {
    test_route_plans();
    test_mmid_plans();
    test_select();
    return test_finish("fw_kernels_moe_host");
}
