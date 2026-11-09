// zz_kind_stubs.cpp — weak default definitions of the M7 kind handlers (tsr/exec.hpp kind_add ...
// kind_unary; docs/design/06-software-verification.md "SW" §7.2, §7.3). Each one refuses its
// record (ctx.ret = BAD_RECORD, no push: "this kind is not in the image"), so the executor's kind
// table can name every M7 handler, and an image built without
// a kind's source still links and answers BAD_RECORD for its records (SW §7.3 return codes).
//
// A kernel source defines the handler strongly (tsr::kind_<name> with the kind_fn signature), which
// replaces the stub:
//   - RISC-V images link every object of sw/ctrl/kernels (an object library): the strong
//     definition wins;
//   - the host build (tsr_fw_host, a static archive, SW §6.8) extracts, for an undefined symbol, the
//     first member in archive order that defines it, and the members are in sorted file order: this
//     file's name sorts after every kernel source (k_*.cpp, ...), so a member with the strong
//     definition is extracted first and the stub is used only when no member defines the handler.
// Not compiled for speed (not a hot-path source, SW §6.1): a stub runs at most once per launch.
#include "tsr/exec.hpp"

namespace tsr {

namespace {
[[gnu::noinline, gnu::cold]] ticket kind_missing(kind_ctx & ctx) { return ctx.fail(opl::RET_BAD_RECORD); }
}  // namespace

#define TSR_KIND_STUB(name)                                                                                 \
    [[gnu::weak]] ticket name(const volatile uint32_t *, uint32_t, uint32_t, deps, kind_ctx & ctx) {        \
        return kind_missing(ctx);                                                                           \
    }
TSR_KIND_STUB(kind_mulmat_scale)
TSR_KIND_STUB(kind_mulmat_softcap)
TSR_KIND_STUB(kind_add)
TSR_KIND_STUB(kind_mul)
TSR_KIND_STUB(kind_scale)
TSR_KIND_STUB(kind_rms_norm)
TSR_KIND_STUB(kind_glu)
TSR_KIND_STUB(kind_rope)
TSR_KIND_STUB(kind_get_rows)
TSR_KIND_STUB(kind_set_rows_kv)
TSR_KIND_STUB(kind_flash_attn)
TSR_KIND_STUB(kind_cpy)
TSR_KIND_STUB(kind_unary)
TSR_KIND_STUB(kind_moe_route)
TSR_KIND_STUB(kind_mul_mat_id)
TSR_KIND_STUB(kind_softplus)
TSR_KIND_STUB(kind_ssm_conv)
TSR_KIND_STUB(kind_gated_delta_net)
#undef TSR_KIND_STUB

// The fast-path entries (exec.hpp kind_<name>_fast): weakly "not taken", so the
// executor falls back to the kind's handler (the stub above when the kind is missing too).
#define TSR_FAST_STUB(name)                                                                                 \
    [[gnu::weak]] uint32_t name(const volatile uint32_t *, uint32_t, deps, kind_ctx &) { return KIND_NOT_FAST; }
TSR_FAST_STUB(kind_add_fast)
TSR_FAST_STUB(kind_mul_fast)
TSR_FAST_STUB(kind_scale_fast)
TSR_FAST_STUB(kind_rms_norm_fast)
TSR_FAST_STUB(kind_glu_fast)
TSR_FAST_STUB(kind_rope_fast)
TSR_FAST_STUB(kind_get_rows_fast)
TSR_FAST_STUB(kind_set_rows_kv_fast)
TSR_FAST_STUB(kind_cpy_fast)
TSR_FAST_STUB(kind_unary_fast)
#undef TSR_FAST_STUB

}  // namespace tsr
