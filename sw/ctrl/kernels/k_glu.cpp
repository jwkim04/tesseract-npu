// k_glu.cpp — the GLU kind of the graph executor (op list kind 0x15, the split form; docs/design/
// 06-software-verification.md "SW" §7.2, §7.5; VU §5.2, §6.5; handler interface tsr/exec.hpp).
//
// dst = act(gate) * up per element, with the variant (word 8): 0 SWIGLU (VOP.SWIGLU: FM(SILU(gate),
// up), tsr_m_silu), 1 GEGLU16 (VOP.GEGLU16: the ggml-CPU fp16 GELU of VU §6.5, tsr_m_gelu16, what
// llama.cpp graphs use, SW-31), 2 GEGLU (VOP.GEGLU: FM(GELU(gate), up) with the exact tanh form,
// tsr_m_gelu_tanh; PyTorch). Tile-op sequence (k_elem.cpp's engine, plan_elem.hpp EB_ROWS):
// per row tile, LD the gate rows, LD the up rows (operand b, rows form: b stride L), the VOP in place
// on the gate rows, ST; sw/ctrl/ref/elem_ref.cpp composes the same tsr_ref_vop calls. The decode
// fast path (one tile, dense or single rows) is ek::ew2_fast (k_elem.hpp), the four pushes LD, LD,
// VOP, ST.
#include "k_elem.hpp"

// The op list kinds this source handles (SW §6.9: .tsr_kinds entries at tsr/opl.hpp kind_rev).
TSR_KIND(GLU);

namespace tsr {

namespace ek {

namespace glu = opl::glu;

[[gnu::noinline, gnu::cold]] ticket glu_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                kind_ctx & ctx) {
    uint32_t w[glu::NARGS];
    for (uint32_t i = 0; i < glu::NARGS; ++i) w[i] = a[i];
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    ew_op e{};
    if (!plan_glu(w, n_args, ctx.pl, &e.kp)) return ctx.fail(e.kp.status);
    e.func = GLU_FUNC[w[glu::VARIANT]];
    e.dst = from_handle(w[glu::DST]);
    e.ds = w[glu::DST_STRIDE];
    e.a = from_handle(w[glu::GATE]);
    e.as = w[glu::GATE_STRIDE];
    e.b = from_handle(w[glu::UP]);
    e.bs = w[glu::UP_STRIDE];
    e.n = w[glu::N];
    e.rows = w[glu::ROWS];
    e.pf = ctx.push_flags;
    return ew_run(e, d, ctx);
}

}  // namespace ek

// The fast entry (exec.hpp kind_glu_fast): ew2_fast (k_elem.hpp; always the rows form, plan_elem.hpp
// EB_ROWS: VOP b stride L, so an infinite period; an up stride of 0, which ew2_fast would take as the
// broadcast form, goes to the general path), with RING: a decode row too large for the one-tile
// block (Qwen2.5 1.5B-32B, Gemma 4 12B: 1,120-3,456 lines against 1,024) takes a ring block there.
// The handler: the fast entry, else the general path.
[[gnu::noinline]] uint32_t kind_glu_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t var = a[opl::glu::VARIANT];
    if (kflags != 0u || var > opl::glu::VARIANT_MAX || a[opl::glu::UP_STRIDE] == 0u) [[unlikely]] return ek::NOT_FAST;
    return ek::ew2_fast<ek::ew_words, true>(a, ek::GLU_FUNC[var], 0u, 0u, 0xFFFF'FFFFu, d.a.v, d.b.v, ctx);
}

ticket kind_glu(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t t = kind_glu_fast(a, kflags, d, ctx);
    if (t != ek::NOT_FAST) return ticket{t};
    return ek::glu_general(a, n_args, kflags, d, ctx);
}

}  // namespace tsr
