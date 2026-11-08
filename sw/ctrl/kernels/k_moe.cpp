// k_moe.cpp — the MoE kinds of the graph executor: MOE_ROUTE (0x1E) and MUL_MAT_ID (0x1F)
// (docs/design/06-software-verification.md "SW" §7.2, §7.5, SW-60 to SW-65; docs/spec.md §13; handler
// interface tsr/exec.hpp, planners tsr/plan_moe.hpp).
//
// Both kinds read data the device computed, with the RISC-V, so each handler joins inside the op
// stream: the one place a kind waits (exec.hpp says kinds never join; the router's top-k is the reason,
// spec §13 `join(issue(TOPK(...)))`). The wait is on the handler's own tile ops, which the units run
// while the RISC-V waits, and every record before it is already pushed; what stops is only the issue of
// the records after it.
//
// MOE_ROUTE, per tile of tr = min(T, 64) tokens (tsr/plan_moe.hpp has the ring block):
//   1. LD the logits rows (n_expert f32 each) into P;
//   2. the softmax on the VU, as tsr_ref_moe_route composes it: RED.MAX -> V, VOP.EXPSUBC (c = V) in
//      place, RED.SUM with post RCP -> V, VOP.MULC (c = V) in place: P holds the probabilities;
//   3. join the last VOP; the RISC-V selects, per token, the k largest probabilities by integer
//      compares on the bits (the VU TOPK's order: the DAZ'd value under the total order, NaN above
//      everything, ties by the smaller index; the TOPK RTL stays disabled) and writes the selected
//      values to the line SEL_i (words 0 .. k-1) and the expert numbers to the line IDL_i (words
//      k .. 15 are 0);
//   4. RED.SUM with post RCP over the k selected values (rows = tr, one line per row) -> V,
//      VOP.MULC (c = V) in place: SEL_i holds the weights;
//   5. join the last VOP; with tr > 1 the RISC-V packs the weights, k words per token, into PK (the
//      tensor the graph reads is packed, [k, T] f32); ST the ids lines (IDS + token * IDS_STRIDE lines,
//      one ST per token unless the rows are dense) and the weights.
// The numerics are tsr_ref_moe_route's: sw/ctrl/ref/moe_ref.cpp composes the same references.
//
// MUL_MAT_ID: per (token, slot) the MUL_MAT sequence of SW §7.4 with T = 1 (LD x, QACT, MVM, ST) against
// the weights of expert ids[token][slot], i.e. W_OFF + id * EXP_STRIDE (a 64-bit sum; an id outside
// [0, N_EXPERT) gets W_OFF 0xFFFF_FFFF, so the MVM fails with DRAM_RANGE and the launch ends with
// TILE_ERR on this op), optionally followed by VOP.MULC with s14 = SCALE[id] on the y block (one f32
// multiply, ggml's `res * s`). The ids come from DRAM: the handler LDs a chunk of them (and the scale
// table) into the one-tile block, joins, and reads them with the RISC-V; a token's pushes follow its
// ids. A token takes one ring block [x | QA | y_0 .. y_(k-1)] (the x row and the QA image are shared by
// the k slots when X_SLOT_STRIDE is 0), pushed so that the LD, QACT and MVM of slot j + 1 are queued
// before the ST of slot j (MUL_MAT's double buffering: the DMA queue is in order, an ST that waits for
// its MVM must not hold back the next LD), and the next token's block is allocated while this one's STs
// are pending (the ring's release tickets order its reuse). The returned ticket is the last ST.
// Per-token weight reuse is not attempted (SW-65: prefill reads each token's experts separately).
#include "k_mulmat.hpp"
#include "tsr/plan_moe.hpp"

// The op list kinds this source handles (SW §6.9: .tsr_kinds entries at tsr/opl.hpp kind_rev).
TSR_KIND(MOE_ROUTE);
TSR_KIND(MUL_MAT_ID);

namespace tsr {

namespace {

namespace mr = opl::moer;
namespace mi = opl::mmid;

constexpr uint32_t S_SCALE = 14u;   // the s register of the per-expert scale

// ---------------------------------------------------------------------------------------------
// The selection (integer only, SW-63). Same order as tsr_ref_topk (VU §6.8): x^ = DAZ(x) with every
// NaN as 0x7FC00000; larger x^ first under the total order (NaN above everything); a tie goes to the
// smaller index.
// ---------------------------------------------------------------------------------------------
constexpr uint32_t CANON_NAN = 0x7FC00000u;

constexpr uint32_t sel_value(uint32_t b) {
    const uint32_t e = b & 0x7F800000u;
    if (e == 0x7F800000u) return (b & 0x007FFFFFu) != 0u ? CANON_NAN : b;
    return e == 0u ? (b & 0x80000000u) : b;
}
// Order key of a selection value: the VU's key (sign ? ~bits : bits | 2^31) for numbers, the maximum for a NaN.
constexpr uint32_t sel_key(uint32_t v) {
    if (v == CANON_NAN) return 0xFFFFFFFFu;
    return (v & 0x80000000u) != 0u ? ~v : (v | 0x80000000u);
}

// The k largest of the n words at p (SPM, k <= 16): ids[j] and the values x^ in selection order.
[[gnu::noinline]] void select_topk(const volatile uint32_t * p, uint32_t n, uint32_t k, uint32_t * ids, uint32_t * vals) {
    uint32_t sk[mr::MAX_USED], si[mr::MAX_USED], sv[mr::MAX_USED];
    uint32_t cnt = 0u, kmin = 0u;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t v = sel_value(p[i]);
        const uint32_t key = sel_key(v);
        if (cnt == k && key <= kmin) continue;
        uint32_t pos = cnt < k ? cnt++ : k - 1u;
        while (pos > 0u && key > sk[pos - 1u]) {
            sk[pos] = sk[pos - 1u];
            si[pos] = si[pos - 1u];
            sv[pos] = sv[pos - 1u];
            --pos;
        }
        sk[pos] = key;
        si[pos] = i;
        sv[pos] = v;
        if (cnt == k) kmin = sk[k - 1u];
    }
    for (uint32_t j = 0; j < k; ++j) {
        ids[j] = si[j];
        vals[j] = sv[j];
    }
}

TSR_HOT op red_op(uint32_t kind, uint32_t post, spm d, spm a, uint32_t m, uint32_t rows, uint32_t a_stride) {
    red_f f{};
    f.kind = kind;
    f.post = post;
    f.d = d.line;
    f.a = a.line;
    f.m = m;
    f.rows = rows;
    f.a_stride = a_stride;
    return RED(f);
}
// d = func(a, c) per row, c_r = word r of the vector at line c (cmode 2); in place when d = a.
TSR_HOT op vop_vec(uint32_t func, spm d, spm a, spm c, uint32_t m, uint32_t rows, uint32_t stride) {
    vop_f f{};
    f.cmode = 2u;
    f.func = func;
    f.d = d.line;
    f.a = a.line;
    f.c = c.line;
    f.m = m;
    f.rows = rows;
    f.d_stride = stride;
    f.a_stride = stride;
    return VOP(f);
}

// ---------------------------------------------------------------------------------------------
// MOE_ROUTE
// ---------------------------------------------------------------------------------------------
[[gnu::noinline, gnu::cold]] ticket route_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                  kind_ctx & ctx) {
    uint32_t w[mr::NARGS];
    for (uint32_t i = 0; i < mr::NARGS; ++i) w[i] = a[i];
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    kind_plan kp{};
    if (!plan_moe_route(w, n_args, ctx.pl, &kp)) return ctx.fail(kp.status);
    const uint32_t n = w[mr::N_EXPERT], k = w[mr::N_USED], T = w[mr::T], lstride = w[mr::LOGITS_STRIDE],
                   istride = w[mr::IDS_STRIDE];
    const plan::route_geom g = plan::route_geometry(n, k, T);
    const dram logits = from_handle(w[mr::LOGITS]), ids_out = from_handle(w[mr::IDS]), w_out = from_handle(w[mr::W]);
    const uint32_t pf = ctx.push_flags;
    spm_ring & ring = *ctx.ring;
    ticket last{};
    bool first = true;
    for (uint32_t t0 = 0; t0 < T; t0 += mr::TILE_TOKENS) {
        const uint32_t tr = T - t0 < mr::TILE_TOKENS ? T - t0 : mr::TILE_TOKENS;
        const uint32_t pk_lines = plan::vu_lines(tr * k);
        deps rd;
        const uint32_t vl = plan::vu_lines(tr);
        const spm blk = kf::alloc_g(ring, tr * g.ln + vl + 2u * tr + (tr > 1u ? pk_lines : 0u), &rd);
        if (!valid(blk)) return ctx.fail(opl::RET_SCRATCH);
        const spm P = blk, V = P + tr * g.ln, SEL = V + vl, IDL = SEL + tr, PK = tr > 1u ? IDL + tr : SEL;
        // 1-2: the logits and the softmax
        const deps f = first ? mmk::first_deps(d, rd) : kf::drop_dma(rd);
        first = false;
        const ticket ld = kf::load_rows(kf::chk_t{}, P, logits, lstride, g.ln, t0, tr, f, pf);
        kf::push_g(red_op(RED_KIND_MAX, RED_POST_NONE, V, P, n, tr, g.ln), deps{ld}, pf);
        kf::push_g(vop_vec(VOP_FUNC_EXPSUBC, P, P, V, n, tr, g.ln), deps{}, pf);
        kf::push_g(red_op(RED_KIND_SUM, RED_POST_RCP, V, P, n, tr, g.ln), deps{}, pf);
        const ticket sm = kf::push_g(vop_vec(VOP_FUNC_MULC, P, P, V, n, tr, g.ln), deps{}, pf);
        join(sm);
        // 3: the selection
        uint32_t ids[mr::MAX_USED], vals[mr::MAX_USED];
        for (uint32_t i = 0; i < tr; ++i) {
            select_topk(spm_lb_ptr<uint32_t>(P + i * g.ln), n, k, ids, vals);
            volatile uint32_t * sel = spm_ptr<uint32_t>(SEL + i);
            volatile uint32_t * idl = spm_ptr<uint32_t>(IDL + i);
            for (uint32_t j = 0; j < k; ++j) {
                sel[j] = vals[j];
                idl[j] = ids[j];
            }
            for (uint32_t j = k; j < 16u; ++j) idl[j] = 0u;
        }
        // 4: the weights
        kf::push_g(red_op(RED_KIND_SUM, RED_POST_RCP, V, SEL, k, tr, 1u), deps{}, pf);
        const ticket rn = kf::push_g(vop_vec(VOP_FUNC_MULC, SEL, SEL, V, k, tr, 1u), deps{}, pf);
        join(rn);
        // 5: pack and store
        if (tr > 1u) {
            volatile uint32_t * pk = spm_ptr<uint32_t>(PK);
            for (uint32_t i = 0; i < tr; ++i) {
                const volatile uint32_t * s = spm_lb_ptr<uint32_t>(SEL + i);
                uint32_t v[mr::MAX_USED];
                for (uint32_t j = 0; j < k; ++j) v[j] = s[j];
                for (uint32_t j = 0; j < k; ++j) pk[i * k + j] = v[j];
            }
            for (uint32_t x = tr * k; x < 16u * pk_lines; ++x) pk[x] = 0u;
        }
        if (istride == 1u || tr == 1u) {
            kf::push_g(ST(IDL, ids_out + t0 * istride, tr == 1u ? 1u : tr), deps{}, pf);
        } else {
            for (uint32_t i = 0; i < tr; ++i) kf::push_g(ST(IDL + i, ids_out + (t0 + i) * istride, 1u), deps{}, pf);
        }
        last = kf::push_g(ST(PK, w_out + ((t0 * k) >> 4), tr > 1u ? pk_lines : 1u), deps{rn}, pf);
        ring.release(blk, last);
    }
    return last;
}

// ---------------------------------------------------------------------------------------------
// MUL_MAT_ID
// ---------------------------------------------------------------------------------------------
struct mmid_op {
    dram dst, x, ids, scale;
    uint32_t dst_stride, x_slot_stride, x_tok_stride, ids_stride, w_off, exp_stride, fmt, N, K, T, nu, ne;
    bool shared, has_scale;
    plan::mmid_geom g;
    uint32_t pf;
};

// The weights base of expert e (beats): W_OFF + e * EXP_STRIDE, or 0xFFFF_FFFF (the MVM fails with
// DRAM_RANGE) for an id outside the tensor.
TSR_HOT uint32_t expert_off(const mmid_op & m, uint32_t e) {
    const uint64_t o = uint64_t(m.w_off) + uint64_t(e) * m.exp_stride;
    return e < m.ne && o < 0xFFFF'FFFFull ? uint32_t(o) : 0xFFFF'FFFFu;
}

// The pushes of one (token, slot) after its MVM: the scale and the ST. Returns the ST.
ticket mmid_finish(const mmid_op & m, kind_ctx & ctx, spm y, uint32_t row, ticket mv, uint32_t scale_bits) {
    ticket t = mv;
    if (m.has_scale) {
        kf::sreg_movi(kf::chk_t{}, ctx, S_SCALE, scale_bits, m.pf);
        vop_f f{};
        f.cmode = 1u;
        f.func = VOP_FUNC_MULC;
        f.si = S_SCALE;
        f.d = y.line;
        f.a = y.line;
        f.m = m.N;
        f.rows = 1u;
        t = kf::push_g(VOP(f), deps{mv}, m.pf);
    }
    return kf::push_g(ST(y, m.dst + row * m.dst_stride, m.g.tl.y_lines), deps{t}, m.pf);
}

// One token's pipeline (header comment); e[j] / sb[j]: the slots' expert numbers and scale bits.
ticket mmid_token(const mmid_op & m, kind_ctx & ctx, uint32_t t, const uint32_t * e, const uint32_t * sb, deps first_in,
                  bool * ok) {
    spm_ring & ring = *ctx.ring;
    const plan::mm_tiling & tl = m.g.tl;
    deps rd;
    const spm blk = kf::alloc_g(ring, m.g.tok_lines, &rd);
    if (!valid(blk)) {
        *ok = false;
        return ticket{};
    }
    const uint32_t xs = m.g.xs;
    const spm X = blk, Q = X + xs * tl.x_lines, Y = Q + xs * tl.qa_lines;
    const dram xsrc = m.x + t * m.x_tok_stride;
    const qa kind = qa_for_fmt(m.fmt);
    const deps f = mmk::first_deps(first_in, rd);
    ticket mv[mr::MAX_USED], ld{}, qt{}, last{};
    auto front = [&](uint32_t j, deps fd) {              // LD, QACT, MVM of slot j
        const uint32_t xi = m.shared ? 0u : j;
        if (!m.shared || j == 0u) {
            ld = kf::push_g(LD(X + xi * tl.x_lines, xsrc + j * m.x_slot_stride, tl.x_lines), fd, m.pf);
            qt = kf::push_g(QACT(kind, Q + xi * tl.qa_lines, X + xi * tl.x_lines, m.K, 1u, tl.x_lines), deps{ld}, m.pf);
        }
        mv[j] = kf::push_g(MVM(Y + j * tl.y_lines, Q + xi * tl.qa_lines, wref{expert_off(m, e[j]), uint8_t(m.fmt)}, m.N,
                               m.K, 1u, tl.y_lines, false), deps{qt}, m.pf);
    };
    front(0u, f);
    for (uint32_t j = 0; j < m.nu; ++j) {
        if (j + 1u < m.nu) front(j + 1u, deps{});
        last = mmid_finish(m, ctx, Y + j * tl.y_lines, t * m.nu + j, mv[j], sb[j]);
    }
    ring.release(blk, last);
    return last;
}

[[gnu::noinline, gnu::cold]] ticket mmid_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                 kind_ctx & ctx) {
    uint32_t w[mi::NARGS];
    for (uint32_t i = 0; i < mi::NARGS; ++i) w[i] = a[i];
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    kind_plan kp{};
    if (!plan_mul_mat_id(w, n_args, ctx.pl, &kp)) return ctx.fail(kp.status);
    mmid_op m{};
    m.dst = from_handle(w[mi::DST]);
    m.x = from_handle(w[mi::X]);
    m.ids = from_handle(w[mi::IDS]);
    m.scale = from_handle(w[mi::SCALE]);
    m.dst_stride = w[mi::DST_STRIDE];
    m.x_slot_stride = w[mi::X_SLOT_STRIDE];
    m.x_tok_stride = w[mi::X_TOK_STRIDE];
    m.ids_stride = w[mi::IDS_STRIDE];
    m.w_off = w[mi::W_OFF];
    m.exp_stride = w[mi::EXP_STRIDE];
    m.fmt = w[mi::FMT];
    m.N = w[mi::N];
    m.K = w[mi::K];
    m.T = w[mi::T];
    m.nu = w[mi::N_USED];
    m.ne = w[mi::N_EXPERT];
    m.shared = m.x_slot_stride == 0u;
    m.has_scale = w[mi::SCALE] != opl::HANDLE_NONE;
    m.pf = ctx.push_flags;
    if (plan::mmid_geometry(ctx.mml, m.fmt, m.N, m.K, m.nu, m.shared, m.ne, m.has_scale, &m.g) != opl::RET_OK ||
        m.g.head_lines > ctx.one_lines)
        return ctx.fail(opl::RET_SCRATCH);
    // The head block is the executor's one-tile block: the ids chunk (ID_CHUNK_LINES) and the scale table.
    const spm idspm = ctx.one, tbl = ctx.one + plan::ID_CHUNK_LINES;
    deps first = mmk::first_deps(d, deps{});
    if (m.has_scale) {
        kf::push_g(LD(tbl, m.scale, plan::vu_lines(m.ne)), first, m.pf);
        first = deps{};
    }
    ticket last{};
    for (uint32_t c0 = 0; c0 < m.T;) {
        // tokens of this chunk: the ids rows [c0, c0 + tc) fit ID_CHUNK_LINES lines
        const uint64_t b0 = uint64_t(c0) * m.ids_stride;
        const uint32_t off0 = uint32_t(b0 & 63u);
        const uint32_t room = plan::ID_CHUNK_LINES * 64u - off0 - 4u * m.nu;
        uint32_t tc = 1u + room / (m.ids_stride != 0u ? m.ids_stride : 1u);
        if (tc > m.T - c0) tc = m.T - c0;
        const uint32_t lines = (off0 + (tc - 1u) * m.ids_stride + 4u * m.nu + 63u) >> 6;
        const ticket ldi = kf::push_g(LD(idspm, m.ids + uint32_t(b0 >> 6), lines), first, m.pf);
        first = deps{};
        join(ldi);
        for (uint32_t t = c0; t < c0 + tc; ++t) {
            uint32_t e[mi::MAX_USED], sb[mi::MAX_USED];
            const volatile uint32_t * ip = spm_lb_ptr<uint32_t>(idspm, off0 + (t - c0) * m.ids_stride);
            for (uint32_t j = 0; j < m.nu; ++j) e[j] = ip[j];
            for (uint32_t j = 0; j < m.nu; ++j) sb[j] = 0u;
            if (m.has_scale) {
                const volatile uint32_t * tp = spm_lb_ptr<uint32_t>(tbl);
                for (uint32_t j = 0; j < m.nu; ++j) sb[j] = e[j] < m.ne ? tp[e[j]] : 0u;
            }
            bool ok = true;
            last = mmid_token(m, ctx, t, e, sb, deps{}, &ok);
            if (!ok) return ctx.fail(opl::RET_SCRATCH);
        }
        c0 += tc;
    }
    return last;
}

}  // namespace

#if defined(TSR_FW_HOST)
// Host builds only (fw_kernels_moe_host): the RISC-V selection on plain memory.
void host_moe_select_topk(const uint32_t * p, uint32_t n, uint32_t k, uint32_t * ids, uint32_t * vals) {
    select_topk(p, n, k, ids, vals);
}
#endif

ticket kind_moe_route(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    return route_general(a, n_args, kflags, d, ctx);
}

ticket kind_mul_mat_id(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    return mmid_general(a, n_args, kflags, d, ctx);
}

}  // namespace tsr
