// k_elem.cpp — the element-wise kinds ADD (0x11), MUL (0x12), SCALE (0x13) and UNARY (0x1B) of the
// graph executor, and the element-wise tile engine they share with GLU (k_glu.cpp)
// (docs/design/06-software-verification.md "SW" §7.2, §7.5; VU §5.2, §6.5, §9; handler interface
// tsr/exec.hpp, SW §7.3; fast path SW §7.3 "Hot path", k_elem.hpp).
//
// Tile-op sequence (the numerical contract, SW §7.5 last paragraph; sw/ctrl/ref/elem_ref.cpp
// composes the same tsr_ref_vu calls). Per row tile [r0, r0 + t) (tsr/plan_elem.hpp: kind_plan
// tile_rows; one tile at decode shapes), in one block [a rows | b part] (the one-tile block for an op's
// only tile, SW-36; else the ring):
//   1. LD the a rows (one LD for dense rows, a stride = L(n), else one per row);
//   2. LD the b part (plan_elem.hpp EB_*): b rows r0 .. (rows form), b row 0 (broadcast), b rows
//      0 .. p-1 (period), the scalar's line (tile 0 only), the per-row scalars' lines from r0/16;
//      with one scalar for more than one row, SOP.LDS s12 <- that word (tile 0);
//   3. the VOP(s), in place on the a rows (d = a, stride L):
//        ADD / MUL   VOP.ADD / MUL, b stride L (rows) or 0 (broadcast); one VOP per period block
//                    of p rows (period form, VU §9); with a scalar b VOP.ADDC / MULC, c = s12
//                    (cmode 1) or the c vector of the tile's rows (cmode 2: one row, per-row
//                    scalars, one VOP per period block of the period form)
//        SCALE       (s12 <- s, and s13 <- b when b != 0 (bits), once per op before tile 0's LD:
//                    SOP.MOVI unless the launch's s-register shadow holds the value, k_elem.hpp)
//                    VOP.MULC c = s12, then VOP.ADDC c = s13 when b != 0: y = FA(FM(x, s), b),
//                    the bits of VOP.MULCADD with a b row of copies of b (SW §7.5), without the
//                    b row (the record carries b as an immediate, which only
//                    an s register can take without a scratch row)
//        UNARY       VOP.TANH / GELU16 / SILU / EXP / GELU / SIGMOID (func 0 .. 5)
//   4. ST the rows (one ST for dense dst rows, else one per row); the VOP tail rule zeroes words
//      [n, 16 L) of each row, and the ST writes whole lines (VU §4.2).
// Numerics per element (VU §6): FA / FM (tsr_fadd / tsr_fmul) for ADD, MUL, ADDC, MULC; SCALE
// FA(FM(x, s), b); UNARY tsr_m_tanh / tsr_m_gelu16 / tsr_m_silu / tsr_m_exp / tsr_m_gelu_tanh; DAZ
// inputs, FTZ, canonical NaN (VU §4.1). Rows and tiles never change a value.
//
// The record's kind flags (word0 [23:16]) must be 0 (none is defined): BAD_RECORD otherwise, as
// every argument the planner refuses. A shape whose one aligned tile does not fit the ring budget
// is RET_SCRATCH.
#include "k_elem.hpp"

// The op list kinds this source handles (SW §6.9: .tsr_kinds entries at tsr/opl.hpp kind_rev).
TSR_KIND(ADD);
TSR_KIND(MUL);
TSR_KIND(SCALE);
TSR_KIND(UNARY);

namespace tsr {

namespace ek {


namespace {

struct ew_tile {
    uint32_t r0, nr;     // first row, rows
    spm blk;             // [a rows | b part]
    ticket ld;           // the tile's last LD
};

uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }

// One in-place VOP (vop_ip) pushed with deps {ld}: the general path's single copy of the builder.
[[gnu::noinline]] ticket push_vop(uint32_t func, uint32_t cmode, uint32_t si, spm A, uint32_t b, uint32_t bs, uint32_t c,
                                  uint32_t n, uint32_t rows, uint32_t L, ticket ld, uint32_t pf) {
    return issue(vop_ip(func, cmode, si, A, b, bs, c, n, rows, L), deps{ld}, pf);
}

// Rows [r0, r0 + nr) of a row tensor (stride lines apart in DRAM; rows of `lines` lines) into dst,
// dense in SPM: one LD when nr = 1 or the rows are dense, else one per row. The first LD carries
// `first`; returns the last.
ticket load_rows(spm dst, dram src, uint32_t stride, uint32_t lines, uint32_t r0, uint32_t nr, deps first, uint32_t pf) {
    if (nr == 1u || stride == lines) return push_g(LD(dst, src + r0 * stride, nr * lines), first, pf);
    ticket last{};
    for (uint32_t i = 0; i < nr; ++i)
        last = push_g(LD(dst + i * lines, src + (r0 + i) * stride, lines), i == 0u ? first : deps{}, pf);
    return last;
}
// The rows of a tile from SPM (dense, `lines` per row) to DRAM; the first ST carries `first`.
ticket store_rows(spm src, dram dst, uint32_t stride, uint32_t lines, uint32_t r0, uint32_t nr, deps first, uint32_t pf) {
    if (nr == 1u || stride == lines) return push_g(ST(src, dst + r0 * stride, nr * lines), first, pf);
    ticket last{};
    for (uint32_t i = 0; i < nr; ++i)
        last = push_g(ST(src + i * lines, dst + (r0 + i) * stride, lines), i == 0u ? first : deps{}, pf);
    return last;
}

// Lines of a tile's block: its a rows and its b part (plan_elem.hpp ew_b_lines; the scalar's line
// in tile 0 only).
uint32_t tile_lines(const ew_op & e, const ew_tile & t) {
    const uint32_t L = e.kp.row_lines, nr = t.nr, p = e.kp.aux[0];
    const uint32_t a = nr * L;
    switch (plan::ew_bmode(e.kp)) {
        case plan::EB_ROWS:    return a + nr * L;
        case plan::EB_BCAST:   return a + L;
        case plan::EB_PERIOD:  return a + min_u(p, nr) * L;
        case plan::EB_C1:      return a + (t.r0 == 0u ? 1u : 0u);
        case plan::EB_CROWS:   return a + plan::vu_lines(nr);
        case plan::EB_CPERIOD: return a + plan::vu_lines(min_u(p, nr));
        default:               return a;
    }
}

// Steps 1-2 of a tile whose block is allocated.
[[gnu::cold]] void load_tile(const ew_op & e, ew_tile & t, deps first) {
    const uint32_t L = e.kp.row_lines, nr = t.nr, p = e.kp.aux[0];
    const spm A = t.blk, B = A + nr * L;
    ticket ld = load_rows(A, e.a, e.as, L, t.r0, nr, first, e.pf);
    switch (plan::ew_bmode(e.kp)) {
        case plan::EB_ROWS:    ld = load_rows(B, e.b, e.bs, L, t.r0, nr, deps{}, e.pf); break;
        case plan::EB_BCAST:   ld = push_g(LD(B, e.b, L), deps{}, e.pf); break;
        case plan::EB_PERIOD:  ld = load_rows(B, e.b, e.bs, L, 0u, min_u(p, nr), deps{}, e.pf); break;
        case plan::EB_C1:
            if (t.r0 == 0u) {
                ld = push_g(LD(B, e.b, 1u), deps{}, e.pf);
                if (e.rows > 1u) {
                    sop_f f{};
                    f.sub = SOP_SUB_LDS;
                    f.sd = S_A;
                    f.addr = 16u * B.line;
                    push_g(SOP(f), deps{ld}, e.pf);
                }
            }
            break;
        case plan::EB_CROWS:   ld = push_g(LD(B, e.b + (t.r0 >> 4), plan::vu_lines(nr)), deps{}, e.pf); break;
        case plan::EB_CPERIOD: ld = push_g(LD(B, e.b, plan::vu_lines(min_u(p, nr))), deps{}, e.pf); break;
        default: break;
    }
    t.ld = ld;
}

// Step 3: the tile's VOP(s); returns the last.
[[gnu::cold]] ticket compute_tile(const ew_op & e, const ew_tile & t) {
    const uint32_t L = e.kp.row_lines, nr = t.nr, p = e.kp.aux[0], n = e.n;
    const spm A = t.blk, B = A + nr * L;
    const uint32_t bm = plan::ew_bmode(e.kp);
    switch (bm) {
        case plan::EB_NONE:
            if (e.scale) {
                ticket v = push_vop(VOP_FUNC_MULC, VOP_CMODE_SREG, S_A, A, 0u, 0u, 0u, n, nr, L, t.ld, e.pf);
                if (e.b_bits != 0u) v = push_vop(VOP_FUNC_ADDC, VOP_CMODE_SREG, S_B, A, 0u, 0u, 0u, n, nr, L, t.ld, e.pf);
                return v;
            }
            return push_vop(e.func, VOP_CMODE_NONE, 0u, A, 0u, 0u, 0u, n, nr, L, t.ld, e.pf);
        case plan::EB_ROWS:  return push_vop(e.func, VOP_CMODE_NONE, 0u, A, B.line, L, 0u, n, nr, L, t.ld, e.pf);
        case plan::EB_BCAST: return push_vop(e.func, VOP_CMODE_NONE, 0u, A, B.line, 0u, 0u, n, nr, L, t.ld, e.pf);
        case plan::EB_C1:
            if (e.rows == 1u) return push_vop(e.cfunc, VOP_CMODE_VEC, 0u, A, 0u, 0u, B.line, n, 1u, L, t.ld, e.pf);
            return push_vop(e.cfunc, VOP_CMODE_SREG, S_A, A, 0u, 0u, 0u, n, nr, L, t.ld, e.pf);
        case plan::EB_CROWS: return push_vop(e.cfunc, VOP_CMODE_VEC, 0u, A, 0u, 0u, B.line, n, nr, L, t.ld, e.pf);
        default: {                                   // EB_PERIOD / EB_CPERIOD: one VOP per period block
            const bool c = bm == plan::EB_CPERIOD;
            ticket v{};
            for (uint32_t k = 0; k < nr; k += p) {
                const uint32_t m = min_u(p, nr - k);
                const spm Ak = A + k * L;
                v = c ? push_vop(e.cfunc, VOP_CMODE_VEC, 0u, Ak, 0u, 0u, B.line, n, m, L, t.ld, e.pf)
                      : push_vop(e.func, VOP_CMODE_NONE, 0u, Ak, B.line, L, 0u, n, m, L, t.ld, e.pf);
            }
            return v;
        }
    }
}

}  // namespace

// The general path (any shape the planner accepts): MUL_MAT's double-buffered tile loop (k_mulmat.cpp
// mulmat_tiles): tile 0's LDs, then per tile k: its VOP(s), the next tile's allocation and LDs, the ST
// of tile k (so the LD of tile k+1 precedes the ST of tile k on the DMA queue and overlaps the VOP of
// tile k); when the ring cannot place tile k+1 beside tile k, tile k is stored and released first.
[[gnu::noinline, gnu::cold]] ticket ew_run(const ew_op & e, deps d, kind_ctx & ctx) {
    spm_ring & ring = *ctx.ring;
    ew_tile tl[2];
    ew_tile * cur = &tl[0], * nxt = &tl[1];
    cur->r0 = 0u;
    cur->nr = min_u(e.kp.tile_rows, e.rows);
    // The op's only tile takes the one-tile block when it fits (kind_ctx::one: DMA-bracketed, no ring
    // record, no deps), as the fast path does; the tiles of a multi-tile op the ring.
    const uint32_t l0 = tile_lines(e, *cur);
    const bool one = cur->nr == e.rows && l0 <= ctx.one_lines;
    deps rd{};
    cur->blk = one ? ctx.one : alloc_g(ring, l0, &rd);
    if (!valid(cur->blk)) {
        ctx.ret = opl::RET_SCRATCH;
        return ticket{};
    }
    const deps first = first_deps(d, rd);
    if (e.scale) {                                   // SCALE's s registers, once per op (the shadow)
        sreg_set(true, ctx, S_A, e.s_bits, e.pf);
        if (e.b_bits != 0u) sreg_set(true, ctx, S_B, e.b_bits, e.pf);
    } else if (plan::ew_bmode(e.kp) == plan::EB_C1 && e.rows > 1u) {
        kf::sreg_clobber(ctx, S_A);                  // tile 0's SOP.LDS writes s12
    }
    load_tile(e, *cur, first);
    const uint32_t L = e.kp.row_lines;
    ticket last{};
    for (;;) {
        const ticket v = compute_tile(e, *cur);
        const uint32_t next_r0 = cur->r0 + cur->nr;
        if (next_r0 >= e.rows) {
            last = store_rows(cur->blk, e.dst, e.ds, L, cur->r0, cur->nr, deps{v}, e.pf);
            if (!one) ring.release(cur->blk, last);
            break;
        }
        nxt->r0 = next_r0;
        nxt->nr = min_u(e.kp.tile_rows, e.rows - next_r0);
        nxt->blk = alloc_g(ring, tile_lines(e, *nxt), &rd);
        bool stored = false;
        if (!valid(nxt->blk)) {
            last = store_rows(cur->blk, e.dst, e.ds, L, cur->r0, cur->nr, deps{v}, e.pf);
            ring.release(cur->blk, last);
            stored = true;
            nxt->blk = alloc_g(ring, tile_lines(e, *nxt), &rd);
            if (!valid(nxt->blk)) {                  // cannot happen: one tile fits the ring
                ctx.ret = opl::RET_SCRATCH;
                return last;
            }
        }
        load_tile(e, *nxt, kf::drop_dma(rd));
        if (!stored) {
            last = store_rows(cur->blk, e.dst, e.ds, L, cur->r0, cur->nr, deps{v}, e.pf);
            ring.release(cur->blk, last);
        }
        ew_tile * x = cur;
        cur = nxt;
        nxt = x;
    }
    return last;
}

// Both the op's deps and ring deps (without DMA tickets): fold to <= 2 with group (SYNC NOPs).
deps first_deps_slow(deps dd, deps rr) { return kf::fold_deps(dd, rr); }

namespace {

// ---------------------------------------------------------------------------------------------
// ADD / MUL
// ---------------------------------------------------------------------------------------------
namespace ew = opl::ew;

// A refused record or a planned op: copies the words (one line fill; nothing was pushed), plans,
// fills e (the caller adds its kind's fields) or sets ctx.ret.
template <uint32_t NARGS>
[[gnu::always_inline]] inline bool general_plan(bool (*planner)(const uint32_t *, uint32_t, const plan_limits &, kind_plan *),
                                                const volatile uint32_t * a, uint32_t n_args, uint32_t kflags,
                                                kind_ctx & ctx, uint32_t * w, ew_op & e) {
    for (uint32_t i = 0; i < NARGS; ++i) w[i] = a[i];
    if (kflags != 0u) {
        ctx.fail(opl::RET_BAD_RECORD);
        return false;
    }
    const plan_limits l = ctx.pl;
    if (!planner(w, n_args, l, &e.kp)) {
        ctx.fail(e.kp.status);
        return false;
    }
    e.pf = ctx.push_flags;
    return true;
}

bool plan_addmul_fn(const uint32_t * a, uint32_t n, const plan_limits & l, kind_plan * o) { return plan_add(a, n, l, o); }
bool plan_scale_fn(const uint32_t * a, uint32_t n, const plan_limits & l, kind_plan * o) { return plan_scale(a, n, l, o); }
bool plan_unary_fn(const uint32_t * a, uint32_t n, const plan_limits & l, kind_plan * o) { return plan_unary(a, n, l, o); }

[[gnu::noinline, gnu::cold]] ticket addmul_general(uint32_t func, uint32_t cfunc, const volatile uint32_t * a,
                                                   uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    uint32_t w[ew::NARGS];
    ew_op e{};
    if (!general_plan<ew::NARGS>(&plan_addmul_fn, a, n_args, kflags, ctx, w, e)) return ticket{};
    e.func = func;
    e.cfunc = cfunc;
    e.dst = from_handle(w[ew::DST]);
    e.ds = w[ew::DST_STRIDE];
    e.a = from_handle(w[ew::A]);
    e.as = w[ew::A_STRIDE];
    e.b = from_handle(w[ew::B]);
    e.bs = w[ew::B_STRIDE];
    e.n = w[ew::N];
    e.rows = w[ew::ROWS];
    return ew_run(e, d, ctx);
}

}  // namespace

// The fast path of ADD (FUNC ADD, CFUNC ADDC) and MUL (MUL, MULC); the kinds' entries
// kind_add_fast / kind_mul_fast below (exec.hpp). One instance per kind (a shared body with the
// function as a value measured ~24 RV cycles more per record: the requirement is the cycles, SW §7.8).
template <uint32_t FUNC, uint32_t CFUNC>
TSR_HOT uint32_t addmul_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t fl = a[ew::FLAGS];
    if ((kflags | (fl & ~ew::FLAG_B_SCALAR)) != 0u) [[unlikely]] return NOT_FAST;
    return ew2_fast<ew_words>(a, FUNC, CFUNC, fl, a[ew::B_PERIOD], d.a.v, d.b.v, ctx);
}

namespace {

// ---------------------------------------------------------------------------------------------
// SCALE
// ---------------------------------------------------------------------------------------------
namespace scl = opl::scl;

[[gnu::noinline, gnu::cold]] ticket scale_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                  kind_ctx & ctx) {
    uint32_t w[scl::NARGS];
    ew_op e{};
    if (!general_plan<scl::NARGS>(&plan_scale_fn, a, n_args, kflags, ctx, w, e)) return ticket{};
    e.func = VOP_FUNC_MULC;
    e.dst = from_handle(w[scl::DST]);
    e.ds = w[scl::DST_STRIDE];
    e.a = from_handle(w[scl::SRC]);
    e.as = w[scl::SRC_STRIDE];
    e.n = w[scl::N];
    e.rows = w[scl::ROWS];
    e.scale = 1u;
    e.s_bits = w[scl::S];
    e.b_bits = w[scl::B];
    return ew_run(e, d, ctx);
}

// The fast path: one tile of one row or dense rows. SOP.MOVI s12 <- s (, s13 <- b), LD x,
// VOP.MULC c = s12 (, VOP.ADDC c = s13), ST: 4 (6) pushes.
TSR_HOT uint32_t scale_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t hd = a[scl::DST], ds = a[scl::DST_STRIDE], hs = a[scl::SRC], ss = a[scl::SRC_STRIDE], n = a[scl::N],
                   rows = a[scl::ROWS], s = a[scl::S], b = a[scl::B];
    if (kflags != 0u || n - 1u >= plan::EW_N_MAX || rows - 1u >= plan::VU_ROWS_MAX || (ctx.pl.caps & plan::CAPS_VU) == 0u) [[unlikely]]
        return NOT_FAST;
    const uint32_t L = (n + 15u) >> 4;
    if (rows != 1u && (ss != L || ds != L)) [[unlikely]] return NOT_FAST;
    const uint32_t al = rows * L;
    if (al > (ctx.pl.budget >> 1)) [[unlikely]] return NOT_FAST;
    kf::block blk;
    if (!kf::alloc_fast(ctx, al, &blk)) [[unlikely]] return NOT_FAST;
    const spm A = blk.at;
    const deps first = kf::first_deps(d);
    const uint32_t pf = ctx.push_flags;
    const uint32_t nv = 1u + kf::sreg_need(ctx, S_A, s) + (b != 0u ? 1u + kf::sreg_need(ctx, S_B, b) : 0u);
    const bool chk = need_chk(2u, nv);
    if (!chk && nv == (b != 0u ? 2u : 1u)) [[likely]] {
        // The common case: one credit read covers the pushes and the launch's s-register
        // shadow holds s (and b): no MOVI, the nochk_t pushes with the staged words reused (kf::push_s,
        // CC §5.1): the MULC's d line is the LD's SPM line (w1); the ADDC differs from the MULC only in
        // w0 (function, s register), so it stages w0 alone; the ST's SPM line is the last VOP's d (w1).
        // The same ops as the form below (fw_kernels_elem_host compares both with the general path).
        kf::stage_reset();
        const ticket ld = kf::push_s(LD(A, from_handle(hs), al), first, pf);
        ticket v;
        if (b == 0u) {
            v = kf::push_s<kf::W1>(vop_ip(VOP_FUNC_MULC, VOP_CMODE_SREG, S_A, A, 0u, 0u, 0u, n, rows, L), deps{ld}, pf);
        } else {
            kf::push_s<kf::W1, false>(vop_ip(VOP_FUNC_MULC, VOP_CMODE_SREG, S_A, A, 0u, 0u, 0u, n, rows, L), deps{ld}, pf);
            v = kf::push_s<kf::W1 | kf::W2 | kf::W3 | kf::W4 | kf::W5 | kf::W6 | kf::W7>(
                vop_ip(VOP_FUNC_ADDC, VOP_CMODE_SREG, S_B, A, 0u, 0u, 0u, n, rows, L), deps{ld}, pf);
        }
        const ticket st = kf::push_s<kf::W1>(ST(A, from_handle(hd), al), deps{v}, pf);
        kf::release(blk, st);
        return st.v;
    }
    sreg_set(chk, ctx, S_A, s, pf);
    if (b != 0u) sreg_set(chk, ctx, S_B, b, pf);
    const ticket ld = kpush<UNIT_DMA>(chk, LD(A, from_handle(hs), al), first, pf);
    ticket v = kpush<UNIT_VU>(chk, vop_ip(VOP_FUNC_MULC, VOP_CMODE_SREG, S_A, A, 0u, 0u, 0u, n, rows, L), deps{ld}, pf);
    if (b != 0u) v = kpush<UNIT_VU>(chk, vop_ip(VOP_FUNC_ADDC, VOP_CMODE_SREG, S_B, A, 0u, 0u, 0u, n, rows, L), deps{ld}, pf);
    const ticket st = kpush<UNIT_DMA>(chk, ST(A, from_handle(hd), al), deps{v}, pf);
    kf::release(blk, st);
    return st.v;
}

// ---------------------------------------------------------------------------------------------
// UNARY
// ---------------------------------------------------------------------------------------------
namespace un = opl::unary;

// VOP function per UNARY func (SW §7.2: 0 TANH, 1 GELU16, 2 SILU, 3 EXP, 4 GELU exact tanh form,
// 5 SIGMOID).
constexpr uint8_t k_unary_func[un::FUNC_MAX + 1u] = {VOP_FUNC_TANH, VOP_FUNC_GELU16, VOP_FUNC_SILU, VOP_FUNC_EXP,
                                                     VOP_FUNC_GELU, VOP_FUNC_SIGMOID};
static_assert(un::TANH == 0u && un::GELU16 == 1u && un::SILU == 2u && un::EXP == 3u && un::GELU == 4u &&
                  un::SIGMOID == 5u,
              "UNARY func codes (SW §7.2)");

[[gnu::noinline, gnu::cold]] ticket unary_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                                                  kind_ctx & ctx) {
    uint32_t w[un::NARGS];
    ew_op e{};
    if (!general_plan<un::NARGS>(&plan_unary_fn, a, n_args, kflags, ctx, w, e)) return ticket{};
    e.func = k_unary_func[w[un::FUNC]];
    e.dst = from_handle(w[un::DST]);
    e.ds = w[un::DST_STRIDE];
    e.a = from_handle(w[un::SRC]);
    e.as = w[un::SRC_STRIDE];
    e.n = w[un::N];
    e.rows = w[un::ROWS];
    return ew_run(e, d, ctx);
}

// The fast path: one tile of one row or dense rows. LD, VOP, ST.
TSR_HOT uint32_t unary_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t hd = a[un::DST], ds = a[un::DST_STRIDE], hs = a[un::SRC], ss = a[un::SRC_STRIDE], n = a[un::N],
                   rows = a[un::ROWS], fn = a[un::FUNC];
    if (kflags != 0u || fn > un::FUNC_MAX || n - 1u >= plan::EW_N_MAX || rows - 1u >= plan::VU_ROWS_MAX ||
        (ctx.pl.caps & plan::CAPS_VU) == 0u) [[unlikely]]
        return NOT_FAST;
    const uint32_t L = (n + 15u) >> 4;
    if (rows != 1u && (ss != L || ds != L)) [[unlikely]] return NOT_FAST;
    const uint32_t al = rows * L;
    if (al > (ctx.pl.budget >> 1)) [[unlikely]] return NOT_FAST;
    kf::block blk;
    if (!kf::alloc_fast(ctx, al, &blk)) [[unlikely]] return NOT_FAST;
    const spm A = blk.at;
    const op v = vop_ip(k_unary_func[fn], VOP_CMODE_NONE, 0u, A, 0u, 0u, 0u, n, rows, L);
    const deps first = kf::first_deps(d);
    const uint32_t pf = ctx.push_flags;
    const bool chk = need_chk(2u, 1u);
    const ticket ld = kpush<UNIT_DMA>(chk, LD(A, from_handle(hs), al), first, pf);
    const ticket vt = kpush<UNIT_VU>(chk, v, deps{ld}, pf);
    const ticket st = kpush<UNIT_DMA>(chk, ST(A, from_handle(hd), al), deps{vt}, pf);
    kf::release(blk, st);
    return st.v;
}

}  // namespace

#if defined(TSR_FW_HOST)
// Host builds only (fw_kernels_elem_host, k_elem.hpp): a kind's general path alone and its fast path
// alone, which the test compares push for push on twin simulators.
ticket host_general(uint32_t kind, const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    switch (kind) {
        case opl::K_ADD:   return addmul_general(VOP_FUNC_ADD, VOP_FUNC_ADDC, a, n_args, kflags, d, ctx);
        case opl::K_MUL:   return addmul_general(VOP_FUNC_MUL, VOP_FUNC_MULC, a, n_args, kflags, d, ctx);
        case opl::K_SCALE: return scale_general(a, n_args, kflags, d, ctx);
        case opl::K_UNARY: return unary_general(a, n_args, kflags, d, ctx);
        case opl::K_GLU:   return glu_general(a, n_args, kflags, d, ctx);
        default:           return host_general_rows(kind, a, n_args, kflags, d, ctx);
    }
}
uint32_t host_fast(uint32_t kind, const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    switch (kind) {
        case opl::K_ADD:
        case opl::K_MUL: {
            const uint32_t fl = a[ew::FLAGS];
            if ((kflags | (fl & ~ew::FLAG_B_SCALAR)) != 0u) [[unlikely]] return NOT_FAST;
            const bool mul = kind == opl::K_MUL;
            return ew2_fast<ew_words>(a, mul ? VOP_FUNC_MUL : VOP_FUNC_ADD, mul ? VOP_FUNC_MULC : VOP_FUNC_ADDC, fl,
                                      a[ew::B_PERIOD], d.a.v, d.b.v, ctx);
        }
        case opl::K_SCALE: return scale_fast(a, kflags, d, ctx);
        case opl::K_UNARY: return unary_fast(a, kflags, d, ctx);
        case opl::K_GLU: {
            const uint32_t var = a[opl::glu::VARIANT];
            if (kflags != 0u || var > opl::glu::VARIANT_MAX || a[opl::glu::UP_STRIDE] == 0u) [[unlikely]] return NOT_FAST;
            return ew2_fast<ew_words, true>(a, GLU_FUNC[var], 0u, 0u, 0xFFFF'FFFFu, d.a.v, d.b.v, ctx);
        }
        default:           return host_fast_rows(kind, a, kflags, d, ctx);
    }
}
#endif
}  // namespace ek

// ---------------------------------------------------------------------------------------------
// The fast entries (exec.hpp kind_<name>_fast: the executor's first call, leaves in their common
// path) and the handlers (exec.hpp kind_fn: the same fast path first, else the general path).
// ---------------------------------------------------------------------------------------------
[[gnu::noinline]] uint32_t kind_add_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    return ek::addmul_fast<VOP_FUNC_ADD, VOP_FUNC_ADDC>(a, kflags, d, ctx);
}
[[gnu::noinline]] uint32_t kind_mul_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    return ek::addmul_fast<VOP_FUNC_MUL, VOP_FUNC_MULC>(a, kflags, d, ctx);
}
[[gnu::noinline]] uint32_t kind_scale_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    return ek::scale_fast(a, kflags, d, ctx);
}
[[gnu::noinline]] uint32_t kind_unary_fast(const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx) {
    return ek::unary_fast(a, kflags, d, ctx);
}

ticket kind_add(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t t = kind_add_fast(a, kflags, d, ctx);
    if (t != ek::NOT_FAST) return ticket{t};
    return ek::addmul_general(VOP_FUNC_ADD, VOP_FUNC_ADDC, a, n_args, kflags, d, ctx);
}

ticket kind_mul(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t t = kind_mul_fast(a, kflags, d, ctx);
    if (t != ek::NOT_FAST) return ticket{t};
    return ek::addmul_general(VOP_FUNC_MUL, VOP_FUNC_MULC, a, n_args, kflags, d, ctx);
}

ticket kind_scale(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t t = kind_scale_fast(a, kflags, d, ctx);
    if (t != ek::NOT_FAST) return ticket{t};
    return ek::scale_general(a, n_args, kflags, d, ctx);
}

ticket kind_unary(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    const uint32_t t = kind_unary_fast(a, kflags, d, ctx);
    if (t != ek::NOT_FAST) return ticket{t};
    return ek::unary_general(a, n_args, kflags, d, ctx);
}

}  // namespace tsr
