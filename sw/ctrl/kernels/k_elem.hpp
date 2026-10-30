// k_elem.hpp — shared pieces of the element-wise and row kinds of the graph executor (OPL v1 ADD
// 0x11, MUL 0x12, SCALE 0x13, GLU 0x15, GET_ROWS 0x17, CPY 0x1A, UNARY 0x1B; docs/v2/
// 06-software-verification.md "SW" §7.2, §7.3 "Hot path", §7.5, §7.8; VU §5.2, §9). Kernel-private:
// included by k_elem.cpp (ADD, MUL, SCALE, UNARY and the element-wise tile engine), k_glu.cpp (GLU)
// and k_rows.cpp (GET_ROWS, CPY).
//
// Every kind has two paths that push the same ops for the records both take:
//   - the decode fast path (V2-SW-35's form, k_fast.hpp): the argument words read in place into
//     locals, a one-tile plan by shifts and compares (no divide) that is plan_elem.hpp's for the
//     record, the one-tile block from kf::alloc_fast (V2-SW-36: no deps, no ring record; the
//     general path takes it for an op's only tile too; GLU's larger rows a ring block), one
//     ISSUE_CREDIT read for the op's pushes when it covers them (kf::credits_for; else issue()'s wait
//     per push: kpush below; ADD / MUL / GLU and SCALE's common case wait until one read covers them
//     and reuse staged words, kf::push_s: W2b fix), the VU words by kf::vop_fast, no call. It returns
//     kf::NOT_FAST, having changed nothing, for anything else: several tiles, strided rows, a b
//     period, one scalar for several rows, (GLU) a ring wrap or a block released by a non-DMA
//     ticket, a record the planner refuses;
//   - the general path ([[gnu::cold]], compiled for size): the planner of tsr/plan_elem.hpp (shared
//     with the host's supports_op, V2-SW-20), any number of tiles, double-buffered like MUL_MAT's
//     (SW §7.4), spm_ring::alloc.
// Deps (MUL_MAT's rule, SW §7.4): the op's deps (and its first block's ring deps) go on its first LD
// without DMA tickets (queue order implies them); each VOP depends on its tile's last LD (DMA,
// implying the tile's earlier LDs); the first ST of a tile on the tile's last VOP (copies: none, the
// DMA queue order puts it after the tile's LDs); the returned ticket is the op's last ST, which
// implies every tile op (I1: it follows the last VOP, which follows every earlier VU op, and every
// earlier DMA op by queue order). A block is released after its tile's last ST.
//
// s registers (VU §4.4): SCALE sets s12 (s) and s13 (b, when nonzero) once per op through the
// launch's s-register shadow (exec.hpp kind_ctx::sreg_valid: an SOP.MOVI only when the shadow does
// not already hold the value, e.g. the second of a list's equal SCALEs); ADD / MUL with one scalar b
// for more than one row load it into s12 with SOP.LDS and clear s12 from the shadow (sreg_clobber,
// the rule of exec.hpp). The VU runs its queue in order, so every VOP reads the value its record set.
#pragma once

#include "tsr/exec.hpp"
#include "tsr/plan.hpp"

#include "k_fast.hpp"

namespace tsr {
namespace ek {

using kf::NOT_FAST;
constexpr uint32_t S_A = 12u, S_B = 13u;      // the s registers of SCALE / scalar-b ADD and MUL

// d and the general path's ring deps for a first LD, both without DMA tickets, folded to <= 2
// (group: SYNC NOPs when more than two units remain; k_elem.cpp).
deps first_deps_slow(deps dd, deps rr);
TSR_HOT deps first_deps(deps d, deps rd) {
    const deps dd = kf::drop_dma(d);
    const deps rr = kf::drop_dma(rd);
    if (rr.a.none()) return dd;
    if (dd.a.none()) return rr;
    return first_deps_slow(dd, rr);
}

// One in-place VOP (d = a = A, both stride L) over rows rows of n elements; cmode / si / c / b / bs
// as VU §5.2 (a field the function does not read is 0). kf::vop_fast's words (no run-time range
// checks): every field is planned (SPM lines < 2^24, n <= EW_N_MAX, rows of a tile and L < 2^16).
TSR_HOT op vop_ip(uint32_t func, uint32_t cmode, uint32_t si, spm A, uint32_t b, uint32_t bs, uint32_t c, uint32_t n,
                  uint32_t rows, uint32_t L) {
    vop_f f{};
    f.cmode = cmode;
    f.func = func;
    f.si = si;
    f.d = A.line;
    f.a = A.line;
    f.b = b;
    f.c = c;
    f.m = n;
    f.rows = rows;
    f.d_stride = L;
    f.a_stride = L;
    f.b_stride = bs;
    return kf::vop_fast(f);
}

// The general paths' push and ring allocation, out of line (one copy for every kind of these files;
// the fast paths inline their own pushes): issue() and spm_ring::alloc(lines, any parity).
using kf::alloc_g;   // k_fast.hpp: one copy per image
using kf::push_g;

// GLU's VOP function per variant (SW §7.2: 0 SWIGLU, 1 GEGLU16, 2 GEGLU) and its general path (k_glu.cpp).
constexpr uint8_t GLU_FUNC[opl::glu::VARIANT_MAX + 1u] = {VOP_FUNC_SWIGLU, VOP_FUNC_GEGLU16, VOP_FUNC_GEGLU};
static_assert(opl::glu::SWIGLU == 0u && opl::glu::GEGLU16 == 1u && opl::glu::GEGLU == 2u, "GLU variant codes (SW §7.2)");
ticket glu_general(const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);

// An element-wise op for the general path (k_elem.cpp ew_run): the record's words decoded and planned.
struct ew_op {
    uint32_t func;         // the VOP function (rows / broadcast / period forms; UNARY's; GLU's)
    uint32_t cfunc;        // with a scalar b: the c form (ADDC / MULC)
    dram dst, a, b;
    uint32_t ds, as, bs;   // DRAM strides in lines
    uint32_t n, rows;
    uint32_t scale;        // SCALE: 1 (VOP.MULC s12, then VOP.ADDC s13 when b_bits != 0)
    uint32_t s_bits, b_bits;
    kind_plan kp;          // plan_<kind>: row_lines = L, tile_rows, flags >> KP_KIND_LSB = EB_*, aux[0] = p
    uint32_t pf;           // push flags (kind_ctx::push_flags)
};
// Pushes the whole op (tiles, double buffering); ctx.ret = RET_SCRATCH if a tile cannot be placed.
ticket ew_run(const ew_op & e, deps d, kind_ctx & ctx);

// The fast paths' push: issue_nc, preceded by issue()'s credit wait (the same pushes) when the op's
// one ISSUE_CREDIT read (kf::credits_for) did not cover the record's pushes: one copy of the push
// sequence instead of k_fast.hpp's two forms (ITCM, SW §7.8), a not-taken branch per push. The op's
// unit U is the caller's (a constant: DMA for LD / ST; VU for VOP / SOP, whose CAPS bit the fast
// path checked, so the issue path accepts them on the VU, CC §7.4): no opcode table lookup, which
// could call detail::push_unit_slow and make the fast path a non-leaf function.
template <uint32_t U> TSR_HOT ticket kpush(bool chk, const op & o, deps d, uint32_t pf) {
    if (chk) [[unlikely]] {
        while (detail::credit_of(hw::cw_read(CW_ISSUE_CREDIT), U) == 0u) hw::pause();
    }
    return issue_nc(o, d, pf);
}
// true when the credit read does not cover n_dma DMA and n_vu VU pushes (then kpush waits per push).
TSR_HOT bool need_chk(uint32_t n_dma, uint32_t n_vu) { return !kf::credits_for(credits(), n_dma, n_vu); }

// The launch's s-register shadow (exec.hpp kind_ctx::sreg_valid, k_fast.hpp sreg_movi): s[i] = bits
// for the VU ops pushed after this, with an SOP.MOVI only when the shadow does not hold bits.
TSR_HOT void sreg_set(bool chk, kind_ctx & ctx, uint32_t i, uint32_t bits, uint32_t pf) {
    if (kf::sreg_holds(ctx, i, bits)) return;
    kpush<UNIT_VU>(chk, kf::sop_movi(i, bits), deps{}, pf);
    ctx.sreg_val[i] = bits;
    ctx.sreg_valid |= 1u << i;
}

// The decode fast path of the two-input kinds: ADD / MUL (func = ADD / MUL, cfunc = ADDC / MULC) and
// GLU (func by its variant, period ~0: the rows form only). One tile (plan_elem.hpp: its block
// within half the budget): one row, or dense a / dst rows (and dense b rows in the rows form);
// operand b as plan_elem.hpp decides it: EB_BCAST (p = 1 or b stride 0: VOP b stride 0), EB_ROWS
// (VOP b stride L), EB_C1 with one row and EB_CROWS (the scalars as a c vector, cmode 2). Returns
// the op's ticket (its ST) or NOT_FAST (nothing changed). Inlined into each kind's handler (one
// instance per kind: a shared body measured ~24 RV cycles more per record); the handles are read
// once the block is placed (before the first control-window access: the credit read invalidates the
// SPM line buffer), and the VOP's words are vop_fast's (k_fast.hpp: the plan bounded every field).
// RING (GLU; W2b fix): a tile larger than the one-tile block (plan.hpp ONE_BLOCK_MAX: the GLU rows of
// Qwen2.5 1.5B-32B and Gemma 4 12B) takes a ring block from spm_ring::alloc_dma_first instead (the
// block and records ew_run's alloc gives the op's only tile, nothing to wait for: NOT_FAST in every
// other case), released after the ST, as mulmat_t1_fast; the other kinds' decode tiles fit the block.
template <class W, bool RING = false>
TSR_HOT uint32_t ew2_fast(const volatile uint32_t * a, uint32_t func, uint32_t cfunc, uint32_t scalar, uint32_t p,
                                    uint32_t t0, uint32_t t1, kind_ctx & ctx) {
    const uint32_t n = a[W::N], rows = a[W::ROWS];
    if (n - 1u >= plan::EW_N_MAX || rows - 1u >= plan::VU_ROWS_MAX || p == 0u || (ctx.pl.caps & plan::CAPS_VU) == 0u) [[unlikely]]
        return NOT_FAST;
    const uint32_t L = (n + 15u) >> 4;
    if (rows != 1u && (a[W::A_STRIDE] != L || a[W::DST_STRIDE] != L)) [[unlikely]] return NOT_FAST;
    const uint32_t half = ctx.pl.budget >> 1;
    const uint32_t al = rows * L;                  // < 2^32: rows, L < 2^16
    uint32_t bl, bstr = 0u;
    if (scalar) {                                  // EB_C1 (one row) / EB_CROWS: the c vector
        if (rows != 1u && p < rows) [[unlikely]] return NOT_FAST;
        bl = (rows + 15u) >> 4;
    } else {
        const uint32_t bs = a[W::B_STRIDE];
        if (p == 1u || bs == 0u) {                 // EB_BCAST
            bl = L;
        } else if (p >= rows && (rows == 1u || bs == L)) {   // EB_ROWS
            bl = al;
            bstr = L;
        } else [[unlikely]] {
            return NOT_FAST;                       // EB_PERIOD / strided b rows
        }
    }
    if (al > half || al + bl > half) [[unlikely]] return NOT_FAST;
    kf::block blk;
    uint32_t * rel = nullptr;
    if (!kf::alloc_fast(ctx, al + bl, &blk)) [[unlikely]] {
        if constexpr (!RING) return NOT_FAST;
        blk.at = ctx.ring->alloc_dma_first(al + bl, &rel);   // its ring deps: DMA tickets only (implied)
        if (!valid(blk.at)) return NOT_FAST;
    }
    // Committed: the one-tile block or (RING) a ring block whose deps the first LD's queue order implies.
    const spm A = blk.at, B = A + al;
    const uint32_t ha = a[W::A], hb = a[W::B], hd = a[W::DST];
    const uint32_t pf = ctx.push_flags;
    // One ISSUE_CREDIT read covering the four pushes (W2b fix: wait until it does, as k_norm.cpp's
    // wait_credits; it waits only while the RISC-V is ahead of the units, and pushes the same ops as
    // issue()'s per-push wait), then the nochk_t pushes with the staged words reused (kf::push_s): the
    // ST's SPM line is the VOP's d (w1 = A); no ISSUE_TICKET read for the a LD.
    while (need_chk(3u, 1u)) hw::pause();
    kf::stage_reset();
    kf::push_s<0u, false>(LD(A, from_handle(ha), al), kf::first_deps(deps{ticket{t0}, ticket{t1}}), pf);
    const ticket lb = kf::push_s(LD(B, from_handle(hb), bl), deps{}, pf);
    vop_f f{};
    f.cmode = scalar ? VOP_CMODE_VEC : VOP_CMODE_NONE;
    f.func = scalar ? cfunc : func;
    f.d = A.line;
    f.a = A.line;
    f.b = scalar ? 0u : B.line;
    f.c = scalar ? B.line : 0u;
    f.m = n;
    f.rows = rows;
    f.d_stride = L;
    f.a_stride = L;
    f.b_stride = bstr;
    const ticket vt = kf::push_s(kf::vop_fast(f), deps{lb}, pf);
    const ticket st = kf::push_s<kf::W1>(ST(A, from_handle(hd), al), deps{vt}, pf);
    kf::release(blk, st);
    if constexpr (RING) {
        if (rel != nullptr) spm_ring::release_at(rel, st);
    }
    return st.v;
}
// The argument word indices of ew2_fast: ADD / MUL's, which are GLU's for words 0-7 (gate = a,
// up = b).
struct ew_words {
    static constexpr uint32_t DST = opl::ew::DST, DST_STRIDE = opl::ew::DST_STRIDE, A = opl::ew::A,
                              A_STRIDE = opl::ew::A_STRIDE, B = opl::ew::B, B_STRIDE = opl::ew::B_STRIDE, N = opl::ew::N,
                              ROWS = opl::ew::ROWS;
};
static_assert(opl::glu::DST == opl::ew::DST && opl::glu::DST_STRIDE == opl::ew::DST_STRIDE && opl::glu::GATE == opl::ew::A &&
                  opl::glu::GATE_STRIDE == opl::ew::A_STRIDE && opl::glu::UP == opl::ew::B &&
                  opl::glu::UP_STRIDE == opl::ew::B_STRIDE && opl::glu::N == opl::ew::N && opl::glu::ROWS == opl::ew::ROWS,
              "GLU's words 0-7 are ADD / MUL's (SW §7.2)");

#if defined(TSR_FW_HOST)
// Host builds only (fw_kernels_elem_host): a kind's general path alone (k_elem.cpp host_general,
// which calls the GLU / row ones), and its fast path alone (NOT_FAST when it does not take the
// record; nothing changed then), as k_mulmat's mulmat_general.
ticket host_general(uint32_t kind, const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx);
uint32_t host_fast(uint32_t kind, const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx);
ticket host_general_rows(uint32_t kind, const volatile uint32_t * a, uint32_t n_args, uint32_t kflags, deps d,
                         kind_ctx & ctx);
uint32_t host_fast_rows(uint32_t kind, const volatile uint32_t * a, uint32_t kflags, deps d, kind_ctx & ctx);
#endif

}  // namespace ek
}  // namespace tsr
