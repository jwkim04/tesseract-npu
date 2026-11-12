// gdn_kinds.cpp — the Gated DeltaNet kinds of the graph executor: SOFTPLUS (0x2D), SSM_CONV (0x2E) and
// GATED_DELTA_NET (0x2F) (docs/design/06-software-verification.md "SW" §7.10; handler interface
// tsr/exec.hpp, SW §7.3; planning tsr/plan_gdn.hpp; references sw/ctrl/ref/gdn_ref.cpp, which
// composes the same tsr_ref_vu calls in the same order). They are built from existing VU functions
// and the DMA; nothing here needs a new RTL op (decision SW-67).
//
// Deps (exec.hpp contract). The op's predecessors d (and the ring block's release tickets, DMA
// tickets dropped: queue order implies them) go on the op's first LD. A VU op depends on the LD of
// the data it reads (a DMA ticket implies the earlier LDs of the queue); the VU queue order orders
// the rest (VU §4.5). A ST depends on the last VU op that writes its source. The returned ticket is
// the op's last ST, which implies every tile op (I1). None of the kinds has a fast path: a record
// is long (SOFTPLUS 20 VOPs, SSM_CONV 7 + 1 per channel tile, GATED_DELTA_NET 13 VU ops per
// head and token).
//
// SOFTPLUS tile-op sequence (per row tile of nr rows in one block of five nr L-line buffers
// B0 .. B4; x = B0):
//   VOP.MULC B1 = x * s4(-1); VOP.MIN B1 = MINT(x, B1) (= -|x|); VOP.EXP B2 = exp(B1) (= h);
//   VOP.ADDC B3 = B2 + s5(2); VOP.RECIP B3; VOP.MUL B3 = B2 * B3 (= s); VOP.MUL B2 = B3 * B3 (= z);
//   VOP.MULC B4 = B2 * s7(c5); VOP.ADDC B4 += s8(c4); for k = 3 .. 0: VOP.MUL B4 *= B2,
//   VOP.ADDC B4 += s(c_k); VOP.MUL B3 = B3 * B4 (= y); VOP.SUB B1 = x - B1 (= x + |x|);
//   VOP.MULCADD B3 = FA(FM(B1, s6(1/2)), B3); ST B3.
// s4 .. s11 hold the constants (opl::softplus COEF, NEG_ONE, TWO, HALF; s5 = c0 = 2).
//
// SSM_CONV, per sequence and channel tile (lj lines of C / 16, plan_gdn.hpp), one block
// [window rows | tap rows | ACC rows | TMP rows] of lj-line rows: LD the d_conv taps (rows of the W
// tensor), the d_conv - 1 state rows, the T token rows (one LD each); VOP.MUL ACC = window rows
// 0 .. T-1 * tap 0 (the tap row with stride 0 over the T token rows); for w = 1 .. d_conv - 1:
// VOP.MUL TMP = window rows w .. w+T-1 * tap w, VOP.ADD ACC += TMP; with FLAG_SILU VOP.SILU ACC;
// ST the T output rows and the last d_conv - 1 window rows (the new state).
//
// GATED_DELTA_NET, on one of two paths chosen at run time from CAPS (plan::gdn::use_fused, SW-78): the composition of
// revision 1 (gdn_ref_step in ref/gdn_ref.hpp: the numerics, 13 VU ops per head and token), or, when the device has
// the fused op (CAPS_GDN) and S_v >= 64, the VU's GDN op (VU §7.12; gdn_ref_step_fused: the state update and the
// output in one op, then the scale). Two pipelines push them, chosen by the plan (plan_gdn.hpp, kp.aux[2] = R):
//   - gdn_run, head by head (the composition, and a fused record whose token block pipeline block does not fit the
//     budget): one ring block holds two state tiles (the next head's state loads while the VU works on this head's),
//     the product tile (composition only) and two token slots (token i + 2 loads while token i computes);
//   - gdn_run_tok, the token block pipeline of a fused record (below, before its definition): the operands of a group
//     of heads load together, so a head is two SOP.LDS and one GDN op and the units pace the loop.
// The numerics do not depend on the pipeline: every (head, token) runs the same op on the same operands.
#include "k_elem.hpp"

#include "tsr/plan_gdn.hpp"

// The op list kinds this source handles (SW §6.9: .tsr_kinds entries at tsr/opl.hpp kind_rev).
TSR_KIND(SOFTPLUS);
TSR_KIND(SSM_CONV);
TSR_KIND(GATED_DELTA_NET);

namespace tsr {

namespace {

using kf::alloc_g;
using kf::push_g;

constexpr uint32_t SR = VOP_CMODE_SREG, VEC = VOP_CMODE_VEC, NOC = VOP_CMODE_NONE;

uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }

// The VU ops of these kinds are tables of vspec (code size: the ITCM budget has little room). A role
// names an SPM operand; line[role] is its first line, and the roles in `tile` (a bit mask) are
// matrices with row stride L lines, the others vectors or broadcasts with stride 0. RN: no operand.
constexpr uint8_t RN = 15u;
struct vspec {
    uint8_t func;    // VOP function
    uint8_t cs;      // cmode (bits [1:0]) | s register (bits [7:4])
    uint8_t d, a, b, c;   // roles
    uint8_t rows1;   // 1: one row (a vector op), else the matrix rows
};
[[gnu::noinline]] ticket pv(const vspec & s, const uint32_t * line, uint32_t tile, uint32_t m, uint32_t rows, uint32_t L,
                            deps dd, uint32_t pf) {
    auto ln = [&](uint8_t r) { return r == RN ? 0u : line[r]; };
    auto st = [&](uint8_t r) { return r != RN && ((tile >> r) & 1u) != 0u && s.rows1 == 0u ? L : 0u; };
    vop_f f{};
    f.cmode = s.cs & 3u;
    f.func = s.func;
    f.si = s.cs >> 4;
    f.d = ln(s.d);
    f.a = ln(s.a);
    f.b = ln(s.b);
    f.c = ln(s.c);
    f.m = m;
    f.rows = s.rows1 ? 1u : rows;
    f.d_stride = st(s.d);
    f.a_stride = st(s.a);
    f.b_stride = st(s.b);
    return push_g(kf::vop_fast(f), dd, pf);
}
[[gnu::noinline]] ticket pred(uint32_t d, uint32_t a, uint32_t m, uint32_t L, deps dd, uint32_t pf) {
    red_f f{};
    f.kind = RED_KIND_SUM;
    f.d = d;
    f.a = a;
    f.m = m;
    f.rows = m;
    f.a_stride = L;
    return push_g(kf::red_fast(f), dd, pf);
}
[[gnu::noinline]] ticket psop(uint32_t sub, uint32_t sd, uint32_t sa, uint32_t addr, deps dd, uint32_t pf) {
    sop_f f{};
    f.sub = sub;
    f.sd = sd;
    f.sa = sa;
    f.addr = addr;
    return push_g(SOP(f), dd, pf);
}

[[gnu::noinline]] ticket pld(spm dst, dram src, uint32_t lines, deps dd, uint32_t pf) { return push_g(LD(dst, src, lines), dd, pf); }
[[gnu::noinline]] ticket pst(spm src, dram dst, uint32_t lines, deps dd, uint32_t pf) { return push_g(ST(src, dst, lines), dd, pf); }

// Rows [r0, r0 + nr) of a DRAM row tensor (stride lines apart; `lines` per row) into dst, packed: one
// LD when dense or nr = 1, else one per row. The first LD carries `first`; returns the last.
ticket load_rows(spm dst, dram src, uint32_t stride, uint32_t lines, uint32_t r0, uint32_t nr, deps first, uint32_t pf) {
    if (nr == 1u || stride == lines) return pld(dst, src + r0 * stride, nr * lines, first, pf);
    ticket last{};
    for (uint32_t i = 0; i < nr; ++i)
        last = pld(dst + i * lines, src + (r0 + i) * stride, lines, i == 0u ? first : deps{}, pf);
    return last;
}
ticket store_rows(spm src, dram dst, uint32_t stride, uint32_t lines, uint32_t r0, uint32_t nr, deps first, uint32_t pf) {
    if (nr == 1u || stride == lines) return pst(src, dst + r0 * stride, nr * lines, first, pf);
    ticket last{};
    for (uint32_t i = 0; i < nr; ++i)
        last = pst(src + i * lines, dst + (r0 + i) * stride, lines, i == 0u ? first : deps{}, pf);
    return last;
}

// ---------------------------------------------------------------------------------------------
// SOFTPLUS
// ---------------------------------------------------------------------------------------------
namespace sp = opl::softplus;
constexpr uint32_t S_NEG1 = 4u, S_TWO = 5u, S_HALF = 6u, S_C5 = 7u, S_C4 = 8u, S_C3 = 9u, S_C2 = 10u, S_C1 = 11u;
constexpr uint8_t CS(uint32_t cmode, uint32_t si) { return uint8_t(cmode | (si << 4)); }

// The 20 VOPs of the composition (header comment) on the roles B0 .. B4 of a tile.
constexpr vspec SPSTEP[20] = {
    {VOP_FUNC_MULC, CS(SR, S_NEG1), 1, 0, RN, RN, 0},   // B1 = x * -1
    {VOP_FUNC_MIN, 0, 1, 0, 1, RN, 0},                  // B1 = MINT(x, B1)
    {VOP_FUNC_EXP, 0, 2, 1, RN, RN, 0},                 // B2 = exp(B1)
    {VOP_FUNC_ADDC, CS(SR, S_TWO), 3, 2, RN, RN, 0},    // B3 = B2 + 2
    {VOP_FUNC_RECIP, 0, 3, 3, RN, RN, 0},               // B3 = 1 / B3
    {VOP_FUNC_MUL, 0, 3, 2, 3, RN, 0},                  // B3 = B2 * B3 (= s)
    {VOP_FUNC_MUL, 0, 2, 3, 3, RN, 0},                  // B2 = B3 * B3 (= z)
    {VOP_FUNC_MULC, CS(SR, S_C5), 4, 2, RN, RN, 0},     // B4 = z c5
    {VOP_FUNC_ADDC, CS(SR, S_C4), 4, 4, RN, RN, 0},     // B4 += c4
    {VOP_FUNC_MUL, 0, 4, 4, 2, RN, 0},                  // k = 3
    {VOP_FUNC_ADDC, CS(SR, S_C3), 4, 4, RN, RN, 0},
    {VOP_FUNC_MUL, 0, 4, 4, 2, RN, 0},                  // k = 2
    {VOP_FUNC_ADDC, CS(SR, S_C2), 4, 4, RN, RN, 0},
    {VOP_FUNC_MUL, 0, 4, 4, 2, RN, 0},                  // k = 1
    {VOP_FUNC_ADDC, CS(SR, S_C1), 4, 4, RN, RN, 0},
    {VOP_FUNC_MUL, 0, 4, 4, 2, RN, 0},                  // k = 0 (c0 = 2: s5)
    {VOP_FUNC_ADDC, CS(SR, S_TWO), 4, 4, RN, RN, 0},
    {VOP_FUNC_MUL, 0, 3, 3, 4, RN, 0},                  // B3 = s * acc (= y)
    {VOP_FUNC_SUB, 0, 1, 0, 1, RN, 0},                  // B1 = x - B1
    {VOP_FUNC_MULCADD, CS(SR, S_HALF), 3, 1, 3, RN, 0}, // B3 = FA(FM(B1, 1/2), B3)
};

[[gnu::noinline, gnu::cold]] ticket softplus_run(const uint32_t * w, const kind_plan & kp, deps d, kind_ctx & ctx) {
    const uint32_t pf = ctx.push_flags;
    const dram dst = from_handle(w[sp::DST]), src = from_handle(w[sp::SRC]);
    const uint32_t ds = w[sp::DST_STRIDE], ss = w[sp::SRC_STRIDE], n = w[sp::N], rows = w[sp::ROWS];
    const uint32_t L = kp.row_lines, t = kp.tile_rows;
    static constexpr uint32_t REG[8] = {S_NEG1, S_TWO, S_HALF, S_C5, S_C4, S_C3, S_C2, S_C1};
    static constexpr uint32_t VAL[8] = {sp::NEG_ONE, sp::TWO, sp::HALF, sp::COEF[5], sp::COEF[4], sp::COEF[3], sp::COEF[2], sp::COEF[1]};
    for (uint32_t i = 0; i < 8u; ++i) ek::sreg_set(true, ctx, REG[i], VAL[i], pf);
    spm_ring & ring = *ctx.ring;
    ticket last{};
    bool first = true;
    for (uint32_t r0 = 0; r0 < rows; r0 += t) {
        const uint32_t nr = min_u(t, rows - r0), tl = nr * L;
        deps rd{};
        const spm B = alloc_g(ring, plan::gdn::sp_block(nr, L), &rd);
        if (!valid(B)) return ctx.fail(opl::RET_SCRATCH);
        const deps f = first ? ek::first_deps(d, rd) : kf::drop_dma(rd);
        first = false;
        const uint32_t line[5] = {B.line, B.line + tl, B.line + 2u * tl, B.line + 3u * tl, B.line + 4u * tl};
        const ticket ld = load_rows(B, src, ss, L, r0, nr, f, pf);
        ticket vt{};
        for (uint32_t i = 0; i < 20u; ++i) vt = pv(SPSTEP[i], line, 0x1Fu, n, nr, L, i == 0u ? deps{ld} : deps{}, pf);
        last = store_rows(spm{line[3]}, dst, ds, L, r0, nr, deps{vt}, pf);
        ring.release(B, last);
    }
    return last;
}

// ---------------------------------------------------------------------------------------------
// SSM_CONV
// ---------------------------------------------------------------------------------------------
namespace sc = opl::sconv;

[[gnu::noinline, gnu::cold]] ticket sconv_run(const uint32_t * w, const kind_plan & kp, deps d, kind_ctx & ctx) {
    const uint32_t pf = ctx.push_flags;
    const dram dst = from_handle(w[sc::DST]), xs = from_handle(w[sc::X]), sin = from_handle(w[sc::STATE_IN]),
               sout = from_handle(w[sc::STATE_OUT]), taps = from_handle(w[sc::W]);
    const uint32_t dts = w[sc::DST_TOKEN_STRIDE], xts = w[sc::X_TOKEN_STRIDE];
    const uint32_t dc = w[sc::D_CONV], T = w[sc::N_TOKENS], ns = w[sc::N_SEQS], Lc = kp.row_lines;
    const uint32_t lt = kp.aux[0], per_seq = kp.aux[1], nw = dc - 1u + T;
    const bool silu = (w[sc::FLAGS] & sc::FLAG_SILU) != 0u;
    // Roles: 0 window, 1 taps, 2 ACC, 3 TMP; matrices with stride lj (b of the products is a tap row with stride 0).
    static constexpr vspec MUL0 = {VOP_FUNC_MUL, 0, 2, 0, 1, RN, 0}, MULK = {VOP_FUNC_MUL, 0, 3, 0, 1, RN, 0},
                           ADDK = {VOP_FUNC_ADD, 0, 2, 2, 3, RN, 0}, SILU = {VOP_FUNC_SILU, 0, 2, 2, RN, RN, 0};
    spm_ring & ring = *ctx.ring;
    ticket last{};
    bool first = true;
    for (uint32_t s = 0; s < ns; ++s) {
        for (uint32_t j = 0; j < per_seq; ++j) {
            const uint32_t j0 = j * lt, lj = min_u(lt, Lc - j0), m = 16u * lj;
            deps rd{};
            const spm B = alloc_g(ring, (nw + dc + 2u * T) * lj, &rd);
            if (!valid(B)) return ctx.fail(opl::RET_SCRATCH);
            const spm WIN = B, TAP = B + nw * lj, ACC = TAP + dc * lj, TMP = ACC + T * lj;
            deps f = first ? ek::first_deps(d, rd) : kf::drop_dma(rd);
            first = false;
            ticket ld{};
            for (uint32_t k = 0; k < dc; ++k, f = deps{}) ld = pld(TAP + k * lj, taps + k * Lc + j0, lj, f, pf);
            for (uint32_t k = 0; k + 1u < dc; ++k)
                ld = pld(WIN + k * lj, sin + (s * (dc - 1u) + k) * Lc + j0, lj, deps{}, pf);
            for (uint32_t t = 0; t < T; ++t)
                ld = pld(WIN + (dc - 1u + t) * lj, xs + (s * T + t) * xts + j0, lj, deps{}, pf);
            // The new state rows (window rows T .. T + d_conv - 2) leave as soon as the window is loaded.
            if (lj == Lc) {
                pst(WIN + T * lj, sout + s * (dc - 1u) * Lc, (dc - 1u) * lj, deps{}, pf);
            } else {
                for (uint32_t k = 0; k + 1u < dc; ++k)
                    pst(WIN + (T + k) * lj, sout + (s * (dc - 1u) + k) * Lc + j0, lj, deps{}, pf);
            }
            // acc = w0; acc += wk (k = 1 ..): the window operand of tap k starts at window row k.
            ticket vt{};
            for (uint32_t k = 0; k < dc; ++k) {
                const uint32_t line[4] = {WIN.line + k * lj, TAP.line + k * lj, ACC.line, TMP.line};
                vt = pv(k == 0u ? MUL0 : MULK, line, 0xDu, m, T, lj, k == 0u ? deps{ld} : deps{}, pf);   // matrices: window, ACC, TMP
                if (k != 0u) vt = pv(ADDK, line, 0xCu, m, T, lj, deps{}, pf);
            }
            if (silu) {
                const uint32_t line[4] = {0u, 0u, ACC.line, 0u};
                vt = pv(SILU, line, 0x4u, m, T, lj, deps{}, pf);
            }
            if (lj == Lc && dts == Lc) {
                last = pst(ACC, dst + s * T * dts, T * lj, deps{vt}, pf);
            } else {
                for (uint32_t t = 0; t < T; ++t)
                    last = pst(ACC + t * lj, dst + (s * T + t) * dts + j0, lj, t == 0u ? deps{vt} : deps{}, pf);
            }
            ring.release(B, last);
        }
    }
    return last;
}

// ---------------------------------------------------------------------------------------------
// GATED_DELTA_NET
// ---------------------------------------------------------------------------------------------
namespace gd = opl::gdn;
constexpr uint32_t S_G = 12u, S_SCALE = 13u, S_BETA = 14u, S_ALPHA = 15u;
// Roles: 0 S (the head's state tile), 1 P, 2 k, 3 q, 4 v, 5 u, 6 w; the matrices (stride L) are S and P.
constexpr uint8_t R_S = 0, R_P = 1, R_K = 2, R_Q = 3, R_V = 4, R_U = 5, R_W = 6;
constexpr uint32_t TILES = (1u << R_S) | (1u << R_P);
// Dependency classes of a step's ops: index into the step's ticket table.
enum : uint8_t { D_NONE = 0, D_STATE = 1, D_K = 2, D_V = 3, D_Q = 4 };
struct gstep {
    vspec v;
    uint8_t dep;
};
// The 9 VOPs of a token step around the two RED.SUMs (the numerics of ref/gdn_ref.cpp gdn_ref_step).
constexpr gstep GA[2] = {   // before the first RED
    {{VOP_FUNC_MULC, CS(SR, S_ALPHA), R_S, R_S, RN, RN, 0}, D_STATE},   // S *= alpha
    {{VOP_FUNC_MUL, 0, R_P, R_S, R_K, RN, 0}, D_K},                     // P = S * k
};
constexpr gstep GB[4] = {   // after the first RED (u = rowsums): delta, update, product with q
    {{VOP_FUNC_SUB, 0, R_U, R_V, R_U, RN, 1}, D_V},                     // u = v - u
    {{VOP_FUNC_MULC, CS(SR, S_BETA), R_U, R_U, RN, RN, 1}, D_NONE},     // delta = u * beta
    {{VOP_FUNC_MULCADD, CS(VEC, 0), R_S, R_K, R_S, R_U, 0}, D_NONE},    // S = FA(FM(k, delta_j), S)
    {{VOP_FUNC_MUL, 0, R_P, R_S, R_Q, RN, 0}, D_Q},                     // P = S * q
};
constexpr vspec GC = {VOP_FUNC_MULC, CS(SR, S_SCALE), R_W, R_W, RN, RN, 1};   // o = w * scale

struct slot_t {
    uint32_t k, q, v, g, b, u, w;   // lines of the token slot (5 L + 2 lines)
    ticket ld_g, ld_b, ld_k, ld_v, ld_q;
    ticket done;               // the last VU op reading or writing the slot
    ticket st_o;               // the ST that reads the slot's output vector w
};

// What a token step returns: the last VU op that writes the state tile (the state ST of the head's last token
// waits for it) and the last VU op of the step (it ends every read and write of the slot).
struct step_out {
    ticket state, last;
};

// One token of one head, composed path (revision 1): alpha = exp(g), beta, 9 VOPs and two RED.SUMs.
// hl: the lane of the head in its g / beta line; st_ld: the state LD when this is the head's first token.
[[gnu::noinline, gnu::cold]] step_out gdn_step_composed(uint32_t pf, uint32_t sv, uint32_t L, uint32_t hl, const slot_t & s,
                                                        uint32_t S_line, uint32_t P_line, ticket st_ld) {
    const uint32_t line[7] = {S_line, P_line, s.k, s.q, s.v, s.u, s.w};
    const ticket dep[5] = {ticket{}, st_ld, s.ld_k, s.ld_v, s.ld_q};
    ticket v_mulcadd{};
    psop(SOP_SUB_LDS, S_G, 0u, 16u * s.g + hl, deps{s.ld_g}, pf);          // alpha = exp(g), beta
    psop(SOP_SUB_EXP, S_ALPHA, S_G, 0u, deps{}, pf);
    psop(SOP_SUB_LDS, S_BETA, 0u, 16u * s.b + hl, deps{s.ld_b}, pf);
    for (const gstep & g : GA) pv(g.v, line, TILES, sv, sv, L, deps{dep[g.dep]}, pf);
    pred(s.u, P_line, sv, L, deps{}, pf);                                   // u = rowsums
    for (const gstep & g : GB) {
        const ticket tk = pv(g.v, line, TILES, sv, sv, L, deps{dep[g.dep]}, pf);
        if (g.v.func == VOP_FUNC_MULCADD) v_mulcadd = tk;
    }
    pred(s.w, P_line, sv, L, deps{s.st_o}, pf);                             // w = rowsums (w: previous ST done)
    return {v_mulcadd, pv(GC, line, TILES, sv, sv, L, deps{}, pf)};
}

// The same token, fused path (revision 2): one GDN op updates the state tile in place and writes the output
// vector (rows = m = S_v), then VOP.MULC scales it. The op reads the state tile, k, q and v and writes the
// tile and w. The DMA tickets (the state LD, the LDs of k, v and q, the ST that last read w) are one unit:
// the latest of them covers the others (I1).
[[gnu::noinline, gnu::cold]] step_out gdn_step_fused(uint32_t pf, uint32_t sv, uint32_t L, uint32_t hl, const slot_t & s,
                                                     uint32_t S_line, ticket st_ld) {
    psop(SOP_SUB_LDS, S_G, 0u, 16u * s.g + hl, deps{s.ld_g}, pf);
    psop(SOP_SUB_EXP, S_ALPHA, S_G, 0u, deps{}, pf);
    psop(SOP_SUB_LDS, S_BETA, 0u, 16u * s.b + hl, deps{s.ld_b}, pf);
    gdn_f f{};
    f.sa = S_ALPHA;
    f.sb = S_BETA;
    f.d = f.a = S_line;
    f.k = s.k;
    f.q = s.q;
    f.v = s.v;
    f.o = s.w;
    f.m = sv;
    f.rows = sv;
    const ticket dma = later(later(later(st_ld, s.ld_k), later(s.ld_v, s.ld_q)), s.st_o);
    const ticket gt = push_g(GDN(f), deps{dma}, pf);
    const uint32_t line[7] = {S_line, 0u, s.k, s.q, s.v, s.u, s.w};
    return {gt, pv(GC, line, TILES, sv, sv, L, deps{}, pf)};
}

[[gnu::noinline, gnu::cold]] ticket gdn_run(const uint32_t * a, const kind_plan & kp, deps d, kind_ctx & ctx) {
    const uint32_t pf = ctx.push_flags;
    const uint32_t sv = a[gd::S_V], hk = a[gd::H_K], hv = a[gd::H_V], T = a[gd::N_TOKENS], ns = a[gd::N_SEQS];
    const uint32_t L = kp.row_lines, SL = sv * L, slot_lines = kp.aux[1];
    const bool fused = plan::gdn::use_fused(ctx.pl.caps, sv);
    const dram dst = from_handle(a[gd::DST]), sout = from_handle(a[gd::STATE_OUT]), sin = from_handle(a[gd::STATE_IN]),
               qd = from_handle(a[gd::Q]), kd = from_handle(a[gd::K]), vd = from_handle(a[gd::V]),
               gd_ = from_handle(a[gd::G]), bd = from_handle(a[gd::BETA]);
    spm_ring & ring = *ctx.ring;
    deps rd{};
    const spm B = alloc_g(ring, kp.scratch_lines, &rd);
    if (!valid(B)) return ctx.fail(opl::RET_SCRATCH);
    const spm St[2] = {B, B + SL}, P = B + 2u * SL;
    slot_t sl[2] = {};
    for (uint32_t i = 0; i < 2u; ++i) {
        const uint32_t s0 = B.line + 3u * SL + i * slot_lines;
        sl[i].k = s0;
        sl[i].q = s0 + L;
        sl[i].v = s0 + 2u * L;
        sl[i].g = s0 + 3u * L;
        sl[i].b = s0 + 3u * L + 1u;
        sl[i].u = s0 + 3u * L + 2u;
        sl[i].w = s0 + 4u * L + 2u;
    }
    ek::sreg_set(true, ctx, S_SCALE, a[gd::SCALE], pf);
    kf::sreg_clobber(ctx, S_G);
    kf::sreg_clobber(ctx, S_BETA);
    kf::sreg_clobber(ctx, S_ALPHA);

    const uint32_t nheads = ns * hv, nsteps = nheads * T;
    ticket st_ld[2] = {};                       // the state LD of each tile's current head
    ticket last{};

    // Loads of step i into slot i & 1: g, beta, k, v, q (this DMA order; a VU op waits for the LD of
    // what it reads). The slot's previous user (step i - 2) must be done with it.
    auto load_step = [&](uint32_t i) {
        slot_t & s = sl[i & 1u];
        const uint32_t hidx = i / T, t = i - hidx * T, sq = hidx / hv, h = hidx - sq * hv, hq = h % hk;
        const uint32_t n = sq * T + t;
        const deps wait = i >= 2u ? deps{s.done} : deps{};
        s.ld_g = pld(spm{s.g}, gd_ + n * a[gd::G_TOKEN_STRIDE] + (h >> 4), 1u, wait, pf);
        s.ld_b = pld(spm{s.b}, bd + n * a[gd::BETA_TOKEN_STRIDE] + (h >> 4), 1u, deps{}, pf);
        s.ld_k = pld(spm{s.k}, kd + hq * a[gd::K_HEAD_STRIDE] + n * a[gd::K_TOKEN_STRIDE], L, deps{}, pf);
        s.ld_v = pld(spm{s.v}, vd + h * a[gd::V_HEAD_STRIDE] + n * a[gd::V_TOKEN_STRIDE], L, deps{}, pf);
        s.ld_q = pld(spm{s.q}, qd + hq * a[gd::Q_HEAD_STRIDE] + n * a[gd::Q_TOKEN_STRIDE], L, deps{}, pf);
    };
    // The state of head hidx into tile hidx & 1 (its previous user's ST precedes in the DMA queue).
    auto load_state = [&](uint32_t hidx, deps first) {
        st_ld[hidx & 1u] = pld(St[hidx & 1u], sin + hidx * SL, SL, first, pf);
    };

    load_state(0u, ek::first_deps(d, rd));
    load_step(0u);
    if (nsteps > 1u) load_step(1u);

    for (uint32_t hidx = 0; hidx < nheads; ++hidx) {
        const uint32_t sq = hidx / hv, h = hidx - sq * hv;
        const spm S = St[hidx & 1u];
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t i = hidx * T + t;
            slot_t & s = sl[i & 1u];
            const uint32_t n = sq * T + t;
            if (t + 1u == T && hidx + 1u < nheads) load_state(hidx + 1u, deps{});   // prefetch the next head's state
            const ticket first_ld = t == 0u ? st_ld[hidx & 1u] : ticket{};
            const step_out so = fused ? gdn_step_fused(pf, sv, L, h & 15u, s, S.line, first_ld)
                                      : gdn_step_composed(pf, sv, L, h & 15u, s, S.line, P.line, first_ld);
            s.done = so.last;
            if (t + 1u == T) pst(S, sout + hidx * SL, SL, deps{so.state}, pf);
            last = s.st_o = pst(spm{s.w}, dst + uint32_t((uint64_t(n) * hv + h) * L), L, deps{so.last}, pf);
            if (i + 2u < nsteps) load_step(i + 2u);
        }
    }
    // The last ST of the DMA queue is the last head's output row (its state ST precedes it, in order).
    ring.release(B, last);
    return last;
}

// ---------------------------------------------------------------------------------------------
// GATED_DELTA_NET, the token block pipeline of the fused path (SW-70, plan_gdn.hpp)
// ---------------------------------------------------------------------------------------------
// One block of the ring for the record: R state tiles of S_v L lines, then two token blocks of
//   [ g (Lg) | beta (Lg) | alpha (Lg) | k (H_k L) | q (H_k L) | v (Gs L) | o (Gs L) ]      Lg = ceil(H_v / 16)
// (kp.aux[1] lines each). A step is one (sequence, token) of a group of Gs heads: the operands of all its heads load
// together (g and beta: one LD each of Lg lines; k and q: the H_k key rows; v: the rows of the group's heads), VOP.EXP
// turns the H_v words of g into alpha (the function SOP.EXP computes, so the head step's SOP.LDS reads the bits that
// SOP.LDS then SOP.EXP gives), and each head is SOP.LDS alpha, SOP.LDS beta and one GDN op, whose output row goes to
// row (h - h0) of the o rows. After the last head one VOP.MULC scales all the o rows and one ST writes them (the
// rows of a token are contiguous in the result). The step after the next loads into the block when this step's VOP.MULC,
// the last VU op that reads the block, is done. A GDN op depends on the latest DMA op of what it reads (the state
// LD, the last LD of the block and, which the LD already implies, the ST that last read the o rows): q and v are not
// among the op's extents (VU-35), so these dependencies order them.
// T = 1: R = 2, Gs = H_v, a step is a sequence; the head after next loads into a tile after the ST of the state of
// the head that used it (DMA queue order). T > 1: R = Gs, the state tiles of a group stay in the SPM while the T
// tokens run, tokens outer and heads inner. During the first token the state of head j + 2 loads after the step of
// head j (the first two load before the group's first step); during the last token the state ST follows each GDN
// op. The next group's first state LDs come after those STs in the DMA queue, as do the LDs of the rows of
// a step into a block after the ST of its o rows. Nothing in the head loop divides or multiplies by a run-time
// value: lines and addresses are carried along.
// Cannot deadlock: every dependency names an op that was pushed earlier, so the oldest op that has not
// completed has every dependency and every op ahead of it in its queue done, whatever the queue depths (DMA 8, VU 16)
// make the pushes wait for.
struct tblk {
    uint32_t g, b, a, k, q, v, o;   // lines
    ticket ld_g, ld_b, ld;          // the LDs of g and beta, the last LD of the block's rows
    ticket mulc;                    // the VOP.MULC of the block's last step (the last VU op that reads it)
    ticket st_o;                    // the ST that last read the o rows
};
struct gcur {
    uint32_t sq, h0, gn, t;         // the step: sequence, first head of the group, its heads, token
};

[[gnu::noinline, gnu::cold]] ticket gdn_run_tok(const uint32_t * a, const kind_plan & kp, deps d, kind_ctx & ctx) {
    const uint32_t pf = ctx.push_flags;
    const uint32_t sv = a[gd::S_V], hk = a[gd::H_K], hv = a[gd::H_V], T = a[gd::N_TOKENS], ns = a[gd::N_SEQS];
    const uint32_t L = kp.row_lines, SL = sv * L, Lg = (hv + 15u) >> 4, Gs = kp.aux[3];
    const bool res = T > 1u;                    // the state tiles stay for the tokens (R = Gs), else two tiles in turn
    const uint32_t nheads = ns * hv;
    const dram dst = from_handle(a[gd::DST]), sout = from_handle(a[gd::STATE_OUT]), sin = from_handle(a[gd::STATE_IN]),
               qd = from_handle(a[gd::Q]), kd = from_handle(a[gd::K]), vd = from_handle(a[gd::V]),
               gd_ = from_handle(a[gd::G]), bd = from_handle(a[gd::BETA]);
    spm_ring & ring = *ctx.ring;
    deps rd{};
    const spm B = alloc_g(ring, kp.scratch_lines, &rd);
    if (!valid(B)) return ctx.fail(opl::RET_SCRATCH);
    tblk tb[2] = {};
    for (uint32_t i = 0; i < 2u; ++i) {
        tblk & b = tb[i];
        b.g = B.line + kp.aux[2] * SL + i * kp.aux[1];
        b.b = b.g + Lg;
        b.a = b.b + Lg;
        b.k = b.a + Lg;
        b.q = b.k + hk * L;
        b.v = b.q + hk * L;
        b.o = b.v + Gs * L;
    }
    ek::sreg_set(true, ctx, S_SCALE, a[gd::SCALE], pf);
    kf::sreg_clobber(ctx, S_BETA);
    kf::sreg_clobber(ctx, S_ALPHA);

    // The step after c; gn is the heads of its group.
    auto adv = [&](gcur & c) {
        if (++c.t == T) {
            c.t = 0;
            c.h0 += Gs;
            if (c.h0 >= hv) {
                c.h0 = 0;
                ++c.sq;
            }
        }
        c.gn = min_u(Gs, hv - c.h0);
    };
    // The rows of step c into block b: g, beta, k, q, v in this DMA order; the first LD carries `first`.
    auto load_block = [&](tblk & b, const gcur & c, deps first) {
        const uint32_t n = c.sq * T + c.t;
        b.ld_g = pld(spm{b.g}, gd_ + n * a[gd::G_TOKEN_STRIDE], Lg, first, pf);
        b.ld_b = pld(spm{b.b}, bd + n * a[gd::BETA_TOKEN_STRIDE], Lg, deps{}, pf);
        load_rows(spm{b.k}, kd + n * a[gd::K_TOKEN_STRIDE], a[gd::K_HEAD_STRIDE], L, 0u, hk, deps{}, pf);
        load_rows(spm{b.q}, qd + n * a[gd::Q_TOKEN_STRIDE], a[gd::Q_HEAD_STRIDE], L, 0u, hk, deps{}, pf);
        b.ld = load_rows(spm{b.v}, vd + n * a[gd::V_TOKEN_STRIDE], a[gd::V_HEAD_STRIDE], L, c.h0, c.gn, deps{}, pf);
    };
    ticket ldc{}, ldn{};                        // the state LDs of the current head and of the next one
    // The states of the first two heads of a group into tiles 0 and 1 (T = 1: of the record).
    auto group_start = [&](uint32_t x0, uint32_t gn) {
        const uint32_t lim = res ? gn : nheads;
        ldc = pld(B, sin + x0 * SL, SL, deps{}, pf);
        ldn = lim > 1u ? pld(B + SL, sin + (x0 + 1u) * SL, SL, deps{}, pf) : ticket{};
    };

    gcur lc{0u, 0u, min_u(Gs, hv), 0u};         // the step whose rows load next
    load_block(tb[0], lc, ek::first_deps(d, rd));
    adv(lc);
    group_start(0u, lc.gn);
    if (lc.sq < ns) {
        load_block(tb[1], lc, deps{});
        adv(lc);
    }

    ticket last{};
    uint32_t bi = 0u, x0 = 0u;                  // the block of the step, the first head of the group (over sequences)
    for (uint32_t sq = 0; sq < ns; ++sq) {
        for (uint32_t h0 = 0; h0 < hv; h0 += Gs) {
            const uint32_t gn = min_u(Gs, hv - h0), hq0 = h0 % hk;
            if (res && x0 != 0u) group_start(x0, gn);
            for (uint32_t t = 0; t < T; ++t) {
                tblk & b = tb[bi];
                bi ^= 1u;
                const bool ft = t == 0u, lt = t + 1u == T;
                const uint32_t n = sq * T + t;
                vop_f e{};                      // alpha = exp(g) over the H_v words of g
                e.cmode = NOC;
                e.func = VOP_FUNC_EXP;
                e.d = b.a;
                e.a = b.g;
                e.m = hv;
                e.rows = 1u;
                push_g(kf::vop_fast(e), deps{b.ld_g}, pf);
                uint32_t hq = hq0, kq = hq0 * L, tl = B.line, xl = x0 * SL, vo = 0u;
                for (uint32_t j = 0; j < gn; ++j) {
                    if (!res) tl = B.line + (((x0 + j) & 1u) != 0u ? SL : 0u);
                    psop(SOP_SUB_LDS, S_ALPHA, 0u, 16u * b.a + h0 + j, deps{}, pf);
                    psop(SOP_SUB_LDS, S_BETA, 0u, 16u * b.b + h0 + j, j == 0u ? deps{b.ld_b} : deps{}, pf);
                    gdn_f f{};
                    f.sa = S_ALPHA;
                    f.sb = S_BETA;
                    f.d = f.a = tl;
                    f.k = b.k + kq;
                    f.q = b.q + kq;
                    f.v = b.v + vo;
                    f.o = b.o + vo;
                    f.m = sv;
                    f.rows = sv;
                    ticket dma = j == 0u ? later(b.ld, b.st_o) : ticket{};
                    if (ft) dma = later(dma, ldc);
                    const ticket gt = push_g(GDN(f), deps{dma}, pf);
                    if (lt) pst(spm{tl}, sout + xl, SL, deps{gt}, pf);
                    // The state of the head two on, into the tile this head's ST has read (T = 1) or into its own.
                    ticket nl{};
                    if (res ? (ft && j + 2u < gn) : (x0 + j + 2u < nheads))
                        nl = pld(spm{res ? tl + 2u * SL : tl}, sin + (xl + 2u * SL), SL, deps{}, pf);
                    ldc = ldn;
                    ldn = nl;
                    if (res) tl += SL;
                    xl += SL;
                    vo += L;
                    kq += L;
                    if (++hq == hk) {
                        hq = 0u;
                        kq = 0u;
                    }
                }
                vop_f m{};                      // o = o' * scale over the Gs rows (one vector of Gs S_v words)
                m.cmode = SR;
                m.func = VOP_FUNC_MULC;
                m.si = S_SCALE;
                m.d = b.o;
                m.a = b.o;
                m.m = gn * sv;
                m.rows = 1u;
                b.mulc = push_g(kf::vop_fast(m), deps{}, pf);
                last = b.st_o = pst(spm{b.o}, dst + (n * hv + h0) * L, gn * L, deps{b.mulc}, pf);
                if (lc.sq < ns) {               // the step after the next one, into this block
                    load_block(b, lc, deps{b.mulc});
                    adv(lc);
                }
            }
            x0 += gn;
        }
    }
    // The last DMA op is the o rows' ST of the last step; the state STs of the last token precede it in the queue.
    ring.release(B, last);
    return last;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------------------------
ticket kind_softplus(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    uint32_t w[sp::NARGS];
    for (uint32_t i = 0; i < sp::NARGS; ++i) w[i] = args[i];
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    kind_plan kp{};
    if (!plan_softplus(w, n_args, ctx.pl, &kp)) return ctx.fail(kp.status);
    return softplus_run(w, kp, d, ctx);
}

ticket kind_ssm_conv(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    uint32_t w[sc::NARGS];
    for (uint32_t i = 0; i < sc::NARGS; ++i) w[i] = args[i];
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    kind_plan kp{};
    if (!plan_ssm_conv(w, n_args, ctx.pl, &kp)) return ctx.fail(kp.status);
    return sconv_run(w, kp, d, ctx);
}

ticket kind_gated_delta_net(const volatile uint32_t * args, uint32_t n_args, uint32_t kflags, deps d, kind_ctx & ctx) {
    uint32_t w[gd::NARGS];
    for (uint32_t i = 0; i < gd::NARGS; ++i) w[i] = args[i];
    if (kflags != 0u) return ctx.fail(opl::RET_BAD_RECORD);
    kind_plan kp{};
    if (!plan_gated_delta_net(w, n_args, ctx.pl, &kp)) return ctx.fail(kp.status);
    return kp.aux[2] != 0u ? gdn_run_tok(w, kp, d, ctx) : gdn_run(w, kp, d, ctx);
}

}  // namespace tsr
