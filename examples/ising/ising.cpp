// Example kernel: an Ising machine. Discrete simulated bifurcation (dSB) on the device.
//
// The problem is s_i = +-1 with the energy E = -1/2 sum_ij J_ij s_i s_j (J symmetric, zero diagonal).
// Every spin has a position x_i and a momentum y_i. One step, for t replicas side by side:
//
//     s = sgn(x)                          the signs (0 for x = 0)
//     f = J s                             the field: a matrix-vector product on the device
//     y += cA x + cF f                    cA rises from -a0 dt to 0 over the run: the bifurcation
//     x += cD y
//     where |x| >= 1:  x = sgn(x), y = 0  a wall at +-1
//
// The answer is sgn(x) at the end. The control core has no floating point: the host passes every
// constant as f32 bits, and every operation below is one operation of the vector unit, rounded
// separately (docs/design/02-vector-unit.md §6), so the host reference (ising_ref.cpp) follows the
// device bit for bit.
//
// The field is exact. QACT turns s = +-1 into q = +-127 and stores the block scale 1/127 rounded to
// fp16, which is not 1/127 (spec §6). An MVM accepts any bytes as its activation image (spec §7), so
// after the QACT the kernel overwrites the scale words with the f32 nearest to 1/127 (0x3C010204).
// With J as Q8_0 with block scale 1.0 and small integer values, the term of a block is
// (1.0 * D) * f32(127 m) = m exactly (m = the sum of J s over the block: D * 127 m rounds to m for
// every |m| < 2^15), and f is the exact integer J s as long as n * max|J| stays below 2^24.
//
// Parameters: a block in DRAM (ising_abi.hpp) that the runtime loads into the SPM (ARGBLK). For given
// parameters the kernel is a fixed stream of ops. Only a soft stop (polled once per step) ends it early.
#include "tsr/kernel.hpp"
#include "tsr/rt.hpp"

#include "ising_abi.hpp"

using namespace tsr;

TSR_KERNEL(ising, 1, TSR_KFLAG_ARGBLK, ism::PARAM_LINES);

namespace {

// f32 bits of the constants of the step.
constexpr uint32_t BIG = 0x71800000u;        // 2^100: clip(x * 2^100, -1, 1) = sgn(x) unless 0 < |x| < 2^-100
constexpr uint32_t ONE = 0x3F800000u;        // +1
constexpr uint32_t MINUS_ONE = 0xBF800000u;  // -1
constexpr uint32_t SCALE = 0x3C010204u;      // the f32 nearest to 1/127
constexpr uint32_t P24 = 0x4B800000u;        // 2^24
constexpr uint32_t M24 = 0xCB800000u;        // -2^24

// Scalar registers: s0 BIG, s1 SCALE, s2 cA, s3 dA, s4 cF, s5 cD, s6 -2^24, s7 2^24, s8 +1, s9 -1.
enum : uint32_t { S_BIG = 0, S_SCALE, S_CA, S_DA, S_CF, S_CD, S_M24, S_P24, S_ONE, S_MONE };

void movi(uint32_t sd, uint32_t bits) { issue(SOP({.sub = SOP_SUB_MOVI, .sd = sd, .imm = bits})); }

}  // namespace

extern "C" int32_t ising(const tsr_args * args) {
    // The parameter block, copied out of the SPM, and its checks against this device.
    ism::params p;
    {
        const volatile uint32_t * w = reinterpret_cast<const volatile uint32_t *>(args->blk);
        uint32_t * d = reinterpret_cast<uint32_t *>(&p);
        for (uint32_t i = 0; i < sizeof(p) / 4; ++i) d[i] = w[i];
    }
    const uint32_t cfg0 = core_csr(CSR_CFG0), cfg2 = core_csr(CSR_CFG2);
    ism::layout lay;
    const int32_t st = ism::make_layout(p, argblk_line(), field_get(cfg0, CFG0_T_MAX, CFG0_T_MAX_LSB),
                                          field_get(cfg2, CFG2_K_MAX_256, CFG2_K_MAX_256_LSB) * 256u, &lay);
    if (st != ism::ST_OK) return st;

    const uint32_t n = p.n, t = p.t, L = lay.L, tl = t * L, nblk = lay.nblk;
    const bool energy = (p.flags & ism::FLAG_ENERGY) != 0u;

    // One VOP over the t rows of n floats: d = func(a, b, c). The rows of d and a are L lines apart; b is
    // one row for all replicas (b_stride 0) or L lines apart; c is the scalar register si. An operand
    // that the function does not read is 0 (VU §5.2).
    auto vu = [&](uint32_t func, uint32_t d, uint32_t a, uint32_t b, uint32_t b_stride, uint32_t si, deps dp = {}) {
        const uint32_t rd = vop_func_reads(func);
        const bool ra = (rd & VOP_READS_A) != 0u, rb = (rd & VOP_READS_B) != 0u, rc = (rd & VOP_READS_C) != 0u;
        return issue(VOP({.cmode = rc ? VOP_CMODE_SREG : 0u, .func = func, .si = rc ? si : 0u, .d = d,
                          .a = ra ? a : 0u, .b = rb ? b : 0u, .m = n, .rows = t, .d_stride = L,
                          .a_stride = ra ? L : 0u, .b_stride = rb ? b_stride : 0u}), dp);
    };

    // ---- prologue: the scalar registers, the two constant rows, the state ---------------------------
    movi(S_BIG, BIG);
    movi(S_SCALE, SCALE);
    movi(S_CA, p.ca0);
    movi(S_DA, p.da);
    movi(S_CF, p.cf);
    movi(S_CD, p.cd);
    movi(S_M24, M24);
    movi(S_P24, P24);
    movi(S_ONE, ONE);
    movi(S_MONE, MINUS_ONE);
    issue(VOP({.cmode = VOP_CMODE_SREG, .func = VOP_FUNC_SETC, .si = S_ONE, .d = lay.one, .m = n, .rows = 1}));
    issue(VOP({.cmode = VOP_CMODE_SREG, .func = VOP_FUNC_SETC, .si = S_MONE, .d = lay.neg, .m = n, .rows = 1}));
    issue(LD(spm{lay.x}, from_handle(p.x), tl));
    const ticket ld_y = issue(LD(spm{lay.y}, from_handle(p.y), tl));   // the DMA unit completes in order: ld_y covers the LD of x

    // Tickets that later ops wait for. The VU runs its ops in queue order, and so does every unit, so an op
    // only waits for ops of other units: the loads, the QACTs that read s, the MVMs that read the QA
    // images, the snapshot that reads x.
    ticket ld = ld_y;               // the first op that reads x waits for the loads
    ticket q_last;                  // the last QACT of the previous step
    ticket mvm[ism::MAX_BLK];     // the MVM of each column tensor in the previous step
    ticket snap;                    // the last snapshot ST
    ticket v_x = ld_y, v_y = ld_y;  // the last VU op that wrote x, y
    ticket v_e;                     // the last energy RED
    uint32_t done_steps = 0, snaps = 0;

    for (uint32_t step = 0; step < p.steps; ++step) {
        if (stop_requested()) break;   // the steps pushed so far finish and the state is stored below (EX §4.9)

        // ---- s = clip(x * 2^100, -1, 1): the signs ---------------------------------------------------
        // s replaces the signs of the previous step: wait for the QACTs that read them.
        vu(VOP_FUNC_MULC, lay.s, lay.x, 0, 0, S_BIG, after(q_last, ld));
        ld = ticket{};
        vu(VOP_FUNC_MIN, lay.s, lay.s, lay.one, 0, 0);
        const ticket v_s = vu(VOP_FUNC_MAX, lay.s, lay.s, lay.neg, 0, 0);

        // ---- one QA image of s per column tensor, with the exact scale -------------------------------
        // A QACT replaces its image: wait for the MVM of the previous step that read it. The SETC then
        // overwrites the scale words (K / 32 per image, QA_LINES apart).
        ticket setc[ism::MAX_BLK];
        for (uint32_t b = 0; b < nblk; ++b) {
            const uint32_t k = lay.kb[b];
            q_last = issue(QACT(qa::a32, spm{lay.qa[b]}, spm{lay.s + b * (p.kblk / 16u)}, k, t, L), after(v_s, mvm[b]));
            setc[b] = issue(VOP({.cmode = VOP_CMODE_SREG, .func = VOP_FUNC_SETC, .si = S_SCALE,
                                 .d = lay.qa[b] + ism::qa_scale_line(k), .m = k / 32u, .rows = t,
                                 .d_stride = ism::qa_lines(k)}), after(q_last));
        }

        // ---- y1 = x cA + y: it does not need the field, so it runs while the MVMs stream the weights ---
        vu(VOP_FUNC_MULCADD, lay.y, lay.x, lay.y, L, S_CA);

        // ---- f = J s: one MVM per column tensor, the later ones accumulate (f += J_b s_b) -------------
        // Each MVM waits for its SETC. The MVMs of one unit see each other's results in queue order,
        // so an accumulating MVM needs no ticket for the one before it (docs/ctrl-riscv.md §5.5).
        ticket m_last;
        for (uint32_t b = 0; b < nblk; ++b) {
            m_last = issue(MVM(spm{lay.f}, spm{lay.qa[b]}, wref{p.w_off + b * p.tensor_stride, FMT_Q8_0}, n, lay.kb[b], t, L,
                               b != 0u), after(setc[b]));
            mvm[b] = m_last;
        }

        // ---- e = sum_i s_i f_i per replica (the energy of the signs that entered the step is -e / 2) ---
        // The first VU op that reads f waits for the last MVM.
        deps wait_f = after(m_last);
        if (energy) {
            vu(VOP_FUNC_MUL, lay.tmp, lay.s, lay.f, L, 0, wait_f);
            wait_f = deps{};
            v_e = issue(RED({.kind = RED_KIND_SUM, .d = lay.e + done_steps, .a = lay.tmp, .m = n, .rows = t, .a_stride = L}));
        }

        // ---- y2 = f cF + y1, x1 = y2 cD + x -----------------------------------------------------------
        vu(VOP_FUNC_MULCADD, lay.y, lay.f, lay.y, L, S_CF, wait_f);
        // x is read by the last snapshot: wait for it before x is written.
        vu(VOP_FUNC_MULCADD, lay.x, lay.y, lay.x, L, S_CD, after(snap));
        snap = ticket{};

        // ---- the wall: x2 = clip(x1, -1, 1) ------------------------------------------------------------
        vu(VOP_FUNC_MIN, lay.x, lay.x, lay.one, 0, 0);
        v_x = vu(VOP_FUNC_MAX, lay.x, lay.x, lay.neg, 0, 0);
        ++done_steps;
        if (p.snap_every != 0u && done_steps % p.snap_every == 0u) {
            snap = issue(ST(spm{lay.x}, from_handle(p.snap) + snaps * tl, tl), after(v_x));
            ++snaps;
        }

        // ---- y3 = y2 * m, m = 1 inside the walls and 0 at |x2| = 1 ---------------------------------------
        // m = min(2^24 - 2^24 x2 x2, 1). For |x2| < 1 the rounded 2^24 x2^2 is at most 2^24 - 2, so m = 1.
        vu(VOP_FUNC_MULCB, lay.tmp, lay.x, lay.x, L, S_M24);
        vu(VOP_FUNC_ADDC, lay.tmp, lay.tmp, 0, 0, S_P24);
        vu(VOP_FUNC_MIN, lay.tmp, lay.tmp, lay.one, 0, 0);
        v_y = vu(VOP_FUNC_MUL, lay.y, lay.y, lay.tmp, L, 0);

        // ---- cA += dA --------------------------------------------------------------------------------------
        issue(SOP({.sub = SOP_SUB_ADD, .sd = S_CA, .sa = S_CA, .sb = S_DA}));
    }

    // ---- store the state, and the energy lines of the steps done ---------------------------------------
    issue(ST(spm{lay.x}, from_handle(p.x), tl), after(v_x));
    issue(ST(spm{lay.y}, from_handle(p.y), tl), after(v_y));
    if (energy && done_steps != 0u) issue(ST(spm{lay.e}, from_handle(p.energy), done_steps), after(v_e));
    return done_steps < p.steps ? ism::ST_STOPPED_FLAG | int32_t(done_steps) : ism::ST_OK;
}
