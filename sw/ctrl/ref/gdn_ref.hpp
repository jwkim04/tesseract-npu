// gdn_ref.hpp — host C++ references of the Gated DeltaNet kinds SOFTPLUS (0x2D), SSM_CONV (0x2E) and
// GATED_DELTA_NET (0x2F) (docs/design/06-software-verification.md "SW" §7.9, decisions SW-60 to
// SW-63). Like the other references of sw/ctrl/ref they compose the tsr_ref_vu calls that the
// kernels' tile ops execute, in the kernels' order, so the kernel tests compare the firmware against
// them bit for bit. Rows, tiles and heads never change a value: every element is computed by the
// same sequence whatever the tiling.
//
// SOFTPLUS, per element x (VU rules: DAZ, RNE, FTZ, canonical NaN; FM, FA, FS, RC, MINT, EXPK of
// VU §6), the stable form softplus(x) = max(x, 0) + log1p(exp(-|x|)) with log1p(h) = 2 atanh(s),
// s = h / (2 + h) in (0, 1/3], the odd series of atanh to z^5 (z = s s):
//   nx  = FM(x, -1)             nax = MINT(x, nx)  (= -|x|)        h = EXPK(nax)
//   den = FA(h, 2)              r   = RC(den)                      s = FM(h, r)       z = FM(s, s)
//   acc = FA(FM(z, c5), c4); acc = FA(FM(acc, z), c_k) for k = 3, 2, 1, 0      (c_k = 2 / (2k + 1))
//   y   = FM(s, acc)            m   = FS(x, nax)  (= x + |x|)
//   out = FA(FM(m, 1/2), y)
// 20 VOPs. -inf gives NaN (FS(-inf, -inf)); +inf gives +inf; the error against the exact softplus is
// below 5 ulp for results above 2^-20 and below 4.2e-7 relative everywhere (gdn_accuracy test).
//
// SSM_CONV: the channel-vector form of ggml_ssm_conv (record words opl::sconv).
// GATED_DELTA_NET: ggml_gated_delta_net with a scalar gate and K = 1 (record words opl::gdn), the
// per-token step gdn_ref_step.
#pragma once

#include <cstddef>
#include <cstdint>

#include "mulmat_ref.hpp"   // dram_view

namespace tsr {
namespace ref {

constexpr int GD_OK = 0;
constexpr int GD_BAD_RECORD = -1;   // arguments the kernel rejects (tsr/plan_gdn.hpp)
constexpr int GD_RANGE = -2;        // a row outside the DRAM view (the device would fail with DRAM_RANGE)

// SOFTPLUS of n words (one row); y gets the zero tail up to 16 L(n) words. x and y may be the same row.
void softplus_ref_row(uint32_t n, uint32_t * y, const uint32_t * x);
// Applies one SOFTPLUS record (the 6 words of opl::softplus) to the DRAM view.
int softplus_ref(const dram_view & d, const uint32_t args[6]);

// Applies one SSM_CONV record (the 12 words of opl::sconv).
int sconv_ref(const dram_view & d, const uint32_t args[12]);

// One token of one head. state: S_v rows of 16 L words (L = S_v / 16), updated in place; q, k, v:
// S_v words each; g, beta: f32 bits; scale: f32 bits; o: S_v words out. Sequence (decision SW-60):
//   alpha = EXPK(g)                         (SOP.EXP)
//   S     = FM(S, alpha)                    (VOP.MULC)
//   u_j   = RED.SUM(FM(S_j, k))             (VOP.MUL, RED.SUM: the 128-partial tree of VU §6.2)
//   delta = FM(FS(v, u), beta)              (VOP.SUB, VOP.MULC)
//   S_ji  = FA(FM(k_i, delta_j), S_ji)      (VOP.MULCADD)
//   w_j   = RED.SUM(FM(S_j, q))             (VOP.MUL, RED.SUM)
//   o     = FM(w, scale)                    (VOP.MULC)
void gdn_ref_step(uint32_t S_v, uint32_t * state, const uint32_t * q, const uint32_t * k, const uint32_t * v,
                  uint32_t g, uint32_t beta, uint32_t scale, uint32_t * o);

// Applies one GATED_DELTA_NET record (the 22 words of opl::gdn) to the DRAM view: every state is read
// before it is written (STATE_OUT may equal STATE_IN).
int gdn_ref(const dram_view & d, const uint32_t args[22]);

}  // namespace ref
}  // namespace tsr
