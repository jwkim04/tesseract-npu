// tsr_ref_math.h — the VU's approximate and composite math, defined bit for bit.
//
// Normative source: docs/v2/02-vector-unit.md §6 (EXPK §6.3, composites §6.5, SIN/COS §6.6,
// constants Appendix A). The simulator, the RTL benches (ctypes on libtsr_ref.so) and the
// firmware's host build call these functions; the RTL must match them bit for bit.
//
// Rules (docs/v2/06-software-verification.md §4.5):
//   - every f32 operation is a tsr_fadd / tsr_fsub / tsr_fmul / tsr_frecip call (DAZ on the
//     inputs, RNE, FTZ, canonical NaN 0x7FC00000; see tsr_ref.h); everything else is integer
//     bit manipulation. No libm, no float operators, no __int128 (RV32-clean).
//   - every function is pure and thread-safe.
//   - no result is ever subnormal; every NaN result is 0x7FC00000. EXPK never returns -0.
//
// Reductions (RED, the z() rule), ROPE (both angle forms), TOPK, VMASK and AT's vector ops are
// not here: VU §6.1 places them in tsr_ref.c.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// κ codes of tsr_m_expk, in the order of the VU §6.3 table. EXPK(v, κ) = exp(K·v):
//   P1: K = +1 (EXP, EXPSUB)       N1: K = -1 (SIGMOID, SILU)
//   N2: K = -2 (TANH)              NG: K = -C2G, C2G = 2·sqrt(2/pi)·0.044715 (GELU)
#define TSR_M_P1 0u
#define TSR_M_N1 1u
#define TSR_M_N2 2u
#define TSR_M_NG 3u

float tsr_m_expk(float v, unsigned kappa);   // EXPK(v, κ), VU §6.3; κ > TSR_M_NG -> 0x7FC00000
float tsr_m_exp(float x);                    // EXPK(x, P1)
float tsr_m_sigmoid(float x);                // SIG(x)  = RC(FA(1, EXPK(x, N1)))
float tsr_m_silu(float x);                   // SILU(x) = FM(x, SIG(x))
float tsr_m_gelu_tanh(float x);              // GELU(x), tanh form as x·σ(2u), VU §6.5
float tsr_m_gelu16(float x);                 // GELU16(x), the ggml-CPU fp16 form, VU §6.5
float tsr_m_tanh(float x);                   // TANH(x), VU §6.5
void  tsr_m_sincos(float t, float * s, float * c);   // SIN and COS, VU §6.6
float tsr_m_sin(float t);
float tsr_m_cos(float t);

// Binary element functions of VU §6.5 with their own special-case rules.
float tsr_m_geglu16(float a, float b);       // (DAZ(a) <= -10) ? +0 : FM(GELU16(a), b)
float tsr_m_expsub(float a, float x);        // (DAZ(a) == -inf) ? +0 : EXPK(FS(a, x), P1)

// binary16 helpers used by GELU16 (VU §6.1). F16: DAZ, then RNE with gradual underflow
// (fp16 subnormals kept), overflow -> +-inf, NaN -> sign | 0x7E00. F32: exact widening.
uint16_t tsr_f32_to_f16_rne(float x);
float    tsr_f16_to_f32(uint16_t h);

#ifdef __cplusplus
}
#endif
