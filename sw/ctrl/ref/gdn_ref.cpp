// gdn_ref.cpp — see gdn_ref.hpp (SW §7.9: the Gated DeltaNet kinds' tile-op sequences as tsr_ref_vu
// calls).
#include "gdn_ref.hpp"

#include <cstring>
#include <vector>

#include "tsr/opl.hpp"
#include "tsr/plan.hpp"
#include "tsr_ref.h"
#include "tsr_ref_vu.h"

namespace tsr {
namespace ref {

namespace {

constexpr uint64_t LINE = 64u;

uint8_t * at(const dram_view & d, uint32_t h, uint64_t lines, uint64_t bytes) {
    const uint32_t ch = h >> 28;
    const uint64_t off = (uint64_t(h & 0x0FFFFFFFu) + lines) * LINE;
    if (ch >= d.num_ch || off + bytes > d.ch_bytes) return nullptr;
    return d.ch[ch] + off;
}

uint32_t rd32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
void wr32(uint8_t * p, uint32_t v) {
    p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
}
uint32_t lines_of(uint32_t n) { return (n >> 4) + ((n & 15u) != 0u ? 1u : 0u); }

constexpr plan_limits ref_limits() {
    plan_limits l = plan::LIMITS_U200;
    l.budget = plan::SPM_LINE_MAX;   // the reference has no ring: only BAD_RECORD matters
    return l;
}

// n words of a DRAM row (line `line` past handle h) into w (zero tail up to 16 L words).
bool get_words(const dram_view & d, uint32_t h, uint64_t line, uint32_t n, uint32_t * w) {
    const uint32_t L = lines_of(n);
    const uint8_t * p = at(d, h, line, uint64_t(L) * LINE);
    if (p == nullptr) return false;
    for (uint32_t j = 0; j < 16u * L; ++j) w[j] = j < n ? rd32(p + 4u * j) : 0u;
    return true;
}
bool put_words(const dram_view & d, uint32_t h, uint64_t line, uint32_t n, const uint32_t * w) {
    const uint32_t L = lines_of(n);
    uint8_t * p = at(d, h, line, uint64_t(L) * LINE);
    if (p == nullptr) return false;
    for (uint32_t j = 0; j < 16u * L; ++j) wr32(p + 4u * j, j < n ? w[j] : 0u);
    return true;
}

// One VOP on single rows (rows = 1, m = n): the strides are unused.
void vop1(uint32_t func, uint32_t cmode, uint32_t n, uint32_t * dst, const uint32_t * a, const uint32_t * b,
          const uint32_t * c) {
    tsr_ref_vop(func, cmode, 1u, n, dst, 0u, a, 0u, b, 0u, c);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// SOFTPLUS
// ---------------------------------------------------------------------------------------------
void softplus_ref_row(uint32_t n, uint32_t * y, const uint32_t * x) {
    namespace s = opl::softplus;
    const uint32_t W = 16u * lines_of(n);
    std::vector<uint32_t> b0(W, 0u), b1(W, 0u), b2(W, 0u), b3(W, 0u), b4(W, 0u);
    for (uint32_t j = 0; j < n; ++j) b0[j] = x[j];
    const uint32_t SR = TSR_VOP_CMODE_SREG;
    uint32_t c = s::NEG_ONE;
    vop1(TSR_VOP_FUNC_MULC, SR, n, b1.data(), b0.data(), nullptr, &c);                    // nx
    vop1(TSR_VOP_FUNC_MIN, TSR_VOP_CMODE_NONE, n, b1.data(), b0.data(), b1.data(), nullptr);   // nax = -|x|
    vop1(TSR_VOP_FUNC_EXP, TSR_VOP_CMODE_NONE, n, b2.data(), b1.data(), nullptr, nullptr);     // h
    c = s::TWO;
    vop1(TSR_VOP_FUNC_ADDC, SR, n, b3.data(), b2.data(), nullptr, &c);                    // den
    vop1(TSR_VOP_FUNC_RECIP, TSR_VOP_CMODE_NONE, n, b3.data(), b3.data(), nullptr, nullptr);   // r
    vop1(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, n, b3.data(), b2.data(), b3.data(), nullptr);   // s = h r
    vop1(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, n, b2.data(), b3.data(), b3.data(), nullptr);   // z = s s
    c = s::COEF[5];
    vop1(TSR_VOP_FUNC_MULC, SR, n, b4.data(), b2.data(), nullptr, &c);                    // z c5
    c = s::COEF[4];
    vop1(TSR_VOP_FUNC_ADDC, SR, n, b4.data(), b4.data(), nullptr, &c);                    // acc
    for (int k = 3; k >= 0; --k) {
        vop1(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, n, b4.data(), b4.data(), b2.data(), nullptr);
        c = s::COEF[k];
        vop1(TSR_VOP_FUNC_ADDC, SR, n, b4.data(), b4.data(), nullptr, &c);
    }
    vop1(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, n, b3.data(), b3.data(), b4.data(), nullptr);   // y = s acc
    vop1(TSR_VOP_FUNC_SUB, TSR_VOP_CMODE_NONE, n, b1.data(), b0.data(), b1.data(), nullptr);   // m = x - nax
    c = s::HALF;
    vop1(TSR_VOP_FUNC_MULCADD, SR, n, b3.data(), b1.data(), b3.data(), &c);               // FA(FM(m, 1/2), y)
    for (uint32_t j = 0; j < W; ++j) y[j] = b3[j];
}

int softplus_ref(const dram_view & d, const uint32_t a[6]) {
    namespace s = opl::softplus;
    kind_plan kp{};
    if (!plan_softplus(a, s::NARGS, ref_limits(), &kp)) return GD_BAD_RECORD;
    const uint32_t n = a[s::N], rows = a[s::ROWS], L = lines_of(n);
    std::vector<uint32_t> x(size_t(rows) * 16u * L), y(size_t(rows) * 16u * L);
    for (uint32_t r = 0; r < rows; ++r)
        if (!get_words(d, a[s::SRC], uint64_t(r) * a[s::SRC_STRIDE], n, x.data() + size_t(r) * 16u * L)) return GD_RANGE;
    for (uint32_t r = 0; r < rows; ++r) softplus_ref_row(n, y.data() + size_t(r) * 16u * L, x.data() + size_t(r) * 16u * L);
    for (uint32_t r = 0; r < rows; ++r)
        if (!put_words(d, a[s::DST], uint64_t(r) * a[s::DST_STRIDE], n, y.data() + size_t(r) * 16u * L)) return GD_RANGE;
    return GD_OK;
}

// ---------------------------------------------------------------------------------------------
// SSM_CONV
// ---------------------------------------------------------------------------------------------
int sconv_ref(const dram_view & d, const uint32_t a[12]) {
    namespace c = opl::sconv;
    kind_plan kp{};
    if (!plan_ssm_conv(a, c::NARGS, ref_limits(), &kp)) return GD_BAD_RECORD;
    const uint32_t C = a[c::N_CH], dc = a[c::D_CONV], T = a[c::N_TOKENS], ns = a[c::N_SEQS], Lc = C >> 4, W = C;
    const bool silu = (a[c::FLAGS] & c::FLAG_SILU) != 0u;
    std::vector<uint32_t> taps(size_t(dc) * W);
    for (uint32_t w = 0; w < dc; ++w)
        if (!get_words(d, a[c::W], uint64_t(w) * Lc, C, taps.data() + size_t(w) * W)) return GD_RANGE;
    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t nw = dc - 1u + T;
        std::vector<uint32_t> win(size_t(nw) * W);
        for (uint32_t w = 0; w + 1u < dc; ++w)
            if (!get_words(d, a[c::STATE_IN], (uint64_t(s) * (dc - 1u) + w) * Lc, C, win.data() + size_t(w) * W))
                return GD_RANGE;
        for (uint32_t t = 0; t < T; ++t)
            if (!get_words(d, a[c::X], uint64_t(s * T + t) * a[c::X_TOKEN_STRIDE], C, win.data() + size_t(dc - 1u + t) * W))
                return GD_RANGE;
        std::vector<uint32_t> acc(W), tmp(W);
        for (uint32_t t = 0; t < T; ++t) {
            vop1(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, C, acc.data(), win.data() + size_t(t) * W, taps.data(), nullptr);
            for (uint32_t w = 1; w < dc; ++w) {
                vop1(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, C, tmp.data(), win.data() + size_t(t + w) * W,
                     taps.data() + size_t(w) * W, nullptr);
                vop1(TSR_VOP_FUNC_ADD, TSR_VOP_CMODE_NONE, C, acc.data(), acc.data(), tmp.data(), nullptr);
            }
            if (silu) vop1(TSR_VOP_FUNC_SILU, TSR_VOP_CMODE_NONE, C, acc.data(), acc.data(), nullptr, nullptr);
            if (!put_words(d, a[c::DST], uint64_t(s * T + t) * a[c::DST_TOKEN_STRIDE], C, acc.data())) return GD_RANGE;
        }
        for (uint32_t w = 0; w + 1u < dc; ++w)
            if (!put_words(d, a[c::STATE_OUT], (uint64_t(s) * (dc - 1u) + w) * Lc, C, win.data() + size_t(T + w) * W))
                return GD_RANGE;
    }
    return GD_OK;
}

// ---------------------------------------------------------------------------------------------
// GATED_DELTA_NET
// ---------------------------------------------------------------------------------------------
void gdn_ref_step(uint32_t S_v, uint32_t * state, const uint32_t * q, const uint32_t * k, const uint32_t * v,
                  uint32_t g, uint32_t beta, uint32_t scale, uint32_t * o) {
    const uint32_t L = S_v >> 4, row = 16u * L;
    std::vector<uint32_t> p(size_t(S_v) * row), u(16u * lines_of(S_v), 0u);
    const uint32_t SR = TSR_VOP_CMODE_SREG, VEC = TSR_VOP_CMODE_VEC;
    uint32_t alpha = tsr_ref_sop(TSR_SOP_SUB_EXP, g, 0u, 0u, 0u);
    tsr_ref_vop(TSR_VOP_FUNC_MULC, SR, S_v, S_v, state, L, state, L, nullptr, 0u, &alpha);          // S *= alpha
    tsr_ref_vop(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, S_v, S_v, p.data(), L, state, L, k, 0u, nullptr);
    tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_NONE, S_v, S_v, u.data(), p.data(), L, 0u, 0u);       // u = S k
    vop1(TSR_VOP_FUNC_SUB, TSR_VOP_CMODE_NONE, S_v, u.data(), v, u.data(), nullptr);                  // v - u
    vop1(TSR_VOP_FUNC_MULC, SR, S_v, u.data(), u.data(), nullptr, &beta);                             // delta
    tsr_ref_vop(TSR_VOP_FUNC_MULCADD, VEC, S_v, S_v, state, L, k, 0u, state, L, u.data());            // S += k delta
    tsr_ref_vop(TSR_VOP_FUNC_MUL, TSR_VOP_CMODE_NONE, S_v, S_v, p.data(), L, state, L, q, 0u, nullptr);
    tsr_ref_red(TSR_RED_KIND_SUM, TSR_RED_POST_NONE, S_v, S_v, u.data(), p.data(), L, 0u, 0u);       // w = S q
    vop1(TSR_VOP_FUNC_MULC, SR, S_v, u.data(), u.data(), nullptr, &scale);                            // o = w scale
    for (uint32_t j = 0; j < S_v; ++j) o[j] = u[j];
}

int gdn_ref(const dram_view & d, const uint32_t a[22]) {
    namespace g = opl::gdn;
    kind_plan kp{};
    if (!plan_gated_delta_net(a, g::NARGS, ref_limits(), &kp)) return GD_BAD_RECORD;
    const uint32_t sv = a[g::S_V], hk = a[g::H_K], hv = a[g::H_V], T = a[g::N_TOKENS], ns = a[g::N_SEQS], L = sv >> 4;
    std::vector<uint32_t> st(size_t(sv) * 16u * L), q(16u * L), k(16u * L), v(16u * L), o(sv), gw(16u * lines_of(hv)),
        bw(16u * lines_of(hv));
    for (uint32_t s = 0; s < ns; ++s) {
        for (uint32_t h = 0; h < hv; ++h) {
            const uint64_t sline = (uint64_t(s) * hv + h) * sv * L;
            for (uint32_t j = 0; j < sv; ++j)
                if (!get_words(d, a[g::STATE_IN], sline + uint64_t(j) * L, sv, st.data() + size_t(j) * 16u * L)) return GD_RANGE;
            const uint32_t hq = h % hk;
            for (uint32_t t = 0; t < T; ++t) {
                const uint64_t n = uint64_t(s) * T + t;
                if (!get_words(d, a[g::Q], hq * uint64_t(a[g::Q_HEAD_STRIDE]) + n * a[g::Q_TOKEN_STRIDE], sv, q.data()) ||
                    !get_words(d, a[g::K], hq * uint64_t(a[g::K_HEAD_STRIDE]) + n * a[g::K_TOKEN_STRIDE], sv, k.data()) ||
                    !get_words(d, a[g::V], h * uint64_t(a[g::V_HEAD_STRIDE]) + n * a[g::V_TOKEN_STRIDE], sv, v.data()) ||
                    !get_words(d, a[g::G], n * a[g::G_TOKEN_STRIDE], hv, gw.data()) ||
                    !get_words(d, a[g::BETA], n * a[g::BETA_TOKEN_STRIDE], hv, bw.data()))
                    return GD_RANGE;
                gdn_ref_step(sv, st.data(), q.data(), k.data(), v.data(), gw[h], bw[h], a[g::SCALE], o.data());
                if (!put_words(d, a[g::DST], (n * hv + h) * L, sv, o.data())) return GD_RANGE;
            }
            for (uint32_t j = 0; j < sv; ++j)
                if (!put_words(d, a[g::STATE_OUT], sline + uint64_t(j) * L, sv, st.data() + size_t(j) * 16u * L)) return GD_RANGE;
        }
    }
    return GD_OK;
}

}  // namespace ref
}  // namespace tsr
