// elem_ref.cpp — see elem_ref.hpp (SW §7.5: the element-wise and row kinds' tile-op sequences as
// tsr_ref_vu calls; VU §5.2 VOP, §6.5 composites).
#include "elem_ref.hpp"

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

// Byte address of line `lines` past handle h in the view, if bytes fit; else nullptr.
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

// n words of a DRAM row into w (false: outside the view).
bool get_row(const dram_view & d, uint32_t h, uint64_t line, uint32_t n, std::vector<uint32_t> & w) {
    const uint8_t * p = at(d, h, line, 4ull * n);
    if (p == nullptr) return false;
    w.resize(n);
    for (uint32_t j = 0; j < n; ++j) w[j] = rd32(p + 4u * j);
    return true;
}

constexpr plan_limits ref_limits() {
    plan_limits l = plan::LIMITS_U200;
    l.budget = plan::SPM_LINE_MAX;   // the reference has no ring: only BAD_RECORD matters
    return l;
}

}  // namespace

uint32_t elem_vop_func(uint32_t kind, const uint32_t * a) {
    switch (kind) {
        case opl::K_ADD: return (a[opl::ew::FLAGS] & opl::ew::FLAG_B_SCALAR) ? TSR_VOP_FUNC_ADDC : TSR_VOP_FUNC_ADD;
        case opl::K_MUL: return (a[opl::ew::FLAGS] & opl::ew::FLAG_B_SCALAR) ? TSR_VOP_FUNC_MULC : TSR_VOP_FUNC_MUL;
        case opl::K_SCALE: return TSR_VOP_FUNC_MULC;
        case opl::K_GLU: {
            static const uint32_t f[3] = {TSR_VOP_FUNC_SWIGLU, TSR_VOP_FUNC_GEGLU16, TSR_VOP_FUNC_GEGLU};
            return a[opl::glu::VARIANT] <= 2u ? f[a[opl::glu::VARIANT]] : ~0u;
        }
        case opl::K_UNARY: {
            static const uint32_t f[5] = {TSR_VOP_FUNC_TANH, TSR_VOP_FUNC_GELU16, TSR_VOP_FUNC_SILU, TSR_VOP_FUNC_EXP,
                                          TSR_VOP_FUNC_GELU};
            return a[opl::unary::FUNC] <= 4u ? f[a[opl::unary::FUNC]] : ~0u;
        }
        default: return ~0u;
    }
}

void elem_ref_row(uint32_t kind, uint32_t func, uint32_t n, uint32_t * y, const uint32_t * a, const uint32_t * b,
                  uint32_t c, bool scalar, uint32_t c2) {
    if (kind == opl::K_SCALE) {
        // VOP.MULC c = s12 = s, then VOP.ADDC c = s13 = b in place (b != 0 as bits).
        tsr_ref_vop(TSR_VOP_FUNC_MULC, TSR_VOP_CMODE_SREG, 1u, n, y, 0u, a, 0u, nullptr, 0u, &c);
        if (c2 != 0u) tsr_ref_vop(TSR_VOP_FUNC_ADDC, TSR_VOP_CMODE_SREG, 1u, n, y, 0u, y, 0u, nullptr, 0u, &c2);
        return;
    }
    if (scalar) {
        // The c form (cmode 1: s12, or cmode 2: the row's word of the c vector; the same value).
        tsr_ref_vop(func, TSR_VOP_CMODE_SREG, 1u, n, y, 0u, a, 0u, nullptr, 0u, &c);
        return;
    }
    tsr_ref_vop(func, TSR_VOP_CMODE_NONE, 1u, n, y, 0u, a, 0u, b, 0u, nullptr);
}

int elem_ref(const dram_view & d, uint32_t kind, const uint32_t * a, uint32_t n_args, const uint8_t * pre,
             size_t pre_bytes) {
    const plan_limits l = ref_limits();
    kind_plan kp{};
    if (!plan::plan_kind(l, kind, a, n_args, &kp)) return kp.status == opl::RET_BAD_RECORD ? EL_BAD_RECORD : EL_RANGE;

    if (kind == opl::K_CPY || kind == opl::K_GET_ROWS) {
        const bool gr = kind == opl::K_GET_ROWS;
        const uint32_t rl = gr ? a[opl::getrows::ROW_LINES] : a[opl::cpy::ROW_LINES];
        const uint32_t rows = gr ? a[opl::getrows::N_IDS] : a[opl::cpy::ROWS];
        const uint32_t src = gr ? a[opl::getrows::SRC] : a[opl::cpy::SRC], ss = gr ? a[opl::getrows::SRC_STRIDE] : a[opl::cpy::SRC_STRIDE];
        const uint32_t dst = gr ? a[opl::getrows::DST] : a[opl::cpy::DST], ds = gr ? a[opl::getrows::DST_STRIDE] : a[opl::cpy::DST_STRIDE];
        const size_t rb = size_t(rl) * LINE;
        std::vector<uint8_t> buf(rb * rows);
        for (uint32_t r = 0; r < rows; ++r) {
            uint64_t line = uint64_t(r) * ss;
            if (gr) {
                const uint32_t off = a[opl::getrows::IDS] + 4u * r;
                if (pre == nullptr || size_t(off) + 4u > pre_bytes) return EL_BAD_RECORD;
                const uint32_t id = rd32(pre + off);
                // The device's rule (k_rows.cpp): a negative id, or a beat beyond 2^32, is DRAM_RANGE.
                if ((id & 0x80000000u) != 0u || uint64_t(src & 0x0FFFFFFFu) + uint64_t(id) * ss > 0xFFFFFFFFull) return EL_RANGE;
                line = uint64_t(id) * ss;
            }
            const uint8_t * p = at(d, src, line, rb);
            if (p == nullptr) return EL_RANGE;
            memcpy(buf.data() + rb * r, p, rb);
        }
        for (uint32_t r = 0; r < rows; ++r) {
            uint8_t * p = at(d, dst, uint64_t(r) * ds, rb);
            if (p == nullptr) return EL_RANGE;
            memcpy(p, buf.data() + rb * r, rb);
        }
        return EL_OK;
    }

    // Element-wise kinds: (dst, dst stride, a, a stride, b, b stride, n, rows) in their word order.
    uint32_t dst, dsr, ha, as, hb = 0, bs = 0, n, rows, p = 1u, scalar = 0u, c = 0u, c2 = 0u;
    switch (kind) {
        case opl::K_ADD:
        case opl::K_MUL: {
            namespace e = opl::ew;
            dst = a[e::DST]; dsr = a[e::DST_STRIDE]; ha = a[e::A]; as = a[e::A_STRIDE]; hb = a[e::B]; bs = a[e::B_STRIDE];
            n = a[e::N]; rows = a[e::ROWS]; p = a[e::B_PERIOD]; scalar = a[e::FLAGS] & e::FLAG_B_SCALAR;
            break;
        }
        case opl::K_GLU: {
            namespace g = opl::glu;
            dst = a[g::DST]; dsr = a[g::DST_STRIDE]; ha = a[g::GATE]; as = a[g::GATE_STRIDE]; hb = a[g::UP];
            bs = a[g::UP_STRIDE]; n = a[g::N]; rows = a[g::ROWS]; p = 0xFFFFFFFFu;
            break;
        }
        case opl::K_SCALE: {
            namespace s = opl::scl;
            dst = a[s::DST]; dsr = a[s::DST_STRIDE]; ha = a[s::SRC]; as = a[s::SRC_STRIDE]; n = a[s::N]; rows = a[s::ROWS];
            c = a[s::S]; c2 = a[s::B];
            break;
        }
        case opl::K_UNARY: {
            namespace u = opl::unary;
            dst = a[u::DST]; dsr = a[u::DST_STRIDE]; ha = a[u::SRC]; as = a[u::SRC_STRIDE]; n = a[u::N]; rows = a[u::ROWS];
            break;
        }
        default: return EL_BAD_RECORD;
    }
    const uint32_t func = elem_vop_func(kind, a);
    const uint32_t L = lines_of(n);
    const bool has_b = kind == opl::K_ADD || kind == opl::K_MUL || kind == opl::K_GLU;
    std::vector<uint32_t> out(size_t(rows) * 16u * L, 0u), ra, rbv;
    for (uint32_t r = 0; r < rows; ++r) {
        if (!get_row(d, ha, uint64_t(r) * as, n, ra)) return EL_RANGE;
        const uint32_t * bp = nullptr;
        if (has_b && scalar) {
            const uint32_t k = p == 1u ? 0u : r % p;
            const uint8_t * q = at(d, hb, 0u, 4ull * (k + 1u));
            if (q == nullptr) return EL_RANGE;
            c = rd32(q + 4u * k);
        } else if (has_b) {
            const uint64_t br = (p == 1u || bs == 0u) ? 0u : uint64_t(p >= rows ? r : r % p) * bs;
            if (!get_row(d, hb, br, n, rbv)) return EL_RANGE;
            bp = rbv.data();
        }
        elem_ref_row(kind, func, n, out.data() + size_t(r) * 16u * L, ra.data(), bp, c, scalar != 0u, c2);
    }
    for (uint32_t r = 0; r < rows; ++r) {
        uint8_t * q = at(d, dst, uint64_t(r) * dsr, uint64_t(L) * LINE);
        if (q == nullptr) return EL_RANGE;
        for (uint32_t j = 0; j < 16u * L; ++j) wr32(q + 4u * j, out[size_t(r) * 16u * L + j]);
    }
    return EL_OK;
}

}  // namespace ref
}  // namespace tsr
