// moe_ref.cpp — see moe_ref.hpp (SW §7.5: the MoE kinds' tile-op sequences as tsr_ref calls).
#include "moe_ref.hpp"

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
// Byte-granular variant (the ids matrix has an arbitrary multiple-of-4 row stride).
uint8_t * at_bytes(const dram_view & d, uint32_t h, uint64_t bytes_off, uint64_t bytes) {
    const uint32_t ch = h >> 28;
    const uint64_t off = uint64_t(h & 0x0FFFFFFFu) * LINE + bytes_off;
    if (ch >= d.num_ch || off + bytes > d.ch_bytes) return nullptr;
    return d.ch[ch] + off;
}

uint32_t rd32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
void wr32(uint8_t * p, uint32_t v) {
    p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
}
float rdf(const uint8_t * p) {
    const uint32_t u = rd32(p);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int moe_route_ref(const dram_view & d, const uint32_t a[9]) {
    namespace r = opl::moer;
    const uint32_t n = a[r::N_EXPERT], k = a[r::N_USED], T = a[r::T];
    if (a[r::FLAGS] != 0u || k == 0u || k > r::MAX_USED || n > r::MAX_EXPERTS || k > n || T == 0u) return MM_BAD_RECORD;
    const uint32_t ln = plan::vu_lines(n);
    std::vector<uint32_t> logits(n);
    // the packed weights, whole lines per tile (the kernel's STs)
    const uint32_t w_words = 16u * plan::vu_lines(T * k);
    std::vector<uint32_t> wp(w_words, 0u);
    for (uint32_t t = 0; t < T; ++t) {
        const uint8_t * lp = at(d, a[r::LOGITS], uint64_t(t) * a[r::LOGITS_STRIDE], uint64_t(ln) * LINE);
        uint8_t * ip = at(d, a[r::IDS], uint64_t(t) * a[r::IDS_STRIDE], LINE);
        if (lp == nullptr || ip == nullptr) return MM_RANGE;
        for (uint32_t i = 0; i < n; ++i) logits[i] = rd32(lp + 4u * i);
        uint32_t ids[r::MAX_USED], w[r::MAX_USED];
        tsr_ref_moe_route(n, k, logits.data(), ids, w, nullptr);
        for (uint32_t j = 0; j < 16u; ++j) wr32(ip + 4u * j, j < k ? ids[j] : 0u);
        for (uint32_t j = 0; j < k; ++j) wp[t * k + j] = w[j];
    }
    uint8_t * wd = at(d, a[r::W], 0, uint64_t(w_words) * 4u);
    if (wd == nullptr) return MM_RANGE;
    for (uint32_t i = 0; i < w_words; ++i) wr32(wd + 4u * i, wp[i]);
    return MM_OK;
}

int mul_mat_id_ref(const dram_view & d, const uint32_t a[17], uint32_t threads) {
    namespace m = opl::mmid;
    const uint32_t fmt = a[m::FMT], N = a[m::N], K = a[m::K], T = a[m::T], nu = a[m::N_USED], ne = a[m::N_EXPERT];
    if (a[m::FLAGS] != 0u || nu == 0u || nu > m::MAX_USED || ne == 0u || T == 0u) return MM_BAD_RECORD;
    const plan::dev_cfg cfg = {65536u, 8u, 32768u};
    if (plan::mm_plan(cfg, fmt, N, K, 1u, nullptr) == opl::RET_BAD_RECORD) return MM_BAD_RECORD;
    const uint32_t x_lines = K / 16u, y_lines = plan::ceil_div(N, 16u);
    const bool has_scale = a[m::SCALE] != opl::HANDLE_NONE;
    std::vector<float> x(K), y(N);
    std::vector<uint32_t> scale(has_scale ? ne : 0u);
    if (has_scale) {
        const uint8_t * sp = at(d, a[m::SCALE], 0, uint64_t(plan::vu_lines(ne)) * LINE);
        if (sp == nullptr) return MM_RANGE;
        for (uint32_t i = 0; i < ne; ++i) scale[i] = rd32(sp + 4u * i);
    }
    const uint64_t row_beats = tsr_row_beats(fmt, K);
    const uint64_t w_need = uint64_t(tsr_rows_per_ch(N, d.num_ch)) * row_beats;
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < nu; ++j) {
            const uint8_t * ip = at_bytes(d, a[m::IDS], uint64_t(t) * a[m::IDS_STRIDE] + 4u * j, 4u);
            if (ip == nullptr) return MM_RANGE;
            const uint32_t e = rd32(ip);
            if (e >= ne) return MM_RANGE;
            const uint64_t w_off = uint64_t(a[m::W_OFF]) + uint64_t(e) * a[m::EXP_STRIDE];
            if ((w_off + w_need) * LINE > d.ch_bytes) return MM_RANGE;
            const uint8_t * xp = at(d, a[m::X], uint64_t(t) * a[m::X_TOK_STRIDE] + uint64_t(j) * a[m::X_SLOT_STRIDE],
                                    uint64_t(x_lines) * LINE);
            uint8_t * yp = at(d, a[m::DST], uint64_t(t * nu + j) * a[m::DST_STRIDE], uint64_t(y_lines) * LINE);
            if (xp == nullptr || yp == nullptr) return MM_RANGE;
            for (uint32_t i = 0; i < K; ++i) x[i] = rdf(xp + 4u * i);
            mulmat_ref_token(d, fmt, N, K, w_off, x.data(), nullptr, y.data(), threads);
            for (uint32_t i = 0; i < N; ++i) {
                uint32_t v;
                std::memcpy(&v, &y[i], 4);
                if (has_scale) v = tsr_ref_vop_elem(TSR_VOP_FUNC_MULC, v, 0u, scale[e]);
                wr32(yp + 4u * i, v);
            }
            std::memset(yp + 4u * uint64_t(N), 0, size_t(uint64_t(y_lines) * LINE - 4u * uint64_t(N)));
        }
    }
    return MM_OK;
}

}  // namespace ref
}  // namespace tsr
