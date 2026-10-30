// rope_ref.cpp — see rope_ref.hpp (SW §7.5: the ROPE kind's tile op as the tsr_ref_vu ROPE call).
#include "rope_ref.hpp"

#include <cstring>
#include <vector>

#include "tsr/opl.hpp"
#include "tsr/plan.hpp"

namespace tsr {
namespace ref {

namespace {

constexpr uint64_t LINE = 64u;
constexpr uint32_t F32_ONE = 0x3F800000u;

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

}  // namespace

tsr_ref_rope_cfg rope_cfg_of(const uint32_t a[14], uint32_t T) {
    namespace r = opl::rope;
    const uint32_t mode = a[r::MODE];
    const bool atab = (mode & r::MODE_ATAB) != 0u, table = a[r::TABLE] != opl::HANDLE_NONE;
    tsr_ref_rope_cfg c{};
    c.mode = (mode & r::MODE_NEOX) ? TSR_ROPE_MODE_NEOX : TSR_ROPE_MODE_NORM;
    c.atab = atab ? 1u : 0u;
    c.ff = (table && !atab) ? 1u : 0u;
    c.fs = (!atab && a[r::FREQ_SCALE] != F32_ONE) ? 1u : 0u;
    c.m = a[r::M];
    c.ts = atab ? 0u : a[r::TS];
    c.freq_scale = c.fs ? a[r::FREQ_SCALE] : 0u;
    c.head_dim = a[r::HEAD_DIM];
    c.n_dims = a[r::N_DIMS];
    c.n_heads = a[r::N_HEADS];
    c.T = T;
    return c;
}

int rope_ref(const dram_view & d, const uint32_t a[14], const uint8_t * pre, uint32_t pre_bytes) {
    namespace r = opl::rope;
    const uint32_t hd = a[r::HEAD_DIM], nd = a[r::N_DIMS], nh = a[r::N_HEADS], T = a[r::N_TOKENS], mode = a[r::MODE];
    const bool table = a[r::TABLE] != opl::HANDLE_NONE;
    // The kernel's argument rules (plan_norm.hpp rope_check; the positions inside the preload area:
    // k_rope.cpp's run-time check).
    if (plan::norm::rope_check(mode, table, hd, nd, nh, T, a[r::DST_STRIDE], a[r::POS]) != opl::RET_OK ||
        uint64_t(a[r::POS]) + 4ull * T > pre_bytes)
        return NR_BAD_RECORD;
    const uint32_t H = (nh * hd) >> 4;
    const size_t tw = size_t(H) * 16u;
    std::vector<uint32_t> x(size_t(T) * tw), pos(T), tab;
    for (uint32_t t = 0; t < T; ++t) {
        const uint8_t * p8 = at(d, a[r::SRC], uint64_t(t) * a[r::SRC_STRIDE], uint64_t(H) * LINE);
        if (p8 == nullptr) return NR_RANGE;
        for (size_t j = 0; j < tw; ++j) x[size_t(t) * tw + j] = rd32(p8 + 4u * j);
        pos[t] = rd32(pre + a[r::POS] + 4u * t);
    }
    if (table) {
        const uint8_t * p8 = at(d, a[r::TABLE], 0u, uint64_t(nd / 2u) * 4u);
        if (p8 == nullptr) return NR_RANGE;
        tab.resize(nd / 2u);
        for (uint32_t i = 0; i < nd / 2u; ++i) tab[i] = rd32(p8 + 4u * i);
    }
    const tsr_ref_rope_cfg cfg = rope_cfg_of(a, T);
    tsr_ref_rope(&cfg, x.data(), H, x.data(), H, pos.data(), table ? tab.data() : nullptr);   // in place
    for (uint32_t t = 0; t < T; ++t) {
        uint8_t * p8 = at(d, a[r::DST], uint64_t(t) * a[r::DST_STRIDE], uint64_t(H) * LINE);
        if (p8 == nullptr) return NR_RANGE;
        for (size_t j = 0; j < tw; ++j) wr32(p8 + 4u * j, x[size_t(t) * tw + j]);
    }
    return NR_OK;
}

}  // namespace ref
}  // namespace tsr
