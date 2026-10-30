// fattn_ref.cpp — host reference of the FLASH_ATTN kind (kvfa_ref.hpp; SW §7.5, AT §6.6, §13.1):
// tsr_ref_fa on a record's inputs.
#include "kvfa_ref.hpp"

#include <cstring>
#include <vector>

#include "tsr/opl.hpp"

namespace tsr {
namespace ref {

namespace {
uint32_t rd32(const uint8_t * p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
uint8_t * at_handle(const dram_view & d, uint32_t h, uint64_t beats) { return d.ch[h >> 28] + (uint64_t(h & 0x0FFFFFFFu) + beats) * 64u; }
}  // namespace

int flash_attn_ref_out(const dram_view & d, const uint32_t a[22], const uint8_t * pre, size_t pre_bytes,
                       const uint8_t * const kplane[], const uint8_t * const vplane[], const tsr_att_cfg & cfg,
                       float * out, tsr_fa_stats * stats) {
    namespace fa = opl::fa;
    const uint32_t D = a[fa::D], G = a[fa::G], T = a[fa::N_TOKENS], n_kv = a[fa::N_KV];
    const uint32_t E = opl::geom_e64(a[fa::GEOM]) << 6, LG = opl::geom_lg(a[fa::GEOM]);
    if (a[fa::SOFTCAP] != 0u || a[fa::HEAD_DIM] != D || D == 0u || E == 0u || E % D != 0u || a[fa::N_HEAD_KV] != E / D ||
        a[fa::N_HEAD] != G * (E / D) || a[fa::Q_HEAD_STRIDE] != D / 16u || T == 0u)
        return KV_BAD_RECORD;
    const uint32_t fl = a[fa::FLAGS];
    if ((fl & 0xFEu) != 0u || ((fl & 1u) == 0u && fl != 0u)) return KV_BAD_RECORD;
    // The MSUM (64-aligned in the preload area, its header matching the record).
    const uint32_t mo = a[fa::MSUM];
    const size_t mw = tsr_msum_words(T);
    if ((mo & 63u) != 0u || uint64_t(mo) + 4ull * mw > pre_bytes) return KV_BAD_RECORD;
    std::vector<uint32_t> words(mw);
    for (size_t i = 0; i < mw; ++i) words[i] = rd32(pre + mo + 4u * i);
    tsr_msum ms;
    if (tsr_msum_view(words.data(), mw, &ms) != 0 || ms.n_q != T || ms.n_kv != n_kv) return KV_BAD_RECORD;
    // Q rows [T][n_head][D].
    const uint32_t K = a[fa::N_HEAD] * D;
    std::vector<float> q(size_t(T) * K);
    for (uint32_t t = 0; t < T; ++t)
        memcpy(q.data() + size_t(t) * K, at_handle(d, a[fa::Q], uint64_t(t) * a[fa::Q_TOKEN_STRIDE]), size_t(K) * 4u);
    // The f32 mask copy (rows of ceil(n_kv/16) lines).
    const float * mask = nullptr;
    if (a[fa::MASK] != opl::HANDLE_NONE) mask = reinterpret_cast<const float *>(at_handle(d, a[fa::MASK], 0u));
    // The new cells as runs of consecutive cells in index order.
    std::vector<uint32_t> runs;
    if (fl & 1u) {
        const uint32_t off = ((fl >> 8) & 0xFFu) << 6, n = fl >> 16;
        if (uint64_t(off) + 8ull * n > pre_bytes) return KV_BAD_RECORD;
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t * p = pre + off + 8u * i;
            if (rd32(p + 4) != 0u) continue;            // >= 2^32: in no chunk (the kernel's rule)
            const uint32_t c = rd32(p);
            if (!runs.empty() && runs.back() == c) runs.back() = c + 1u;
            else { runs.push_back(c); runs.push_back(c + 1u); }
        }
    }
    const tsr_fa_args fa_args = { T, D, G, tsr_bits_f32(a[fa::SCALE]), q.data(), E, LG, kplane, vplane, &ms, mask,
                                  runs.empty() ? nullptr : reinterpret_cast<const uint32_t(*)[2]>(runs.data()),
                                  uint32_t(runs.size() / 2u), out, stats };
    return tsr_ref_fa(&cfg, &fa_args);
}

int flash_attn_ref(const dram_view & d, const uint32_t a[22], const uint8_t * pre, size_t pre_bytes,
                   const uint8_t * const kplane[], const uint8_t * const vplane[], const tsr_att_cfg & cfg,
                   tsr_fa_stats * stats) {
    namespace fa = opl::fa;
    const uint32_t T = a[fa::N_TOKENS], K = a[fa::N_HEAD] * a[fa::D];
    std::vector<float> out(size_t(T) * K);
    const int rc = flash_attn_ref_out(d, a, pre, pre_bytes, kplane, vplane, cfg, out.data(), stats);
    if (rc != 0) return rc;
    for (uint32_t t = 0; t < T; ++t)
        memcpy(at_handle(d, a[fa::DST], uint64_t(t) * a[fa::DST_STRIDE]), out.data() + size_t(t) * K, size_t(K) * 4u);
    return 0;
}

}  // namespace ref
}  // namespace tsr
