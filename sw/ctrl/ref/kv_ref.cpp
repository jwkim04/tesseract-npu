// kv_ref.cpp — host reference of the SET_ROWS_KV kind (kvfa_ref.hpp; SW §7.5, AT §5.1, §6.1).
#include "kvfa_ref.hpp"

#include <cstring>
#include <vector>

extern "C" {
#include "tsr_ref.h"
}

#include "tsr/opl.hpp"

namespace tsr {
namespace ref {

namespace {
uint32_t rd32(const uint8_t * p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
}  // namespace

int set_rows_kv_ref(const dram_view & d, const uint32_t a[12], const uint8_t * pre, size_t pre_bytes,
                    uint8_t * const plane[], uint32_t C) {
    namespace sr = opl::setrows;
    const uint32_t e64 = opl::geom_e64(a[sr::GEOM]), LG = opl::geom_lg(a[sr::GEOM]), E = e64 << 6;
    if (a[sr::WHICH] > 1u || a[sr::PLANE_KIND] != 0u || e64 == 0u || e64 > 511u || LG == 0u || a[sr::ROW_ELEMS] != E)
        return KV_BAD_RECORD;
    const uint32_t n = a[sr::N_ROWS];
    if ((a[sr::IDX] & 7u) != 0u || uint64_t(a[sr::IDX]) + 8ull * n > pre_bytes) return KV_BAD_RECORD;
    const tsr_kv_geom g = { E, C, LG };
    const uint64_t cells = 16ull * C * LG;
    for (uint32_t i = 0; i < n; ++i) {   // every index first (the kernel refuses before any push)
        const uint8_t * p = pre + a[sr::IDX] + 8u * i;
        if (rd32(p + 4) != 0u || rd32(p) >= cells) return KV_BAD_RECORD;
    }
    const uint32_t ch = a[sr::SRC] >> 28;
    const uint64_t beat = a[sr::SRC] & 0x0FFFFFFFu;
    std::vector<float> x(E);
    std::vector<uint8_t> qa(size_t(tsr_qa_lines(TSR_QA_A32, E)) * 64u);
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t * row = d.ch[ch] + (beat + uint64_t(i) * a[sr::SRC_STRIDE]) * 64u;
        memcpy(x.data(), row, size_t(E) * 4u);
        tsr_qact(TSR_QA_A32, x.data(), E, qa.data());
        tsr_ref_kvst(a[sr::WHICH], &g, plane, rd32(pre + a[sr::IDX] + 8u * i), 1u, qa.data(), 0u, nullptr);
    }
    return KV_OK;
}

}  // namespace ref
}  // namespace tsr
