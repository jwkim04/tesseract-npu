// tsr-rec.cpp — see tsr-rec.h
#include "tsr-rec.h"

#include "tsr/plan.hpp"

namespace opl = tsr::opl;

void tsr_note_kv_write(const tsr_rec & r, tsr_fanew_map * m) {
    if (r.kind == opl::K_SET_ROWS_KV && r.kv_idx_n != 0) {
        (*m)[r.kv_plane] = { r.kv_idx_src, r.kv_idx_n };
    }
}

uint32_t tsr_encode(tsr_opl_builder & b, tsr_rec & r, int32_t tag, const tsr_fanew_map & fanew) {
    std::vector<std::pair<uint32_t, uint32_t>> ents;
    for (const tsr_pre_ref & p : r.pre) {
        ents.push_back({ p.src, p.bytes });
    }
    const auto nw = r.kind == opl::K_FLASH_ATTN ? fanew.find(r.kv_plane) : fanew.end();
    if (nw != fanew.end()) {
        ents.push_back({ nw->second.first, opl::PRE_KVIDX_BYTES * nw->second.second });
    }
    if (b.preload_lines_with(ents) > opl::PRE_MAX_LINES) {
        return tsr_opl_builder::NO_ROOM;
    }
    for (const tsr_pre_ref & p : r.pre) {
        const uint32_t off = b.preload(p.src, p.bytes);
        if (off == tsr_opl_builder::NO_ROOM || off == tsr_opl_builder::FAILED) {
            return off;
        }
        r.args[p.word] = off;
    }
    if (r.kind == opl::K_FLASH_ATTN) {
        r.args[opl::fa::FLAGS] = 0;
        if (nw != fanew.end()) {
            // FA_NEW: the cells SET_ROWS_KV wrote into this K plane (AT §6.6 rule 7, §12)
            const uint32_t off = b.preload(nw->second.first, opl::PRE_KVIDX_BYTES * nw->second.second);
            if (off == tsr_opl_builder::NO_ROOM || off == tsr_opl_builder::FAILED) {
                return off;
            }
            r.args[opl::fa::FLAGS] = tsr::plan::attn::fa_flags(off, nw->second.second);
        }
    }
    std::vector<tsr_dram_range> rd, wr;
    b.kind_ranges(r.kind, r.args, r.aux, rd, wr);
    return b.add(r.kind, r.args, r.n_args, rd, wr, tag);
}
