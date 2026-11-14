// opl_ref.cpp — see opl_ref.hpp
#include "opl_ref.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "elem_ref.hpp"
#include "kvfa_ref.hpp"
#include "rmsnorm_ref.hpp"
#include "rope_ref.hpp"
#include "tsr/opl.hpp"
#include "tsr_isa.h"

namespace tsr {
namespace ref {

namespace {

constexpr uint64_t LINE = 64u;

int fail(std::string * why, int rc, uint32_t idx, uint32_t kind, const char * fmt, ...) __attribute__((format(printf, 5, 6)));
int fail(std::string * why, int rc, uint32_t idx, uint32_t kind, const char * fmt, ...) {
    if (why != nullptr) {
        char buf[256];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        char head[96];
        snprintf(head, sizeof(head), "op list record %u (%s): ", idx, opl::kind_name(kind));
        *why = std::string(head) + buf;
    }
    return rc;
}

// The plane pointers of a K / V plane at `beat` in every channel; false when a plane leaves the view.
bool planes(const dram_view & d, uint32_t beat, uint32_t geom, uint8_t ** out) {
    const tsr_kv_geom g = { opl::geom_e64(geom) << 6, d.num_ch, opl::geom_lg(geom) };
    if (g.E == 0 || g.LG == 0) return false;
    const uint64_t end = (uint64_t(beat) + tsr_kv_plane_beats(&g)) * LINE;
    if (end > d.ch_bytes) return false;
    for (uint32_t i = 0; i < d.num_ch; ++i) out[i] = d.ch[i] + uint64_t(beat) * LINE;
    return true;
}

}  // namespace

int opl_exec(const uint8_t * blob, size_t bytes, const dram_view & d, const opl_exec_cfg & cfg, std::string * why,
             opl_exec_stats * st) {
    opl::reader rd;
    const uint32_t e = rd.open(blob, uint32_t(bytes), true);
    if (e != opl::RET_OK) return fail(why, OX_BAD_BLOB, 0, opl::K_NOP, "the header or CRC is bad (executor code 0x%08x)", e);
    if (d.num_ch == 0 || d.num_ch > TSR_NUM_CH_MAX) return fail(why, OX_RANGE, 0, opl::K_NOP, "%u channels", d.num_ch);

    // The preload area: the entries' bytes from DRAM, before record 0.
    std::vector<uint8_t> pre(size_t(rd.hdr().pre_lines) * LINE, 0u);
    for (uint32_t i = 0; i < rd.hdr().n_pre; ++i) {
        const opl::pre_entry pe = rd.pre(i);
        const uint32_t ch = pe.src >> 28;
        const uint64_t off = uint64_t(pe.src & 0x0FFFFFFFu) * LINE;
        if (ch >= d.num_ch || off + pe.bytes > d.ch_bytes || size_t(pe.pre_byte) + pe.bytes > pre.size()) {
            return fail(why, OX_RANGE, 0, opl::K_NOP, "preload entry %u (handle 0x%08x, %u bytes) is outside the DRAM view", i, pe.src,
                        pe.bytes);
        }
        memcpy(pre.data() + pe.pre_byte, d.ch[ch] + off, pe.bytes);
    }

    const tsr_att_cfg att = { d.num_ch, cfg.t_max, cfg.pcap, cfg.k_max };
    opl::record r{};
    for (uint32_t i = 0; i < rd.hdr().n_ops; ++i) {
        const uint32_t ne = rd.next(r);
        if (ne != opl::RET_OK) return fail(why, OX_BAD_BLOB, i, r.kind, "record check 0x%08x", ne);
        if (rd.check_args(r) != opl::RET_OK) return fail(why, OX_BAD_BLOB, i, r.kind, "argument rules refuse it");
        uint32_t a[255] = {};
        r.words(a, r.n_args);
        int rc = 0;
        switch (r.kind) {
            case opl::K_NOP:
            case opl::K_JOIN:
            case opl::K_MUL_MAT_END:
                break;
            case opl::K_MUL_MAT:
            case opl::K_MUL_MAT_BEGIN:
                rc = mulmat_ref(d, a, cfg.mm_threads);
                break;
            case opl::K_RMS_NORM:
                rc = rmsnorm_ref(d, a);
                break;
            case opl::K_ROPE:
                rc = rope_ref(d, a, pre.data(), uint32_t(pre.size()));
                break;
            case opl::K_SET_ROWS_KV: {
                uint8_t * pl[TSR_NUM_CH_MAX];
                if (!planes(d, a[opl::setrows::PLANE], a[opl::setrows::GEOM], pl))
                    return fail(why, OX_RANGE, i, r.kind, "the KV plane is outside the DRAM view");
                rc = set_rows_kv_ref(d, a, pre.data(), pre.size(), pl, d.num_ch);
                break;
            }
            case opl::K_FLASH_ATTN: {
                uint8_t * kp[TSR_NUM_CH_MAX];
                uint8_t * vp[TSR_NUM_CH_MAX];
                if (!planes(d, a[opl::fa::K_PLANE], a[opl::fa::GEOM], kp) || !planes(d, a[opl::fa::V_PLANE], a[opl::fa::GEOM], vp))
                    return fail(why, OX_RANGE, i, r.kind, "a KV plane is outside the DRAM view");
                const uint8_t * ck[TSR_NUM_CH_MAX];
                const uint8_t * cv[TSR_NUM_CH_MAX];
                for (uint32_t c = 0; c < d.num_ch; ++c) {
                    ck[c] = kp[c];
                    cv[c] = vp[c];
                }
                rc = flash_attn_ref(d, a, pre.data(), pre.size(), ck, cv, att);
                break;
            }
            case opl::K_ADD:
            case opl::K_MUL:
            case opl::K_SCALE:
            case opl::K_GLU:
            case opl::K_UNARY:
            case opl::K_CPY:
            case opl::K_GET_ROWS:
                rc = elem_ref(d, r.kind, a, r.n_args, pre.data(), pre.size());
                break;
            default:
                return fail(why, OX_NO_REF, i, r.kind, "no host reference for this kind");
        }
        if (rc != 0) return fail(why, rc == -2 ? OX_RANGE : OX_RECORD, i, r.kind, "the reference refuses it (status %d)", rc);
        if (st != nullptr) {
            st->records++;
            if (r.kind < 64) st->by_kind[r.kind]++;
        }
    }
    return OX_OK;
}

}  // namespace ref
}  // namespace tsr
