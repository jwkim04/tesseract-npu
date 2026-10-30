// tsr-msum.cpp — see tsr-msum.h (AT §7.3, §5.5, §6.6; SW §9.3 M7 "Mask summaries", V2-SW-32).
#include "tsr-msum.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

uint64_t pad64(uint64_t v) { return (v + 63u) & ~63ull; }

// ASMAX / AMASK need a finite, normal, positive scale (tsr_msum_mask_f32 aborts otherwise;
// supports_op refuses such an FA, AT §2.6).
bool scale_ok(uint32_t b) {
    const uint32_t e = (b >> 23) & 0xFFu;
    return (b & 0x80000000u) == 0u && e != 0u && e != 0xFFu;
}

// A value that makes fdiv(m, scale) depend on the scale: finite and nonzero (+-0 and -inf, the only
// values llama.cpp writes, AT §2.4, are unchanged by the division; +inf and NaN stay +inf / NaN).
bool scale_dependent(const std::vector<uint16_t> & f16) {
    for (uint16_t h : f16) {
        if ((h & 0x7C00u) != 0x7C00u && (h & 0x7FFFu) != 0u) {
            return true;
        }
    }
    return false;
}

float f32_of(uint32_t b) {
    float f;
    memcpy(&f, &b, 4);
    return f;
}

}  // namespace

bool tsr_msum_is_mask_name(const char * name) {
    return name != nullptr && strstr(name, "kq_mask") != nullptr;
}

bool tsr_msum_layout_of(const char * name, bool is_f16, const int64_t ne[4], const size_t nb[4], size_t nbytes,
                        tsr_msum_layout * out) {
    if (!tsr_msum_is_mask_name(name) || !is_f16) {
        return false;
    }
    if (ne[0] < 1 || ne[0] > (1 << 20) || ne[1] < 1 || ne[1] > (int64_t) UINT32_MAX / 16 || ne[2] != 1 || ne[3] != 1) {
        return false;
    }
    if (nb[0] != 2 || nb[1] % 2 != 0 || nb[1] < 2 * (size_t) ne[0]) {
        return false;
    }
    tsr_msum_layout L;
    L.n_kv       = (uint32_t) ne[0];
    L.n_q        = (uint32_t) ne[1];
    L.row_stride = nb[1] / 2;
    L.mask_bytes = pad64(nbytes);
    L.msum_off   = L.mask_bytes;
    L.msum_bytes = pad64(4ull * tsr_msum_words(L.n_q));
    L.copy_off   = L.msum_off + L.msum_bytes;
    L.copy_bytes = (uint64_t) L.n_q * TSR_LINE_BYTES * ((L.n_kv + 15u) / 16u);
    L.total      = L.copy_off + L.copy_bytes;
    if (out != nullptr) {
        *out = L;
    }
    return true;
}

void tsr_msum_store::init(const tsr_att_cfg & cfg, tsr_dram_write_fn write, alloc_fn alloc, free_fn release) {
    clear();
    cfg_ = cfg;
    write_ = std::move(write);
    alloc_ = std::move(alloc);
    release_ = std::move(release);
}

void tsr_msum_store::drop_side(mask & m) {
    for (const copy_slot & s : m.slots) {
        if (s.side && release_) {
            release_(s.ch, s.off);
        }
    }
    m.slots.clear();
}

void tsr_msum_store::forget(uint64_t lo, uint64_t hi) {
    for (auto it = masks_.lower_bound(lo); it != masks_.end() && it->first < hi;) {
        drop_side(it->second);
        it = masks_.erase(it);
    }
}

void tsr_msum_store::clear() {
    for (auto & kv : masks_) {
        drop_side(kv.second);
    }
    masks_.clear();
    for (auto & kv : nomask_) {
        if (release_) {
            release_(kv.second.ch, kv.second.off);
        }
    }
    nomask_.clear();
}

bool tsr_msum_store::last_msum(uint64_t key, std::vector<uint32_t> * words) const {
    auto it = masks_.find(key);
    if (it == masks_.end() || it->second.words.empty()) {
        return false;
    }
    *words = it->second.words;
    return true;
}

void tsr_msum_store::on_set(uint64_t key, uint32_t ch, uint64_t off, const tsr_msum_layout & L, const uint16_t * f16) {
    mask & m = masks_[key];
    if (m.L.n_kv != L.n_kv || m.L.n_q != L.n_q || m.L.total != L.total || m.ch != ch || m.off != off) {
        drop_side(m);             // a new mask at this address: its readers come with the next build
        m = mask();
        m.L = L;
        m.ch = ch;
        m.off = off;
    }
    m.f16.resize((size_t) L.n_q * L.n_kv);           // the host copy, rows dense (stride n_kv)
    for (uint32_t q = 0; q < L.n_q; ++q) {
        memcpy(m.f16.data() + (size_t) q * L.n_kv, f16 + (size_t) q * L.row_stride, (size_t) L.n_kv * 2u);
    }
    m.have_data = true;
    produce(m);
}

// One production (AT §7.3): the MSUM, the GENERAL promotion of refused tiles, the f32 copies.
void tsr_msum_store::produce(mask & m) {
    const tsr_msum_layout & L = m.L;
    const uint32_t slot0 = tsr_dram_handle(m.ch, m.off + L.copy_off);   // the reservation's copy
    m.words.assign(tsr_msum_words(L.n_q), 0u);
    const uint32_t flags = tsr_msum_build_f16(m.f16.data(), L.n_kv, L.n_kv, L.n_q, slot0, m.words.data());
    bool general = (flags & TSR_MSUM_GENERAL) != 0u;
    if (general) {
        st_general++;
    }

    // Scales whose copy some tile reads: run the shared planner (tsr_fa_plan, the firmware's
    // source) on every query tile of every distinct reader shape, as the FLASH_ATTN kernel will.
    std::vector<bool> need(m.slots.size(), general);
    if (!general && !m.readers.empty()) {
        tsr_msum ms;
        if (tsr_msum_view(m.words.data(), m.words.size(), &ms) != 0) {
            fprintf(stderr, "tsr-msum: internal error: bad MSUM image\n");
            abort();
        }
        std::vector<tsr_fa_chunk> chunks(64);
        bool refused = false;
        for (size_t i = 0; i < m.readers.size() && !refused; ++i) {
            const tsr_msum_reader & r = m.readers[i];
            bool seen = false;                                     // same shape as an earlier reader
            for (size_t j = 0; j < i && !seen; ++j) {
                seen = m.readers[j].E == r.E && m.readers[j].D == r.D && m.readers[j].G == r.G;
            }
            bool gen_tile = false;
            if (seen) {
                for (size_t j = 0; j < i; ++j) {                   // inherit: the plan is the same
                    const tsr_msum_reader & o = m.readers[j];
                    if (o.E == r.E && o.D == r.D && o.G == r.G) {
                        for (size_t s = 0; s < m.slots.size(); ++s) {
                            gen_tile = gen_tile || (m.slots[s].scale_bits == o.scale_bits && need[s]);
                        }
                    }
                }
            } else {
                uint32_t GS = 0, Tq = 0;
                tsr_fa_pass_rule(&cfg_, r.G == 0 ? 1u : r.G, &GS, &Tq);
                for (uint32_t q0 = 0; q0 < L.n_q && Tq != 0 && !refused; q0 += Tq) {
                    const uint32_t tq = std::min(Tq, L.n_q - q0);
                    long n = tsr_fa_plan(&cfg_, r.E, r.D, &ms, q0, tq, nullptr, 0, chunks.data(), chunks.size());
                    if (n > (long) chunks.size()) {
                        chunks.resize((size_t) n);
                        n = tsr_fa_plan(&cfg_, r.E, r.D, &ms, q0, tq, nullptr, 0, chunks.data(), chunks.size());
                    }
                    if (n == -3) {
                        refused = true;                            // AT §6.6 rule 5 / 6: data-dependent
                    } else if (n > 0) {
                        for (long k = 0; k < n; ++k) {
                            gen_tile = gen_tile || chunks[(size_t) k].general != 0u;
                        }
                    }
                    // n = -1 / -2: a shape the planner cannot take; supports_op refuses such an FA
                }
            }
            if (gen_tile) {
                for (size_t s = 0; s < m.slots.size(); ++s) {
                    need[s] = need[s] || m.slots[s].scale_bits == r.scale_bits;
                }
            }
        }
        if (refused) {
            // Promote: every tile of every reader takes the GENERAL path (rule 1), which never
            // refuses for mask data; the copies make it valid.
            m.words[3] |= TSR_MSUM_GENERAL;
            m.words[4] = slot0;
            general = true;
            need.assign(m.slots.size(), true);
            st_promoted++;
        }
    }
    write_(m.ch, m.off + L.msum_off, m.words.data(), m.words.size() * 4u);
    st_produced++;

    const bool dep = scale_dependent(m.f16);
    std::vector<float> copy;
    uint32_t copy_scale = 0;
    for (size_t s = 0; s < m.slots.size(); ++s) {
        if (!need[s] || !scale_ok(m.slots[s].scale_bits)) {
            continue;
        }
        const uint32_t sb = dep ? m.slots[s].scale_bits : 0x3F800000u;   // scale-free mask: one image
        if (copy.empty() || sb != copy_scale) {
            copy.assign((size_t) (L.copy_bytes / 4u), 0.0f);
            tsr_msum_mask_f32(m.f16.data(), L.n_kv, L.n_kv, L.n_q, f32_of(sb), copy.data());
            copy_scale = sb;
        }
        write_(m.slots[s].ch, m.slots[s].off, copy.data(), (size_t) L.copy_bytes);
        st_copies++;
    }
}

const tsr_msum_store::copy_slot & tsr_msum_store::slot_for(mask & m, uint32_t scale_bits) {
    for (const copy_slot & s : m.slots) {
        if (s.scale_bits == scale_bits) {
            return s;
        }
    }
    copy_slot s;
    s.scale_bits = scale_bits;
    if (m.slots.empty()) {
        s.ch = m.ch;                           // the reservation next to the mask
        s.off = m.off + m.L.copy_off;
        s.side = false;
    } else {
        if (!alloc_ || !alloc_(m.L.copy_bytes, &s.ch, &s.off)) {
            fprintf(stderr, "tsr-msum: no device memory for a further f32 mask copy (%.2f MiB; a mask read by FLASH_ATTN "
                    "ops with %zu different scales)\n", m.L.copy_bytes / 1048576.0, m.slots.size() + 1);
            abort();
        }
        s.side = true;
    }
    m.slots.push_back(s);
    return m.slots.back();
}

tsr_msum_ref tsr_msum_store::ref_of(const mask & m, const copy_slot & s) const {
    tsr_msum_ref r;
    r.valid = true;
    r.n_q = m.L.n_q;
    r.n_kv = m.L.n_kv;
    r.msum_handle = tsr_dram_handle(m.ch, m.off + m.L.msum_off);
    r.msum_range.ch = m.ch;
    r.msum_range.lo = m.off + m.L.msum_off;
    r.msum_range.hi = r.msum_range.lo + 4ull * tsr_msum_words(m.L.n_q);
    r.mask_handle = tsr_dram_handle(s.ch, s.off);
    r.copy_range.ch = s.ch;
    r.copy_range.lo = s.off;
    r.copy_range.hi = s.off + m.L.copy_bytes;
    r.flags = m.words.size() > 3 ? m.words[3] : 0u;
    return r;
}

tsr_msum_ref tsr_msum_store::attach(uint64_t key, const tsr_msum_reader & r) {
    auto it = masks_.find(key);
    if (it == masks_.end() || !it->second.have_data) {
        return tsr_msum_ref();
    }
    mask & m = it->second;
    const bool known = std::find(m.readers.begin(), m.readers.end(), r) != m.readers.end();
    if (!known) {
        m.readers.push_back(r);
        slot_for(m, r.scale_bits);
        produce(m);                            // the new reader's tiles and copy (from the host copy)
        st_reattach++;
    }
    return ref_of(m, slot_for(m, r.scale_bits));
}

tsr_msum_ref tsr_msum_store::nomask(uint32_t n_q, uint32_t n_kv) {
    tsr_msum_ref r;
    if (n_q == 0 || n_kv == 0) {
        return r;
    }
    const auto key = std::make_pair(n_q, n_kv);
    auto it = nomask_.find(key);
    const uint64_t bytes = 4ull * tsr_msum_words(n_q);
    if (it == nomask_.end()) {
        nomask_entry e;
        if (!alloc_ || !alloc_(pad64(bytes), &e.ch, &e.off)) {
            return r;
        }
        std::vector<uint32_t> w(tsr_msum_words(n_q), 0u);
        w[0] = TSR_MSUM_MAGIC;
        w[1] = n_q;
        w[2] = n_kv;
        for (uint32_t q = 0; q < n_q; ++q) {
            w[TSR_MSUM_HDR_WORDS + TSR_MSUM_ROW_WORDS * q + 1] = n_kv;   // one run [0, n_kv)
        }
        write_(e.ch, e.off, w.data(), w.size() * 4u);
        it = nomask_.emplace(key, e).first;
    }
    r.valid = true;
    r.n_q = n_q;
    r.n_kv = n_kv;
    r.msum_handle = tsr_dram_handle(it->second.ch, it->second.off);
    r.msum_range.ch = it->second.ch;
    r.msum_range.lo = it->second.off;
    r.msum_range.hi = it->second.off + bytes;
    r.mask_handle = tsr::opl::HANDLE_NONE;
    r.copy_range = r.msum_range;           // nothing else is read; a valid (non-empty) range
    r.flags = 0;
    return r;
}
