// tsr-kvbuf.cpp — see tsr-kvbuf.h (SW §9.3 M7 "KV buffers", V2-SW-17; AT §4, §4.5, §4.6).
#include "tsr-kvbuf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t Q8_0_BLOCK      = 34u;          // sizeof(block_q8_0): fp16 d + int8 qs[32]

[[noreturn]] void kv_fatal(const char * what, uint64_t a, uint64_t b) {
    fprintf(stderr, "tsr-kvbuf: %s (0x%" PRIx64 ", 0x%" PRIx64 ")\n", what, a, b);
    abort();
}

uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

}  // namespace

// ---------------------------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------------------------
bool tsr_kv_parse_name(const char * name, uint32_t * which, int * layer) {
    static const char prefix[] = "cache_";
    const size_t np = sizeof(prefix) - 1;
    if (name == nullptr || strncmp(name, prefix, np) != 0) {
        return false;
    }
    const size_t n = strlen(name);
    size_t d = n;                                   // first digit of the trailing number
    while (d > 0 && name[d - 1] >= '0' && name[d - 1] <= '9') {
        --d;
    }
    // "cache_" <tag> ('k' | 'v') "_l" <digits>, 1-9 digits
    if (d == n || n - d > 9 || d < np + 3 || name[d - 1] != 'l' || name[d - 2] != '_') {
        return false;
    }
    const char kv = name[d - 3];
    if (kv != 'k' && kv != 'v') {
        return false;
    }
    if (which != nullptr) {
        *which = kv == 'k' ? TSR_KV_K : TSR_KV_V;
    }
    if (layer != nullptr) {
        *layer = atoi(name + d);
    }
    return true;
}

bool tsr_kv_identify(const char * name, bool is_q8_0, const int64_t ne[4], const size_t nb[4], uint32_t num_ch,
                     uint32_t k_max, tsr_kv_shape * out, std::string * why) {
    auto no = [why](const char * w) {
        if (why != nullptr) {
            *why = w;
        }
        return false;
    };
    tsr_kv_shape s;
    if (!tsr_kv_parse_name(name, &s.which, &s.layer)) return no("name is not cache_<tag>k_l<n> / cache_<tag>v_l<n>");
    if (!is_q8_0)                                       return no("type is not q8_0 (AT §2.6: q8_0 K and V only)");
    if (num_ch < 1 || num_ch > TSR_NUM_CH_MAX)          return no("bad channel count");
    if (ne[0] < 64 || ne[0] % 64 != 0 || ne[0] / 64 > 511) return no("E = ne[0] must be a multiple of 64 in [64, 511*64]");
    if (k_max != 0 && (uint64_t) ne[0] > k_max)         return no("E = ne[0] exceeds K_MAX");
    if (ne[1] < 1 || ne[1] > (int64_t) UINT32_MAX)      return no("kv_size = ne[1] out of range");
    if (ne[2] < 1 || ne[2] > 256)                       return no("n_stream = ne[2] out of range");
    if (ne[3] != 1)                                     return no("not 3D (ne[3] != 1)");
    s.E         = (uint32_t) ne[0];
    s.kv_size   = (uint32_t) ne[1];
    s.n_stream  = (uint32_t) ne[2];
    s.C         = num_ch;
    s.row_bytes = (size_t) (s.E / 32u) * Q8_0_BLOCK;
    if (nb[0] != Q8_0_BLOCK || nb[1] != s.row_bytes || nb[2] != s.row_bytes * s.kv_size ||
        nb[3] != nb[2] * s.n_stream) {
        return no("not a contiguous q8_0 tensor");
    }
    s.LG = tsr_kv_lg_min(s.kv_size, num_ch);
    if (s.LG >= (1u << 16))                             return no("LG >= 2^16 (GEOM's LG field)");
    const tsr_kv_geom g = { s.E, s.C, s.LG };
    const uint64_t beats = tsr_kv_plane_beats(&g);
    if (beats >= (1ull << 28))                          return no("plane of 2^28 beats or more (a handle's beat field)");
    s.plane_bytes = beats * TSR_BEAT_BYTES;
    if (out != nullptr) {
        *out = s;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Planes
// ---------------------------------------------------------------------------------------------
void tsr_kv_plane::ranges(std::vector<tsr_dram_range> & out) const {
    for (uint32_t c = 0; c < geom.C; ++c) {
        tsr_dram_range r;
        r.ch = c;
        r.lo = off;
        r.hi = off + bytes;
        out.push_back(r);
    }
}

tsr_kvbuf::tsr_kvbuf(uint32_t num_ch, tsr_range_alloc * heap, tsr_dram_read_fn read, tsr_dram_write_fn write)
    : num_ch_(num_ch), heap_(heap), read_(std::move(read)), write_(std::move(write)) {}

tsr_kvbuf::~tsr_kvbuf() {
    release();
}

void tsr_kvbuf::add(const std::string & name, uint64_t vaddr, size_t nbytes, const tsr_kv_shape & shape) {
    if (placed_) {
        kv_fatal("add() after place()", vaddr, nbytes);
    }
    if (shape.C != num_ch_ || nbytes != (size_t) shape.n_stream * shape.kv_size * shape.row_bytes) {
        kv_fatal("tensor shape does not match the buffer", vaddr, nbytes);
    }
    tsr_kv_tensor t;
    t.name = name;
    t.vaddr = vaddr;
    t.nbytes = nbytes;
    t.shape = shape;
    auto it = std::lower_bound(tensors_.begin(), tensors_.end(), vaddr,
                               [](const tsr_kv_tensor & a, uint64_t v) { return a.vaddr < v; });
    if ((it != tensors_.end() && it->vaddr < vaddr + nbytes) ||
        (it != tensors_.begin() && std::prev(it)->vaddr + std::prev(it)->nbytes > vaddr)) {
        kv_fatal("overlapping KV tensors", vaddr, nbytes);
    }
    tensors_.insert(it, std::move(t));
}

bool tsr_kvbuf::place(std::string * err) {
    if (placed_) {
        return true;
    }
    std::vector<uint64_t> taken;
    for (tsr_kv_tensor & t : tensors_) {
        t.planes.clear();
        for (uint32_t s = 0; s < t.shape.n_stream; ++s) {
            tsr_kv_plane p;
            p.which   = t.shape.which;
            p.layer   = t.shape.layer;
            p.stream  = s;
            p.kv_size = t.shape.kv_size;
            p.geom    = { t.shape.E, num_ch_, t.shape.LG };
            p.bytes   = t.shape.plane_bytes;
            if (!heap_->alloc_top(p.bytes, &p.off)) {
                for (uint64_t o : taken) {
                    heap_->release(o);
                }
                for (tsr_kv_tensor & u : tensors_) {
                    u.planes.clear();
                }
                if (err != nullptr) {
                    char buf[256];
                    snprintf(buf, sizeof buf, "no room for KV plane %s stream %u (%.2f MiB per channel; heap free %.2f MiB, "
                             "largest %.2f MiB)", t.name.c_str(), s, p.bytes / 1048576.0, heap_->free_bytes() / 1048576.0,
                             heap_->largest_free() / 1048576.0);
                    *err = buf;
                }
                return false;
            }
            taken.push_back(p.off);
            t.planes.push_back(p);
        }
    }
    placed_ = true;
    return true;
}

void tsr_kvbuf::release() {
    if (!placed_) {
        return;
    }
    for (tsr_kv_tensor & t : tensors_) {
        for (const tsr_kv_plane & p : t.planes) {
            heap_->release(p.off);
        }
        t.planes.clear();
    }
    placed_ = false;
}

uint64_t tsr_kvbuf::device_bytes_per_ch() const {
    uint64_t s = 0;
    for (const tsr_kv_tensor & t : tensors_) {
        for (const tsr_kv_plane & p : t.planes) {
            s += p.bytes;
        }
    }
    return s;
}

tsr_kv_tensor * tsr_kvbuf::find(uint64_t addr) {
    auto it = std::upper_bound(tensors_.begin(), tensors_.end(), addr,
                               [](uint64_t v, const tsr_kv_tensor & a) { return v < a.vaddr; });
    if (it == tensors_.begin()) {
        return nullptr;
    }
    --it;
    return addr < it->vaddr + it->nbytes ? &*it : nullptr;
}

bool tsr_kvbuf::resolve(uint64_t addr, tsr_kv_view * out) const {
    tsr_kv_tensor * t = const_cast<tsr_kvbuf *>(this)->find(addr);
    if (t == nullptr || !placed_) {
        return false;
    }
    const uint64_t rel = addr - t->vaddr;
    const uint64_t sb  = (uint64_t) t->shape.kv_size * t->shape.row_bytes;
    const uint32_t s   = (uint32_t) (rel / sb);
    const uint64_t r   = rel % sb;
    out->t           = t;
    out->plane       = &t->planes[s];
    out->stream      = s;
    out->cell0       = (uint32_t) (r / t->shape.row_bytes);
    out->byte_in_row = (uint32_t) (r % t->shape.row_bytes);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------------------------
void tsr_kvbuf::stream_access(const tsr_kv_tensor & t, const tsr_kv_plane & p, uint64_t rel, size_t n,
                              const uint8_t * src, uint8_t * dst, uint8_t value) {
    if (n == 0) {
        return;
    }
    const bool get = dst != nullptr;
    const uint32_t C = num_ch_, E = t.shape.E, LG = t.shape.LG;
    const uint64_t rb = t.shape.row_bytes;
    const uint64_t gb_bytes = (uint64_t) tsr_kv_group_beats(E) * TSR_BEAT_BYTES;   // one local group, one channel
    const uint64_t cells_per_l = 16ull * C;                                         // cells of one local group index
    const uint64_t c_lo = rel / rb, c_hi = ceil_div(rel + n, rb);
    const uint64_t l_lo = (c_lo / 16) / C, l_hi = (ceil_div(c_hi, 16) - 1) / C + 1;
    const uint64_t w_max = std::max<uint64_t>(1, window_bytes_ / (gb_bytes * C));  // local groups per window
    const uint32_t slices = E / 64u;

    std::vector<std::vector<uint8_t>> win(C);
    std::vector<uint8_t *> wp(C);
    std::vector<uint8_t> fillbuf;
    for (uint64_t la = l_lo; la < l_hi; la += w_max) {
        const uint64_t lb = std::min<uint64_t>(la + w_max, l_hi);
        const uint64_t nl = lb - la;
        const uint64_t cell_a = la * cells_per_l;                          // first plane cell of the window
        const uint64_t wbytes = nl * gb_bytes;                             // per channel
        const uint64_t b_lo = std::max<uint64_t>(rel, cell_a * rb);        // stream bytes of the access in it
        const uint64_t b_hi = std::min<uint64_t>(rel + n, lb * cells_per_l * rb);
        if (b_lo >= b_hi) {
            continue;
        }
        // whole rows of every ggml cell of the window (cells >= kv_size are padding): no read
        const uint64_t full_hi = std::min<uint64_t>(lb * cells_per_l, t.shape.kv_size) * rb;
        const bool need_read = get || b_lo != cell_a * rb || b_hi != full_hi;
        for (uint32_t c = 0; c < C; ++c) {
            win[c].assign((size_t) wbytes, clear_value_);
            wp[c] = win[c].data();
        }
        // device <-> window: K one range per channel, V one per slice and channel (AT §4.3, §4.4)
        auto xfer = [&](bool to_dev) {
            for (uint32_t c = 0; c < C; ++c) {
                if (p.which == TSR_KV_K) {
                    const uint64_t dev = p.off + la * gb_bytes;
                    if (to_dev) write_(c, dev, wp[c], (size_t) wbytes);
                    else        read_(c, dev, wp[c], (size_t) wbytes);
                } else {
                    const uint64_t seg = nl * 17u * TSR_BEAT_BYTES;
                    for (uint32_t s = 0; s < slices; ++s) {
                        const uint64_t dev = p.off + ((uint64_t) s * LG + la) * 17u * TSR_BEAT_BYTES;
                        if (to_dev) write_(c, dev, wp[c] + s * seg, (size_t) seg);
                        else        read_(c, dev, wp[c] + s * seg, (size_t) seg);
                    }
                }
            }
        };
        if (need_read) {
            xfer(false);
            st_windows_read++;
            st_bytes_in += wbytes * C;
        }
        st_windows++;
        // the window as a plane of nl local groups: plane cell j <-> window cell j - cell_a
        const tsr_kv_geom wg = { E, C, (uint32_t) nl };
        const size_t woff = (size_t) (b_lo - cell_a * rb), wn = (size_t) (b_hi - b_lo);
        if (get) {
            tsr_kv_get_bytes(p.which, &wg, wp.data(), dst + (b_lo - rel), woff, wn);
            continue;
        }
        const uint8_t * from = src != nullptr ? src + (b_lo - rel) : nullptr;
        if (from == nullptr) {
            fillbuf.assign(wn, value);
            from = fillbuf.data();
        }
        tsr_kv_set_bytes(p.which, &wg, wp.data(), from, woff, wn);
        xfer(true);
        st_bytes_out += wbytes * C;
    }
}

void tsr_kvbuf::set(uint64_t addr, const void * src, size_t n) {
    const uint8_t * s8 = (const uint8_t *) src;
    while (n > 0) {
        tsr_kv_tensor * t = find(addr);
        if (t == nullptr || !placed_) {
            kv_fatal("set outside the KV tensors (or before placement)", addr, n);
        }
        const uint64_t sb = (uint64_t) t->shape.kv_size * t->shape.row_bytes;
        const uint64_t rel = addr - t->vaddr;
        const uint32_t s = (uint32_t) (rel / sb);
        const size_t k = (size_t) std::min<uint64_t>(n, sb - rel % sb);
        stream_access(*t, t->planes[s], rel % sb, k, s8, nullptr, 0);
        addr += k;
        n -= k;
        s8 += k;
    }
}

void tsr_kvbuf::get(uint64_t addr, void * dst, size_t n) {
    uint8_t * d8 = (uint8_t *) dst;
    while (n > 0) {
        tsr_kv_tensor * t = find(addr);
        if (t == nullptr || !placed_) {
            kv_fatal("get outside the KV tensors (or before placement)", addr, n);
        }
        const uint64_t sb = (uint64_t) t->shape.kv_size * t->shape.row_bytes;
        const uint64_t rel = addr - t->vaddr;
        const uint32_t s = (uint32_t) (rel / sb);
        const size_t k = (size_t) std::min<uint64_t>(n, sb - rel % sb);
        stream_access(*t, t->planes[s], rel % sb, k, nullptr, d8, 0);
        addr += k;
        n -= k;
        d8 += k;
    }
}

void tsr_kvbuf::fill(uint64_t addr, uint8_t value, size_t n) {
    while (n > 0) {
        tsr_kv_tensor * t = find(addr);
        if (t == nullptr || !placed_) {
            kv_fatal("memset outside the KV tensors (or before placement)", addr, n);
        }
        const uint64_t sb = (uint64_t) t->shape.kv_size * t->shape.row_bytes;
        const uint64_t rel = addr - t->vaddr;
        const uint32_t s = (uint32_t) (rel / sb);
        const size_t k = (size_t) std::min<uint64_t>(n, sb - rel % sb);
        stream_access(*t, t->planes[s], rel % sb, k, nullptr, nullptr, value);
        addr += k;
        n -= k;
    }
}

void tsr_kvbuf::clear(uint8_t value) {
    if (!placed_) {
        kv_fatal("clear before placement", 0, 0);
    }
    clear_value_ = value;
    std::vector<uint8_t> buf;
    for (const tsr_kv_tensor & t : tensors_) {
        for (const tsr_kv_plane & p : t.planes) {
            // every byte of the plane is a byte of some cell (meta and quant beats are full, AT §4.3,
            // §4.4), so the plane of a constant ggml image is that constant, padding cells included
            for (uint32_t c = 0; c < num_ch_; ++c) {
                for (uint64_t done = 0; done < p.bytes;) {
                    const size_t k = (size_t) std::min<uint64_t>(p.bytes - done, std::max<uint64_t>(window_bytes_, 4096));
                    if (buf.size() < k) {
                        buf.assign(k, value);
                    }
                    write_(c, p.off + done, buf.data(), k);
                    done += k;
                }
            }
            st_bytes_out += p.bytes * num_ch_;
        }
    }
}
