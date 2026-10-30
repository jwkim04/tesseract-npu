// tsr-opl.cpp — see tsr-opl.h: DRAM handles, the builder and the cache (no transport calls; the
// `graph` launcher is tsr-opl-dev.cpp, so the unit tests link without a device).
#include "tsr-opl.h"

#include <algorithm>
#include <cstring>

#include "tsr_ref.h"

namespace opl = tsr::opl;
namespace mm  = tsr::opl::mm;

uint32_t tsr_dram_handle(uint32_t ch, uint64_t byte_off) {
    return (ch << 28) | (uint32_t) ((byte_off / TSR_BEAT_BYTES) & 0x0FFFFFFFu);
}

tsr::plan::dev_cfg tsr_plan_cfg(const tsr_config & cfg) {
    return tsr::plan::dev_cfg{ cfg.spm_bytes / TSR_LINE_BYTES, cfg.t_max, cfg.k_max };
}

// ---------------------------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------------------------

tsr_opl_builder::tsr_opl_builder(uint32_t num_ch, uint32_t flags) : num_ch_(num_ch), flags_(flags), w_(flags) {}

void tsr_opl_builder::reset() {
    w_ = opl::writer(flags_);
    ops_.clear();
    anc_.clear();
    n_joins_ = n_deps_ = n_dropped_ = 0;
}

tsr_dram_range tsr_rows_range(uint32_t ch, uint64_t lo, uint64_t n, uint64_t row, uint64_t stride) {
    tsr_dram_range r;
    r.ch = ch;
    r.lo = lo;
    if (n <= 1 || stride <= row) {
        r.hi = lo + (n > 1 ? (n - 1) * stride : 0) + row;     // contiguous (or the same row again)
        r.hi = std::max(r.hi, lo + row);
        return r;
    }
    r.hi = lo + (n - 1) * stride + row;
    r.row = row;
    r.stride = stride;
    return r;
}

// Is some row of strided b inside [p, q)? Row j = [b.lo + j s, b.lo + j s + w) meets [p, q) iff
// p - w < b.lo + j s < q, for 0 <= j < rows.
static bool tsr_strided_meets(const tsr_dram_range & b, uint64_t p, uint64_t q) {
    const uint64_t s = b.stride, w = b.row;
    const uint64_t rows = (b.hi - b.lo - w) / s + 1;
    // smallest j with b.lo + j s + w > p
    uint64_t j = 0;
    if (p + 1 > b.lo + w) {
        j = (p + 1 - b.lo - w + s - 1) / s;
    }
    return j < rows && b.lo + j * s < q;
}

bool tsr_ranges_overlap(const tsr_dram_range & a, const tsr_dram_range & b) {
    if (a.ch != b.ch || !(a.lo < b.hi && b.lo < a.hi)) {
        return false;
    }
    if (a.row == 0 && b.row == 0) {
        return true;
    }
    if (a.row == 0) {
        return tsr_strided_meets(b, a.lo, a.hi);
    }
    if (b.row == 0) {
        return tsr_strided_meets(a, b.lo, b.hi);
    }
    // both strided: every row of the one with fewer rows against the other
    const uint64_t na = (a.hi - a.lo - a.row) / a.stride + 1, nb = (b.hi - b.lo - b.row) / b.stride + 1;
    const tsr_dram_range & x = na <= nb ? a : b;
    const tsr_dram_range & y = na <= nb ? b : a;
    const uint64_t n = std::min(na, nb);
    if (n > 65536) {
        return true;        // conservative: an extra dependency, never a missing one
    }
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t p = x.lo + i * x.stride;
        if (p >= y.hi) {
            break;
        }
        if (p + x.row > y.lo && tsr_strided_meets(y, p, p + x.row)) {
            return true;
        }
    }
    return false;
}

// Rows [0, T) of `lines` lines each, `stride` lines apart (stride 0: row 0 for every token).
static tsr_dram_range tsr_rows_of(uint32_t h, uint32_t T, uint32_t stride, uint32_t lines) {
    return tsr_rows_range(tsr_handle_ch(h), tsr_handle_byte(h), T, (uint64_t) lines * TSR_LINE_BYTES,
                          (uint64_t) stride * TSR_LINE_BYTES);
}

void tsr_opl_builder::mm_ranges(const uint32_t * a, std::vector<tsr_dram_range> & rd, std::vector<tsr_dram_range> & wr) const {
    rd.clear();
    wr.clear();
    const uint32_t N = a[mm::N], K = a[mm::K], T = a[mm::T], fmt = a[mm::FMT];
    const uint32_t xl = (uint32_t) tsr_ceil_div_u64((uint64_t) K * 4, TSR_LINE_BYTES);
    const uint32_t yl = (uint32_t) tsr_ceil_div_u64((uint64_t) N * 4, TSR_LINE_BYTES);
    rd.push_back(tsr_rows_of(a[mm::X], T, a[mm::X_STRIDE], xl));
    if (a[mm::ADD] != opl::HANDLE_NONE) {
        rd.push_back(tsr_rows_of(a[mm::ADD], T, a[mm::ADD_STRIDE], yl));
    }
    const tsr_dram_range y = tsr_rows_of(a[mm::Y], T, a[mm::Y_STRIDE], yl);
    if (a[mm::FLAGS] & mm::FLAG_ACC_DST) {
        rd.push_back(y);
    }
    // the weight rows on every channel (read only during a launch; listed so that an op writing
    // into a weight's bytes would be ordered too)
    if (tsr::plan::fmt_valid(fmt) && K > 0 && K % tsr::plan::fmt_block(fmt) == 0 && N > 0 && num_ch_ > 0) {
        const uint64_t lo = (uint64_t) a[mm::W_OFF] * TSR_BEAT_BYTES;
        const uint64_t bytes = tsr_tensor_ch_beats(fmt, N, K, num_ch_) * TSR_BEAT_BYTES;
        for (uint32_t c = 0; c < num_ch_; ++c) {
            rd.push_back(tsr_rows_range(c, lo, 1, bytes, 0));
        }
    }
    wr.push_back(y);
}

bool tsr_opl_builder::is_ancestor(uint32_t a, uint32_t b) const {
    const std::vector<uint64_t> & s = anc_[b];
    return a / 64 < s.size() && ((s[a / 64] >> (a % 64)) & 1u) != 0;
}

uint32_t tsr_opl_builder::push_record(uint32_t kind, const uint32_t * args, uint32_t n_args, uint32_t d0, uint32_t d1,
                                      int32_t tag) {
    const uint32_t idx = w_.add_raw(kind, args, n_args, d0, d1);
    if (idx == opl::HANDLE_NONE) {
        return FAILED;
    }
    std::vector<uint64_t> s((idx + 63) / 64, 0);
    for (uint32_t d : { d0, d1 }) {
        if (d == opl::DEP_NONE) {
            continue;
        }
        const std::vector<uint64_t> & sd = anc_[d];
        for (size_t i = 0; i < sd.size(); ++i) {
            s[i] |= sd[i];
        }
        s[d / 64] |= 1ull << (d % 64);
    }
    anc_.push_back(std::move(s));
    op_info oi;
    oi.kind = kind;
    oi.tag = tag;
    oi.dep0 = d0;
    oi.dep1 = d1;
    ops_.push_back(std::move(oi));
    return idx;
}

uint32_t tsr_opl_builder::add(uint32_t kind, const uint32_t * args, uint32_t n_args, const std::vector<tsr_dram_range> & rd,
                              const std::vector<tsr_dram_range> & wr, int32_t tag) {
    if (!w_.ok()) {
        return FAILED;
    }
    // Conflicting earlier ops (RAW, WAR, WAW), latest first.
    std::vector<uint32_t> cand;
    for (uint32_t i = (uint32_t) ops_.size(); i-- > 0;) {
        const op_info & o = ops_[i];
        bool hit = false;
        for (const tsr_dram_range & r : rd) {
            for (const tsr_dram_range & q : o.wr) hit = hit || tsr_ranges_overlap(r, q);
        }
        for (const tsr_dram_range & r : wr) {
            for (const tsr_dram_range & q : o.wr) hit = hit || tsr_ranges_overlap(r, q);
            for (const tsr_dram_range & q : o.rd) hit = hit || tsr_ranges_overlap(r, q);
        }
        if (hit) {
            cand.push_back(i);
        }
    }
    // Transitive reduction: a conflicting op that is an ancestor of a later kept one is implied.
    std::vector<uint32_t> kept;
    uint32_t dropped = 0;
    for (uint32_t c : cand) {
        bool implied = false;
        for (uint32_t k : kept) {
            implied = implied || is_ancestor(c, k);
        }
        if (implied) {
            dropped++;
        } else {
            kept.push_back(c);
        }
    }
    std::sort(kept.begin(), kept.end());
    const uint32_t joins = kept.size() > 2 ? (uint32_t) kept.size() - 2 : 0;
    if (ops_.size() + joins + 1 > opl::MAX_OPS) {
        return NO_ROOM;
    }
    // More than two predecessors: JOIN records, folded as tsr::opl::writer::add does (SW §7.2): the
    // first two go into a JOIN, whose index replaces them at the end of the list, until two remain.
    std::vector<uint32_t> q = kept;
    while (q.size() > 2) {
        const uint32_t j = push_record(opl::K_JOIN, nullptr, 0, q[0], q[1], -1);
        if (j == FAILED) {
            return FAILED;
        }
        n_joins_++;
        q.erase(q.begin(), q.begin() + 2);
        q.push_back(j);
    }
    const uint32_t idx = push_record(kind, args, n_args, q.size() > 0 ? q[0] : opl::DEP_NONE,
                                     q.size() > 1 ? q[1] : opl::DEP_NONE, tag);
    if (idx == FAILED) {
        return FAILED;
    }
    ops_[idx].rd = rd;
    ops_[idx].wr = wr;
    n_deps_ += (uint32_t) kept.size();
    n_dropped_ += dropped;
    return idx;
}

uint32_t tsr_opl_builder::mul_mat(const uint32_t * args, int32_t tag) {
    std::vector<tsr_dram_range> rd, wr;
    mm_ranges(args, rd, wr);
    return add(opl::K_MUL_MAT, args, mm::NARGS, rd, wr, tag);
}

bool tsr_opl_builder::touches(const tsr_dram_range & r, bool any_access) const {
    for (const op_info & o : ops_) {
        for (const tsr_dram_range & q : o.wr) {
            if (tsr_ranges_overlap(r, q)) return true;
        }
        if (any_access) {
            for (const tsr_dram_range & q : o.rd) {
                if (tsr_ranges_overlap(r, q)) return true;
            }
        }
    }
    return false;
}

std::vector<uint8_t> tsr_opl_builder::finish() const {
    std::vector<uint8_t> blob = w_.finish();
    if (blob.size() >= opl::HDR_BYTES) {
        // graph_id (logged by the executor) = the CRC of the body: equal lists are equal blobs. The
        // header is not covered by the CRC.
        opl::writer::wr32(blob.data() + 4u * opl::hw::GRAPH_ID, opl::rd32(blob.data() + 4u * opl::hw::CRC));
    }
    return blob;
}

// ---------------------------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------------------------

uint64_t tsr_opl_cache::hash(const uint8_t * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

void tsr_opl_cache::init(uint32_t ch, uint64_t base, uint64_t size, size_t max_entries) {
    ch_ = ch;
    max_entries_ = std::max<size_t>(1, max_entries);
    lru_.clear();
    map_.clear();
    alloc_.init(base, size, TSR_BEAT_BYTES);
    st_ = stats();
}

void tsr_opl_cache::clear() {
    for (const entry & e : lru_) {
        alloc_.release(e.off);
    }
    lru_.clear();
    map_.clear();
}

void tsr_opl_cache::evict_lru() {
    auto it = std::prev(lru_.end());
    auto range = map_.equal_range(it->key);
    for (auto m = range.first; m != range.second; ++m) {
        if (m->second == it) {
            map_.erase(m);
            break;
        }
    }
    alloc_.release(it->off);
    lru_.erase(it);
    st_.evictions++;
}

bool tsr_opl_cache::place(const std::vector<uint8_t> & blob, const tsr_dram_write_fn & write, uint64_t * off, bool * hit) {
    const uint64_t key = hash(blob.data(), blob.size());
    auto range = map_.equal_range(key);
    for (auto m = range.first; m != range.second; ++m) {
        if (m->second->blob == blob) {
            lru_.splice(lru_.begin(), lru_, m->second);
            *off = m->second->off;
            *hit = true;
            st_.hits++;
            return true;
        }
    }
    if (blob.empty() || blob.size() > alloc_.total()) {
        return false;
    }
    while (lru_.size() >= max_entries_) {
        evict_lru();
    }
    uint64_t o = 0;
    while (!alloc_.alloc(blob.size(), &o)) {
        if (lru_.empty()) {
            return false;
        }
        evict_lru();
    }
    try {
        write(ch_, o, blob.data(), blob.size());
    } catch (...) {
        alloc_.release(o);
        throw;
    }
    lru_.push_front(entry{ key, o, blob });
    map_.emplace(key, lru_.begin());
    st_.misses++;
    st_.bytes_written += blob.size();
    *off = o;
    *hit = false;
    return true;
}


// ---------------------------------------------------------------------------------------------
// The op a failed launch names (SW §9.3 step 5)
// ---------------------------------------------------------------------------------------------
int32_t tsr_opl_failed_op(const tsr_opl_run_result & r, uint32_t n_ops) {
    if (r.rc != 0 || r.res.type != TSR_LCQ_R0_TYPE_DONE || r.res.status != TSR_LCQ_R0_STATUS_TILE_ERR) {
        return -1;
    }
    // RET = failing record index + 1 (SW §6.4, §7.3); 0 or beyond the list: the executor did not
    // name a record of it.
    if (r.res.ret < 1 || (uint32_t) r.res.ret > n_ops) {
        return -1;
    }
    return r.res.ret - 1;
}

int32_t tsr_opl_failed_tag(const tsr_opl_builder & b, const tsr_opl_run_result & r) {
    const int32_t op = tsr_opl_failed_op(r, b.n_ops());
    return op < 0 ? -1 : b.tag_of((uint32_t) op);
}
