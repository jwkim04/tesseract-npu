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
    pre_.clear();
    ops_.clear();
    anc_.clear();
    n_joins_ = n_deps_ = n_dropped_ = 0;
    std::fill(std::begin(n_kind_), std::end(n_kind_), 0u);
    pend_ = pending_mm();
    open_.clear();
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

// Lines of n f32 values (L(n), VU §4.2).
static uint32_t tsr_f32_lines(uint32_t n) {
    return (n >> 4) + ((n & 15u) != 0u ? 1u : 0u);
}

// The plane bytes of every channel (AT §4.5: the same beat range [B, B + LG·GB) in each channel).
static void tsr_plane_ranges(uint32_t num_ch, uint32_t base_beat, uint32_t geom, std::vector<tsr_dram_range> & out) {
    const uint64_t e64 = opl::geom_e64(geom), lg = opl::geom_lg(geom);
    const uint64_t beats = lg * 17u * e64;              // tsr_kv_plane_beats: LG · GB, GB = 17·E/64
    if (beats == 0) {
        return;
    }
    for (uint32_t c = 0; c < num_ch; ++c) {
        out.push_back(tsr_rows_range(c, (uint64_t) base_beat * TSR_BEAT_BYTES, 1, beats * TSR_BEAT_BYTES, 0));
    }
}

void tsr_opl_builder::kind_ranges(uint32_t kind, const uint32_t * a, uint64_t aux, std::vector<tsr_dram_range> & rd,
                                  std::vector<tsr_dram_range> & wr) const {
    rd.clear();
    wr.clear();
    switch (kind) {
        case opl::K_MUL_MAT:
        case opl::K_MUL_MAT_SCALE:
        case opl::K_MUL_MAT_SOFTCAP:
        case opl::K_MUL_MAT_BEGIN:      // the whole MUL_MAT: y is its write (WAR / WAW with the records before it)
            mm_ranges(a, rd, wr);
            return;
        case opl::K_MUL_MAT_END:        // word 0 is an op index; the builder gives its ranges (emit_end)
            return;
        case opl::K_ADD:
        case opl::K_MUL: {
            namespace e = opl::ew;
            const uint32_t L = tsr_f32_lines(a[e::N]), rows = a[e::ROWS], p = std::max<uint32_t>(a[e::B_PERIOD], 1u);
            rd.push_back(tsr_rows_of(a[e::A], rows, a[e::A_STRIDE], L));
            if (a[e::FLAGS] & e::FLAG_B_SCALAR) {
                // the per-row scalars: words 0 .. min(p, rows) - 1 of the f32 vector at b (SW §7.2)
                rd.push_back(tsr_rows_of(a[e::B], 1, 0, tsr_f32_lines(std::min(p, rows))));
            } else if (p == 1u || a[e::B_STRIDE] == 0u) {
                rd.push_back(tsr_rows_of(a[e::B], 1, 0, L));
            } else {
                rd.push_back(tsr_rows_of(a[e::B], std::min(p, rows), a[e::B_STRIDE], L));
            }
            wr.push_back(tsr_rows_of(a[e::DST], rows, a[e::DST_STRIDE], L));
            return;
        }
        case opl::K_SCALE: {
            namespace s = opl::scl;
            const uint32_t L = tsr_f32_lines(a[s::N]);
            rd.push_back(tsr_rows_of(a[s::SRC], a[s::ROWS], a[s::SRC_STRIDE], L));
            wr.push_back(tsr_rows_of(a[s::DST], a[s::ROWS], a[s::DST_STRIDE], L));
            return;
        }
        case opl::K_RMS_NORM: {
            namespace r = opl::rms;
            const uint32_t L = tsr_f32_lines(a[r::N]), rows = a[r::ROWS];
            rd.push_back(tsr_rows_of(a[r::SRC], rows, a[r::SRC_STRIDE], L));
            if (a[r::W] != opl::HANDLE_NONE) {
                // weight rows r mod p, dense (L lines apart, SW §7.2): min(p, rows) of them
                const uint32_t wr_rows = std::min(std::max<uint32_t>(a[r::W_PERIOD], 1u), rows);
                rd.push_back(tsr_rows_of(a[r::W], wr_rows, L, L));
            }
            wr.push_back(tsr_rows_of(a[r::DST], rows, a[r::DST_STRIDE], L));
            return;
        }
        case opl::K_GLU: {
            namespace g = opl::glu;
            const uint32_t L = tsr_f32_lines(a[g::N]), rows = a[g::ROWS];
            rd.push_back(tsr_rows_of(a[g::GATE], rows, a[g::GATE_STRIDE], L));
            rd.push_back(tsr_rows_of(a[g::UP], rows, a[g::UP_STRIDE], L));
            wr.push_back(tsr_rows_of(a[g::DST], rows, a[g::DST_STRIDE], L));
            return;
        }
        case opl::K_ROPE: {
            namespace r = opl::rope;
            const uint32_t H = (uint32_t) (((uint64_t) a[r::N_HEADS] * a[r::HEAD_DIM]) >> 4), T = a[r::N_TOKENS];
            rd.push_back(tsr_rows_of(a[r::SRC], T, a[r::SRC_STRIDE], H));
            if (a[r::TABLE] != opl::HANDLE_NONE) {
                rd.push_back(tsr_rows_of(a[r::TABLE], 1, 0, tsr_f32_lines(a[r::N_DIMS] >> 1)));
            }
            wr.push_back(tsr_rows_of(a[r::DST], T, a[r::DST_STRIDE], H));
            return;
        }
        case opl::K_GET_ROWS: {
            namespace g = opl::getrows;
            const uint32_t rl = a[g::ROW_LINES], n = a[g::N_IDS];
            rd.push_back(tsr_rows_of(a[g::SRC], (uint32_t) std::min<uint64_t>(aux != 0 ? aux : n, UINT32_MAX), a[g::SRC_STRIDE], rl));
            wr.push_back(tsr_rows_of(a[g::DST], n, a[g::DST_STRIDE], rl));
            return;
        }
        case opl::K_SET_ROWS_KV: {
            namespace s = opl::setrows;
            rd.push_back(tsr_rows_of(a[s::SRC], a[s::N_ROWS], a[s::SRC_STRIDE], tsr_f32_lines(a[s::ROW_ELEMS])));
            tsr_plane_ranges(num_ch_, a[s::PLANE], a[s::GEOM], wr);
            return;
        }
        case opl::K_FLASH_ATTN: {
            namespace f = opl::fa;
            const uint32_t T = a[f::N_TOKENS], ql = (uint32_t) (((uint64_t) a[f::N_HEAD] * a[f::D]) >> 4);
            rd.push_back(tsr_rows_of(a[f::Q], T, a[f::Q_TOKEN_STRIDE], ql));
            tsr_plane_ranges(num_ch_, a[f::K_PLANE], a[f::GEOM], rd);
            tsr_plane_ranges(num_ch_, a[f::V_PLANE], a[f::GEOM], rd);
            if (a[f::MASK] != opl::HANDLE_NONE) {
                const uint32_t ml = tsr_f32_lines(a[f::N_KV]);     // the f32 copy: rows of ceil(n_kv/16) lines
                rd.push_back(tsr_rows_of(a[f::MASK], T, ml, ml));
            }
            wr.push_back(tsr_rows_of(a[f::DST], T, a[f::DST_STRIDE], ql));
            return;
        }
        case opl::K_CPY: {
            namespace c = opl::cpy;
            rd.push_back(tsr_rows_of(a[c::SRC], a[c::ROWS], a[c::SRC_STRIDE], a[c::ROW_LINES]));
            wr.push_back(tsr_rows_of(a[c::DST], a[c::ROWS], a[c::DST_STRIDE], a[c::ROW_LINES]));
            return;
        }
        case opl::K_UNARY: {
            namespace u = opl::unary;
            const uint32_t L = tsr_f32_lines(a[u::N]);
            rd.push_back(tsr_rows_of(a[u::SRC], a[u::ROWS], a[u::SRC_STRIDE], L));
            wr.push_back(tsr_rows_of(a[u::DST], a[u::ROWS], a[u::DST_STRIDE], L));
            return;
        }
        case opl::K_MOE_ROUTE: {
            // reads the logits rows; writes the first line of each ids row and the packed weights (SW §7.5)
            namespace r = opl::moer;
            const uint32_t T = a[r::T], k = a[r::N_USED];
            rd.push_back(tsr_rows_of(a[r::LOGITS], T, a[r::LOGITS_STRIDE], tsr_f32_lines(a[r::N_EXPERT])));
            wr.push_back(tsr_rows_of(a[r::IDS], T, a[r::IDS_STRIDE], 1u));
            wr.push_back(tsr_rows_of(a[r::W], 1, 0, tsr_f32_lines((uint32_t) std::min<uint64_t>((uint64_t) T * k, UINT32_MAX))));
            return;
        }
        case opl::K_MUL_MAT_ID: {
            // reads x (per token the span of its slot rows: a superset when the slots are not dense), the ids
            // (N_USED words per token, rows IDS_STRIDE bytes apart), the scale table and every expert's rows on
            // every channel; writes the T * N_USED dst rows (SW §7.5)
            namespace m = opl::mmid;
            const uint32_t T = a[m::T], nu = a[m::N_USED], fmt = a[m::FMT], N = a[m::N], K = a[m::K];
            const uint32_t xl = (uint32_t) tsr_ceil_div_u64((uint64_t) K * 4, TSR_LINE_BYTES);
            const uint32_t yl = (uint32_t) tsr_ceil_div_u64((uint64_t) N * 4, TSR_LINE_BYTES);
            const uint32_t span = a[m::X_SLOT_STRIDE] == 0u ? xl : (nu - 1u) * a[m::X_SLOT_STRIDE] + xl;
            rd.push_back(tsr_rows_of(a[m::X], T, a[m::X_TOK_STRIDE], span));
            rd.push_back(tsr_rows_range(tsr_handle_ch(a[m::IDS]), tsr_handle_byte(a[m::IDS]), T, (uint64_t) 4 * nu, a[m::IDS_STRIDE]));
            if (a[m::SCALE] != opl::HANDLE_NONE) {
                rd.push_back(tsr_rows_of(a[m::SCALE], 1, 0, tsr_f32_lines(a[m::N_EXPERT])));
            }
            if (tsr::plan::fmt_valid(fmt) && K > 0 && K % tsr::plan::fmt_block(fmt) == 0 && N > 0 && num_ch_ > 0) {
                const uint64_t lo = (uint64_t) a[m::W_OFF] * TSR_BEAT_BYTES;
                const uint64_t bytes = (uint64_t) a[m::EXP_STRIDE] * TSR_BEAT_BYTES * a[m::N_EXPERT];
                for (uint32_t c = 0; c < num_ch_; ++c) {
                    rd.push_back(tsr_rows_range(c, lo, 1, bytes, 0));
                }
            }
            wr.push_back(tsr_rows_of(a[m::DST], (uint32_t) std::min<uint64_t>((uint64_t) T * nu, UINT32_MAX), a[m::DST_STRIDE], yl));
            return;
        }
        case opl::K_SOFTPLUS: {
            namespace u = opl::softplus;
            const uint32_t L = tsr_f32_lines(a[u::N]);
            rd.push_back(tsr_rows_of(a[u::SRC], a[u::ROWS], a[u::SRC_STRIDE], L));
            wr.push_back(tsr_rows_of(a[u::DST], a[u::ROWS], a[u::DST_STRIDE], L));
            return;
        }
        case opl::K_SSM_CONV: {
            namespace c = opl::sconv;
            const uint32_t Lc = a[c::N_CH] >> 4, dc = a[c::D_CONV], nt = a[c::N_TOKENS] * a[c::N_SEQS],
                           ns = a[c::N_SEQS];
            rd.push_back(tsr_rows_of(a[c::X], nt, a[c::X_TOKEN_STRIDE], Lc));
            rd.push_back(tsr_rows_of(a[c::STATE_IN], ns * (dc - 1u), Lc, Lc));
            rd.push_back(tsr_rows_of(a[c::W], dc, Lc, Lc));
            wr.push_back(tsr_rows_of(a[c::DST], nt, a[c::DST_TOKEN_STRIDE], Lc));
            wr.push_back(tsr_rows_of(a[c::STATE_OUT], ns * (dc - 1u), Lc, Lc));
            return;
        }
        case opl::K_GATED_DELTA_NET: {
            namespace g = opl::gdn;
            const uint32_t sv = a[g::S_V], L = sv >> 4, hk = a[g::H_K], hv = a[g::H_V], nt = a[g::N_TOKENS] * a[g::N_SEQS],
                           Lg = tsr_f32_lines(hv);
            // per token row: the lines from head 0's first to the last head's end (a superset when the heads are
            // not adjacent)
            rd.push_back(tsr_rows_of(a[g::Q], nt, a[g::Q_TOKEN_STRIDE], (hk - 1u) * a[g::Q_HEAD_STRIDE] + L));
            rd.push_back(tsr_rows_of(a[g::K], nt, a[g::K_TOKEN_STRIDE], (hk - 1u) * a[g::K_HEAD_STRIDE] + L));
            rd.push_back(tsr_rows_of(a[g::V], nt, a[g::V_TOKEN_STRIDE], (hv - 1u) * a[g::V_HEAD_STRIDE] + L));
            rd.push_back(tsr_rows_of(a[g::G], nt, a[g::G_TOKEN_STRIDE], Lg));
            rd.push_back(tsr_rows_of(a[g::BETA], nt, a[g::BETA_TOKEN_STRIDE], Lg));
            const uint32_t srows = a[g::N_SEQS] * hv * sv;
            rd.push_back(tsr_rows_of(a[g::STATE_IN], srows, L, L));
            wr.push_back(tsr_rows_of(a[g::DST], nt * hv, L, L));
            wr.push_back(tsr_rows_of(a[g::STATE_OUT], srows, L, L));
            return;
        }
        default:
            return;
    }
}

uint32_t tsr_opl_builder::preload_lines_with(const std::vector<std::pair<uint32_t, uint32_t>> & entries) const {
    uint64_t end = (uint64_t) w_.pre_lines() * opl::PRE_ALIGN;     // entries start 64-aligned (the writer)
    std::vector<std::pair<uint32_t, uint32_t>> seen;
    for (const auto & e : entries) {
        bool have = false;
        for (const auto & p : pre_) have = have || p.first == e;
        for (const auto & p : seen) have = have || p == e;
        if (have || e.second == 0) {
            continue;
        }
        seen.push_back(e);
        end = (end + opl::PRE_ALIGN - 1) / opl::PRE_ALIGN * opl::PRE_ALIGN + e.second;
    }
    return (uint32_t) ((end + opl::SLOT_BYTES - 1) / opl::SLOT_BYTES);
}

uint32_t tsr_opl_builder::preload(uint32_t src, uint32_t bytes) {
    if (bytes == 0) {
        return FAILED;
    }
    for (const auto & p : pre_) {
        if (p.first.first == src && p.first.second == bytes) {
            return p.second;
        }
    }
    if (preload_lines_with({ { src, bytes } }) > opl::PRE_MAX_LINES) {
        return NO_ROOM;
    }
    const uint32_t at = w_.add_preload(src, bytes);
    if (at == opl::HANDLE_NONE) {
        return FAILED;
    }
    pre_.push_back({ { src, bytes }, at });
    return at;
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
    if (kind < opl::N_KINDS) {
        n_kind_[kind]++;
    }
    return idx;
}

// The conflicting earlier ops (RAW, WAR, WAW) of a record with ranges rd, wr, and pred, reduced transitively: a
// conflicting op that is an ancestor of a later kept one is implied.
void tsr_opl_builder::deps_of(const ranges & rd, const ranges & wr, int32_t pred, std::vector<uint32_t> & kept,
                              uint32_t & dropped) const {
    std::vector<uint32_t> cand;       // latest first
    for (uint32_t i = (uint32_t) ops_.size(); i-- > 0;) {
        const op_info & o = ops_[i];
        bool hit = pred >= 0 && (uint32_t) pred == i;
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
    kept.clear();
    dropped = 0;
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
}

// Adds the record with the dependencies `kept`: JOIN records for more than two, folded as tsr::opl::writer::add does
// (SW §7.2): the first two go into a JOIN, whose index replaces them at the end of the list, until two remain. NO_ROOM
// (nothing added) unless the record, its JOINs and `reserve` further records (the ENDs of the open BEGINs still owed)
// fit the list.
uint32_t tsr_opl_builder::emit(uint32_t kind, const uint32_t * args, uint32_t n_args, const ranges & rd, const ranges & wr,
                               int32_t tag, const std::vector<uint32_t> & kept, uint32_t dropped, uint32_t reserve) {
    const uint32_t joins = kept.size() > 2 ? (uint32_t) kept.size() - 2 : 0;
    if (ops_.size() + joins + 1 + reserve > opl::MAX_OPS) {
        return NO_ROOM;
    }
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

static bool tsr_hits_any(const tsr_dram_range & y, const std::vector<tsr_dram_range> & v) {
    for (const tsr_dram_range & r : v) {
        if (tsr_ranges_overlap(y, r)) return true;
    }
    return false;
}

// The MUL_MAT_END of open_[i]: no ranges read, y written, the BEGIN as explicit predecessor. The records between the
// BEGIN and now do not touch y (resolve), so the BEGIN is its only dependency.
uint32_t tsr_opl_builder::emit_end(size_t i) {
    const open_begin ob = open_[i];
    const uint32_t a[opl::mmend::NARGS] = { ob.idx };
    const ranges wr = { ob.y };
    std::vector<uint32_t> kept;
    uint32_t dropped = 0;
    deps_of({}, wr, (int32_t) ob.idx, kept, dropped);
    const uint32_t r = emit(opl::K_MUL_MAT_END, a, opl::mmend::NARGS, {}, wr, ob.tag, kept, dropped, (uint32_t) open_.size() - 1u);
    if (r == NO_ROOM || r == FAILED) {
        return r;
    }
    open_.erase(open_.begin() + (std::ptrdiff_t) i);
    return 0;
}

// The pending MUL_MAT becomes a MUL_MAT_BEGIN (may_begin: the next record leaves y alone, and the list has room for
// its END), else a MUL_MAT. With MAX_OPEN BEGINs open the oldest gets its END first.
uint32_t tsr_opl_builder::emit_pending(bool may_begin) {
    if (may_begin && open_.size() >= MAX_OPEN) {
        const uint32_t r = emit_end(0);
        if (r != 0) return r;
    }
    pending_mm p = std::move(pend_);
    pend_ = pending_mm();
    const uint32_t joins = p.kept.size() > 2 ? (uint32_t) p.kept.size() - 2 : 0;
    const bool begin = may_begin && ops_.size() + joins + 1 + open_.size() + 1 <= opl::MAX_OPS;
    const uint32_t idx = emit(begin ? opl::K_MUL_MAT_BEGIN : opl::K_MUL_MAT, p.args, opl::mm::NARGS, p.rd, p.wr, p.tag, p.kept,
                              p.dropped, (uint32_t) open_.size() + (begin ? 1u : 0u));
    if (idx == NO_ROOM || idx == FAILED) {
        pend_ = std::move(p);
        return idx;
    }
    if (begin) {
        open_.push_back({ idx, p.wr[0], p.tag });
    }
    return 0;
}

// What a record with ranges rd, wr needs first: the pending MUL_MAT is emitted (a MUL_MAT if the record touches its y,
// else a BEGIN), and every open BEGIN whose y the record touches gets its END. 0, NO_ROOM or FAILED.
uint32_t tsr_opl_builder::resolve(const ranges & rd, const ranges & wr) {
    if (pend_.valid) {
        const bool touch = tsr_hits_any(pend_.wr[0], rd) || tsr_hits_any(pend_.wr[0], wr);
        const uint32_t r = emit_pending(!touch);
        if (r != 0) return r;
    }
    for (size_t i = 0; i < open_.size();) {
        if (tsr_hits_any(open_[i].y, rd) || tsr_hits_any(open_[i].y, wr)) {
            const uint32_t r = emit_end(i);
            if (r != 0) return r;
        } else {
            ++i;
        }
    }
    return 0;
}

bool tsr_opl_builder::settle() {
    if (!w_.ok()) {
        return false;
    }
    if (pend_.valid && emit_pending(false) != 0) {
        return false;
    }
    while (!open_.empty()) {
        if (emit_end(0) != 0) {
            return false;
        }
    }
    return true;
}

uint32_t tsr_opl_builder::add(uint32_t kind, const uint32_t * args, uint32_t n_args, const std::vector<tsr_dram_range> & rd,
                              const std::vector<tsr_dram_range> & wr, int32_t tag, int32_t pred) {
    if (!w_.ok()) {
        return FAILED;
    }
    if (pend_.valid || !open_.empty()) {
        const uint32_t r = resolve(rd, wr);
        if (r != 0) return r;
    }
    std::vector<uint32_t> kept;
    uint32_t dropped = 0;
    deps_of(rd, wr, pred, kept, dropped);
    return emit(kind, args, n_args, rd, wr, tag, kept, dropped, (uint32_t) open_.size());
}

uint32_t tsr_opl_builder::mul_mat(const uint32_t * args, int32_t tag, uint32_t kind, bool defer) {
    std::vector<tsr_dram_range> rd, wr;
    mm_ranges(args, rd, wr);
    if (!defer || !defer_ || kind != opl::K_MUL_MAT) {
        return add(kind, args, opl::kind_nargs(kind), rd, wr, tag);
    }
    if (!w_.ok()) {
        return FAILED;
    }
    if (pend_.valid || !open_.empty()) {
        const uint32_t r = resolve(rd, wr);
        if (r != 0) return r;
    }
    pending_mm p;
    deps_of(rd, wr, -1, p.kept, p.dropped);
    const uint32_t joins = p.kept.size() > 2 ? (uint32_t) p.kept.size() - 2 : 0;
    if (ops_.size() + joins + 1 + open_.size() > opl::MAX_OPS) {     // as a MUL_MAT, with the ENDs of the open BEGINs
        return NO_ROOM;
    }
    p.valid = true;
    std::copy(args, args + opl::mm::NARGS, p.args);
    p.tag = tag;
    p.rd = std::move(rd);
    p.wr = std::move(wr);
    pend_ = std::move(p);
    return PENDING;
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
    if (pend_.valid) {                   // the pending MUL_MAT is part of the list
        if (tsr_hits_any(r, pend_.wr) || (any_access && tsr_hits_any(r, pend_.rd))) return true;
    }
    return false;
}

std::vector<uint8_t> tsr_opl_builder::finish() const {
    if (pend_.valid) {
        return {};                       // a pending MUL_MAT would be lost: settle() first
    }
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
