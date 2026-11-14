// tsr_rt_graph.cpp — the graph API of tsr::rt::device (see tsr_rt.h "Graphs").
//
// A graph is a straight-line program over f32 [T, n] values. For each T (and mask shape) the program is planned once:
//   - memory: the positions, the KV cell indices and the mask summaries ("static" part) sit at the bottom of the IO region
//     of channel 0 and are never written by a record (they are preload sources); every in, tmp and out value is placed above
//     them by liveness with tsr_range_alloc; constants live in AUX; the KV planes of each attn op (K and V) at the top of the
//     WEIGHT heap;
//   - records: each op becomes the records of the framework-free encoders (tsr-kinds.h) in program order, added to a
//     tsr_opl_builder (hazards, JOINs, preload table, MUL_MAT deferral); a list that is full is cut and the next one started;
//   - the plan keeps the finished lists (blobs) and the byte ranges the host reference needs from the device.
// A run writes the inputs, positions, cell indices and mask summaries, launches the lists and reads the out values.
// graph_run_ref executes the same blobs on a host image with tsr::ref::opl_exec.
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <list>
#include <set>
#include <thread>
#include <tuple>

#include "opl_ref.hpp"
#include "tsr_rt.h"
#include "tsr_rt_internal.h"

namespace tsr::rt {

namespace opl = tsr::opl;

const char * gop_name(gop op) {
    switch (op) {
        case gop::mul_mat:  return "mul_mat";
        case gop::add:      return "add";
        case gop::mul:      return "mul";
        case gop::scale:    return "scale";
        case gop::rms_norm: return "rms_norm";
        case gop::glu:      return "glu";
        case gop::unary:    return "unary";
        case gop::rope:     return "rope";
        case gop::attn:     return "attn";
        case gop::cpy:      return "cpy";
    }
    return "?";
}

const char * record_kind_name(uint32_t kind) { return kind < opl::N_KINDS ? opl::kind_name(kind) : "?"; }

namespace {

constexpr uint64_t LINE = TSR_LINE_BYTES;
constexpr uint32_t MAX_VALUES = 1u << 20;
constexpr uint64_t XFER_CHUNK = 8u << 20;   // bytes per transport call when a large range is written or read

uint64_t pad64(uint64_t b) { return (b + 63u) / 64u * 64u; }
uint32_t L(uint64_t n) { return tsr_k_lines(n); }
uint32_t H0(uint64_t byte_off) { return tsr_dram_handle(0, byte_off); }
uint32_t bits_of(float f) { return tsr_f32_bits(f); }

bool mm_defer_enabled() {
    static const bool on = [] {
        const char * e = getenv("TSR_MM_DEFER");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// The slot of one mask summary: the summary depends on the mask, the shape of the attention and its scale.
struct mask_key {
    int32_t mask;                 // the mask value, -1: causal
    uint32_t E, D, G, scale;      // scale: float bits
    bool operator<(const mask_key & o) const {
        return std::tie(mask, E, D, G, scale) < std::tie(o.mask, o.E, o.D, o.G, o.scale);
    }
};

struct dev_range {                // bytes the host reference reads from the device
    uint32_t ch;
    uint64_t off, bytes;
};

struct list_t {
    std::vector<uint8_t> blob;
    std::vector<int32_t> tags;    // op index of each record, -1 for a JOIN
    uint32_t n_ops = 0, n_joins = 0;
};

// The part of the IO layout that does not depend on the plan: positions, KV cell indices, mask summaries and f32 mask copies.
struct slot_t {
    mask_key key;
    uint64_t msum_off = 0, f32_off = 0;
};
struct layout_t {
    std::vector<uint64_t> pos_off;      // per value (only pos values)
    uint64_t idx_off = 0;
    std::vector<slot_t> slots;
    uint64_t end = 0;                   // first byte above, ch0 byte offset
};

struct plan_t {
    uint32_t T = 0;
    std::vector<uint8_t> sig;           // per mask slot: the summary needs the f32 copy
    std::vector<uint64_t> off;          // per value: ch0 byte offset (in, tmp; pos: its array; konst: AUX)
    std::vector<list_t> lists;
    std::vector<dev_range> reads;       // the host reference reads these from the device (constants, weights)
    graph_plan_info info;
};

}  // namespace

struct graph_obj {
    graph_desc desc;
    bool probe = false;                 // a probe: planes are not allocated (fake offsets)
    std::vector<int32_t> last_use;      // per value: last op that reads it, -1 none
    std::vector<uint32_t> attn_ops;     // op indices of the attn ops
    std::vector<int32_t> attn_slot;     // per op: mask slot of an attn op
    std::vector<mask_key> slot_keys;
    std::vector<tsr_k_kv> kv;           // per attn op (index in attn_ops)
    std::vector<uint64_t> plane_k, plane_v;   // per attn op: byte offset of its K and of its V plane (the same in every channel)
    std::vector<int> arg_value;         // per argument: the value id
    std::vector<uint32_t> out_value;    // per out: the value id
    uint64_t kv_bytes = 0;              // per channel, all planes
    std::list<std::shared_ptr<plan_t>> plans;   // LRU, front = most recent
    std::map<std::pair<uint32_t, uint32_t>, std::shared_ptr<tsr_k_fa_mask>> causal;   // (T, slot)
};

namespace {

std::string op_name(const graph_desc & g, uint32_t i) {
    return strf("op #%u (tag %d, %s)", i, g.ops[i].tag, gop_name(g.ops[i].op));
}

[[noreturn]] void bad_op(const graph_desc & g, uint32_t i, const char * what) {
    throw std::invalid_argument(strf("tsr_rt: graph: %s: %s", op_name(g, i).c_str(), what));
}

// ---------------------------------------------------------------------------------------------
// Static checks
// ---------------------------------------------------------------------------------------------
void validate(const graph_desc & g, bool need_handles, const device::impl & d) {
    const size_t NV = g.values.size();
    if (NV == 0 || NV > MAX_VALUES) throw std::invalid_argument("tsr_rt: graph: no values or too many");
    if (g.ops.empty()) throw std::invalid_argument("tsr_rt: graph: no ops");
    bool any_out = false;
    for (size_t v = 0; v < NV; ++v) {
        const graph_value & x = g.values[v];
        if ((x.kind == gv::in || x.kind == gv::tmp || x.kind == gv::konst) && (x.n == 0 || x.n > (1u << 24))) {
            throw std::invalid_argument(strf("tsr_rt: graph: value %zu: n = %u out of range", v, x.n));
        }
        if (x.kind == gv::konst) {
            if (x.rows == 0) throw std::invalid_argument(strf("tsr_rt: graph: const value %zu has no rows", v));
            if (need_handles) {
                auto it = d.consts.find(x.handle);
                if (it == d.consts.end()) throw std::invalid_argument(strf("tsr_rt: graph: const value %zu: unknown const handle %" PRIu64, v, x.handle));
                if (it->second.rows != x.rows || it->second.n != x.n) {
                    throw std::invalid_argument(strf("tsr_rt: graph: const value %zu is [%u x %u], the uploaded const [%u x %u]", v, x.rows, x.n,
                                                     it->second.rows, it->second.n));
                }
            }
        }
        if (x.out && x.kind != gv::tmp) throw std::invalid_argument(strf("tsr_rt: graph: value %zu: only a tmp value can be an out", v));
        any_out = any_out || x.out;
    }
    if (!any_out) throw std::invalid_argument("tsr_rt: graph: no out value");
    if (g.kv_size == 0 && std::any_of(g.ops.begin(), g.ops.end(), [](const graph_op & o) { return o.op == gop::attn; })) {
        throw std::invalid_argument("tsr_rt: graph: an attn op needs kv_size >= 1");
    }

    std::vector<uint8_t> defined(NV, 0), written(NV, 0);
    for (size_t v = 0; v < NV; ++v) defined[v] = g.values[v].kind != gv::tmp;
    for (uint32_t i = 0; i < g.ops.size(); ++i) {
        const graph_op & o = g.ops[i];
        auto vid = [&](int32_t id, const char * what) -> const graph_value & {
            if (id < 0 || (size_t) id >= NV) bad_op(g, i, strf("%s: value id %d out of range", what, id).c_str());
            return g.values[(size_t) id];
        };
        // an activation operand: in, or a tmp written by an earlier op; n values per row
        auto act = [&](int32_t id, const char * what, uint32_t n) {
            const graph_value & x = vid(id, what);
            if (x.kind != gv::in && x.kind != gv::tmp) bad_op(g, i, strf("%s must be an in or tmp value", what).c_str());
            if (!defined[(size_t) id]) bad_op(g, i, strf("%s is read before it is written", what).c_str());
            if (n != 0 && x.n != n) bad_op(g, i, strf("%s has %u values per row, %u expected", what, x.n, n).c_str());
        };
        auto cst = [&](int32_t id, const char * what, uint32_t n) {
            const graph_value & x = vid(id, what);
            if (x.kind != gv::konst || x.rows != 1) bad_op(g, i, strf("%s must be a one-row const", what).c_str());
            if (x.n != n) bad_op(g, i, strf("%s has %u values, %u expected", what, x.n, n).c_str());
        };
        auto dst = [&](int32_t id, uint32_t n) {
            const graph_value & x = vid(id, "dst");
            if (x.kind != gv::tmp) bad_op(g, i, "dst must be a tmp value");
            if (written[(size_t) id]) bad_op(g, i, "dst is written by an earlier op");
            if (x.n != n) bad_op(g, i, strf("dst has %u values per row, %u expected", x.n, n).c_str());
        };
        auto distinct = [&](std::initializer_list<int32_t> ids) {
            for (int32_t a : ids) if (a == o.dst) bad_op(g, i, "dst is also an operand");
        };
        switch (o.op) {
            case gop::mul_mat: {
                uint32_t N = 0, K = 0;
                if (o.w != 0) {
                    auto it = d.weights.find(o.w);
                    if (it == d.weights.end()) bad_op(g, i, strf("unknown weight handle %" PRIu64, o.w).c_str());
                    N = it->second.N;
                    K = it->second.K;
                } else if (need_handles) {
                    bad_op(g, i, "weight handle 0");
                } else {
                    N = vid(o.dst, "dst").n;
                    K = vid(o.x, "x").n;
                }
                act(o.x, "x", K);
                dst(o.dst, N);
                distinct({o.x, o.addend});
                if (o.addend >= 0) {
                    const graph_value & a = vid(o.addend, "addend");
                    if (a.kind == gv::konst) cst(o.addend, "addend", N);
                    else act(o.addend, "addend", N);
                }
                break;
            }
            case gop::add:
            case gop::mul: {
                const graph_value & a = vid(o.a, "a");
                act(o.a, "a", 0);
                const graph_value & b = vid(o.b, "b");
                if (b.kind == gv::konst) cst(o.b, "b", a.n);
                else act(o.b, "b", a.n);
                dst(o.dst, a.n);
                distinct({o.a, o.b});
                break;
            }
            case gop::scale:
            case gop::cpy: {
                const graph_value & x = vid(o.x, "x");
                act(o.x, "x", 0);
                dst(o.dst, x.n);
                distinct({o.x});
                break;
            }
            case gop::rms_norm: {
                const graph_value & x = vid(o.x, "x");
                act(o.x, "x", 0);
                if (o.weight >= 0) cst(o.weight, "weight", x.n);
                dst(o.dst, x.n);
                distinct({o.x});
                if (!(o.eps >= 0.0f)) bad_op(g, i, "eps must be >= 0");
                break;
            }
            case gop::glu: {
                const graph_value & x = vid(o.gate, "gate");
                act(o.gate, "gate", 0);
                act(o.up, "up", x.n);
                dst(o.dst, x.n);
                distinct({o.gate, o.up});
                if (o.variant != opl::glu::SWIGLU && o.variant != opl::glu::GEGLU) bad_op(g, i, "variant must be 0 (SWIGLU) or 2 (GEGLU)");
                break;
            }
            case gop::unary: {
                const graph_value & x = vid(o.x, "x");
                act(o.x, "x", 0);
                dst(o.dst, x.n);
                distinct({o.x});
                if (o.func != opl::unary::TANH && o.func != opl::unary::SILU && o.func != opl::unary::EXP &&
                    o.func != opl::unary::GELU && o.func != opl::unary::SIGMOID) {
                    bad_op(g, i, "func must be 0 (TANH), 2 (SILU), 3 (EXP), 4 (GELU) or 5 (SIGMOID)");
                }
                break;
            }
            case gop::rope: {
                if (o.n_heads == 0 || o.head_dim == 0 || o.n_dims == 0 || o.n_dims > o.head_dim || o.n_dims % 2 != 0) {
                    bad_op(g, i, "n_heads, head_dim, n_dims: 0 < n_dims <= head_dim, n_dims even");
                }
                const uint64_t n = (uint64_t) o.n_heads * o.head_dim;
                if (n > (1u << 24)) bad_op(g, i, "n_heads * head_dim out of range");
                act(o.x, "x", (uint32_t) n);
                dst(o.dst, (uint32_t) n);
                distinct({o.x});
                cst(o.table, "table", o.n_dims / 2);
                if (vid(o.pos, "pos").kind != gv::pos) bad_op(g, i, "pos must be a pos value");
                break;
            }
            case gop::attn: {
                if (o.d == 0 || o.n_head == 0 || o.n_head_kv == 0 || o.n_head % o.n_head_kv != 0) {
                    bad_op(g, i, "d, n_head, n_head_kv: n_head must be a multiple of n_head_kv");
                }
                const uint64_t nq = (uint64_t) o.n_head * o.d, nk = (uint64_t) o.n_head_kv * o.d;
                if (nq > (1u << 24)) bad_op(g, i, "n_head * d out of range");
                act(o.q, "q", (uint32_t) nq);
                act(o.k, "k", (uint32_t) nk);
                act(o.v, "v", (uint32_t) nk);
                dst(o.dst, (uint32_t) nq);
                distinct({o.q, o.k, o.v});
                if (o.mask >= 0 && vid(o.mask, "mask").kind != gv::mask) bad_op(g, i, "mask must be a mask value or -1");
                if (!(o.scale > 0.0f) || !std::isfinite(o.scale)) bad_op(g, i, "scale must be finite and positive");
                break;
            }
        }
        written[(size_t) o.dst] = 1;
        defined[(size_t) o.dst] = 1;
    }
    for (size_t v = 0; v < NV; ++v) {
        if (g.values[v].out && !written[v]) throw std::invalid_argument(strf("tsr_rt: graph: out value %zu is never written", v));
    }
}

// The activation values an op reads (not consts, pos, mask).
void operands(const graph_op & o, std::vector<int32_t> & v) {
    v.clear();
    switch (o.op) {
        case gop::mul_mat:  v = { o.x, o.addend }; break;
        case gop::add:
        case gop::mul:      v = { o.a, o.b }; break;
        case gop::scale:
        case gop::rms_norm:
        case gop::unary:
        case gop::rope:
        case gop::cpy:      v = { o.x }; break;
        case gop::glu:      v = { o.gate, o.up }; break;
        case gop::attn:     v = { o.q, o.k, o.v }; break;
    }
}

// ---------------------------------------------------------------------------------------------
// Graph objects
// ---------------------------------------------------------------------------------------------
std::shared_ptr<graph_obj> make_obj(const device::impl & d, const graph_desc & g, bool probe) {
    auto o = std::make_shared<graph_obj>();
    o->desc = g;
    o->probe = probe;
    const size_t NV = g.values.size();
    o->last_use.assign(NV, -1);
    std::vector<int32_t> ops;
    for (uint32_t i = 0; i < g.ops.size(); ++i) {
        operands(g.ops[i], ops);
        for (int32_t v : ops) if (v >= 0) o->last_use[(size_t) v] = (int32_t) i;
    }
    for (size_t v = 0; v < NV; ++v) {
        if (g.values[v].kind == gv::in || g.values[v].kind == gv::pos || g.values[v].kind == gv::mask) o->arg_value.push_back((int) v);
        if (g.values[v].kind == gv::tmp && g.values[v].out) o->out_value.push_back((uint32_t) v);
    }
    o->attn_slot.assign(g.ops.size(), -1);
    std::map<mask_key, int32_t> slot_of;
    for (uint32_t i = 0; i < g.ops.size(); ++i) {
        const graph_op & op = g.ops[i];
        if (op.op != gop::attn) continue;
        const uint32_t E = op.n_head_kv * op.d;
        std::string why;
        tsr_k_kv kv;
        if (!tsr_k_kv_geom(E, g.kv_size, d.cfg.num_ch, &kv, &why)) {
            throw error(strf("tsr_rt: graph: %s: KV planes: %s", op_name(g, i).c_str(), why.c_str()));
        }
        o->attn_ops.push_back(i);
        o->kv.push_back(kv);
        o->kv_bytes += 2 * kv.bytes;
        const mask_key key{ op.mask, E, op.d, op.n_head / op.n_head_kv, bits_of(op.scale) };
        auto it = slot_of.find(key);
        if (it == slot_of.end()) {
            it = slot_of.emplace(key, (int32_t) o->slot_keys.size()).first;
            o->slot_keys.push_back(key);
        }
        o->attn_slot[i] = it->second;
    }
    o->plane_k.assign(o->attn_ops.size(), 0);
    o->plane_v.assign(o->attn_ops.size(), 0);
    if (probe) {
        for (size_t a = 0; a < o->plane_k.size(); ++a) {
            o->plane_k[a] = TSR_REGION_WEIGHT_OFF;
            o->plane_v[a] = TSR_REGION_WEIGHT_OFF + (o->kv[a].bytes + 4095u) / 4096u * 4096u;
        }
    }
    return o;
}

// ---------------------------------------------------------------------------------------------
// Static IO layout
// ---------------------------------------------------------------------------------------------
layout_t static_layout(const device::impl & d, const graph_obj & g, uint32_t T) {
    layout_t l;
    uint64_t at = d.io_off;
    const graph_desc & desc = g.desc;
    l.pos_off.assign(desc.values.size(), 0);
    for (size_t v = 0; v < desc.values.size(); ++v) {
        if (desc.values[v].kind != gv::pos) continue;
        l.pos_off[v] = at;
        at += pad64((uint64_t) T * 4);
    }
    if (!g.attn_ops.empty()) {
        l.idx_off = at;
        at += pad64((uint64_t) T * opl::PRE_KVIDX_BYTES);
    }
    for (const mask_key & k : g.slot_keys) {
        slot_t s;
        s.key = k;
        s.msum_off = at;
        at += pad64(opl::msum_bytes(T));
        s.f32_off = at;
        at += (uint64_t) T * L(T) * LINE;
        l.slots.push_back(s);
    }
    l.end = at;
    return l;
}

[[noreturn]] void no_fit(const char * fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
[[noreturn]] void no_fit(const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    throw error(std::string("tsr_rt: graph: ") + buf);
}

void check_T(const device::impl & d, const graph_obj & g, uint32_t T) {
    if (T == 0) throw std::invalid_argument("tsr_rt: graph: T must be >= 1");
    if (!g.attn_ops.empty()) {
        if (T > g.desc.kv_size) no_fit("T = %u does not fit the KV planes (kv_size %u)", T, g.desc.kv_size);
        if (T > graph_max_attn_tokens) {
            no_fit("T = %u does not fit an attn op (its mask summary and cell indices take the 256-line preload area: T <= %u)", T,
                   graph_max_attn_tokens);
        }
    }
    (void) d;
}

// ---------------------------------------------------------------------------------------------
// Masks
// ---------------------------------------------------------------------------------------------
// The summaries of every slot for T. data: the run's arguments (null for a probe or graph_info: a user mask is not built).
std::vector<std::shared_ptr<tsr_k_fa_mask>> build_masks(device::impl & d, graph_obj & g, uint32_t T, const layout_t & lay,
                                                        const graph_inputs * in) {
    std::vector<std::shared_ptr<tsr_k_fa_mask>> res(g.slot_keys.size());
    std::map<int32_t, std::vector<uint16_t>> m16;   // per mask value
    for (size_t s = 0; s < g.slot_keys.size(); ++s) {
        const mask_key & k = g.slot_keys[s];
        const uint32_t f32_h = H0(lay.slots[s].f32_off);
        std::string why;
        // the op that owns the slot, for the error text
        uint32_t op_i = 0;
        for (uint32_t i = 0; i < g.desc.ops.size(); ++i) {
            if (g.attn_slot[i] == (int32_t) s) { op_i = i; break; }
        }
        const float scale = [&] { float f; memcpy(&f, &k.scale, 4); return f; }();
        if (k.mask < 0) {
            auto it = g.causal.find({ T, (uint32_t) s });
            if (it == g.causal.end()) {
                auto m = std::make_shared<tsr_k_fa_mask>();
                if (!tsr_k_fa_mask_build(d.kenv, k.E, k.D, k.G, scale, T, T, f32_h, nullptr, 0, 0, m.get(), &why)) {
                    throw error(strf("tsr_rt: graph: %s: causal mask: %s", op_name(g.desc, op_i).c_str(), why.c_str()));
                }
                if (g.causal.size() >= 64) g.causal.clear();
                it = g.causal.emplace(std::make_pair(T, (uint32_t) s), std::move(m)).first;
            }
            res[s] = it->second;
            continue;
        }
        if (in == nullptr) continue;
        auto it = m16.find(k.mask);
        if (it == m16.end()) {
            // the argument of this mask value
            size_t ai = 0;
            while (g.arg_value[ai] != k.mask) ++ai;
            const graph_arg & a = in->args[ai];
            const float * mp = (const float *) a.data;
            const size_t st = a.stride != 0 ? a.stride : T;
            std::vector<uint16_t> v((size_t) T * T);
            for (uint32_t q = 0; q < T; ++q) {
                for (uint32_t j = 0; j < T; ++j) {
                    const float x = mp[q * st + j];
                    if (x == 0.0f) v[(size_t) q * T + j] = 0x0000u;
                    else if (std::isinf(x) && x < 0.0f) v[(size_t) q * T + j] = 0xFC00u;
                    else {
                        no_fit("%s: mask value %g at [%u, %u] does not fit the device's mask form (every value must be 0 or -inf)",
                               op_name(g.desc, op_i).c_str(), (double) x, q, j);
                    }
                }
            }
            it = m16.emplace(k.mask, std::move(v)).first;
        }
        auto m = std::make_shared<tsr_k_fa_mask>();
        if (!tsr_k_fa_mask_build(d.kenv, k.E, k.D, k.G, scale, T, T, f32_h, it->second.data(), T, 0, m.get(), &why)) {
            throw error(strf("tsr_rt: graph: %s: mask: %s", op_name(g.desc, op_i).c_str(), why.c_str()));
        }
        res[s] = std::move(m);
    }
    return res;
}

// ---------------------------------------------------------------------------------------------
// The planner
// ---------------------------------------------------------------------------------------------
struct planner {
    device::impl & d;
    graph_obj & g;
    const graph_desc & desc;
    const uint32_t T;
    const layout_t & lay;
    plan_t & pl;

    tsr_range_alloc dyn;
    uint64_t hw_end = 0;                       // highest byte used above io_off
    tsr_opl_builder b;
    tsr_fanew_map fanew;
    uint32_t cur_op = 0;
    std::vector<uint8_t> have_off;             // per value: placed
    std::set<const_handle> read_consts;
    std::set<weight_handle> read_weights;

    planner(device::impl & dd, graph_obj & gg, uint32_t TT, const layout_t & ll, plan_t & pp)
        : d(dd), g(gg), desc(gg.desc), T(TT), lay(ll), pl(pp), b(dd.cfg.num_ch, opl::F_STOP_ON_ERROR) {
        hw_end = lay.end;
        dyn.init(lay.end, d.io_size - (lay.end - d.io_off), LINE);
        pl.off.assign(desc.values.size(), 0);
        have_off.assign(desc.values.size(), 0);
    }

    [[noreturn]] void fail(const char * what) { throw error(strf("tsr_rt: graph: %s: %s", op_name(desc, cur_op).c_str(), what)); }
    [[noreturn]] void fail_s(const std::string & what) { fail(what.c_str()); }

    uint64_t alloc(uint64_t bytes) {
        uint64_t off = 0;
        if (!dyn.alloc(bytes, &off)) {
            no_fit("%s: the values of T = %u: %.1f MiB more does not fit the IO region (%.1f MiB in use, %.1f MiB free, largest %.1f MiB)",
                   op_name(desc, cur_op).c_str(), T, mib(bytes), mib(dyn.used() + (lay.end - d.io_off)), mib(dyn.free_bytes()),
                   mib(dyn.largest_free()));
        }
        hw_end = std::max(hw_end, off + pad64(bytes));
        return off;
    }
    uint64_t alloc_rows(uint32_t n) { return alloc((uint64_t) T * L(n) * LINE); }

    void place(int32_t v) {
        const graph_value & x = desc.values[(size_t) v];
        if (have_off[(size_t) v]) return;
        have_off[(size_t) v] = 1;
        switch (x.kind) {
            case gv::in:
            case gv::tmp:   pl.off[(size_t) v] = alloc_rows(x.n); break;
            case gv::pos:   pl.off[(size_t) v] = lay.pos_off[(size_t) v]; break;
            case gv::konst: {
                if (x.handle != 0) {
                    const auto ci = d.consts.find(x.handle);
                    if (ci == d.consts.end()) {
                        throw std::invalid_argument(strf("tsr_rt: graph: const value %zu: unknown const handle %" PRIu64, (size_t) v, x.handle));
                    }
                    pl.off[(size_t) v] = ci->second.off;
                    read_consts.insert(x.handle);
                } else {
                    pl.off[(size_t) v] = TSR_REGION_AUX_OFF;     // probe
                }
                break;
            }
            case gv::mask:  break;
        }
    }
    void release(int32_t v) {
        const graph_value & x = desc.values[(size_t) v];
        if ((x.kind == gv::in || x.kind == gv::tmp) && !x.out && have_off[(size_t) v]) {
            dyn.release(pl.off[(size_t) v]);
            have_off[(size_t) v] = 0;
        }
    }

    tsr_k_rows rows(int32_t v) const { return { H0(pl.off[(size_t) v]), L(desc.values[(size_t) v].n) }; }

    // -- lists ----------------------------------------------------------------
    void finish_list() {
        if (!b.settle()) fail_s(strf("op list: %s", b.error()));
        if (b.n_ops() == 0) return;
        list_t l;
        l.blob = b.finish();
        if (l.blob.empty()) fail_s(strf("op list build failed: %s", b.error()));
        l.n_ops = b.n_ops();
        l.n_joins = b.n_joins();
        for (uint32_t i = 0; i < l.n_ops; ++i) l.tags.push_back(b.tag_of(i));
        pl.info.records += l.n_ops;
        pl.info.joins += l.n_joins;
        pl.info.mul_mat_begin += b.n_of_kind(opl::K_MUL_MAT_BEGIN);
        for (uint32_t k = 0; k < opl::N_KINDS; ++k) pl.info.by_kind[k] += b.n_of_kind(k);
        pl.info.preload_lines = std::max(pl.info.preload_lines, b.pre_lines());
        pl.lists.push_back(std::move(l));
        b.reset();
    }

    void emit(tsr_rec & r, bool use_fanew, bool defer = false) {
        b.set_defer(defer);
        uint32_t idx = tsr_k_emit(b, r, (int32_t) cur_op, use_fanew ? &fanew : nullptr, defer);
        if (idx == tsr_opl_builder::NO_ROOM) {
            finish_list();
            b.set_defer(defer);
            idx = tsr_k_emit(b, r, (int32_t) cur_op, use_fanew ? &fanew : nullptr, defer);
        }
        if (idx == tsr_opl_builder::NO_ROOM || idx == tsr_opl_builder::FAILED) fail_s(strf("op list: %s", b.error()));
    }

    // -- ops --------------------------------------------------------------------
    // The weight of a mul_mat: the device's, or (a probe, handle 0) a Q8_0 weight of the shape of the values.
    const weight & weight_of(const graph_op & o, weight & probe_w) {
        if (o.w != 0) {
            read_weights.insert(o.w);
            return d.weights.at(o.w);
        }
        const graph_value & dv = desc.values[(size_t) o.dst];
        const graph_value & xv = desc.values[(size_t) o.x];
        weight_options wo;
        probe_w = weight();
        probe_w.src = dtype::f32;
        probe_w.N = dv.n;
        probe_w.K = xv.n;
        if (const char * why = make_plan(d.cfg, dtype::f32, probe_w.N, probe_w.K, wo, &probe_w.pl)) {
            fail_s(strf("weight [%u x %u]: %s", probe_w.N, probe_w.K, why));
        }
        for (uint32_t i = 0; i < probe_w.pl.n_parts; ++i) {
            for (uint32_t j = 0; j < probe_w.pl.k_parts; ++j) {
                part p;
                p.r0 = i * probe_w.pl.n_part;
                p.n = std::min(probe_w.pl.n_part, probe_w.N - p.r0);
                p.k0 = j * probe_w.pl.k_part;
                p.k = std::min(probe_w.pl.k_part, probe_w.K - p.k0);
                p.w_off = 0;
                probe_w.parts.push_back(p);
            }
        }
        return probe_w;
    }

    void op_mul_mat(const graph_op & o) {
        weight probe_w;
        const weight & w = weight_of(o, probe_w);
        if (w.pl.n_parts > 1 && w.pl.n_part % 16 != 0) {
            fail_s(strf("the weight [%u x %u] is split into N parts of %u rows, not a multiple of 16 (a part's results cannot be placed "
                        "side by side in the output rows)", w.N, w.K, w.pl.n_part));
        }
        const tsr_k_rows x = rows(o.x), y = rows(o.dst);
        const bool add_const = o.addend >= 0 && desc.values[(size_t) o.addend].kind == gv::konst;
        const bool addend_in_mm = o.addend >= 0 && w.pl.k_parts == 1;
        // several K parts and an addend: the parts go to a temporary, one ADD adds the addend
        uint64_t tmp_off = 0;
        tsr_k_rows ymm = y;
        if (o.addend >= 0 && !addend_in_mm) {
            tmp_off = alloc_rows(w.N);
            ymm = { H0(tmp_off), L(w.N) };
        }
        for (uint32_t i = 0; i < w.pl.n_parts; ++i) {
            for (uint32_t j = 0; j < w.pl.k_parts; ++j) {
                const part & p = w.at(i, j);
                tsr_kd_mm m;
                m.x = { x.h + p.k0 / 16u, x.stride };
                m.y = { ymm.h + p.r0 / 16u, ymm.stride };
                if (addend_in_mm) {
                    const tsr_k_rows a = rows(o.addend);
                    m.add = { a.h + p.r0 / 16u, add_const ? 0u : a.stride };
                }
                m.w_off = (uint32_t) (p.w_off / TSR_BEAT_BYTES);
                m.fmt = w.pl.fmt;
                m.N = p.n;
                m.K = p.k;
                m.T = T;
                m.acc_dst = j > 0;
                tsr_rec r;
                std::string why;
                if (!tsr_k_mul_mat(d.kenv, m, &r, &why)) fail_s(strf("N part %u, K part %u: %s", i, j, why.c_str()));
                const bool defer = mm_defer_enabled() && w.pl.k_parts == 1 && tsr_k_mm_deferrable(d.kenv, r);
                emit(r, false, defer);
            }
        }
        if (o.addend >= 0 && !addend_in_mm) {
            tsr_kd_ew e;
            e.kind = opl::K_ADD;
            e.dst = y;
            e.a = ymm;
            const tsr_k_rows a = rows(o.addend);
            e.b = { a.h, add_const ? 0u : a.stride };
            e.b_period = add_const ? 1u : T;
            e.n = w.N;
            e.rows = T;
            tsr_rec r;
            std::string why;
            if (!tsr_k_add(d.kenv, e, &r, &why)) fail_s(strf("addend: %s", why.c_str()));
            emit(r, false);
            dyn.release(tmp_off);
        }
    }

    void op_add(const graph_op & o) {
        const graph_value & a = desc.values[(size_t) o.a];
        const bool b_const = desc.values[(size_t) o.b].kind == gv::konst;
        tsr_kd_ew e;
        e.kind = o.op == gop::add ? opl::K_ADD : opl::K_MUL;
        e.dst = rows(o.dst);
        e.a = rows(o.a);
        const tsr_k_rows bb = rows(o.b);
        e.b = { bb.h, b_const ? 0u : bb.stride };
        e.b_period = b_const ? 1u : T;
        e.n = a.n;
        e.rows = T;
        tsr_rec r;
        std::string why;
        if (!tsr_k_add(d.kenv, e, &r, &why)) fail_s(why);
        emit(r, false);
    }

    void op_scale(const graph_op & o) {
        tsr_kd_scale s;
        s.dst = rows(o.dst);
        s.src = rows(o.x);
        s.n = desc.values[(size_t) o.x].n;
        s.rows = T;
        s.s = o.s;
        s.b = o.sb;
        tsr_rec r;
        std::string why;
        if (!tsr_k_scale(d.kenv, s, &r, &why)) fail_s(why);
        emit(r, false);
    }

    void op_rms(const graph_op & o) {
        tsr_kd_rms s;
        s.dst = rows(o.dst);
        s.src = rows(o.x);
        s.n = desc.values[(size_t) o.x].n;
        s.rows = T;
        s.eps = o.eps;
        if (o.weight >= 0) s.weight = rows(o.weight).h;
        s.w_period = 1;
        tsr_rec r;
        std::string why;
        if (!tsr_k_rms_norm(d.kenv, s, &r, &why)) fail_s(why);
        emit(r, false);
    }

    void op_glu(const graph_op & o) {
        tsr_kd_glu s;
        s.dst = rows(o.dst);
        s.gate = rows(o.gate);
        s.up = rows(o.up);
        s.n = desc.values[(size_t) o.gate].n;
        s.rows = T;
        s.variant = o.variant;
        tsr_rec r;
        std::string why;
        if (!tsr_k_glu(d.kenv, s, &r, &why)) fail_s(why);
        emit(r, false);
    }

    void op_unary(const graph_op & o) {
        tsr_kd_unary s;
        s.dst = rows(o.dst);
        s.src = rows(o.x);
        s.n = desc.values[(size_t) o.x].n;
        s.rows = T;
        s.func = o.func;
        tsr_rec r;
        std::string why;
        if (!tsr_k_unary(d.kenv, s, &r, &why)) fail_s(why);
        emit(r, false);
    }

    void op_cpy(const graph_op & o) {
        tsr_kd_cpy s;
        s.dst = rows(o.dst);
        s.src = rows(o.x);
        s.row_lines = L(desc.values[(size_t) o.x].n);
        s.rows = T;
        tsr_rec r;
        std::string why;
        if (!tsr_k_cpy(d.kenv, s, &r, &why)) fail_s(why);
        emit(r, false);
    }

    void op_rope(const graph_op & o) {
        tsr_kd_rope s;
        s.dst = rows(o.dst);
        s.src = rows(o.x);
        s.n_heads = o.n_heads;
        s.head_dim = o.head_dim;
        s.n_dims = o.n_dims;
        s.T = T;
        s.pos_src = H0(pl.off[(size_t) o.pos]);
        s.table = rows(o.table).h;
        s.neox = true;
        s.attn_factor = o.factor;
        tsr_rec r;
        std::string why;
        if (!tsr_k_rope(d.kenv, s, &r, &why)) fail_s(why);
        emit(r, false);
    }

    void op_attn(const graph_op & o, uint32_t ai) {
        const tsr_k_kv & kv = g.kv[ai];
        const slot_t & slot = lay.slots[(size_t) g.attn_slot[cur_op]];
        const bool need_f32 = pl.sig[(size_t) g.attn_slot[cur_op]] != 0;
        const uint32_t kp = (uint32_t) (g.plane_k[ai] / TSR_BEAT_BYTES);
        const uint32_t vp = (uint32_t) (g.plane_v[ai] / TSR_BEAT_BYTES);
        tsr_kd_set_rows sk;
        sk.src = rows(o.k);
        sk.n_rows = T;
        sk.kv = kv;
        sk.which = opl::setrows::WHICH_K;
        sk.plane = kp;
        sk.idx_src = H0(lay.idx_off);
        tsr_kd_set_rows sv = sk;
        sv.src = rows(o.v);
        sv.which = opl::setrows::WHICH_V;
        sv.plane = vp;
        tsr_kd_flash_attn fa;
        fa.dst = rows(o.dst);
        fa.q = rows(o.q);
        fa.n_kv = T;
        fa.D = o.d;
        fa.n_head = o.n_head;
        fa.n_head_kv = o.n_head_kv;
        fa.T = T;
        fa.scale = o.scale;
        fa.kv = kv;
        fa.k_plane = kp;
        fa.v_plane = vp;
        fa.msum_h = H0(slot.msum_off);
        fa.mask_h = need_f32 ? H0(slot.f32_off) : opl::HANDLE_NONE;
        tsr_rec rk, rv, rf;
        std::string why;
        if (!tsr_k_set_rows_kv(d.kenv, sk, &rk, &why)) fail_s(strf("SET_ROWS_KV K: %s", why.c_str()));
        if (!tsr_k_set_rows_kv(d.kenv, sv, &rv, &why)) fail_s(strf("SET_ROWS_KV V: %s", why.c_str()));
        if (!tsr_k_flash_attn(d.kenv, fa, &rf, &why)) fail_s(strf("FLASH_ATTN: %s", why.c_str()));
        emit(rk, true);
        emit(rv, true);
        emit(rf, true);
        // tsr_encode added FA_NEW (the cells the two SET_ROWS_KV records wrote) to the FLASH_ATTN record's flags word
        const uint32_t fl = rf.args[opl::fa::FLAGS];
        if ((fl & tsr::plan::attn::FA_NEW) != 0u && tsr::plan::attn::fa_n_new(fl) == T) pl.info.fa_new++;
    }

    void run() {
        // the inputs are placed first, in value order; every other value when its writer is reached
        for (size_t v = 0; v < desc.values.size(); ++v) {
            const gv k = desc.values[v].kind;
            if (k == gv::in || k == gv::pos) place((int32_t) v);
        }
        std::vector<int32_t> ops;
        for (uint32_t i = 0; i < desc.ops.size(); ++i) {
            cur_op = i;
            const graph_op & o = desc.ops[i];
            operands(o, ops);
            // an input or a temporary is placed before its first read; without a slot here its slot was released early
            for (int32_t v : ops) {
                if (v < 0) continue;
                const gv k = desc.values[(size_t) v].kind;
                if ((k == gv::in || k == gv::tmp) && !have_off[(size_t) v]) {
                    fail_s(strf("value %d is read after its slot was released", v));
                }
                place(v);
            }
            // consts and the destination
            for (int32_t v : { o.table, o.weight, o.addend, o.b }) {
                if (v >= 0 && desc.values[(size_t) v].kind == gv::konst) place(v);
            }
            place(o.dst);
            switch (o.op) {
                case gop::mul_mat:  op_mul_mat(o); break;
                case gop::add:
                case gop::mul:      op_add(o); break;
                case gop::scale:    op_scale(o); break;
                case gop::rms_norm: op_rms(o); break;
                case gop::glu:      op_glu(o); break;
                case gop::unary:    op_unary(o); break;
                case gop::rope:     op_rope(o); break;
                case gop::attn: {
                    uint32_t ai = 0;
                    while (g.attn_ops[ai] != i) ++ai;
                    op_attn(o, ai);
                    break;
                }
                case gop::cpy:      op_cpy(o); break;
            }
            // values whose last reader this was
            for (int32_t v : ops) if (v >= 0 && g.last_use[(size_t) v] == (int32_t) i) release(v);
            if (g.last_use[(size_t) o.dst] < 0) release(o.dst);
        }
        cur_op = (uint32_t) desc.ops.size() - 1;
        finish_list();
        pl.info.lists = (uint32_t) pl.lists.size();
        pl.info.io_bytes = hw_end - d.io_off;
        for (const_handle h : read_consts) {
            const konst & c = d.consts.at(h);
            pl.reads.push_back({ 0, c.off, c.bytes });
        }
        for (weight_handle h : read_weights) {
            for (const part & p : d.weights.at(h).parts) {
                for (uint32_t c = 0; c < d.cfg.num_ch; ++c) pl.reads.push_back({ c, p.w_off, p.ch_bytes });
            }
        }
    }
};

std::shared_ptr<plan_t> build_plan(device::impl & d, graph_obj & g, uint32_t T, const layout_t & lay, const std::vector<uint8_t> & sig) {
    auto pl = std::make_shared<plan_t>();
    pl->T = T;
    pl->sig = sig;
    pl->info.T = T;
    planner p(d, g, T, lay, *pl);
    p.run();
    return pl;
}

// Checks the layout fits the IO region.
void check_layout(const device::impl & d, const graph_obj & g, uint32_t T, const layout_t & lay) {
    if (lay.end - d.io_off > d.io_size) {
        no_fit("T = %u: the positions, cell indices and mask summaries need %.1f MiB: does not fit the IO region (%.1f MiB)", T,
               mib(lay.end - d.io_off), mib(d.io_size));
    }
    (void) g;
}

// ---------------------------------------------------------------------------------------------
// A host image of the device's DRAM (graph_run_ref)
// ---------------------------------------------------------------------------------------------
struct host_image {
    std::vector<uint8_t *> ch;
    uint64_t bytes = 0;
    host_image(uint32_t num_ch, uint64_t ch_bytes) : bytes(ch_bytes) {
        for (uint32_t c = 0; c < num_ch; ++c) {
            void * p = mmap(nullptr, ch_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) {
                for (uint8_t * q : ch) munmap(q, bytes);
                throw error(strf("tsr_rt: graph_run_ref: cannot map the host image of channel %u (%.1f GiB)", c,
                                 (double) ch_bytes / (1024.0 * 1024.0 * 1024.0)));
            }
            ch.push_back((uint8_t *) p);
        }
    }
    ~host_image() {
        for (uint8_t * q : ch) munmap(q, bytes);
    }
    host_image(const host_image &) = delete;
    host_image & operator=(const host_image &) = delete;
};

}  // namespace

// ---------------------------------------------------------------------------------------------
// device: constants
// ---------------------------------------------------------------------------------------------
const_handle device::const_upload(const float * data, uint32_t rows, uint32_t n, size_t row_stride_floats) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    if (rows == 0 || n == 0) throw std::invalid_argument("tsr_rt: const_upload: rows and n must be >= 1");
    if (data == nullptr) throw std::invalid_argument("tsr_rt: const_upload: null data");
    const size_t rs = row_stride_floats != 0 ? row_stride_floats : n;
    if (rs < n) throw std::invalid_argument("tsr_rt: const_upload: row stride < n");
    konst c;
    c.rows = rows;
    c.n = n;
    c.bytes = (uint64_t) rows * L(n) * LINE;
    if (!d.aux.alloc(c.bytes, &c.off)) {
        throw error(strf("tsr_rt: const_upload: the AUX region does not fit %.1f MiB (%.1f MiB free)", mib(c.bytes), mib(d.aux.free_bytes())));
    }
    try {
        std::vector<float> buf((size_t) rows * L(n) * 16u, 0.0f);
        for (uint32_t r = 0; r < rows; ++r) memcpy(&buf[(size_t) r * L(n) * 16u], data + (size_t) r * rs, (size_t) n * 4);
        d.write(0, c.off, buf.data(), buf.size() * 4);
    } catch (...) {
        d.aux.release(c.off);
        throw;
    }
    const const_handle h = d.next_const++;
    d.consts.emplace(h, c);
    d.cnt.consts++;
    return h;
}

void device::const_free(const_handle h) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    auto it = p_->consts.find(h);
    if (it == p_->consts.end()) throw std::invalid_argument(strf("tsr_rt: unknown const handle %" PRIu64, h));
    p_->aux.release(it->second.off);
    p_->consts.erase(it);
}

// ---------------------------------------------------------------------------------------------
// device: graphs
// ---------------------------------------------------------------------------------------------
namespace {

// The plans of T = 1 and T = T_MAX + 1 (a probe: nothing is allocated; each throws tsr::rt::error).
void probe_plans(device::impl & d, graph_obj & g) {
    std::vector<uint32_t> Ts = { 1u, d.cfg.t_max + 1u };
    if (!g.attn_ops.empty()) {
        for (uint32_t & T : Ts) T = std::min({ T, g.desc.kv_size, graph_max_attn_tokens });
    }
    if (Ts[1] == Ts[0]) Ts.pop_back();
    for (uint32_t T : Ts) {
        const layout_t lay = static_layout(d, g, T);
        check_layout(d, g, T, lay);
        const auto masks = build_masks(d, g, T, lay, nullptr);
        std::vector<uint8_t> sig(masks.size(), 0);
        for (size_t s = 0; s < masks.size(); ++s) sig[s] = masks[s] && masks[s]->need_f32;
        build_plan(d, g, T, lay, sig);
    }
}

}  // namespace

const char * device::graph_unsupported(const graph_desc & g) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    validate(g, false, d);
    try {
        auto o = make_obj(d, g, true);
        if (!o->attn_ops.empty() && d.heap.free_bytes() < o->kv_bytes) {
            throw error(strf("tsr_rt: graph: the KV planes need %.1f MiB per channel: does not fit the WEIGHT heap (%.1f MiB free)",
                             mib(o->kv_bytes), mib(d.heap.free_bytes())));
        }
        probe_plans(d, *o);
    } catch (const error & e) {
        thread_local std::string reason;     // per thread: concurrent probes do not share the answer
        reason = e.what();
        return reason.c_str();
    }
    return nullptr;
}

graph_handle device::graph_create(const graph_desc & g) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    validate(g, true, d);
    auto o = make_obj(d, g, true);
    probe_plans(d, *o);
    o->probe = false;
    // the planes: K and V of each attn op, from the top of the heap, zero-filled in every channel
    std::vector<uint64_t> allocated;
    try {
        for (size_t a = 0; a < o->attn_ops.size(); ++a) {
            uint64_t off[2] = { 0, 0 };
            for (int w = 0; w < 2; ++w) {
                if (!d.heap.alloc_top(o->kv[a].bytes, &off[w])) {
                    throw error(strf("tsr_rt: graph: %s: the KV planes need %.1f MiB per channel: does not fit the WEIGHT heap (%.1f MiB "
                                     "free, largest %.1f MiB)", op_name(g, o->attn_ops[a]).c_str(), mib(2 * o->kv[a].bytes),
                                     mib(d.heap.free_bytes()), mib(d.heap.largest_free())));
                }
                allocated.push_back(off[w]);
            }
            o->plane_k[a] = off[0];
            o->plane_v[a] = off[1];
        }
        std::vector<uint8_t> zero;
        for (size_t a = 0; a < o->attn_ops.size(); ++a) {
            const uint64_t bytes = o->kv[a].bytes;
            zero.assign((size_t) std::min<uint64_t>(bytes, XFER_CHUNK), 0);
            for (uint64_t off : { o->plane_k[a], o->plane_v[a] }) {
                for (uint32_t c = 0; c < d.cfg.num_ch; ++c) {
                    for (uint64_t at = 0; at < bytes; at += zero.size()) {
                        d.write(c, off + at, zero.data(), (size_t) std::min<uint64_t>(zero.size(), bytes - at));
                    }
                }
            }
        }
    } catch (...) {
        for (uint64_t off : allocated) d.heap.release(off);
        throw;
    }
    const graph_handle h = d.next_graph++;
    d.graphs.emplace(h, o);
    d.cnt.graphs++;
    TSR_RT_DBG(1, "graph #%" PRIu64 ": %zu values, %zu ops, %zu attn, KV %.2f MiB per channel\n", h, g.values.size(), g.ops.size(),
               o->attn_ops.size(), mib(o->kv_bytes));
    return h;
}

namespace {

std::shared_ptr<graph_obj> find_graph(const device::impl & d, graph_handle h) {
    auto it = d.graphs.find(h);
    if (it == d.graphs.end()) throw std::invalid_argument(strf("tsr_rt: unknown graph handle %" PRIu64, h));
    return it->second;
}

struct prepared {
    std::shared_ptr<plan_t> plan;
    layout_t lay;
    std::vector<std::shared_ptr<tsr_k_fa_mask>> masks;
};

// The plan of T for the mask shapes of `in` (null: the user masks are taken as needing no f32 copy): checks, the static layout,
// the mask summaries, the cached plan or a new one.
prepared prepare(device::impl & d, graph_obj & g, uint32_t T, const graph_inputs * in) {
    check_T(d, g, T);
    for (const graph_value & v : g.desc.values) {
        if (v.kind == gv::konst && d.consts.find(v.handle) == d.consts.end()) {
            throw std::invalid_argument(strf("tsr_rt: graph: the const handle %" PRIu64 " was freed", v.handle));
        }
    }
    for (uint32_t i = 0; i < g.desc.ops.size(); ++i) {
        const graph_op & o = g.desc.ops[i];
        if (o.op == gop::mul_mat && d.weights.find(o.w) == d.weights.end()) {
            throw std::invalid_argument(strf("tsr_rt: graph: %s: the weight handle %" PRIu64 " was freed", op_name(g.desc, i).c_str(), o.w));
        }
    }
    prepared p;
    p.lay = static_layout(d, g, T);
    check_layout(d, g, T, p.lay);
    p.masks = build_masks(d, g, T, p.lay, in);
    std::vector<uint8_t> sig(p.masks.size(), 0);
    for (size_t s = 0; s < p.masks.size(); ++s) sig[s] = p.masks[s] && p.masks[s]->need_f32;
    for (auto it = g.plans.begin(); it != g.plans.end(); ++it) {
        if ((*it)->T == T && (*it)->sig == sig) {
            g.plans.splice(g.plans.begin(), g.plans, it);
            p.plan = g.plans.front();
            d.cnt.graph_plan_hits++;
            return p;
        }
    }
    p.plan = build_plan(d, g, T, p.lay, sig);
    d.cnt.graph_plan_misses++;
    g.plans.push_front(p.plan);
    while (g.plans.size() > graph_plan_cache) g.plans.pop_back();
    return p;
}

using sink_fn = std::function<void(uint32_t, uint64_t, const void *, size_t)>;

// The inputs, the positions, the KV cell indices and the mask summaries of a run.
void write_inputs(const graph_obj & g, const plan_t & pl, const prepared & p, uint32_t T, const graph_inputs & in, const sink_fn & put) {
    const graph_desc & desc = g.desc;
    std::vector<uint8_t> buf;
    for (size_t ai = 0; ai < g.arg_value.size(); ++ai) {
        const size_t v = (size_t) g.arg_value[ai];
        const graph_value & x = desc.values[v];
        const graph_arg & a = in.args[ai];
        if (x.kind == gv::in) {
            const size_t st = a.stride != 0 ? a.stride : x.n;
            const uint64_t row = (uint64_t) L(x.n) * LINE;
            buf.assign((size_t) T * row, 0);
            for (uint32_t t = 0; t < T; ++t) memcpy(buf.data() + t * row, (const float *) a.data + t * st, (size_t) x.n * 4);
            put(0, pl.off[v], buf.data(), buf.size());
        } else if (x.kind == gv::pos) {
            buf.assign((size_t) pad64((uint64_t) T * 4), 0);
            memcpy(buf.data(), a.data, (size_t) T * 4);
            put(0, p.lay.pos_off[v], buf.data(), buf.size());
        }
    }
    if (!g.attn_ops.empty()) {
        buf.assign((size_t) pad64((uint64_t) T * opl::PRE_KVIDX_BYTES), 0);
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t c = t;
            memcpy(buf.data() + (size_t) t * 8, &c, 8);
        }
        put(0, p.lay.idx_off, buf.data(), buf.size());
    }
    for (size_t s = 0; s < p.masks.size(); ++s) {
        const tsr_k_fa_mask & m = *p.masks[s];
        buf.assign((size_t) pad64(opl::msum_bytes(T)), 0);
        memcpy(buf.data(), m.msum.data(), m.msum.size() * 4);
        put(0, p.lay.slots[s].msum_off, buf.data(), buf.size());
        if (m.need_f32) put(0, p.lay.slots[s].f32_off, m.f32.data(), m.f32.size() * 4);
    }
}

void check_io(const graph_obj & g, uint32_t T, const graph_inputs & in, const graph_outputs & out) {
    if (T == 0) throw std::invalid_argument("tsr_rt: graph: T must be >= 1");
    if (in.args.size() != g.arg_value.size()) {
        throw std::invalid_argument(strf("tsr_rt: graph: %zu arguments given, the graph takes %zu (its in, pos and mask values)",
                                         in.args.size(), g.arg_value.size()));
    }
    for (size_t ai = 0; ai < g.arg_value.size(); ++ai) {
        const graph_value & x = g.desc.values[(size_t) g.arg_value[ai]];
        if (in.args[ai].data == nullptr) throw std::invalid_argument(strf("tsr_rt: graph: argument %zu is null", ai));
        if (x.kind == gv::in && in.args[ai].stride != 0 && in.args[ai].stride < x.n) {
            throw std::invalid_argument(strf("tsr_rt: graph: argument %zu: row stride %zu < n = %u", ai, in.args[ai].stride, x.n));
        }
        if (x.kind == gv::mask && in.args[ai].stride != 0 && in.args[ai].stride < T) {
            throw std::invalid_argument(strf("tsr_rt: graph: argument %zu: row stride %zu < T = %u", ai, in.args[ai].stride, T));
        }
    }
    if (out.outs.size() != g.out_value.size()) {
        throw std::invalid_argument(strf("tsr_rt: graph: %zu outputs given, the graph has %zu out values", out.outs.size(), g.out_value.size()));
    }
    for (size_t i = 0; i < out.outs.size(); ++i) {
        const graph_value & x = g.desc.values[g.out_value[i]];
        if (out.outs[i].data == nullptr) throw std::invalid_argument(strf("tsr_rt: graph: output %zu is null", i));
        if (out.outs[i].stride != 0 && out.outs[i].stride < x.n) {
            throw std::invalid_argument(strf("tsr_rt: graph: output %zu: row stride %zu < n = %u", i, out.outs[i].stride, x.n));
        }
    }
}

void copy_out(const graph_obj & g, uint32_t T, size_t i, const uint8_t * rows, const graph_outputs & out) {
    const graph_value & x = g.desc.values[g.out_value[i]];
    const uint64_t row = (uint64_t) L(x.n) * LINE;
    const size_t st = out.outs[i].stride != 0 ? out.outs[i].stride : x.n;
    for (uint32_t t = 0; t < T; ++t) memcpy(out.outs[i].data + t * st, rows + t * row, (size_t) x.n * 4);
}

std::function<std::string(int32_t)> namer(const graph_desc & desc, const list_t & l) {
    return [&desc, &l](int32_t rec) {
        const int32_t op = rec >= 0 && (size_t) rec < l.tags.size() ? l.tags[(size_t) rec] : -1;
        return op >= 0 ? op_name(desc, (uint32_t) op) : std::string("JOIN");
    };
}

}  // namespace

void device::graph_free(graph_handle h) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    auto g = find_graph(d, h);
    for (size_t a = 0; a < g->attn_ops.size(); ++a) {
        d.heap.release(g->plane_k[a]);
        d.heap.release(g->plane_v[a]);
    }
    d.graphs.erase(h);
}

graph_shape device::graph_shape_of(graph_handle h) const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    const auto g = find_graph(*p_, h);
    graph_shape s;
    for (int v : g->arg_value) {
        const graph_value & x = g->desc.values[(size_t) v];
        s.arg_kind.push_back(x.kind);
        s.arg_n.push_back(x.kind == gv::in ? x.n : 0u);
    }
    for (uint32_t v : g->out_value) s.out_n.push_back(g->desc.values[v].n);
    s.kv_size = g->desc.kv_size;
    s.kv_bytes = g->kv_bytes;
    return s;
}

void device::graph_run(graph_handle h, uint32_t T, const graph_inputs & in, const graph_outputs & out) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const auto g = find_graph(d, h);
    check_io(*g, T, in, out);
    const prepared p = prepare(d, *g, T, &in);
    const plan_t & pl = *p.plan;
    write_inputs(*g, pl, p, T, in, [&d](uint32_t ch, uint64_t off, const void * src, size_t n) { d.write(ch, off, src, n); });
    for (const list_t & l : pl.lists) {
        d.run_blob(l.blob, l.n_ops, l.n_joins, namer(g->desc, l));
        d.cnt.graph_launches++;
        d.cnt.graph_records += l.n_ops;
    }
    std::vector<uint8_t> buf;
    for (size_t i = 0; i < g->out_value.size(); ++i) {
        const uint32_t v = g->out_value[i];
        buf.resize((size_t) T * L(g->desc.values[v].n) * LINE);
        d.read(0, pl.off[v], buf.data(), buf.size());
        copy_out(*g, T, i, buf.data(), out);
    }
    d.cnt.graph_runs++;
}

void device::graph_run_ref(graph_handle h, uint32_t T, const graph_inputs & in, const graph_outputs & out) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const auto g = find_graph(d, h);
    check_io(*g, T, in, out);
    const prepared p = prepare(d, *g, T, &in);
    const plan_t & pl = *p.plan;
    host_image img(d.cfg.num_ch, d.cfg.ch_bytes);
    // the inputs go to the image, the constants and the weights are read from the device; the KV planes stay zero (the lists
    // write every cell they read)
    write_inputs(*g, pl, p, T, in, [&](uint32_t ch, uint64_t off, const void * src, size_t n) {
        if (off + n > d.cfg.ch_bytes) throw error("tsr_rt: graph_run_ref: input outside the DRAM image");
        memcpy(img.ch[ch] + off, src, n);
    });
    for (const dev_range & r : pl.reads) {
        for (uint64_t at = 0; at < r.bytes; at += XFER_CHUNK) {
            d.read(r.ch, r.off + at, img.ch[r.ch] + r.off + at, (size_t) std::min<uint64_t>(XFER_CHUNK, r.bytes - at));
        }
    }
    tsr::ref::opl_exec_cfg xc;
    xc.t_max = d.cfg.t_max;
    xc.pcap = d.pcap;
    xc.k_max = d.cfg.k_max;
    xc.mm_threads = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    const tsr::ref::dram_view view{ img.ch.data(), d.cfg.num_ch, d.cfg.ch_bytes };
    for (size_t li = 0; li < pl.lists.size(); ++li) {
        const list_t & l = pl.lists[li];
        std::string why;
        if (tsr::ref::opl_exec(l.blob, view, xc, &why) != 0) {
            unsigned rec = 0;
            std::string name;
            if (sscanf(why.c_str(), "op list record %u", &rec) == 1) name = namer(g->desc, l)((int32_t) rec);
            throw error(strf("tsr_rt: graph_run_ref: %s%s%s", name.empty() ? "" : name.c_str(), name.empty() ? "" : ": ", why.c_str()));
        }
    }
    for (size_t i = 0; i < g->out_value.size(); ++i) {
        copy_out(*g, T, i, img.ch[0] + pl.off[g->out_value[i]], out);
    }
}

graph_plan_info device::graph_info(graph_handle h, uint32_t T) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const auto g = find_graph(d, h);
    return prepare(d, *g, T, nullptr).plan->info;
}

}  // namespace tsr::rt
