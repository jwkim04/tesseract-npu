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
//
// Cached attention: an attn op with a KV slot (graph_op::kv) owns no planes. A run binds a KV object (kv_create: planes from the
// top of the WEIGHT heap that outlive graphs) to each slot; the op stores the rows of the run into the cells pos0 .. pos0 + T - 1
// and attends over the planes' cells with the mask over the slot's cells. The plan depends on the planes' offsets (not on the
// handles). A graph whose attn ops all have a slot is run in chunks of rows when T does not fit one plan.
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <list>
#include <set>
#include <thread>
#include <tuple>

#include "opl_ref.hpp"
#include "tsr_ref_at.h"
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
    uint32_t cells;               // 0: the op's own planes (T cells); else the cells of its KV slot (cached attention)
    bool operator<(const mask_key & o) const {
        return std::tie(mask, E, D, G, scale, cells) < std::tie(o.mask, o.E, o.D, o.G, o.scale, o.cells);
    }
};

// A refusal because of the shape of the run (limits, memory, a mask that shows what the call does not write): its text contains
// "does not fit". The chunked run halves its chunk on this type only.
struct no_fit_error : error {
    using error::error;
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
    uint64_t idx_off = 0;               // the cell indices 0 .. T-1 of the attn ops with their own planes
    uint64_t idx_kv_off = 0;            // the cell indices pos0 .. pos0 + T-1 of the cached attn ops
    std::vector<slot_t> slots;
    uint64_t end = 0;                   // first byte above, ch0 byte offset
};

struct plan_t {
    uint32_t T = 0;
    std::vector<uint8_t> sig;           // per mask slot: the summary needs the f32 copy
    std::vector<uint64_t> kvoff;        // per KV slot: byte offset of its K plane and of its V plane (k0, v0, k1, v1, ...)
    std::vector<uint32_t> nkv;          // per KV slot: the cells the FLASH_ATTN record covers
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
    std::vector<tsr_k_kv> kv;           // per attn op (index in attn_ops): the geometry of its planes or of its slot
    std::vector<uint64_t> plane_k, plane_v;   // per attn op: byte offset of its K and of its V plane (the same in every channel);
                                              // 0 for an op with a KV slot (its planes belong to the bound KV object)
    std::vector<tsr_k_kv> slot_geom;    // per KV slot
    std::vector<uint32_t> slot_op;      // per KV slot: the op that uses it
    bool has_slots = false, has_unslotted = false;   // attn ops with / without a KV slot
    bool all_cached = false;            // every attn op has a slot (and there is one): the run may be chunked
    bool bound = false;                 // a run bound KV objects: last_k, last_v, last_pos0 (graph_info plans with the plane offsets, and with pos0 unless the graph runs in chunks)
    std::vector<uint64_t> last_k, last_v;
    uint32_t last_pos0 = 0;
    std::vector<int> arg_value;         // per argument: the value id
    std::vector<uint32_t> out_value;    // per out: the value id
    uint64_t kv_bytes = 0;              // per channel, all planes
    std::list<std::shared_ptr<plan_t>> plans;   // LRU, front = most recent
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::shared_ptr<tsr_k_fa_mask>> causal;   // (T, pos0, mask slot)
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
        if (x.cols != 0 && x.kind != gv::mask) throw std::invalid_argument(strf("tsr_rt: graph: value %zu: only a mask value has cols", v));
        any_out = any_out || x.out;
    }
    if (!any_out) throw std::invalid_argument("tsr_rt: graph: no out value");
    if (g.kv_size == 0 && std::any_of(g.ops.begin(), g.ops.end(), [](const graph_op & o) { return o.op == gop::attn && o.kv < 0; })) {
        throw std::invalid_argument("tsr_rt: graph: an attn op without a KV slot needs kv_size >= 1");
    }
    for (size_t s = 0; s < g.kv_slots.size(); ++s) {
        if (g.kv_slots[s] == 0 || g.kv_slots[s] > (1u << 28)) throw std::invalid_argument(strf("tsr_rt: graph: KV slot %zu: %u cells out of range", s, g.kv_slots[s]));
    }
    std::vector<int> slot_used(g.kv_slots.size(), 0);

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
                if (o.kv < -1 || o.kv >= (int32_t) g.kv_slots.size()) bad_op(g, i, strf("kv slot %d out of range (%zu slots)", o.kv, g.kv_slots.size()).c_str());
                if (o.kv >= 0) {
                    if (slot_used[(size_t) o.kv]++ != 0) bad_op(g, i, strf("KV slot %d is used by another attn op (one op per slot)", o.kv).c_str());
                    if (o.mask >= 0 && g.values[(size_t) o.mask].cols != g.kv_slots[(size_t) o.kv]) {
                        bad_op(g, i, strf("the mask has cols = %u, the KV slot %d has %u cells (a cached attention takes a [T, cells] mask)",
                                          g.values[(size_t) o.mask].cols, o.kv, g.kv_slots[(size_t) o.kv]).c_str());
                    }
                } else if (o.mask >= 0 && g.values[(size_t) o.mask].cols != 0) {
                    bad_op(g, i, "a mask with cols != 0 belongs to cached attention (an op with a KV slot)");
                }
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
    for (size_t s = 0; s < slot_used.size(); ++s) {
        if (slot_used[s] == 0) throw std::invalid_argument(strf("tsr_rt: graph: KV slot %zu is used by no attn op", s));
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
        const uint32_t cells = op.kv >= 0 ? g.kv_slots[(size_t) op.kv] : g.kv_size;
        std::string why;
        tsr_k_kv kv;
        if (!tsr_k_kv_geom(E, cells, d.cfg.num_ch, &kv, &why)) {
            throw error(strf("tsr_rt: graph: %s: KV %s: %s", op_name(g, i).c_str(), op.kv >= 0 ? "slot" : "planes", why.c_str()));
        }
        o->attn_ops.push_back(i);
        o->kv.push_back(kv);
        if (op.kv >= 0) {
            o->has_slots = true;
            if (o->slot_geom.size() <= (size_t) op.kv) {
                o->slot_geom.resize((size_t) op.kv + 1);
                o->slot_op.resize((size_t) op.kv + 1, 0);
            }
            o->slot_geom[(size_t) op.kv] = kv;
            o->slot_op[(size_t) op.kv] = i;
        } else {
            o->has_unslotted = true;
            o->kv_bytes += 2 * kv.bytes;
        }
        const mask_key key{ op.mask, E, op.d, op.n_head / op.n_head_kv, bits_of(op.scale), op.kv >= 0 ? cells : 0u };
        auto it = slot_of.find(key);
        if (it == slot_of.end()) {
            it = slot_of.emplace(key, (int32_t) o->slot_keys.size()).first;
            o->slot_keys.push_back(key);
        }
        o->attn_slot[i] = it->second;
    }
    o->all_cached = o->has_slots && !o->has_unslotted;
    o->plane_k.assign(o->attn_ops.size(), 0);
    o->plane_v.assign(o->attn_ops.size(), 0);
    if (probe) {
        for (size_t a = 0; a < o->plane_k.size(); ++a) {
            if (g.ops[o->attn_ops[a]].kv >= 0) continue;
            o->plane_k[a] = TSR_REGION_WEIGHT_OFF;
            o->plane_v[a] = TSR_REGION_WEIGHT_OFF + (o->kv[a].bytes + 4095u) / 4096u * 4096u;
        }
    }
    return o;
}

// What a run supplies to the planner and the writers: the planes of the KV slots, the first cell of the chunk's rows, the chunk's
// first row in the call's arguments, the arguments (null: planning only, a user mask is not built).
struct run_ctx {
    const graph_inputs * in = nullptr;
    uint32_t pos0 = 0;               // cell of the chunk's first row
    uint32_t r0 = 0;                 // row of the chunk's first row in the call
    std::vector<uint64_t> k, v;      // per KV slot: byte offsets of its planes
};

// Fake plane offsets for planning without a bound state (the probe's): nothing is read or written there.
run_ctx fake_ctx(const graph_obj & g) {
    run_ctx cx;
    for (const tsr_k_kv & kv : g.slot_geom) {
        cx.k.push_back(TSR_REGION_WEIGHT_OFF);
        cx.v.push_back(TSR_REGION_WEIGHT_OFF + (kv.bytes + 4095u) / 4096u * 4096u);
    }
    return cx;
}

// The cells a FLASH_ATTN record of a cached attention covers for a chunk of T rows at pos0. Every cell the mask may show is
// below pos0 + T; the summary's runs decide what is streamed (AT §7.3).
uint32_t n_kv_of(uint32_t cells, uint32_t pos0, uint32_t T) {
    (void) pos0;
    (void) T;
    return cells;
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
    if (g.has_unslotted) {
        l.idx_off = at;
        at += pad64((uint64_t) T * opl::PRE_KVIDX_BYTES);
    }
    if (g.has_slots) {
        l.idx_kv_off = at;
        at += pad64((uint64_t) T * opl::PRE_KVIDX_BYTES);
    }
    for (const mask_key & k : g.slot_keys) {
        slot_t s;
        s.key = k;
        s.msum_off = at;
        at += pad64(opl::msum_bytes(T));
        s.f32_off = at;
        at += (uint64_t) T * L(k.cells != 0 ? k.cells : T) * LINE;
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
    throw no_fit_error(std::string("tsr_rt: graph: ") + buf);
}

// The rows pos0 .. pos0 + T - 1 must be cells of every KV slot.
void check_cells(const graph_obj & g, uint32_t T, uint32_t pos0) {
    for (size_t s = 0; s < g.desc.kv_slots.size(); ++s) {
        if ((uint64_t) pos0 + T > g.desc.kv_slots[s]) {
            no_fit("%s: pos0 + T = %" PRIu64 " does not fit the %u cells of the KV slot %zu (pos0 %u, T %u)",
                   op_name(g.desc, g.slot_op[s]).c_str(), (uint64_t) pos0 + T, g.desc.kv_slots[s], s, pos0, T);
        }
    }
}

void check_T(const device::impl & d, const graph_obj & g, uint32_t T, uint32_t pos0) {
    if (T == 0) throw std::invalid_argument("tsr_rt: graph: T must be >= 1");
    if (!g.attn_ops.empty()) {
        if (g.has_unslotted && T > g.desc.kv_size) no_fit("T = %u does not fit the KV planes (kv_size %u)", T, g.desc.kv_size);
        if (T > graph_max_attn_tokens) {
            no_fit("T = %u does not fit an attn op (its mask summary and cell indices take the 256-line preload area: T <= %u)", T,
                   graph_max_attn_tokens);
        }
        check_cells(g, T, pos0);
    }
    (void) d;
}

// ---------------------------------------------------------------------------------------------
// Masks
// ---------------------------------------------------------------------------------------------
// The summaries of every slot for T rows. cx.in: the run's arguments (null for a probe or graph_info: a user mask is not built).
// A cached attention (cells != 0): the rows are the cells cx.pos0 .., the summary covers n_kv_of(cells) cells, a given mask is
// read from row cx.r0 of its argument, [T, cols] with cols = cells.
std::vector<std::shared_ptr<tsr_k_fa_mask>> build_masks(device::impl & d, graph_obj & g, uint32_t T, const layout_t & lay,
                                                        const run_ctx & cx) {
    std::vector<std::shared_ptr<tsr_k_fa_mask>> res(g.slot_keys.size());
    std::map<int32_t, std::vector<uint16_t>> m16;   // per mask value
    for (size_t s = 0; s < g.slot_keys.size(); ++s) {
        const mask_key & k = g.slot_keys[s];
        const uint32_t f32_h = H0(lay.slots[s].f32_off);
        const bool cached = k.cells != 0;
        const uint32_t n_kv = cached ? n_kv_of(k.cells, cx.pos0, T) : T;
        const uint32_t pos0 = cached ? cx.pos0 : 0;
        std::string why;
        // the op that owns the slot, for the error text
        uint32_t op_i = 0;
        for (uint32_t i = 0; i < g.desc.ops.size(); ++i) {
            if (g.attn_slot[i] == (int32_t) s) { op_i = i; break; }
        }
        const float scale = [&] { float f; memcpy(&f, &k.scale, 4); return f; }();
        if (k.mask < 0) {
            auto it = g.causal.find({ T, pos0, (uint32_t) s });
            if (it == g.causal.end()) {
                auto m = std::make_shared<tsr_k_fa_mask>();
                if (!tsr_k_fa_mask_build(d.kenv, k.E, k.D, k.G, scale, T, n_kv, f32_h, nullptr, 0, pos0, m.get(), &why)) {
                    throw error(strf("tsr_rt: graph: %s: causal mask: %s", op_name(g.desc, op_i).c_str(), why.c_str()));
                }
                if (g.causal.size() >= 64) g.causal.clear();
                it = g.causal.emplace(std::make_tuple(T, pos0, (uint32_t) s), std::move(m)).first;
            }
            res[s] = it->second;
            continue;
        }
        if (cx.in == nullptr) continue;
        auto it = m16.find(k.mask);
        if (it == m16.end()) {
            // the argument of this mask value
            size_t ai = 0;
            while (g.arg_value[ai] != k.mask) ++ai;
            const graph_arg & a = cx.in->args[ai];
            const uint32_t cols = cached ? k.cells : T;
            const float * mp = (const float *) a.data;
            const size_t st = a.stride != 0 ? a.stride : cols;
            std::vector<uint16_t> v((size_t) T * n_kv);
            for (uint32_t q = 0; q < T; ++q) {
                for (uint32_t j = 0; j < n_kv; ++j) {
                    const float x = mp[(size_t) (cx.r0 + q) * st + j];
                    if (x == 0.0f) v[(size_t) q * n_kv + j] = 0x0000u;
                    else if (std::isinf(x) && x < 0.0f) v[(size_t) q * n_kv + j] = 0xFC00u;
                    else {
                        no_fit("%s: mask value %g at [%u, %u] does not fit the device's mask form (every value must be 0 or -inf)",
                               op_name(g.desc, op_i).c_str(), (double) x, cx.r0 + q, j);
                    }
                }
            }
            it = m16.emplace(k.mask, std::move(v)).first;
        }
        auto m = std::make_shared<tsr_k_fa_mask>();
        if (!tsr_k_fa_mask_build(d.kenv, k.E, k.D, k.G, scale, T, n_kv, f32_h, it->second.data(), n_kv, 0, m.get(), &why)) {
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
    const run_ctx & cx;

    tsr_range_alloc dyn;
    uint64_t hw_end = 0;                       // highest byte used above io_off
    tsr_opl_builder b;
    tsr_fanew_map fanew;
    uint32_t cur_op = 0;
    std::vector<uint8_t> have_off;             // per value: placed
    bool outs_first = false;                   // place the out values right after the inputs: they form one block
    std::set<const_handle> read_consts;
    std::set<weight_handle> read_weights;

    planner(device::impl & dd, graph_obj & gg, uint32_t TT, const layout_t & ll, plan_t & pp, const run_ctx & cc)
        : d(dd), g(gg), desc(gg.desc), T(TT), lay(ll), pl(pp), cx(cc), b(dd.cfg.num_ch, opl::F_STOP_ON_ERROR) {
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
        const bool cached = o.kv >= 0;
        const uint32_t kp = (uint32_t) ((cached ? cx.k[(size_t) o.kv] : g.plane_k[ai]) / TSR_BEAT_BYTES);
        const uint32_t vp = (uint32_t) ((cached ? cx.v[(size_t) o.kv] : g.plane_v[ai]) / TSR_BEAT_BYTES);
        tsr_kd_set_rows sk;
        sk.src = rows(o.k);
        sk.n_rows = T;
        sk.kv = kv;
        sk.which = opl::setrows::WHICH_K;
        sk.plane = kp;
        sk.idx_src = H0(cached ? lay.idx_kv_off : lay.idx_off);
        tsr_kd_set_rows sv = sk;
        sv.src = rows(o.v);
        sv.which = opl::setrows::WHICH_V;
        sv.plane = vp;
        tsr_kd_flash_attn fa;
        fa.dst = rows(o.dst);
        fa.q = rows(o.q);
        fa.n_kv = cached ? pl.nkv[(size_t) o.kv] : T;
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
        // Out values are never released, so nothing is freed before them: the first-fit allocator puts them side by side and the
        // host reads them in one transfer.
        if (outs_first) {
            for (int32_t v : g.out_value) place(v);
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

// The plane offsets and the covered cells of the KV slots of a run, the part of a plan's key that is not T and the mask shape.
void slot_key(const graph_obj & g, uint32_t T, const run_ctx & cx, std::vector<uint64_t> * kvoff, std::vector<uint32_t> * nkv) {
    kvoff->clear();
    nkv->clear();
    for (size_t s = 0; s < g.desc.kv_slots.size(); ++s) {
        kvoff->push_back(cx.k[s]);
        kvoff->push_back(cx.v[s]);
        nkv->push_back(n_kv_of(g.desc.kv_slots[s], cx.pos0, T));
    }
}

std::shared_ptr<plan_t> build_plan(device::impl & d, graph_obj & g, uint32_t T, const layout_t & lay, const std::vector<uint8_t> & sig,
                                   const run_ctx & cx) {
    auto pl = std::make_shared<plan_t>();
    pl->T = T;
    pl->sig = sig;
    pl->info.T = T;
    slot_key(g, T, cx, &pl->kvoff, &pl->nkv);
    if (g.has_slots) {
        // The out values placed early can raise the peak of the IO region: a plan that fits without it is still taken. A failed
        // planner leaves nothing on the device (its allocator, lists and read sets are its own).
        try {
            planner p(d, g, T, lay, *pl, cx);
            p.outs_first = true;
            p.run();
            return pl;
        } catch (const no_fit_error &) {
            pl = std::make_shared<plan_t>();
            pl->T = T;
            pl->sig = sig;
            pl->info.T = T;
            slot_key(g, T, cx, &pl->kvoff, &pl->nkv);
        }
    }
    planner p(d, g, T, lay, *pl, cx);
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
        uint32_t lim = graph_max_attn_tokens;
        if (g.has_unslotted) lim = std::min(lim, g.desc.kv_size);
        for (uint32_t c : g.desc.kv_slots) lim = std::min(lim, c);
        for (uint32_t & T : Ts) T = std::min(T, lim);
    }
    if (Ts[1] == Ts[0]) Ts.pop_back();
    const run_ctx cx = fake_ctx(g);
    for (uint32_t T : Ts) {
        const layout_t lay = static_layout(d, g, T);
        check_layout(d, g, T, lay);
        const auto masks = build_masks(d, g, T, lay, cx);
        std::vector<uint8_t> sig(masks.size(), 0);
        for (size_t s = 0; s < masks.size(); ++s) sig[s] = masks[s] && masks[s]->need_f32;
        build_plan(d, g, T, lay, sig, cx);
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
            if (g.ops[o->attn_ops[a]].kv >= 0) continue;     // the planes of a KV slot belong to the bound KV object
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
            if (g.ops[o->attn_ops[a]].kv >= 0) continue;
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

const kvobj & find_kv(const device::impl & d, kv_handle h) {
    auto it = d.kvs.find(h);
    if (it == d.kvs.end()) throw std::invalid_argument(strf("tsr_rt: unknown KV handle %" PRIu64, h));
    return it->second;
}

// The byte ranges of a K or V plane (relative to the plane base, the same in every channel) that hold the cells [c0, c1), whole
// groups of 16 cells (AT §4.1: group u = cell / 16 is local group u / C of channel u % C). A K group is GB = 17 E / 64 consecutive
// beats (§4.3); a V group is E / 64 chunks of 17 beats, one per slice, LG chunks apart (§4.4).
template <class F>
void kv_ranges(const tsr_k_kv & kv, bool is_v, uint32_t c0, uint32_t c1, const F & f) {
    if (c1 <= c0) return;
    const uint32_t l_lo = (c0 / 16u) / kv.C, l_hi = ((c1 - 1u) / 16u) / kv.C;
    const uint64_t nl = (uint64_t) l_hi - l_lo + 1u;
    if (!is_v) {
        const uint64_t gb = 17ull * (kv.E / 64u) * TSR_BEAT_BYTES;
        f((uint64_t) l_lo * gb, nl * gb);
    } else {
        for (uint32_t sl = 0; sl < kv.E / 64u; ++sl) f(((uint64_t) sl * kv.LG + l_lo) * 17ull * TSR_BEAT_BYTES, nl * 17ull * TSR_BEAT_BYTES);
    }
}

void read_big(device::impl & d, uint32_t ch, uint64_t off, uint8_t * dst, uint64_t bytes) {
    for (uint64_t at = 0; at < bytes; at += XFER_CHUNK) d.read(ch, off + at, dst + at, (size_t) std::min<uint64_t>(XFER_CHUNK, bytes - at));
}

void zero_planes(device::impl & d, const std::vector<uint64_t> & offs, uint64_t bytes) {
    std::vector<uint8_t> zero((size_t) std::min<uint64_t>(bytes, XFER_CHUNK), 0);
    for (uint64_t off : offs) {
        for (uint32_t c = 0; c < d.cfg.num_ch; ++c) {
            for (uint64_t at = 0; at < bytes; at += zero.size()) {
                d.write(c, off + at, zero.data(), (size_t) std::min<uint64_t>(zero.size(), bytes - at));
            }
        }
    }
}

struct prepared {
    std::shared_ptr<plan_t> plan;
    layout_t lay;
    std::vector<std::shared_ptr<tsr_k_fa_mask>> masks;
};

// The plan of T rows for the mask shapes and the arguments of `cx` (cx.in null: the user masks are taken as needing no f32
// copy): checks, the static layout, the mask summaries, the cached plan or a new one.
prepared prepare(device::impl & d, graph_obj & g, uint32_t T, const run_ctx & cx) {
    check_T(d, g, T, g.has_slots ? cx.pos0 : 0u);
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
    p.masks = build_masks(d, g, T, p.lay, cx);
    std::vector<uint8_t> sig(p.masks.size(), 0);
    for (size_t s = 0; s < p.masks.size(); ++s) sig[s] = p.masks[s] && p.masks[s]->need_f32;
    std::vector<uint64_t> kvoff;
    std::vector<uint32_t> nkv;
    slot_key(g, T, cx, &kvoff, &nkv);
    for (auto it = g.plans.begin(); it != g.plans.end(); ++it) {
        if ((*it)->T == T && (*it)->sig == sig && (*it)->kvoff == kvoff && (*it)->nkv == nkv) {
            g.plans.splice(g.plans.begin(), g.plans, it);
            p.plan = g.plans.front();
            d.cnt.graph_plan_hits++;
            return p;
        }
    }
    p.plan = build_plan(d, g, T, p.lay, sig, cx);
    d.cnt.graph_plan_misses++;
    g.plans.push_front(p.plan);
    while (g.plans.size() > graph_plan_cache) g.plans.pop_back();
    return p;
}

using sink_fn = std::function<void(uint32_t, uint64_t, const void *, size_t)>;

// Gathers the writes of a run whose byte ranges touch into one device write. A write of kBig bytes or more is not copied: it goes
// straight to the device after the span. No gap is ever written.
struct write_sink {
    static constexpr size_t kBig = 64u * 1024u;
    device::impl & d;
    std::vector<uint8_t> span;
    uint32_t ch = 0;
    uint64_t at = 0;                           // device offset of span[0]
    explicit write_sink(device::impl & dd) : d(dd) {}
    void flush() {
        if (!span.empty()) d.write(ch, at, span.data(), span.size());
        span.clear();
    }
    void put(uint32_t c, uint64_t off, const void * src, size_t n) {
        if (n == 0) return;
        if (n >= kBig) {
            flush();
            d.write(c, off, src, n);
            return;
        }
        if (!span.empty() && (c != ch || off != at + span.size())) flush();
        if (span.empty()) {
            ch = c;
            at = off;
        }
        const uint8_t * b = (const uint8_t *) src;
        span.insert(span.end(), b, b + n);
    }
};

// The inputs, the positions, the KV cell indices and the mask summaries of a run of T rows (the chunk of rows cx.r0 .. of the call,
// the cells cx.pos0 ..).
void write_inputs(const graph_obj & g, const plan_t & pl, const prepared & p, uint32_t T, const run_ctx & cx, const graph_inputs & in,
                  const sink_fn & put) {
    const graph_desc & desc = g.desc;
    std::vector<uint8_t> buf;
    // in address order: the positions, the cell indices and the masks (the static layout), then the in rows (the dynamic area)
    for (size_t ai = 0; ai < g.arg_value.size(); ++ai) {
        const size_t v = (size_t) g.arg_value[ai];
        if (desc.values[v].kind != gv::pos) continue;
        const graph_arg & a = in.args[ai];
        buf.assign((size_t) pad64((uint64_t) T * 4), 0);
        memcpy(buf.data(), (const int32_t *) a.data + cx.r0, (size_t) T * 4);
        put(0, p.lay.pos_off[v], buf.data(), buf.size());
    }
    auto indices = [&](uint64_t off, uint64_t first) {
        buf.assign((size_t) pad64((uint64_t) T * opl::PRE_KVIDX_BYTES), 0);
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t c = first + t;
            memcpy(buf.data() + (size_t) t * 8, &c, 8);
        }
        put(0, off, buf.data(), buf.size());
    };
    if (g.has_unslotted) indices(p.lay.idx_off, 0);
    if (g.has_slots) indices(p.lay.idx_kv_off, cx.pos0);
    for (size_t s = 0; s < p.masks.size(); ++s) {
        const tsr_k_fa_mask & m = *p.masks[s];
        buf.assign((size_t) pad64(opl::msum_bytes(T)), 0);
        memcpy(buf.data(), m.msum.data(), m.msum.size() * 4);
        put(0, p.lay.slots[s].msum_off, buf.data(), buf.size());
        if (m.need_f32) put(0, p.lay.slots[s].f32_off, m.f32.data(), m.f32.size() * 4);
    }
    for (size_t ai = 0; ai < g.arg_value.size(); ++ai) {
        const size_t v = (size_t) g.arg_value[ai];
        const graph_value & x = desc.values[v];
        if (x.kind != gv::in) continue;
        const graph_arg & a = in.args[ai];
        const size_t st = a.stride != 0 ? a.stride : x.n;
        const uint64_t row = (uint64_t) L(x.n) * LINE;
        buf.assign((size_t) T * row, 0);
        for (uint32_t t = 0; t < T; ++t) memcpy(buf.data() + t * row, (const float *) a.data + (size_t) (cx.r0 + t) * st, (size_t) x.n * 4);
        put(0, pl.off[v], buf.data(), buf.size());
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
        if (x.kind == gv::mask) {
            const uint32_t cols = x.cols != 0 ? x.cols : T;
            if (in.args[ai].stride != 0 && in.args[ai].stride < cols) {
                throw std::invalid_argument(strf("tsr_rt: graph: argument %zu: row stride %zu < %s = %u", ai, in.args[ai].stride,
                                                 x.cols != 0 ? "cols" : "T", cols));
            }
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

void copy_out(const graph_obj & g, uint32_t T, uint32_t r0, size_t i, const uint8_t * rows, const graph_outputs & out) {
    const graph_value & x = g.desc.values[g.out_value[i]];
    const uint64_t row = (uint64_t) L(x.n) * LINE;
    const size_t st = out.outs[i].stride != 0 ? out.outs[i].stride : x.n;
    for (uint32_t t = 0; t < T; ++t) memcpy(out.outs[i].data + (size_t) (r0 + t) * st, rows + t * row, (size_t) x.n * 4);
}

std::function<std::string(int32_t)> namer(const graph_desc & desc, const list_t & l) {
    return [&desc, &l](int32_t rec) {
        const int32_t op = rec >= 0 && (size_t) rec < l.tags.size() ? l.tags[(size_t) rec] : -1;
        return op >= 0 ? op_name(desc, (uint32_t) op) : std::string("JOIN");
    };
}

// The given masks of the cached attn ops over all T rows of a call: every value 0 or -inf, and no row shows a cell at or above the
// end of its chunk (pos0 + the end row of the chunk of c rows that holds it): the call does not write that cell, so the host cache
// and the planes may differ there.
void check_masks(const graph_obj & g, uint32_t T, uint32_t pos0, uint32_t c, const graph_inputs & in) {
    std::set<int32_t> done;
    for (size_t s = 0; s < g.slot_keys.size(); ++s) {
        const mask_key & k = g.slot_keys[s];
        if (k.cells == 0 || k.mask < 0 || !done.insert(k.mask).second) continue;
        uint32_t op_i = 0;
        for (uint32_t i = 0; i < g.desc.ops.size(); ++i) {
            if (g.attn_slot[i] == (int32_t) s) { op_i = i; break; }
        }
        size_t ai = 0;
        while (g.arg_value[ai] != k.mask) ++ai;
        const graph_arg & a = in.args[ai];
        const float * mp = (const float *) a.data;
        const size_t st = a.stride != 0 ? a.stride : k.cells;
        for (uint32_t r = 0; r < T; ++r) {
            const uint32_t chunk_end = std::min(T, (r / c + 1u) * c);
            const uint64_t end = (uint64_t) pos0 + chunk_end;
            for (uint32_t j = 0; j < k.cells; ++j) {
                const float x = mp[(size_t) r * st + j];
                if (x == 0.0f) {
                    if (j >= end) {
                        no_fit("%s: mask row %u shows cell %u, which is at or above %" PRIu64 " (pos0 %u + %u rows%s): a cell this call does not "
                               "write, where the planes and the host cache may differ, so it does not fit",
                               op_name(g.desc, op_i).c_str(), r, j, end, pos0, chunk_end, chunk_end == T ? "" : ", the end of the row's chunk");
                    }
                } else if (!(std::isinf(x) && x < 0.0f)) {
                    no_fit("%s: mask value %g at [%u, %u] does not fit the device's mask form (every value must be 0 or -inf)",
                           op_name(g.desc, op_i).c_str(), (double) x, r, j);
                }
            }
        }
    }
}

// The KV objects of a run, one per slot of the graph.
std::vector<const kvobj *> bind_state(const device::impl & d, const graph_obj & g, const graph_state & st) {
    const size_t ns = g.desc.kv_slots.size();
    std::vector<const kvobj *> res;
    if (ns == 0) {
        if (!st.kv.empty()) throw std::invalid_argument(strf("tsr_rt: graph: %zu KV handles given, the graph has no KV slots", st.kv.size()));
        return res;
    }
    if (st.kv.size() != ns) throw std::invalid_argument(strf("tsr_rt: graph: %zu KV handles given, the graph has %zu KV slots", st.kv.size(), ns));
    for (size_t s = 0; s < ns; ++s) {
        const kvobj & o = find_kv(d, st.kv[s]);
        const tsr_k_kv & geom = g.slot_geom[s];
        if (o.E != geom.E || o.cells != g.desc.kv_slots[s]) {
            throw std::invalid_argument(strf("tsr_rt: graph: %s: the KV object %" PRIu64 " has E = %u and %u cells, the KV slot %zu needs E = %u and %u cells",
                                             op_name(g.desc, g.slot_op[s]).c_str(), st.kv[s], o.E, o.cells, s, geom.E, g.desc.kv_slots[s]));
        }
        for (size_t t = 0; t < s; ++t) {
            if (st.kv[t] == st.kv[s]) {
                throw std::invalid_argument(strf("tsr_rt: graph: the KV object %" PRIu64 " is bound to the slots %zu and %zu", st.kv[s], t, s));
            }
        }
        res.push_back(&o);
    }
    return res;
}

// graph_run (ref = false) and graph_run_ref (ref = true).
void run_impl(device::impl & d, graph_handle h, uint32_t T, const graph_inputs & in, const graph_outputs & out, const graph_state & st, bool ref) {
    const auto gp = find_graph(d, h);
    graph_obj & g = *gp;
    check_io(g, T, in, out);
    const std::vector<const kvobj *> bound = bind_state(d, g, st);
    run_ctx cx;
    cx.in = &in;
    if (g.has_slots) {
        for (const kvobj * o : bound) {
            cx.k.push_back(o->k_off);
            cx.v.push_back(o->v_off);
        }
        g.bound = true;
        g.last_k = cx.k;
        g.last_v = cx.v;
        g.last_pos0 = st.pos0;
        check_cells(g, T, st.pos0);
    }
    const uint32_t pos0 = g.has_slots ? st.pos0 : 0u;
    // The chunk: a graph whose attn ops all have a slot is cut into the largest chunk that plans; any other graph runs in one piece.
    const bool chunked = g.all_cached;
    uint32_t c = chunked ? std::min(T, graph_max_attn_tokens) : T;
    prepared first;
    for (;;) {
        if (g.has_slots) check_masks(g, T, pos0, c, in);
        cx.r0 = 0;
        cx.pos0 = pos0;
        try {
            first = prepare(d, g, c, cx);
            break;
        } catch (const no_fit_error &) {
            if (!chunked || c <= 1) throw;
            c = (c + 1) / 2;
        }
    }
    std::unique_ptr<host_image> img;
    if (ref) {
        img.reset(new host_image(d.cfg.num_ch, d.cfg.ch_bytes));
        // the constants and the weights are read from the device, and for a cached attention the cells [0, pos0 + T) of the planes
        // (the lists write the cells of the run again); the other bytes of the image stay zero
        for (const dev_range & r : first.plan->reads) read_big(d, r.ch, r.off, img->ch[r.ch] + r.off, r.bytes);
        for (size_t s = 0; s < bound.size(); ++s) {
            for (int w = 0; w < 2; ++w) {
                const uint64_t base = w ? bound[s]->v_off : bound[s]->k_off;
                kv_ranges(bound[s]->geom, w != 0, 0, pos0 + T, [&](uint64_t rel, uint64_t bytes) {
                    for (uint32_t ch = 0; ch < d.cfg.num_ch; ++ch) read_big(d, ch, base + rel, img->ch[ch] + base + rel, bytes);
                });
            }
        }
    }
    tsr::ref::opl_exec_cfg xc;
    xc.t_max = d.cfg.t_max;
    xc.pcap = d.pcap;
    xc.k_max = d.cfg.k_max;
    xc.mm_threads = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    for (uint32_t r0 = 0; r0 < T;) {
        const uint32_t tc = std::min(c, T - r0);
        cx.r0 = r0;
        cx.pos0 = pos0 + r0;
        const prepared cur = r0 == 0 ? std::move(first) : prepare(d, g, tc, cx);
        const plan_t & pl = *cur.plan;
        if (!ref) {
            write_sink ws(d);
            write_inputs(g, pl, cur, tc, cx, in, [&ws](uint32_t ch, uint64_t off, const void * src, size_t n) { ws.put(ch, off, src, n); });
            ws.flush();
            for (const list_t & l : pl.lists) {
                d.run_blob(l.blob, l.n_ops, l.n_joins, namer(g.desc, l));
                d.cnt.graph_launches++;
                d.cnt.graph_records += l.n_ops;
            }
            // the out values whose bytes touch are read in one transfer; a gap is never read
            std::vector<uint8_t> buf;
            auto bytes_of = [&](size_t i) { return (uint64_t) tc * L(g.desc.values[g.out_value[i]].n) * LINE; };
            for (size_t i = 0; i < g.out_value.size();) {
                const uint64_t base = pl.off[g.out_value[i]];
                uint64_t span = bytes_of(i);
                size_t j = i + 1;
                while (j < g.out_value.size() && pl.off[g.out_value[j]] == base + span) span += bytes_of(j++);
                buf.resize((size_t) span);
                d.read(0, base, buf.data(), buf.size());
                for (size_t k = i; k < j; ++k) copy_out(g, tc, r0, k, buf.data() + (pl.off[g.out_value[k]] - base), out);
                i = j;
            }
        } else {
            write_inputs(g, pl, cur, tc, cx, in, [&](uint32_t ch, uint64_t off, const void * src, size_t n) {
                if (off + n > d.cfg.ch_bytes) throw error("tsr_rt: graph_run_ref: input outside the DRAM image");
                memcpy(img->ch[ch] + off, src, n);
            });
            const tsr::ref::dram_view view{ img->ch.data(), d.cfg.num_ch, d.cfg.ch_bytes };
            for (size_t li = 0; li < pl.lists.size(); ++li) {
                const list_t & l = pl.lists[li];
                std::string why;
                if (tsr::ref::opl_exec(l.blob, view, xc, &why) != 0) {
                    unsigned rec = 0;
                    std::string name;
                    if (sscanf(why.c_str(), "op list record %u", &rec) == 1) name = namer(g.desc, l)((int32_t) rec);
                    throw error(strf("tsr_rt: graph_run_ref: %s%s%s", name.empty() ? "" : name.c_str(), name.empty() ? "" : ": ", why.c_str()));
                }
            }
            for (size_t i = 0; i < g.out_value.size(); ++i) copy_out(g, tc, r0, i, img->ch[0] + pl.off[g.out_value[i]], out);
        }
        r0 += tc;
    }
    if (!ref) d.cnt.graph_runs++;
}

}  // namespace

void device::graph_free(graph_handle h) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    auto g = find_graph(d, h);
    for (size_t a = 0; a < g->attn_ops.size(); ++a) {
        if (g->desc.ops[g->attn_ops[a]].kv >= 0) continue;
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
        s.arg_n.push_back(x.kind == gv::in ? x.n : x.kind == gv::mask ? x.cols : 0u);
    }
    for (uint32_t v : g->out_value) s.out_n.push_back(g->desc.values[v].n);
    s.kv_size = g->desc.kv_size;
    s.kv_bytes = g->kv_bytes;
    s.kv_slots = g->desc.kv_slots;
    return s;
}

void device::graph_run(graph_handle h, uint32_t T, const graph_inputs & in, const graph_outputs & out, const graph_state & state) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    run_impl(*p_, h, T, in, out, state, false);
}

void device::graph_run_ref(graph_handle h, uint32_t T, const graph_inputs & in, const graph_outputs & out, const graph_state & state) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    run_impl(*p_, h, T, in, out, state, true);
}

graph_plan_info device::graph_info(graph_handle h, uint32_t T) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const auto g = find_graph(d, h);
    run_ctx cx = fake_ctx(*g);
    if (g->bound) {
        cx.k = g->last_k;
        cx.v = g->last_v;
    }
    if (!g->all_cached) {
        cx.pos0 = g->has_slots ? g->last_pos0 : 0u;
        return prepare(d, *g, T, cx).plan->info;
    }
    // A graph that runs in chunks: the plan the first chunk of a run of T rows starts with, found as run_impl finds it. The
    // plan does not depend on pos0.
    if (T == 0) throw std::invalid_argument("tsr_rt: graph: T must be >= 1");
    check_cells(*g, T, 0);
    uint32_t c = std::min(T, graph_max_attn_tokens);
    for (;;) {
        try {
            return prepare(d, *g, c, cx).plan->info;
        } catch (const no_fit_error &) {
            if (c <= 1) throw;
            c = (c + 1) / 2;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// device: KV objects
// ---------------------------------------------------------------------------------------------
kv_handle device::kv_create(uint32_t E, uint32_t cells) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    if (E == 0 || E % 64u != 0 || E / 64u > 511u) {
        throw std::invalid_argument(strf("tsr_rt: kv_create: E = %u is not a multiple of 64 up to 511 * 64", E));
    }
    if (cells == 0) throw std::invalid_argument("tsr_rt: kv_create: cells must be >= 1");
    kvobj o;
    o.E = E;
    o.cells = cells;
    std::string why;
    if (!tsr_k_kv_geom(E, cells, d.cfg.num_ch, &o.geom, &why)) {
        throw error(strf("tsr_rt: kv_create: E %u, %u cells: %s: does not fit", E, cells, why.c_str()));
    }
    std::vector<uint64_t> got;
    for (int w = 0; w < 2; ++w) {
        uint64_t off = 0;
        if (!d.heap.alloc_top(o.geom.bytes, &off)) {
            for (uint64_t x : got) d.heap.release(x);
            throw error(strf("tsr_rt: kv_create: the KV planes (E %u, %u cells) need %.1f MiB per channel: does not fit the WEIGHT heap (%.1f MiB "
                             "free, largest %.1f MiB)", E, cells, mib(2 * o.geom.bytes), mib(d.heap.free_bytes()), mib(d.heap.largest_free())));
        }
        got.push_back(off);
    }
    o.k_off = got[0];
    o.v_off = got[1];
    try {
        zero_planes(d, got, o.geom.bytes);
    } catch (...) {
        for (uint64_t x : got) d.heap.release(x);
        throw;
    }
    const kv_handle h = d.next_kv++;
    d.kvs.emplace(h, o);
    d.cnt.kv_creates++;
    TSR_RT_DBG(1, "kv #%" PRIu64 ": E %u, %u cells, %.2f MiB per channel\n", h, E, cells, mib(2 * o.geom.bytes));
    return h;
}

void device::kv_free(kv_handle h) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const kvobj & o = find_kv(d, h);
    d.heap.release(o.k_off);
    d.heap.release(o.v_off);
    d.kvs.erase(h);
}

kv_info device::kv_info_of(kv_handle h) const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    const kvobj & o = find_kv(*p_, h);
    kv_info i;
    i.E = o.E;
    i.cells = o.cells;
    i.bytes = 2 * o.geom.bytes;
    i.k_off = o.k_off;
    i.v_off = o.v_off;
    return i;
}

void device::kv_store(kv_handle h, uint32_t cell0, uint32_t n, const float * k, const float * v, size_t k_stride, size_t v_stride) {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const kvobj & o = find_kv(d, h);
    if (n != 0 && (k == nullptr || v == nullptr)) throw std::invalid_argument("tsr_rt: kv_store: null rows");
    if ((uint64_t) cell0 + n > o.cells) {
        if (n == 0) throw std::invalid_argument(strf("tsr_rt: kv_store: cell %u is beyond the %u cells of the KV object", cell0, o.cells));
        throw std::invalid_argument(strf("tsr_rt: kv_store: cells %u .. %" PRIu64 " are beyond the %u cells of the KV object", cell0,
                                         (uint64_t) cell0 + n - 1, o.cells));
    }
    if (n == 0) return;
    const size_t ks = k_stride != 0 ? k_stride : o.E, vs = v_stride != 0 ? v_stride : o.E;
    if (ks < o.E || vs < o.E) throw std::invalid_argument("tsr_rt: kv_store: row stride < E");
    const uint32_t EL = L(o.E);
    // rows per list: the K rows, the V rows and the cell indices share the IO region
    uint32_t c = std::min(n, graph_max_attn_tokens);
    auto need = [&](uint32_t rows) { return 2ull * rows * EL * LINE + pad64((uint64_t) rows * opl::PRE_KVIDX_BYTES); };
    while (c > 1 && need(c) > d.io_size) c = (c + 1) / 2;
    if (need(c) > d.io_size) no_fit("kv_store: a row of E = %u does not fit the IO region (%.1f MiB)", o.E, mib(d.io_size));
    std::vector<uint8_t> buf;
    for (uint32_t r0 = 0; r0 < n; r0 += c) {
        const uint32_t tc = std::min(c, n - r0);
        const uint64_t k_at = d.io_off, v_at = k_at + (uint64_t) tc * EL * LINE, idx_at = v_at + (uint64_t) tc * EL * LINE;
        const uint64_t row = (uint64_t) EL * LINE;
        for (int w = 0; w < 2; ++w) {
            const float * src = w ? v : k;
            const size_t st = w ? vs : ks;
            buf.assign((size_t) tc * row, 0);
            for (uint32_t t = 0; t < tc; ++t) memcpy(buf.data() + t * row, src + (size_t) (r0 + t) * st, (size_t) o.E * 4);
            d.write(0, w ? v_at : k_at, buf.data(), buf.size());
        }
        buf.assign((size_t) pad64((uint64_t) tc * opl::PRE_KVIDX_BYTES), 0);
        for (uint32_t t = 0; t < tc; ++t) {
            const uint64_t cell = (uint64_t) cell0 + r0 + t;
            memcpy(buf.data() + (size_t) t * 8, &cell, 8);
        }
        d.write(0, idx_at, buf.data(), buf.size());
        tsr_opl_builder b(d.cfg.num_ch, opl::F_STOP_ON_ERROR);
        for (int w = 0; w < 2; ++w) {
            tsr_kd_set_rows s;
            s.src = { H0(w ? v_at : k_at), EL };
            s.n_rows = tc;
            s.kv = o.geom;
            s.which = w ? opl::setrows::WHICH_V : opl::setrows::WHICH_K;
            s.plane = (uint32_t) ((w ? o.v_off : o.k_off) / TSR_BEAT_BYTES);
            s.idx_src = H0(idx_at);
            tsr_rec r;
            std::string why;
            if (!tsr_k_set_rows_kv(d.kenv, s, &r, &why)) throw error(strf("tsr_rt: kv_store: SET_ROWS_KV %c: %s", w ? 'V' : 'K', why.c_str()));
            const uint32_t idx = tsr_k_emit(b, r, w, nullptr, false);
            if (idx == tsr_opl_builder::NO_ROOM || idx == tsr_opl_builder::FAILED) throw error(strf("tsr_rt: kv_store: op list: %s", b.error()));
        }
        if (!b.settle()) throw error(strf("tsr_rt: kv_store: op list: %s", b.error()));
        const std::vector<uint8_t> blob = b.finish();
        if (blob.empty()) throw error(strf("tsr_rt: kv_store: op list build failed: %s", b.error()));
        d.run_blob(blob, b.n_ops(), b.n_joins(), [](int32_t rec) { return std::string(rec == 0 ? "kv_store SET_ROWS_KV K" : rec == 1 ? "kv_store SET_ROWS_KV V" : "JOIN"); });
    }
    d.cnt.kv_stores++;
    d.cnt.kv_store_rows += n;
}

void device::kv_read(kv_handle h, uint32_t cell0, uint32_t n, void * k_blocks, void * v_blocks) const {
    std::lock_guard<std::mutex> lk(p_->mtx);
    impl & d = *p_;
    const kvobj & o = find_kv(d, h);
    if (n != 0 && (k_blocks == nullptr || v_blocks == nullptr)) throw std::invalid_argument("tsr_rt: kv_read: null destination");
    if ((uint64_t) cell0 + n > o.cells) {
        if (n == 0) throw std::invalid_argument(strf("tsr_rt: kv_read: cell %u is beyond the %u cells of the KV object", cell0, o.cells));
        throw std::invalid_argument(strf("tsr_rt: kv_read: cells %u .. %" PRIu64 " are beyond the %u cells of the KV object", cell0,
                                         (uint64_t) cell0 + n - 1, o.cells));
    }
    if (n == 0) return;
    const tsr_kv_geom geo = { o.E, d.cfg.num_ch, o.geom.LG };
    const size_t row_bytes = (size_t) (o.E / 32u) * sizeof(tsr_block_q8_0);
    std::vector<tsr_block_q8_0> row(o.E / 32u);
    for (int w = 0; w < 2; ++w) {
        std::vector<std::vector<uint8_t>> img(d.cfg.num_ch, std::vector<uint8_t>((size_t) o.geom.bytes, 0));
        std::vector<const uint8_t *> pl(d.cfg.num_ch);
        const uint64_t base = w ? o.v_off : o.k_off;
        kv_ranges(o.geom, w != 0, cell0, cell0 + n, [&](uint64_t rel, uint64_t bytes) {
            for (uint32_t ch = 0; ch < d.cfg.num_ch; ++ch) read_big(d, ch, base + rel, img[ch].data() + rel, bytes);
        });
        for (uint32_t ch = 0; ch < d.cfg.num_ch; ++ch) pl[ch] = img[ch].data();
        uint8_t * dst = (uint8_t *) (w ? v_blocks : k_blocks);
        for (uint32_t t = 0; t < n; ++t) {
            tsr_kv_unpack_row(w ? TSR_KV_V : TSR_KV_K, &geo, pl.data(), cell0 + t, row.data());
            memcpy(dst + (size_t) t * row_bytes, row.data(), row_bytes);
        }
    }
}

}  // namespace tsr::rt
