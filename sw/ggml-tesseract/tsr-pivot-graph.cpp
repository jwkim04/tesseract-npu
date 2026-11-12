// tsr-pivot-graph.cpp — see tsr-pivot-graph.h
#include "tsr-pivot-graph.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <unordered_map>
#include <unordered_set>

// struct ggml_cgraph and ggml_node_has_n_uses; the header's static helpers are not all used here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "ggml-impl.h"
#pragma GCC diagnostic pop
#include "tsr/plan.hpp"

#include "tsr-build.h"
#include "tsr-weights.h"

namespace {

namespace C = tsr_pivot_costs;

// A node that makes a record: not a no-op and with the COMPUTE flag. Emptiness is not looked at: whether a node has rows depends on
// the batch, not on the graph's structure (llama.cpp's build_rs zeroes the cell of a new sequence with an in-place SCALE of a
// view that has no rows when the sequence goes on), and the units, hence the permutation, must be the same for every batch.
bool is_compute(const ggml_tensor * n) {
    return (n->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 && !tsr_node_is_noop(n);
}

// State plumbing: an in-place SCALE or a CPY into a view of a leaf (the recurrent-state caches: zero a cell, copy the gathered
// extra states). The builder lowers them to no record at all or into the record of the node that uses the state, and they may have no
// rows, so the model counts them as free, with or without rows.
bool is_plumbing(const ggml_tensor * n) {
    return (n->op == GGML_OP_SCALE || n->op == GGML_OP_CPY) && n->view_src != nullptr && n->view_src->op == GGML_OP_NONE;
}

const ggml_tensor * base_of(const ggml_tensor * t) {
    return t->view_src != nullptr ? t->view_src : t;
}

// The first dimension of a node's rows, for the glue cost: the feature axis. The CONCAT of an SSM_CONV group joins the conv state
// to the token rows along the time axis, so its first dimension grows with the token count; its rows are the ne[1] channels.
int64_t row_elems(const ggml_tensor * n) {
    if (n->op == GGML_OP_CONCAT && n->src[1] != nullptr && n->src[1]->op == GGML_OP_TRANSPOSE) {
        return n->ne[1];
    }
    return n->ne[0];
}

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// Order constraints: for every base tensor that a compute node writes through a view or in place, the compute nodes that touch
// it (read it through a source, or write it) in their original order. `nodes` are the graph's nodes in the original order.
std::vector<std::vector<int>> order_chains(const std::vector<ggml_tensor *> & nodes) {
    const int n = (int) nodes.size();
    std::unordered_set<const ggml_tensor *> written;
    for (int i = 0; i < n; ++i) {
        if (is_compute(nodes[i]) && nodes[i]->view_src != nullptr) {
            written.insert(nodes[i]->view_src);
        }
    }
    std::vector<std::vector<int>> chains;
    if (written.empty()) {
        return chains;
    }
    std::unordered_map<const ggml_tensor *, size_t> chain_of;
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * x = nodes[i];
        if (!is_compute(x)) continue;
        const ggml_tensor * touched[GGML_MAX_SRC + 1];
        int nt = 0;
        touched[nt++] = base_of(x);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (x->src[s] != nullptr) touched[nt++] = base_of(x->src[s]);
        }
        for (int t = 0; t < nt; ++t) {
            if (written.count(touched[t]) == 0) continue;
            auto it = chain_of.find(touched[t]);
            if (it == chain_of.end()) {
                it = chain_of.emplace(touched[t], chains.size()).first;
                chains.emplace_back();
            }
            std::vector<int> & c = chains[it->second];
            if (c.empty() || c.back() != i) c.push_back(i);
        }
    }
    return chains;
}

// The cost of one weight MUL_MAT / MUL_MAT_ID tile set: beats per channel of a 2D weight of N rows, K columns.
bool weight_beats(const ggml_tensor * w, uint32_t num_ch, int64_t * beats) {
    if (w == nullptr || w->ne[0] <= 0 || w->ne[1] <= 0 || w->ne[0] > UINT32_MAX || w->ne[1] > UINT32_MAX) return false;
    const uint32_t fmt = tsr_dev_fmt_for_type(w->type);
    const uint32_t rb = tsr::plan::row_beats(fmt, (uint32_t) w->ne[0]);
    if (rb == 0) return false;
    *beats = ceil_div(w->ne[1], num_ch) * (int64_t) rb;
    return true;
}

int64_t mvm_cycles(int64_t beats, const tsr_pivot_dev & dev) {
    const int64_t rate = dev.dual ? C::RATE_PCT_DUAL : C::RATE_PCT_SINGLE;
    return std::max<int64_t>(1, beats * 100 / rate);
}

// The time of one head of a GATED_DELTA_NET (VU §7.12.8: a head that waits for the previous one) plus the margin of the table.
int64_t gdn_head_cycles(int64_t S) {
    return (S + 1) * S / 8 + ceil_div(19 * S, 16) + 9 * S / 2 + 122 + C::GDN_HEAD_EXTRA;
}

// A sequential 64-bit hash for the structure key: the finalizer of splitmix64 over the running state. It depends on the order of
// the values and is the same on every host.
struct key_hash {
    uint64_t h = 0x9e3779b97f4a7c15ull;
    void add(uint64_t v) {
        uint64_t x = h ^ v;
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebull;
        x ^= x >> 31;
        h = x + 0x9e3779b97f4a7c15ull;
    }
};

// The memo's key: the structure, and what else the search depends on (the device's costs, the search parameters).
uint64_t memo_key(uint64_t structure, const tsr_pivot_dev & dev, const tsr_pivot_params & p) {
    key_hash k;
    k.add(structure);
    k.add(((uint64_t) dev.num_ch << 1) | (dev.dual ? 1 : 0));
    k.add(p.window);
    k.add(p.max_moved);
    k.add((uint64_t) p.threshold);
    k.add(p.horizon);
    return k.h;
}

std::string unit_name(const std::vector<ggml_tensor *> & nodes, const std::vector<int> & members) {
    const ggml_tensor * first = nullptr;
    int n_compute = 0;
    for (int i : members) {
        if (is_compute(nodes[i])) {
            if (first == nullptr) first = nodes[i];
            n_compute++;
        }
    }
    std::string s;
    if (first != nullptr) {
        s = first->name[0] != '\0' ? first->name : ggml_op_name(first->op);
        if (n_compute > 1) s += " (+" + std::to_string(n_compute - 1) + ")";
    }
    return s;
}

}   // namespace

bool tsr_pivot_analyze(const ggml_cgraph * cg, const tsr_pivot_dev & dev, tsr_pivot_graph * g) {
    *g = tsr_pivot_graph();
    const int n = cg->n_nodes;
    std::vector<ggml_tensor *> nodes(cg->nodes, cg->nodes + std::max(n, 0));
    std::vector<char> comp((size_t) n, 0);
    std::vector<int> comp_nodes;                                  // node indices of the compute nodes, in order
    for (int i = 0; i < n; ++i) {
        comp[i] = is_compute(nodes[i]);
        if (comp[i]) comp_nodes.push_back(i);
    }
    if (comp_nodes.empty()) {
        return false;
    }
    std::unordered_map<const ggml_tensor *, int> idx;
    idx.reserve((size_t) n * 2);
    for (int i = 0; i < n; ++i) idx[nodes[i]] = i;

    // ---- the groups the builder fuses or lowers without a hazard check, and the nodes whose work another node's record carries
    std::vector<char> absorbed((size_t) n, 0);
    std::vector<char> epilogue((size_t) n, 0);                    // a MUL_MAT whose SCALE / softcap partner may fuse
    std::vector<int32_t> router_experts((size_t) n, 0);           // a SOFT_MAX that heads a router chain: its expert count
    std::vector<std::pair<int, int>> spans;
    std::vector<std::pair<int, int>> before;                      // a before b, as the builder requires it
    for (int i = 0; i < n; ++i) {
        if (!comp[i]) continue;
        const ggml_tensor * node = nodes[i];
        if (is_plumbing(node)) absorbed[i] = 1;
        switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_RMS_NORM: {
            if (!ggml_node_has_n_uses(cg, i, 1)) break;
            const int k = tsr_fusion_user(cg, i);
            if (k < 0 || !comp[k] || !tsr_fuse_pair_shape(node, nodes[k])) break;
            int hi = k;
            absorbed[k] = 1;
            if (node->op == GGML_OP_MUL_MAT && nodes[k]->op == GGML_OP_SCALE) {
                epilogue[i] = 1;
                int mid = -1, tail = -1;
                if (tsr_softcap_shape(cg, k, &mid, &tail)) {
                    hi = std::max(hi, tail);
                    absorbed[mid] = absorbed[tail] = 1;
                }
            }
            spans.push_back({ i, hi });
            break;
        }
        case GGML_OP_SOFT_MAX: {
            tsr_moe_route r;
            if (!tsr_moe_route_match(cg, i, &r)) break;
            int hi = i;
            for (int m : { r.argsort, r.get_rows, r.sum_rows, r.clamp, r.div }) {
                absorbed[m] = 1;
                hi = std::max(hi, m);
            }
            router_experts[i] = (int32_t) node->ne[0];
            spans.push_back({ i, hi });
            break;
        }
        case GGML_OP_MUL_MAT_ID: {
            tsr_moe_scale sc;
            if (!tsr_moe_scale_match(cg, i, &sc)) break;
            int lo = i, hi = i;
            for (int m : { sc.repeat, sc.get_rows, sc.mul }) {
                absorbed[m] = 1;
                lo = std::min(lo, m);
                hi = std::max(hi, m);
            }
            spans.push_back({ lo, hi });
            break;
        }
        case GGML_OP_SSM_CONV: {
            tsr_conv_shape cs;
            if (!tsr_conv_group_shape(cg, i, &cs)) break;
            for (int k : cs.cpys) {
                absorbed[k] = 1;
                before.push_back({ k, i });
            }
            if (cs.silu >= 0) {
                absorbed[cs.silu] = 1;
                spans.push_back({ i, cs.silu });
            }
            break;
        }
        case GGML_OP_GATED_DELTA_NET: {
            // The GET_ROWS that gathers the state operand is plumbing whether or not the builder absorbs it into the record: that
            // depends on the batch (the number of sequences, the snapshot slots, extra states), and the cost must not.
            const ggml_tensor * t = node->src[5];
            while (t != nullptr && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) t = t->src[0];
            if (t != nullptr && t->op == GGML_OP_GET_ROWS) {
                auto it = idx.find(t);
                if (it != idx.end() && comp[it->second]) absorbed[it->second] = 1;
            }
            tsr_gdn_shape sh;
            if (!tsr_gdn_state_shape(cg, i, &sh)) break;
            absorbed[sh.gather] = absorbed[sh.cpy] = 1;
            break;
        }
        default:
            break;
        }
    }

    // ---- units: the compute nodes in runs joined by the spans; a no-op joins the next compute node's unit
    std::vector<char> join(comp_nodes.size(), 0);                 // join[r]: compute node r and r + 1 are one unit
    for (const auto & sp : spans) {
        const size_t a = std::lower_bound(comp_nodes.begin(), comp_nodes.end(), sp.first) - comp_nodes.begin();
        const size_t b = std::upper_bound(comp_nodes.begin(), comp_nodes.end(), sp.second) - comp_nodes.begin();   // one past the last
        for (size_t r = a; r + 1 < b; ++r) join[r] = 1;
        g->groups.push_back(sp);
    }
    std::vector<int> unit_of_rank(comp_nodes.size(), 0);
    int n_units = 0;
    for (size_t r = 0; r < comp_nodes.size(); ++r) {
        unit_of_rank[r] = n_units;
        if (!join[r]) n_units++;
    }
    g->unit_of.assign((size_t) n, 0);
    g->unit_nodes.assign((size_t) n_units, {});
    {
        size_t r = 0;                                              // the next compute node at or after node i
        for (int i = 0; i < n; ++i) {
            while (r < comp_nodes.size() && comp_nodes[r] < i) r++;
            const int u = r < comp_nodes.size() ? unit_of_rank[r] : n_units - 1;
            g->unit_of[i] = u;
            g->unit_nodes[(size_t) u].push_back(i);
        }
    }

    // ---- dependencies: data (every source that is a node of the graph), the builder's order edges, the order constraints
    std::vector<std::vector<uint32_t>> preds((size_t) n_units);
    auto add_edge = [&](int a, int b) -> bool {                    // node a before node b
        const int ua = g->unit_of[a], ub = g->unit_of[b];
        if (ua == ub) return true;
        if (ua > ub) return false;
        preds[(size_t) ub].push_back((uint32_t) ua);
        return true;
    };
    for (int i = 0; i < n; ++i) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (nodes[i]->src[s] == nullptr) continue;
            auto it = idx.find(nodes[i]->src[s]);
            if (it == idx.end()) continue;                         // a leaf, a weight, a node of another split
            if (it->second >= i || !add_edge(it->second, i)) {
                *g = tsr_pivot_graph();                            // not a topological order: leave it alone
                return false;
            }
        }
    }
    for (const auto & e : before) {
        if (!add_edge(e.first, e.second)) { *g = tsr_pivot_graph(); return false; }
    }
    g->chains = order_chains(nodes);
    for (const auto & c : g->chains) {
        for (size_t k = 1; k < c.size(); ++k) {
            if (!add_edge(c[k - 1], c[k])) { *g = tsr_pivot_graph(); return false; }
        }
    }
    g->before = before;
    g->pin_first = nodes[0]->op == GGML_OP_MUL_MAT_ID;
    if (g->pin_first) {
        for (int u = 1; u < n_units; ++u) preds[(size_t) u].push_back(0);
    }

    // ---- costs
    g->units.assign((size_t) n_units, tsr_pivot_unit());
    g->names.assign((size_t) n_units, std::string());
    for (int u = 0; u < n_units; ++u) {
        tsr_pivot_unit & x = g->units[(size_t) u];
        std::sort(preds[(size_t) u].begin(), preds[(size_t) u].end());
        preds[(size_t) u].erase(std::unique(preds[(size_t) u].begin(), preds[(size_t) u].end()), preds[(size_t) u].end());
        x.preds = std::move(preds[(size_t) u]);
        g->names[(size_t) u] = unit_name(nodes, g->unit_nodes[(size_t) u]);

        // the unit's class: the router chain, the recurrence, a MUL_MAT_ID, attention, a weight MUL_MAT, else glue
        const ggml_tensor *route = nullptr, *gdn = nullptr, *mmid = nullptr, *attn = nullptr, *mm = nullptr;
        int mm_node = -1;
        int n_records = 0;
        int64_t glue_body = 0;
        for (int i : g->unit_nodes[(size_t) u]) {
            if (!comp[i]) continue;
            const ggml_tensor * x_n = nodes[i];
            if (router_experts[i] != 0 && route == nullptr) route = x_n;
            if (x_n->op == GGML_OP_GATED_DELTA_NET && gdn == nullptr) gdn = x_n;
            if (x_n->op == GGML_OP_MUL_MAT_ID && mmid == nullptr) mmid = x_n;
            if (x_n->op == GGML_OP_FLASH_ATTN_EXT && attn == nullptr) attn = x_n;
            if (x_n->op == GGML_OP_MUL_MAT && mm == nullptr && x_n->src[0] != nullptr && x_n->src[0]->op == GGML_OP_NONE &&
                x_n->src[0]->ne[2] == 1 && x_n->src[0]->ne[3] == 1) {
                mm = x_n;
                mm_node = i;
            }
            if (!absorbed[i]) {
                n_records++;
                glue_body += C::GLUE_BASE + C::GLUE_PER_16 * (row_elems(x_n) / 16);
            }
        }
        int64_t beats = 0;
        if (route != nullptr) {
            x.cls = TSR_PIVOT_BLOCK;
            x.body = C::ROUTER_BASE + C::ROUTER_PER_EXPERT * route->ne[0];
        } else if (gdn != nullptr && gdn->src[2] != nullptr) {
            x.cls = TSR_PIVOT_BLOCK;
            x.body = C::GDN_BASE + gdn->src[2]->ne[1] * gdn_head_cycles(gdn->src[2]->ne[0]);
        } else if (mmid != nullptr && mmid->src[0] != nullptr && mmid->src[2] != nullptr && weight_beats(mmid->src[0], dev.num_ch, &beats)) {
            const int64_t n_used = std::max<int64_t>(1, mmid->src[2]->ne[0]);
            const int64_t tile = mvm_cycles(beats, dev);
            x.cls = TSR_PIVOT_MVM_ID;
            x.mvm = n_used * tile;
            x.ids = C::MMID_IDS;
            x.tail = std::min(x.mvm, C::MMID_TAIL_TILES * tile);
            x.back = C::MMID_BACK;
        } else if (attn != nullptr) {
            x.cls = TSR_PIVOT_ATTN;
            x.front = C::ATTN_FRONT;
            x.mvm = C::ATTN_MVM;
            x.back = C::ATTN_BACK;
            x.issue = TSR_PIVOT_ISSUE_SHORT;
        } else if (mm != nullptr && weight_beats(mm->src[0], dev.num_ch, &beats)) {
            const int64_t K = mm->src[0]->ne[0], N = mm->src[0]->ne[1];
            x.cls = TSR_PIVOT_MVM;
            x.ld = C::LD_BASE + K / 16;
            x.qact = tsr_dev_fmt_for_type(mm->src[0]->type) == tsr::plan::FMT_Q4_K ? C::QACT_Q4_PER_256 * (K / 256) + C::QACT_Q4_BASE
                                                                                   : C::QACT_Q8_PER_32 * (K / 32) + C::QACT_Q8_BASE;
            x.mvm = mvm_cycles(beats, dev);
            x.back = C::BACK_BASE + N / 16;
            x.defer = epilogue[mm_node] == 0;
            x.issue = TSR_PIVOT_ISSUE_MUL_MAT;
            x.issue_open = TSR_PIVOT_ISSUE_BEGIN;
            x.issue_close = TSR_PIVOT_ISSUE_END;
        } else {
            x.cls = TSR_PIVOT_GLUE;
            x.body = glue_body;
            x.issue = TSR_PIVOT_ISSUE_SHORT * n_records;
        }
    }
    return true;
}

uint64_t tsr_pivot_structure_key(const ggml_cgraph * cg) {
    const int n = std::max(cg->n_nodes, 0);
    std::unordered_map<const ggml_tensor *, int> idx;
    idx.reserve((size_t) n * 2);
    for (int i = 0; i < n; ++i) idx[cg->nodes[i]] = i;
    std::unordered_map<const ggml_tensor *, uint64_t> outside;   // the tensors that are not nodes of this graph, numbered as they come
    key_hash k;
    // a link: 0 (none), 2 + i (node i of this graph), or 1 with the op, the type and the number of a tensor outside this graph
    // (the number says which links reach the same tensor: the order constraints follow the buffers that nodes share)
    auto link = [&](const ggml_tensor * t) {
        if (t == nullptr) {
            k.add(0);
            return;
        }
        auto it = idx.find(t);
        if (it != idx.end()) {
            k.add(2 + (uint64_t) it->second);
            return;
        }
        const uint64_t number = outside.emplace(t, (uint64_t) outside.size()).first->second;
        k.add(1);
        k.add((uint64_t) t->op);
        k.add((uint64_t) t->type);
        k.add(number);
    };
    k.add((uint64_t) n);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * x = cg->nodes[i];
        // Not the COMPUTE flag: llama.cpp selects the branches of a graph by it (ggml_build_forward_select) and ggml-alloc plans a
        // node with or without it, so graphs that differ in the flag alone share a plan and must share an order.
        k.add((uint64_t) x->op);
        k.add(x->op == GGML_OP_UNARY ? 1 + (uint64_t) ggml_get_unary_op(x) : 0);
        k.add((uint64_t) x->type);
        k.add((uint64_t) ggml_node_get_use_count(cg, i));
        link(x->view_src);
        for (int s = 0; s < GGML_MAX_SRC; ++s) link(x->src[s]);
        // the weight of a matrix product: its shape and type set the unit's cost, and they are the model's, not the batch's
        const ggml_tensor * w = (x->op == GGML_OP_MUL_MAT || x->op == GGML_OP_MUL_MAT_ID) ? x->src[0] : nullptr;
        if (w != nullptr && w->op == GGML_OP_NONE) {
            k.add(1 + (uint64_t) w->type);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) k.add((uint64_t) w->ne[d]);
        } else {
            k.add(0);
        }
    }
    return k.h;
}

std::vector<int> tsr_pivot_node_order(const tsr_pivot_graph & g, const std::vector<uint32_t> & unit_order) {
    std::vector<int> out;
    out.reserve(g.unit_of.size());
    for (uint32_t u : unit_order) {
        out.insert(out.end(), g.unit_nodes[u].begin(), g.unit_nodes[u].end());
    }
    return out;
}

bool tsr_pivot_check_order(const std::vector<ggml_tensor *> & orig, const tsr_pivot_graph & g, const std::vector<int> & order,
                           std::string * why) {
    const int n = (int) orig.size();
    auto fail = [&](const std::string & s) {
        if (why != nullptr) *why = s;
        return false;
    };
    auto nm = [&](int i) { return std::string(orig[(size_t) i]->name[0] != '\0' ? orig[(size_t) i]->name : ggml_op_name(orig[(size_t) i]->op)) +
                                  "#" + std::to_string(i); };
    if ((int) order.size() != n) return fail("not a permutation: " + std::to_string(order.size()) + " nodes for " + std::to_string(n));
    std::vector<int> pos((size_t) n, -1);
    for (int p = 0; p < n; ++p) {
        const int i = order[(size_t) p];
        if (i < 0 || i >= n || pos[(size_t) i] >= 0) return fail("not a permutation: node " + std::to_string(i) + " is out of range or twice");
        pos[(size_t) i] = p;
    }
    std::unordered_map<const ggml_tensor *, int> idx;
    idx.reserve((size_t) n * 2);
    for (int i = 0; i < n; ++i) idx[orig[(size_t) i]] = i;
    // every node follows the producers of its sources
    for (int i = 0; i < n; ++i) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = orig[(size_t) i]->src[s];
            if (src == nullptr) continue;
            auto it = idx.find(src);
            if (it != idx.end() && pos[(size_t) it->second] > pos[(size_t) i]) {
                return fail("consumer " + nm(i) + " before its producer " + nm(it->second));
            }
        }
    }
    if (g.pin_first && order[0] != 0) return fail("node 0 is a MUL_MAT_ID and must stay the first node of the split");
    // the order constraints (recomputed from the original order) and the builder's order edges
    for (const auto & c : order_chains(orig)) {
        for (size_t k = 1; k < c.size(); ++k) {
            if (pos[(size_t) c[k - 1]] > pos[(size_t) c[k]]) {
                return fail("order constraint: " + nm(c[k]) + " before " + nm(c[k - 1]) + " (both touch one written buffer)");
            }
        }
    }
    for (const auto & e : g.before) {
        if (pos[(size_t) e.first] > pos[(size_t) e.second]) return fail("order constraint: " + nm(e.second) + " before " + nm(e.first));
    }
    // a fused group stays contiguous: no compute node outside it lies between its first and last node
    for (const auto & gr : g.groups) {
        int lo = n, hi = -1;
        for (int i = gr.first; i <= gr.second; ++i) {
            if (!is_compute(orig[(size_t) i])) continue;
            lo = std::min(lo, pos[(size_t) i]);
            hi = std::max(hi, pos[(size_t) i]);
        }
        for (int p = lo; p <= hi; ++p) {
            const int i = order[(size_t) p];
            if ((i < gr.first || i > gr.second) && is_compute(orig[(size_t) i])) {
                return fail("node " + nm(i) + " lies inside the group " + nm(gr.first) + " .. " + nm(gr.second));
            }
        }
    }
    return true;
}

namespace {

// The first graph of a structure (or every graph, without a memo): analyze, search, apply, validate.
void search_and_apply(ggml_cgraph * cg, const tsr_pivot_dev & dev, const tsr_pivot_params & params, tsr_pivot_report & r,
                      const std::function<void(std::vector<uint32_t> *)> * tamper) {
    if (!tsr_pivot_analyze(cg, dev, &r.graph)) {
        return;
    }
    r.ran = true;
    r.n_units = (uint32_t) r.graph.units.size();
    tsr_pivot_result res = tsr_pivot_run(r.graph.units, params);
    r.cycles_before = res.cycles_before;
    r.cycles_after = res.cycles_after;
    r.passes = res.passes;
    if (!res.ok) {
        r.why = "the core refused the graph: " + res.error;
        r.refused = true;
        return;
    }
    r.unit_order = res.order;
    if (tamper != nullptr) {
        (*tamper)(&r.unit_order);
    }
    for (const tsr_pivot_move & m : res.moves) {
        r.moves.push_back(r.graph.names[m.unit] + " from " + std::to_string(m.from) + " to " + std::to_string(m.to) + " (" +
                          std::to_string(m.moved) + " units, gain " + std::to_string(m.gain) + ")");
    }
    bool identity = true;
    for (size_t k = 0; k < r.unit_order.size(); ++k) identity = identity && r.unit_order[k] == (uint32_t) k;
    if (identity) {
        return;
    }
    {
        std::vector<char> seen(r.unit_order.size(), 0);
        bool perm = r.unit_order.size() == r.graph.units.size();
        for (size_t k = 0; perm && k < r.unit_order.size(); ++k) {
            perm = r.unit_order[k] < seen.size() && !seen[r.unit_order[k]];
            if (perm) seen[r.unit_order[k]] = 1;
        }
        if (!perm) {
            r.refused = true;
            r.why = "the unit order is not a permutation";
            fprintf(stderr, "[tesseract] pivot: the new unit order was refused (%s); the original order stays\n", r.why.c_str());
            return;
        }
    }
    r.node_order = tsr_pivot_node_order(r.graph, r.unit_order);
    // apply, then validate the nodes; a failure puts the original order back
    const std::vector<ggml_tensor *> orig(cg->nodes, cg->nodes + cg->n_nodes);
    for (int p = 0; p < cg->n_nodes; ++p) {
        cg->nodes[p] = orig[(size_t) r.node_order[(size_t) p]];
    }
    std::string why;
    if (!tsr_pivot_check_order(orig, r.graph, r.node_order, &why)) {
        for (int p = 0; p < cg->n_nodes; ++p) cg->nodes[p] = orig[(size_t) p];
        r.refused = true;
        r.why = why;
        r.node_order.clear();
        fprintf(stderr, "[tesseract] pivot: the new node order was refused (%s); the original order stays\n", why.c_str());
    } else {
        r.applied = true;
    }
}

// A later graph of a structure: the node order of the first one, checked against this graph's own analysis and applied. No
// search. An empty order means that the first graph kept its order, and so does this one.
void apply_remembered(ggml_cgraph * cg, const tsr_pivot_dev & dev, const std::vector<int> & order, tsr_pivot_report & r) {
    r.memo_hit = true;
    if (order.empty()) {
        return;
    }
    auto conflict = [&](const std::string & why) {
        r.conflict = r.refused = true;
        r.why = why;
    };
    if (!tsr_pivot_analyze(cg, dev, &r.graph)) {
        conflict("the graph has nothing to schedule");
        return;
    }
    r.ran = true;
    r.n_units = (uint32_t) r.graph.units.size();
    const std::vector<ggml_tensor *> orig(cg->nodes, cg->nodes + cg->n_nodes);
    std::string why;
    if (!tsr_pivot_check_order(orig, r.graph, order, &why)) {
        conflict(why);
        return;
    }
    for (int p = 0; p < cg->n_nodes; ++p) {
        cg->nodes[p] = orig[(size_t) order[(size_t) p]];
    }
    r.node_order = order;
    r.applied = true;
}

}   // namespace

void tsr_pivot_stop_on_conflict(const tsr_pivot_report & rep) {
    if (!rep.conflict) {
        return;
    }
    GGML_ABORT("tesseract: Tesseract Pivot: a graph of %u nodes (structure %016" PRIx64 ") does not accept the node order of the first "
               "graph of its structure: %s. Run with TSR_PIVOT=0.", rep.n_nodes, rep.key, rep.why.c_str());
}

void tsr_pivot_optimize(ggml_cgraph * cg, const tsr_pivot_dev & dev, const tsr_pivot_params & params, tsr_pivot_report * rep,
                        const std::function<void(std::vector<uint32_t> *)> * tamper, tsr_pivot_memo * memo) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    tsr_pivot_report local;
    tsr_pivot_report & r = rep != nullptr ? *rep : local;
    r = tsr_pivot_report();
    r.n_nodes = (uint32_t) std::max(cg->n_nodes, 0);
    if (memo == nullptr) {
        search_and_apply(cg, dev, params, r, tamper);
    } else {
        r.key = tsr_pivot_structure_key(cg);
        const uint64_t mk = memo_key(r.key, dev, params);
        auto it = memo->orders.find(mk);
        if (it != memo->orders.end()) {
            apply_remembered(cg, dev, it->second, r);
        } else if (memo->orders.size() >= tsr_pivot_memo::MAX_ENTRIES) {
            r.memo_full = true;
        } else {
            search_and_apply(cg, dev, params, r, tamper);
            memo->orders.emplace(mk, r.applied ? r.node_order : std::vector<int>());
            if (r.applied && memo->orders.size() < tsr_pivot_memo::MAX_ENTRIES) {
                // The graph as it is now is a structure of its own, and it must stay as it is when the pass sees it again (a scheduler
                // that reserves a graph and then computes the same graph calls the hook twice).
                memo->orders.emplace(memo_key(tsr_pivot_structure_key(cg), dev, params), std::vector<int>());
            }
        }
    }
    r.ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count();
}
