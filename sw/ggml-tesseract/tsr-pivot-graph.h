// tsr-pivot-graph.h — Tesseract Pivot, the graph scheduler: the ggml adapter (the core is tsr-pivot.h).
//
// The backend's graph_optimize hook calls tsr_pivot_optimize on the view of one split before ggml-alloc runs. The adapter
// turns the split's nodes into the units of the core, runs the search, and puts the nodes in the order the core returns.
// Nothing in ggml checks that order, so this file checks it: a node-level validator runs after the permutation and the
// original order is restored if it fails.
//
// Units. A unit is a run of consecutive nodes of the original order that must stay together:
//   - the nodes that one record lowers, or between which the builder makes no hazard check: a MUL_MAT with its fused ADD /
//     SCALE / softcap, an RMS_NORM with its MUL, the MoE router chain (SOFT_MAX to DIV), a MUL_MAT_ID with its scale chain,
//     an SSM_CONV with its SILU. The conditions are the builder's (tsr_fuse_pair_shape, tsr_moe_route_match, ... in
//     tsr-build.h), so both agree on what a group is. When the members of a group are not contiguous the whole span between the
//     first and the last is one unit.
//   - no-op nodes (VIEW, RESHAPE, PERMUTE, TRANSPOSE, NONE, nodes without the COMPUTE flag) join the unit of the next compute node
//     in the original order; the last run joins the unit before it. A node with no rows is a compute node like any other: whether
//     it has rows depends on the batch (build_rs zeroes the cell of a new sequence with an in-place SCALE that has no rows when the
//     sequence goes on), and the units must not.
// Because every unit is an interval of the original order, an edge between nodes (the producer comes first) is an edge from a
// smaller unit to a larger one, as the core requires.
//
// Dependencies. Data: every source of every node of a unit, through the node that produced it. Order constraints: ggml
// orders some nodes only through a buffer, not through a source. All nodes that read or write views of one base tensor which a
// compute node writes through a view or in place keep their original relative order: the KV cache planes (SET_ROWS writes a
// view, FLASH_ATTN reads a view; the builder gives the FLASH_ATTN its FA_NEW word only when the SET_ROWS_KV came first), the
// recurrent state caches (SCALE of a cell, GET_ROWS, GATED_DELTA_NET, CPY), the in-place ops. The state CPYs of an
// SSM_CONV group also stay in front of it, as the builder requires (tsr_conv_group_shape).
// A split whose first node is a MUL_MAT_ID keeps it there: llama.cpp's expert-copy callback reads the first node of a split.
//
// Costs. They come from the structure of the graph (weight shapes, expert counts, head sizes) and from the device (channels,
// the dual weight stream), never from the number of tokens: ggml-alloc replays the allocation of the reserve graph by node
// index, so a graph of another ubatch size must get the same permutation. The table is `tsr_pivot_costs` below.
//
// One order per structure. The costs keep the search from seeing the batch; the memo (tsr_pivot_memo) makes "the same structure,
// the same permutation" a property of the pass and not of every cost. The first graph of a structure (tsr_pivot_structure_key)
// is searched. Every later one gets the node order of the first without a search and is only checked against it. A graph that
// does not accept the remembered order is a conflict: its buffers were planned for that order and no other order is safe, so
// the caller has to stop (the backend aborts).
//
// No device access and no tensor data: the hook runs before ggml-alloc and outside the device mutex. Compiled with ggml-base;
// unit test tests/test-tsr-pivot-graph.cpp (ctest ggml_tesseract_pivot_graph).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ggml.h"

#include "tsr-pivot.h"

// What the costs and the guards need from the device.
struct tsr_pivot_dev {
    uint32_t num_ch = 4;       // DDR channels (each weight is spread over all of them)
    bool     dual  = false;    // CAPS_MVM_DUAL: a decode MVM reads two weight streams per channel
};

// The cost table, in core cycles. The values are coarse: they rank the alternatives, they do not predict a time. Where they come from:
// LD, QACT, MVM and ST of a weight MUL_MAT follow the simulator's cycle functions (cyc_ldst, cyc_qact, cyc_mvm, cyc_st in
// sw/sim/tsr_sim.c) with the weight beats per channel at the DDR rate of the dual weight stream (docs/rtl-mvm.md §15); the Gated
// DeltaNet head is VU §7.12.8 (cyc_gdn); the control-core cost of a record is what fw_exec_perf measures (tsr-pivot.h); the glue
// record, the router chain, the MUL_MAT_ID's ids and tail and the attention are gaps measured in decode profiles of the simulator.
namespace tsr_pivot_costs {
    // GLUE: the LD, VU and ST work of one record on the glue pipeline: a fixed part and a part per 16 elements of the first dimension
    constexpr int64_t GLUE_BASE   = 1300;
    constexpr int64_t GLUE_PER_16 = 3;
    // MVM (a weight MUL_MAT): beats per cycle and channel of the MVM unit (percent), without and with CAPS_MVM_DUAL
    constexpr int64_t RATE_PCT_SINGLE = 100;
    constexpr int64_t RATE_PCT_DUAL   = 141;
    constexpr int64_t LD_BASE         = 64;      // LD of x: 64 + K / 16
    constexpr int64_t QACT_Q8_BASE    = 44;      // QACT of Q8_0: 16 K / 32 + 44
    constexpr int64_t QACT_Q8_PER_32  = 16;
    constexpr int64_t QACT_Q4_BASE    = 72;      // QACT of Q4_K: 17 K / 256 + 72
    constexpr int64_t QACT_Q4_PER_256 = 17;
    constexpr int64_t BACK_BASE       = 110;     // ST of y: 110 + N / 16
    // MVM_ID (a MUL_MAT_ID)
    constexpr int64_t MMID_IDS   = 400;          // LD of the ids and the join
    constexpr int64_t MMID_BACK  = 300;
    constexpr int64_t MMID_TAIL_TILES = 2;       // the control core is busy until two tiles before the MVM end
    // BLOCK: the MoE router chain (6,500 + 96 per expert) and the GATED_DELTA_NET (3,300 + the per-head time)
    constexpr int64_t ROUTER_BASE       = 6500;
    constexpr int64_t ROUTER_PER_EXPERT = 96;
    constexpr int64_t GDN_BASE          = 3300;
    constexpr int64_t GDN_HEAD_EXTRA    = 126;   // the per-head time of VU §7.12.8 plus this, 3,040 for S_v 128
    // ATTN: FLASH_ATTN_EXT (its real time depends on the context)
    constexpr int64_t ATTN_FRONT = 500;
    constexpr int64_t ATTN_MVM   = 6000;
    constexpr int64_t ATTN_BACK  = 300;
}

// The analysis of one graph: its units, and what the node-level validator needs.
struct tsr_pivot_graph {
    std::vector<tsr_pivot_unit>      units;        // original order, preds as the core wants them
    std::vector<std::vector<int>>    unit_nodes;   // the node indices of each unit, in original order (compute nodes and no-ops)
    std::vector<int>                 unit_of;      // node index -> unit
    std::vector<std::string>         names;        // unit -> "name" or "name (+n)" of its first compute node, for the log
    // order constraints: each chain lists node indices in their original order; every node must stay behind the one before it
    std::vector<std::vector<int>>    chains;
    // runs that must stay contiguous: [first, last] node index of each fused group (compute nodes between them are members)
    std::vector<std::pair<int, int>> groups;
    // pairs (a, b) of node indices: a stays before b although no source links them (an SSM_CONV's state CPYs)
    std::vector<std::pair<int, int>> before;
    // node 0 is a MUL_MAT_ID: it stays node 0. llama.cpp's expert-copy callback (llama_context::sched_copy_experts) looks at the
    // first node of a split only; a split that starts with the MUL_MAT_ID of host-resident experts copies just the selected experts.
    bool                             pin_first = false;
};

// Build the units of the graph `cg` (the nodes in their original order). False when there is nothing to schedule (no compute node).
bool tsr_pivot_analyze(const ggml_cgraph * cg, const tsr_pivot_dev & dev, tsr_pivot_graph * g);

// The node-level validator. `orig` is the graph's nodes in the original order, `order` the candidate order as node indices
// into `orig`. True when `order` is a permutation, every node follows the producers of its sources, the order constraints
// hold and no compute node that is not a member lies between the first and the last node of a fused group. *why names the
// rule that fails.
bool tsr_pivot_check_order(const std::vector<ggml_tensor *> & orig, const tsr_pivot_graph & g, const std::vector<int> & order,
                           std::string * why = nullptr);

// The node order for a unit order of the core: the units in that order, each unit's nodes in their original order.
std::vector<int> tsr_pivot_node_order(const tsr_pivot_graph & g, const std::vector<uint32_t> & unit_order);

// The structure key of a graph: a hash of what its nodes are and how they are linked, and of nothing that depends on the batch.
// Per node, in the graph's order: the op (and the unary op of a UNARY), the type, the use count, and for the view base and each
// source whether it is absent, a node of this graph (its position) or a tensor outside it (a leaf, a node of another split: its
// op, its type and a number that is the same for every link to that tensor); for a MUL_MAT or MUL_MAT_ID whose first source is
// a leaf, the weight's type and dimensions. No shape of a node, no offset, no value, no address, no name: llama.cpp keeps the
// topology of a graph constant, so the graphs it builds for one token and for a full ubatch have the same key. The COMPUTE flag
// is left out too: llama.cpp selects the branches of a graph by it (ggml_build_forward_select) and ggml-alloc plans a node with
// or without it, so graphs that differ in the flag alone share a plan. They share a key, and the later one is checked against
// the order of the first.
uint64_t tsr_pivot_structure_key(const ggml_cgraph * cg);

// The node orders the pass has given, by structure (and by device and search parameters). ggml-alloc plans the buffers on the
// reserve graph and replays the plan by node index for every later graph with the same number of nodes and leafs, so within one
// scheduler two graphs of one structure must get the same permutation. One memo belongs to one scheduler: the backend keeps it
// in the backend instance (every llama.cpp context has its own instances and its own scheduler). Not thread safe.
struct tsr_pivot_memo {
    // A full memo learns no new structure: the graphs of a new structure keep their original order (all of them, so they agree).
    static constexpr size_t MAX_ENTRIES = 256;
    std::unordered_map<uint64_t, std::vector<int>> orders;   // -> the node order; empty: the original order stays
};

struct tsr_pivot_report {
    bool        ran = false;           // the graph has units: it was searched, or checked against the remembered order
    bool        applied = false;       // the nodes were permuted (false: nothing moved, or the validator refused)
    bool        refused = false;       // the validator refused the order and the original was restored
    std::string why;                   // the validator's reason
    uint64_t    key = 0;               // the structure key (only with a memo)
    bool        memo_hit = false;      // the order came from the memo: no search (no moves, no cycles, no passes)
    bool        memo_full = false;     // a new structure and a full memo: not searched, the original order stays
    bool        conflict = false;      // memo_hit, and the graph does not accept the remembered order (why). The original order
                                       // stays and refused is set. The caller must stop: ggml-alloc planned the buffers of this
                                       // structure for the remembered order.
    uint32_t    n_units = 0, n_nodes = 0;
    int64_t     cycles_before = 0, cycles_after = 0;
    uint32_t    passes = 0;
    uint64_t    ns = 0;                // the time of the whole call
    std::vector<std::string>      moves;    // "name from -> to (n units, gain g)", for the log
    tsr_pivot_graph               graph;    // kept for the log and the tests
    std::vector<uint32_t>         unit_order;
    std::vector<int>              node_order;
};

// The hook's body: analyze, search, permute cg->nodes in place, validate the nodes and put the original order back if the
// validator refuses. Fills *rep when it is not null. Does nothing (and reports ran = false) for a graph without a compute node.
// tamper (a test hook, null in the backend) may change the core's unit order before it is applied.
// With a memo, only the first graph of a structure takes that path, and the order it ends with is remembered (the original
// order too, when nothing moved or the validator refused). A later graph of the structure gets the remembered node order: it is
// checked against the graph's own analysis and applied, or reported as a conflict (tsr_pivot_report::conflict). A permuted graph
// is remembered as a structure that keeps its order, so a second call on the same graph changes nothing.
void tsr_pivot_optimize(ggml_cgraph * cg, const tsr_pivot_dev & dev, const tsr_pivot_params & params = tsr_pivot_params(),
                        tsr_pivot_report * rep = nullptr, const std::function<void(std::vector<uint32_t> *)> * tamper = nullptr,
                        tsr_pivot_memo * memo = nullptr);

// What the caller of tsr_pivot_optimize does with a conflict: nothing when rep.conflict is false, else the process stops
// (GGML_ABORT) with a message that names the structure, the validator's reason and TSR_PIVOT=0. The buffers of the structure
// are planned for the remembered order and the graph cannot take it, so neither that order nor the graph's own is safe.
void tsr_pivot_stop_on_conflict(const tsr_pivot_report & rep);
