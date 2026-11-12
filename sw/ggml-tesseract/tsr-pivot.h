// tsr-pivot.h — Tesseract Pivot, the graph scheduler: the core (no ggml, no device).
//
// Pivot orders the units of a compute graph so that independent matrix products run on the MVM unit
// while the glue pipeline and the control core are busy with work that does not depend on them (the MoE
// router, the Gated DeltaNet recurrence, the glue between the Q, K and V products). It is a scheduler over a
// dependency graph with a cost model of the device; a front end (the ggml backend's graph_optimize hook) turns
// its graph nodes into units, calls tsr_pivot_run and applies the order it returns.
//
// A unit is what must stay together: the front end merges the nodes that one record lowers, or that need
// no hazard check between them, into one unit. Units are numbered in their original order; `preds` lists
// the units that must come before a unit (data dependencies and order constraints), always with a smaller
// number. The order the search returns is a permutation of the unit numbers in which every unit follows its
// preds.
//
// The machine model (tsr_pivot_eval) is a coarse picture of one device in core cycles (the control core,
// the in-order glue pipeline that stands for the DMA queue, the QACT unit and the MVM unit). It ranks
// orders; it does not predict the time of one. All numbers are integers and every step is integer
// arithmetic, so the same input gives the same result on every host.
//
// The search starts from the original order and tries, for each unit with MVM work, to hoist it (with its
// ancestors) to an earlier position; it keeps a move when the model's time falls by at least a threshold,
// and repeats until a pass moves nothing. The order it ends with is a fixed point: a second run on it moves
// nothing.
//
// Pure C++17, no ggml dependency (unit test tests/test-tsr-pivot.cpp, ctest ggml_tesseract_pivot).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// The class of a unit decides which clocks of the machine model it advances.
enum tsr_pivot_class : uint8_t {
    TSR_PIVOT_GLUE   = 0,   // LD, VU and ST work on the glue pipeline: `body`
    TSR_PIVOT_MVM    = 1,   // a matrix product: LD of x (`ld`), QACT (`qact`), MVM (`mvm`), ST (`back`, may be deferred)
    TSR_PIVOT_MVM_ID = 2,   // a MUL_MAT_ID: ids and join (`ids`), all its tiles on the MVM unit (`mvm`), ST (`back`)
    TSR_PIVOT_BLOCK  = 3,   // a unit that holds the control core and the glue pipeline, the MVM unit free: `body`
    TSR_PIVOT_ATTN   = 4,   // attention: `front`, then the MVM unit (`mvm`), then `back`
};

// A unit of the graph. Every cost is in core cycles; a field a class does not use is zero.
struct tsr_pivot_unit {
    tsr_pivot_class       cls = TSR_PIVOT_GLUE;
    std::vector<uint32_t> preds;     // units that must come before this one (all smaller than its own number)

    int64_t issue       = 0;         // control-core time of the unit's records (a MVM unit that is closed
                                     // by its own next record, or that is not deferred: one MUL_MAT record)
    int64_t issue_open  = 0;         // MVM: control-core time when the unit stays open (a MUL_MAT_BEGIN record)
    int64_t issue_close = 0;         // MVM: control-core time of the record that closes it (a MUL_MAT_END)

    int64_t body  = 0;               // GLUE, BLOCK: time of the unit
    int64_t ld    = 0;               // MVM: LD of x on the glue pipeline
    int64_t qact  = 0;               // MVM: QACT unit
    int64_t ids   = 0;               // MVM_ID: LD of the ids and the join (the control core waits)
    int64_t front = 0;               // ATTN: work before the MVM unit starts
    int64_t mvm   = 0;               // MVM, MVM_ID, ATTN: time on the MVM unit
    int64_t tail  = 0;               // MVM_ID: the control core is busy until this many cycles before the MVM end
    int64_t back  = 0;               // MVM, MVM_ID, ATTN: ST of the result on the glue pipeline
    bool    defer = false;           // MVM: the ST can wait for the first reader (MUL_MAT_BEGIN / MUL_MAT_END)
};

// The control-core cost of one record in core cycles (measured by fw_exec_perf, a clk_rv cycle is two core
// cycles): the values for tsr_pivot_unit::issue, issue_open and issue_close.
constexpr int64_t TSR_PIVOT_ISSUE_MUL_MAT = 830;   // a MUL_MAT record
constexpr int64_t TSR_PIVOT_ISSUE_BEGIN   = 820;   // a MUL_MAT_BEGIN record
constexpr int64_t TSR_PIVOT_ISSUE_END     = 460;   // a MUL_MAT_END record
constexpr int64_t TSR_PIVOT_ISSUE_SHORT   = 700;   // a short record (each record of a GLUE unit, an ATTN unit)

// At most this many deferred units are open at once (the executor's table of open MUL_MAT_BEGIN records, SW §7.3);
// a ninth closes the oldest first.
constexpr uint32_t TSR_PIVOT_MAX_OPEN = 8;

// Search parameters.
struct tsr_pivot_params {
    uint32_t window    = 64;         // a unit is tried at the positions of this many units before it
    uint32_t max_moved = 8;          // a move of more units (the unit and its ancestors) is not tried
    int64_t  threshold = 256;        // a move is taken when it lowers the model's time by at least this (a value below 1 is
                                     // taken as 1: a move that gains nothing could be undone by the next one, and the search
                                     // would not end)
    uint32_t horizon   = 32;         // units after the moved span that the incremental evaluation covers
};

// One accepted move: `unit` and its ancestors at or after position `to` were put, in their relative order,
// at the positions from `to`; `from` is the position of `unit` before the move. Positions refer to the
// order at the time of the move.
struct tsr_pivot_move {
    uint32_t unit  = 0;
    uint32_t from  = 0;
    uint32_t to    = 0;
    uint32_t moved = 0;              // number of units moved (unit and ancestors)
    int64_t  gain  = 0;              // fall of the model's time, in core cycles
};

struct tsr_pivot_result {
    bool                        ok = false;       // false: the input is not a valid graph; `error` says why, `order` is the identity
    std::string                 error;
    std::vector<uint32_t>       order;            // the units in the new order
    int64_t                     cycles_before = 0;   // the model's time of the original order
    int64_t                     cycles_after  = 0;   // the model's time of `order`
    std::vector<tsr_pivot_move> moves;
    uint32_t                    passes = 0;       // passes over the units, the last one moved nothing
};

// True when `order` is a permutation of 0 .. units.size() - 1 and every unit follows its preds. The preds
// are also checked to be in range. On false, *why (if not null) says which rule fails.
bool tsr_pivot_valid(const std::vector<tsr_pivot_unit> & units, const std::vector<uint32_t> & order,
                     std::string * why = nullptr);

// The machine model: the time in core cycles of `order` (a valid order of `units`; see tsr_pivot_valid).
// Units are processed in order. State: t_issue (the control core), t_glue (the in-order glue pipeline),
// t_qact, t_mvm, and per unit `done` (its result is in DRAM) and its MVM end. `open` units are the deferred
// ones that have not stored their result yet.
//   - Every unit adds its issue cost to t_issue before it starts (a deferred unit that stays open adds
//     issue_open, and issue_close when it is closed). A unit does not start before t_issue.
//   - Before a unit, every open unit among its preds is closed in the order it was opened: its ST starts at
//     max(t_glue, mvm end, t_issue) and takes `back`; that sets its `done` and t_glue.
//   - ready = the latest `done` of the preds; s = max(t_glue, t_issue, ready).
//   - GLUE: ends at s + body; t_glue and done take that.
//   - MVM: LD ends at s + ld (t_glue); QACT ends at max(that, t_qact) + qact; the MVM runs from
//     max(QACT end, t_mvm) for `mvm`. If `defer` and the next unit in the order does not read it, the unit
//     stays open; else its ST follows at max(t_glue, mvm end) for `back` (t_glue, done).
//   - MVM_ID: the join ends at s + ids (t_glue, t_issue); the MVM work runs from max(that, t_mvm) for `mvm`;
//     t_issue is at least mvm end - tail; done = t_glue = mvm end + back.
//   - BLOCK: ends at s + body; t_glue, t_issue and done take that.
//   - ATTN: the MVM unit runs from max(s + front, t_mvm) for `mvm`; done = t_glue = mvm end + back.
//   - At most TSR_PIVOT_MAX_OPEN units are open: a deferred unit that would stay open as a ninth closes the
//     oldest first. At the end every open unit is closed. The result is the latest of the clocks.
// `after` is the unit that follows the last unit of `order` in the graph, when `order` is a prefix of one (the deferral rule
// looks at the next unit); the default is the end of the graph.
constexpr uint32_t TSR_PIVOT_NONE = 0xFFFFFFFFu;
int64_t tsr_pivot_eval(const std::vector<tsr_pivot_unit> & units, const std::vector<uint32_t> & order,
                       uint32_t after = TSR_PIVOT_NONE);

// A test hook for tsr_pivot_run: called for every incremental score the search takes (the unchanged order, each candidate and
// the confirmation of the best one). `order` is the current order; `seg[0 .. len)` is the sequence that was evaluated from the
// state saved before position `start` (the units order[0 .. start) come before it); `look` is the unit that followed it and
// `score` the model's time the search got. The score equals tsr_pivot_eval(order[0 .. start) + seg, look).
using tsr_pivot_score_hook = std::function<void(const std::vector<uint32_t> & order, uint32_t start, const uint32_t * seg,
                                                size_t len, uint32_t look, int64_t score)>;

// The search. Deterministic: the same units and parameters give the same result. Validity holds by
// construction: a move takes a unit with its ancestors, so no unit leaves its preds behind. A move is
// evaluated on the units from the position it changes up to `horizon` units after the moved span, from the
// model's state saved at that position; the best one is confirmed on the rest of the graph before it is
// taken, so every accepted move lowers tsr_pivot_eval by at least the threshold.
tsr_pivot_result tsr_pivot_run(const std::vector<tsr_pivot_unit> & units,
                               const tsr_pivot_params & params = tsr_pivot_params(),
                               const tsr_pivot_score_hook * hook = nullptr);
