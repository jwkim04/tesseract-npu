// tsr-prof.h — TSR_DUMP_PLAN: what the ggml backend hands to the profile report (docs/prof-dbg.md "The profile report").
//
// With TSR_DUMP_PLAN=<dir> the backend writes, next to a normal run and without changing a record or a bit of it:
//   plan_<id>.json   one per built plan (a graph_compute that missed the plan cache): the ggml nodes of the split with their
//                    placement (device record, absorbed by a fusion group, no-op, skipped), the fusion groups, the op lists with
//                    every record (kind, argument words, dependencies, DRAM ranges, weight bytes, ideal MVM cycles, SPM tile
//                    plan) and why each list ended (the launch boundaries)
//   launches.jsonl   one line per launch: ordinal, plan and list, cache hit, host times, PERF deltas
//   cpu_fallback.json  the nodes supports_op refused, with the reason (they run on the CPU)
//   trw.raw          the trace writer's records of the whole run (started when the device opens, drained when the backend is
//                    freed; TSR_PROF_TRW_MIB sets the ring size, default 32)
//   meta.json        device, image and clock facts of the run
// TSR_DUMP_PLAN also turns on the descriptor's PROF bit (a NODE_BEGIN marker before every record) unless TSR_OPL_PROF=0:
// the markers are what joins a trace slice to its record. This file knows no ggml types: ggml-tesseract.cpp fills the
// structures below.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tsr/plan.hpp"
#include "tsr_transport.h"

namespace tsr_prof {

bool on();                                   // TSR_DUMP_PLAN is set
const std::string & dir();

struct range {
    uint32_t ch = 0;
    uint64_t lo = 0, hi = 0, row = 0, stride = 0;   // bytes of channel ch: [lo, hi), or rows of `row` bytes every `stride`
};

struct rec {
    uint32_t idx = 0, kind = 0;
    int32_t tag = -1, node = -1;
    std::string name;
    uint32_t dep0 = 0xFFFFFFFFu, dep1 = 0xFFFFFFFFu;
    std::vector<uint32_t> args;
    std::vector<range> rd, wr;
    uint64_t rd_bytes = 0, wr_bytes = 0;     // the record's DRAM traffic by the kind's own formula (weights, rows, KV)
    uint64_t w_bytes = 0;                    // weight bytes the MVMs stream (MUL_MAT, MUL_MAT_ID)
    double   ideal_cycles = 0;               // w_bytes / (num_ch x 64 B per cycle)
    uint32_t n_tiles = 0, scratch_lines = 0, tile_rows = 0;   // the firmware's plan of the record (SPM ring tiles)
};

struct list {
    uint32_t idx = 0, n_ops = 0, n_joins = 0, blob_bytes = 0, graph_id = 0, pre_entries = 0, pre_lines = 0;
    std::string end_reason;
    std::vector<rec> recs;
};

struct src {
    std::string name;
    int32_t node = -1;                       // index of the producing node of this split, else -1
    std::string role;                        // node, weight, input (copied from another backend), kv, leaf
    int64_t ne[4] = {0, 0, 0, 0};
    int type = 0;
    std::string type_name;
};

struct node {
    int32_t idx = 0;
    std::string name, op, type_name;
    int64_t ne[4] = {0, 0, 0, 0};
    std::vector<src> srcs;
    std::string place;                       // device, absorbed, noop, skipped
    int32_t host = -1;                       // absorbed: the node whose record carries it
    int32_t group = -1;
    std::vector<std::pair<int32_t, uint32_t>> recs;   // (list, record index) of the records this node's record(s) became
};

struct group {
    int32_t id = 0;
    std::string kind;                        // MOE_ROUTE, MOE_SCALE, MM_ADD, MM_SCALE, MM_SOFTCAP, RMS_MUL, SSM_CONV, GDN_CELL
    int32_t head = -1;
    std::vector<int32_t> members;
};

struct plan {
    uint64_t id = 0, key = 0;
    int32_t n_nodes = 0;
    std::vector<node> nodes;
    std::vector<group> groups;
    std::vector<list> lists;
    std::vector<src> inputs;                 // tensors the split reads that another backend produced (copied in)
    std::vector<std::string> dynamic;        // reasons the plan is rebuilt every time (staged MUL_MAT, converted weight)
};

struct launch {
    uint64_t ordinal = 0, plan = 0;
    int32_t list = -1;
    uint32_t n_ops = 0;
    bool cache_hit = false;
    uint64_t t0_ns = 0, t1_ns = 0, wall_ns = 0;
    double core_hz = 0;
    int rc = 0;
    bool have_perf = false;
    uint64_t perf[64] = {};                  // PERF deltas over the launch
};

// The firmware's tile plan and the traffic of one record, from its argument words (cfg: the device's planner view).
void fill_record(rec * r, const tsr::plan::dev_cfg & cfg);

void write_plan(const plan & p);
void write_launch(const launch & l);
uint64_t next_plan_id();

// supports_op said no: the node runs on the CPU. `why` is the builder's or the backend's reason.
void note_refusal(const std::string & name, const std::string & op, const int64_t ne[4], const std::string & type_name,
                  const std::string & why);

// Session: start the trace writer after the device opened; at the end drain it and write the files above.
struct session_info {
    // TSR_PROF_TRW_MIB: where the ring goes (a byte offset of channel 0, 1 KiB aligned), from the backend's IO allocator
    uint64_t ring_off = 0;
    std::string image, description;
    uint64_t image_id = 0;
    uint32_t num_ch = 0, caps = 0, t_max = 0, spm_lines = 0, pcap = 0, vl = 0;
    uint64_t kind_mask = 0;
    bool simulated = false;
};
void session_open(tsr_transport * t, const session_info & si);
void session_close(tsr_transport * t);
bool perf_read(tsr_transport * t, uint64_t perf[64], double * core_hz);
uint64_t now_ns();

} // namespace tsr_prof
