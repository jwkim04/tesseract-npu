// tsr-prof.cpp — TSR_DUMP_PLAN (tsr-prof.h): the plan, launch and trace files of the profile report.
#include "tsr-prof.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>

#include "tsr/opl.hpp"

#if defined(__has_include)
#  if __has_include("tsr_dbg_lib.hpp")
#    include "tsr_dbg_lib.hpp"
#    define TSR_PROF_HAVE_TRW 1
#  endif
#endif

namespace tsr_prof {

namespace opl = tsr::opl;
namespace pl  = tsr::plan;

bool on() {
    static const bool v = [] { const char * e = getenv("TSR_DUMP_PLAN"); return e != nullptr && e[0] != '\0'; }();
    return v;
}

const std::string & dir() {
    static const std::string d = [] {
        const char * e = getenv("TSR_DUMP_PLAN");
        return std::string(e != nullptr ? e : "");
    }();
    return d;
}

uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------
// JSON text
// ---------------------------------------------------------------------------------------------
namespace {

std::string esc(const std::string & s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else if ((unsigned char) c < 0x20) {
            char b[8];
            snprintf(b, sizeof(b), "\\u%04x", (unsigned) (unsigned char) c);
            o += b;
        } else {
            o += c;
        }
    }
    return o + "\"";
}

void app(std::string & s, const char * f, ...) __attribute__((format(printf, 2, 3)));
void app(std::string & s, const char * f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    const int n = vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    if (n < (int) sizeof(b)) {
        s.append(b, (size_t) std::max(n, 0));
        return;
    }
    std::string big((size_t) n + 1, '\0');
    va_start(ap, f);
    vsnprintf(&big[0], big.size(), f, ap);
    va_end(ap);
    s.append(big.data(), (size_t) n);
}

bool write_file(const std::string & path, const std::string & s, const char * mode = "w") {
    FILE * f = fopen(path.c_str(), mode);
    if (f == nullptr) {
        fprintf(stderr, "[tesseract] TSR_DUMP_PLAN: cannot write %s\n", path.c_str());
        return false;
    }
    fwrite(s.data(), 1, s.size(), f);
    return fclose(f) == 0;
}

std::string path_in(const std::string & name) { return dir() + "/" + name; }

// Do MVMs of at most T_MAX / 2 tokens read two weight streams per channel? They do when the device has the dual weight stream (CAPS
// bit 16, docs/rtl-mvm.md §15; with T_MAX 1 no MVM qualifies) and its run-time enable is on. No transport call reports the enable, so
// it is read the way the transports read TSR_MVM_DUAL: unset or empty is on, 0 is off (xrt: ADAPTER_CTRL.DUAL_DIS, sim: the cycle
// model's enable).
bool mvm_dual(uint32_t caps, uint32_t t_max) {
    const char * e = getenv("TSR_MVM_DUAL");
    return (caps & TSR_CAPS_MVM_DUAL) != 0 && t_max >= 2 && (e == nullptr || e[0] == '\0' || atoi(e) != 0);
}

// ---------------------------------------------------------------------------------------------
// The argument words of each kind (SW §7.2) as "NAME:t": h DRAM handle, s stride or lines, n count, b beats, p preload byte
// offset, f f32 bits, x flags.
// ---------------------------------------------------------------------------------------------
struct kind_spec {
    uint32_t kind;
    std::vector<const char *> args;
};

const std::vector<kind_spec> & kind_specs() {
    static const std::vector<kind_spec> v = {
        { opl::K_MUL_MAT, { "X:h", "X_STRIDE:s", "Y:h", "Y_STRIDE:s", "W_OFF:b", "FMT:n", "N:n", "K:n", "T:n", "ADD:h", "ADD_STRIDE:s", "FLAGS:x" } },
        { opl::K_MUL_MAT_SCALE, { "X:h", "X_STRIDE:s", "Y:h", "Y_STRIDE:s", "W_OFF:b", "FMT:n", "N:n", "K:n", "T:n", "ADD:h", "ADD_STRIDE:s", "FLAGS:x", "SCALE:f" } },
        { opl::K_MUL_MAT_SOFTCAP, { "X:h", "X_STRIDE:s", "Y:h", "Y_STRIDE:s", "W_OFF:b", "FMT:n", "N:n", "K:n", "T:n", "ADD:h", "ADD_STRIDE:s", "FLAGS:x", "SCALE:f", "POST_SCALE:f" } },
        { opl::K_ADD, { "DST:h", "DST_STRIDE:s", "A:h", "A_STRIDE:s", "B:h", "B_STRIDE:s", "N:n", "ROWS:n", "B_PERIOD:n", "FLAGS:x" } },
        { opl::K_MUL, { "DST:h", "DST_STRIDE:s", "A:h", "A_STRIDE:s", "B:h", "B_STRIDE:s", "N:n", "ROWS:n", "B_PERIOD:n", "FLAGS:x" } },
        { opl::K_SCALE, { "DST:h", "DST_STRIDE:s", "SRC:h", "SRC_STRIDE:s", "N:n", "ROWS:n", "S:f", "B:f" } },
        { opl::K_RMS_NORM, { "DST:h", "DST_STRIDE:s", "SRC:h", "SRC_STRIDE:s", "N:n", "ROWS:n", "EPS:f", "INV_N:f", "W:h", "W_PERIOD:n" } },
        { opl::K_GLU, { "DST:h", "DST_STRIDE:s", "GATE:h", "GATE_STRIDE:s", "UP:h", "UP_STRIDE:s", "N:n", "ROWS:n", "VARIANT:n" } },
        { opl::K_ROPE, { "DST:h", "DST_STRIDE:s", "SRC:h", "SRC_STRIDE:s", "POS:p", "TABLE:h", "TS:f", "M:f", "FREQ_SCALE:f", "HEAD_DIM:n", "N_DIMS:n", "N_HEADS:n", "N_TOKENS:n", "MODE:x" } },
        { opl::K_GET_ROWS, { "DST:h", "DST_STRIDE:s", "SRC:h", "SRC_STRIDE:s", "IDS:p", "N_IDS:n", "ROW_LINES:s" } },
        { opl::K_SET_ROWS_KV, { "CACHE:h", "CACHE_STRIDE:s", "SRC:h", "SRC_STRIDE:s", "IDX:p", "N_ROWS:n", "ROW_ELEMS:n", "WHICH:n", "PLANE:n", "GEOM:x", "PLANE_KIND:n", "STREAM:n" } },
        { opl::K_FLASH_ATTN, { "DST:h", "DST_STRIDE:s", "Q:h", "Q_HEAD_STRIDE:s", "Q_TOKEN_STRIDE:s", "K_CACHE:h", "V_CACHE:h", "N_KV:n", "HEAD_DIM:n", "N_HEAD:n", "N_HEAD_KV:n", "N_TOKENS:n", "MSUM:p", "MASK:h", "SCALE:f", "SOFTCAP:f", "K_PLANE:n", "V_PLANE:n", "GEOM:x", "D:n", "G:n", "FLAGS:x" } },
        { opl::K_CPY, { "DST:h", "DST_STRIDE:s", "SRC:h", "SRC_STRIDE:s", "ROW_LINES:s", "ROWS:n" } },
        { opl::K_UNARY, { "DST:h", "DST_STRIDE:s", "SRC:h", "SRC_STRIDE:s", "N:n", "ROWS:n", "FUNC:n" } },
        { opl::K_MOE_ROUTE, { "LOGITS:h", "LOGITS_STRIDE:s", "IDS:h", "IDS_STRIDE:s", "W:h", "N_EXPERT:n", "N_USED:n", "T:n", "FLAGS:x" } },
        { opl::K_MUL_MAT_BEGIN, { "X:h", "X_STRIDE:s", "Y:h", "Y_STRIDE:s", "W_OFF:b", "FMT:n", "N:n", "K:n", "T:n", "ADD:h", "ADD_STRIDE:s", "FLAGS:x" } },
        { opl::K_MUL_MAT_END, { "BEGIN:n" } },
        { opl::K_MUL_MAT_ID, { "DST:h", "DST_STRIDE:s", "X:h", "X_SLOT_STRIDE:s", "X_TOK_STRIDE:s", "IDS:h", "IDS_STRIDE:n", "W_OFF:b", "EXP_STRIDE:b", "FMT:n", "N:n", "K:n", "T:n", "N_USED:n", "N_EXPERT:n", "SCALE:h", "FLAGS:x" } },
        { opl::K_SOFTPLUS, { "DST:h", "DST_STRIDE:s", "SRC:h", "SRC_STRIDE:s", "N:n", "ROWS:n" } },
        { opl::K_SSM_CONV, { "DST:h", "DST_TOKEN_STRIDE:s", "X:h", "X_TOKEN_STRIDE:s", "STATE_IN:h", "STATE_OUT:h", "W:h", "N_CH:n", "D_CONV:n", "N_TOKENS:n", "N_SEQS:n", "FLAGS:x" } },
        { opl::K_GATED_DELTA_NET, { "DST:h", "STATE_OUT:h", "STATE_IN:h", "Q:h", "Q_HEAD_STRIDE:s", "Q_TOKEN_STRIDE:s", "K:h", "K_HEAD_STRIDE:s", "K_TOKEN_STRIDE:s", "V:h", "V_HEAD_STRIDE:s", "V_TOKEN_STRIDE:s", "G:h", "G_TOKEN_STRIDE:s", "BETA:h", "BETA_TOKEN_STRIDE:s", "S_V:n", "H_K:n", "H_V:n", "N_TOKENS:n", "N_SEQS:n", "SCALE:f" } },
    };
    return v;
}

uint64_t range_bytes(const range & r) {
    if (r.hi <= r.lo) {
        return 0;
    }
    if (r.row == 0 || r.stride == 0 || r.row >= r.stride) {
        return r.hi - r.lo;
    }
    const uint64_t span = r.hi - r.lo;
    const uint64_t n = span < r.row ? 1 : (span - r.row) / r.stride + 1;
    return n * r.row;
}

} // namespace

uint64_t next_plan_id() {
    static uint64_t id = 0;
    return ++id;
}

// ---------------------------------------------------------------------------------------------
// The record: the firmware's tile plan, weight bytes and the traffic by the kind's formula
// ---------------------------------------------------------------------------------------------
void fill_record(rec * r, const pl::dev_cfg & cfg, const rec * begin) {
    const uint32_t * a = r->args.data();
    const uint32_t n = (uint32_t) r->args.size();
    for (const range & x : r->rd) r->rd_bytes += range_bytes(x);
    for (const range & x : r->wr) r->wr_bytes += range_bytes(x);
    const uint32_t C = cfg.num_ch != 0 ? cfg.num_ch : 4u;
    const uint32_t dual_t = mvm_dual(cfg.caps, cfg.t_max) ? cfg.t_max / 2u : 0u;   // an MVM of at most dual_t tokens reads two streams
    // `passes` MVMs of T tokens each. A pass streams the weights once: the slowest channel has rows_ch rows of rb beats, and a weight
    // stream takes one beat per cycle. So the weight phase is rows_ch x rb cycles, and ceil(rows_ch / 2) x rb on two streams. This is
    // mvm_weight_cycles of sw/sim/tsr_sim.c without its memory rate bound: the ideal is the limit of the core's streams, not the memory's.
    auto add_mvm_passes = [&](uint32_t fmt, uint32_t N, uint32_t K, uint64_t passes, uint32_t T) {
        if (passes == 0) return;
        const uint32_t rb = pl::row_beats(fmt, K);
        const uint64_t rows_ch = (N + C - 1u) / C;
        const double cycles = (double) (T <= dual_t ? (rows_ch + 1u) / 2u : rows_ch) * rb;
        r->w_bytes += passes * rows_ch * C * rb * 64ull;
        r->ideal_cycles += (double) passes * cycles;
        r->mvm_passes.emplace_back(passes, cycles);
    };
    switch (r->kind) {
    case opl::K_MUL_MAT_END: {
        // the ST of the BEGIN's y rows: the write bytes belong to this record (the BEGIN has the reads, the weights and the MVMs)
        r->rd_bytes = 0;
        if (begin == nullptr || begin->args.size() < opl::mm::NARGS) break;
        r->wr_bytes = (uint64_t) begin->args[opl::mm::T] * ((begin->args[opl::mm::N] + 15u) / 16u) * 64ull;
        break;
    }
    case opl::K_MUL_MAT:
    case opl::K_MUL_MAT_SCALE:
    case opl::K_MUL_MAT_SOFTCAP:
    case opl::K_MUL_MAT_BEGIN: {
        if (n < opl::mm::NARGS) break;
        const uint32_t fmt = a[opl::mm::FMT], N = a[opl::mm::N], K = a[opl::mm::K], T = a[opl::mm::T];
        pl::mm_tiling t{};
        if (pl::mm_plan(cfg, fmt, N, K, T, &t) != opl::RET_OK) break;
        r->n_tiles = t.n_tiles;
        r->tile_rows = t.t_tile;
        r->scratch_lines = t.t_tile * t.lpt;
        // one MVM per T-tile (SW §7.4): n_tiles - 1 tiles of t_tile tokens, then the last tile with the rest
        add_mvm_passes(fmt, N, K, t.n_tiles - 1u, t.t_tile);
        add_mvm_passes(fmt, N, K, 1u, T - (t.n_tiles - 1u) * t.t_tile);
        r->rd_bytes = r->w_bytes + (uint64_t) T * t.x_lines * 64ull;
        if (a[opl::mm::ADD] != opl::HANDLE_NONE) r->rd_bytes += (uint64_t) T * t.y_lines * 64ull;
        r->wr_bytes = r->kind == opl::K_MUL_MAT_BEGIN ? 0ull : (uint64_t) T * t.y_lines * 64ull;     // a BEGIN stores nothing: its END does
        break;
    }
    case opl::K_MUL_MAT_ID: {
        if (n < opl::mmid::NARGS) break;
        const uint32_t fmt = a[opl::mmid::FMT], N = a[opl::mmid::N], K = a[opl::mmid::K], T = a[opl::mmid::T], U = a[opl::mmid::N_USED];
        pl::kind_plan kp{};
        uint32_t group = 0;
        if (pl::plan_kind(pl::plan_limits_of(cfg), r->kind, a, n, &kp)) {
            r->n_tiles = kp.n_tiles;
            r->tile_rows = kp.tile_rows;
            r->scratch_lines = kp.scratch_lines;
            group = kp.aux[3];
        }
        const uint64_t pairs = (uint64_t) T * U;
        if (group >= 2u) {
            // Grouped record (SW-62, SW-66): the pairs of one expert run in tiles of `group` rows. The ideal is the lower bound,
            // the pairs in full tiles and one tile of the rest; the routing decides the real passes, one partial tile per expert
            // at the end, so a real run reads more weight bytes.
            add_mvm_passes(fmt, N, K, pairs / group, group);
            add_mvm_passes(fmt, N, K, pairs % group != 0 ? 1u : 0u, (uint32_t) (pairs % group));
            r->mvm_passes.clear();   // the trace's MVMs are the real tiles, not these passes: no one-to-one match
            r->grouped = true;
        } else {
            add_mvm_passes(fmt, N, K, pairs, 1u);   // per (token, used expert) the MUL_MAT sequence with T = 1 (SW §7.5)
        }
        r->rd_bytes = r->w_bytes + (uint64_t) T * U * (K / 16u) * 64ull + (uint64_t) T * U * 4ull;
        r->wr_bytes = (uint64_t) T * U * ((N + 15u) / 16u) * 64ull;
        break;
    }
    default: {
        if (r->kind == opl::K_NOP || r->kind == opl::K_JOIN) break;
        pl::kind_plan kp{};
        if (pl::plan_kind(pl::plan_limits_of(cfg), r->kind, a, n, &kp)) {
            r->n_tiles = kp.n_tiles;
            r->tile_rows = kp.tile_rows;
            r->scratch_lines = kp.scratch_lines;
        }
        if (r->kind == opl::K_FLASH_ATTN && n >= opl::fa::NARGS) {
            // K and V of the used cells (Q8_0: 34 bytes per 32 elements), once per query tile of T_MAX tokens
            const uint64_t kv = (uint64_t) a[opl::fa::N_KV] * a[opl::fa::N_HEAD_KV] * a[opl::fa::HEAD_DIM] * 34ull / 32ull * 2ull;
            const uint32_t tm = cfg.t_max != 0 ? cfg.t_max : 1u;
            const uint64_t passes = (a[opl::fa::N_TOKENS] + tm - 1u) / tm;
            r->rd_bytes = passes * kv + (uint64_t) a[opl::fa::N_TOKENS] * a[opl::fa::N_HEAD] * a[opl::fa::HEAD_DIM] * 4ull;
        } else if (r->kind == opl::K_GET_ROWS && n >= opl::getrows::NARGS) {
            r->rd_bytes = (uint64_t) a[opl::getrows::N_IDS] * a[opl::getrows::ROW_LINES] * 64ull;
        } else if (r->kind == opl::K_SET_ROWS_KV && n >= opl::setrows::NARGS) {
            // f32 rows in, Q8_0 rows into the plane (the ranges are the whole plane)
            r->rd_bytes = (uint64_t) a[opl::setrows::N_ROWS] * a[opl::setrows::ROW_ELEMS] * 4ull;
            r->wr_bytes = (uint64_t) a[opl::setrows::N_ROWS] * a[opl::setrows::ROW_ELEMS] * 34ull / 32ull;
        }
        break;
    }
    }
}

// ---------------------------------------------------------------------------------------------
// plan_<id>.json
// ---------------------------------------------------------------------------------------------
namespace {

void json_src(std::string & s, const src & x) {
    app(s, "{\"name\":%s,\"node\":%d,\"role\":%s,\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "],\"type\":%s}", esc(x.name).c_str(), x.node,
        esc(x.role).c_str(), x.ne[0], x.ne[1], x.ne[2], x.ne[3], esc(x.type_name).c_str());
}

void json_ranges(std::string & s, const std::vector<range> & v) {
    s += "[";
    for (size_t i = 0; i < v.size(); ++i) {
        const range & r = v[i];
        app(s, "%s[%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]", i ? "," : "", r.ch, r.lo, r.hi, r.row, r.stride);
    }
    s += "]";
}

} // namespace

void write_plan(const plan & p) {
    if (!on()) return;
    std::string s;
    s.reserve(1 << 20);
    app(s, "{\"format\":\"tsr-plan-1\",\"plan\":%" PRIu64 ",\"key\":\"%016" PRIx64 "\",\"n_nodes\":%d,\n", p.id, p.key, p.n_nodes);
    s += "\"dynamic\":[";
    for (size_t i = 0; i < p.dynamic.size(); ++i) app(s, "%s%s", i ? "," : "", esc(p.dynamic[i]).c_str());
    s += "],\n\"inputs\":[";
    for (size_t i = 0; i < p.inputs.size(); ++i) {
        s += i ? "," : "";
        json_src(s, p.inputs[i]);
    }
    s += "],\n\"groups\":[";
    for (size_t i = 0; i < p.groups.size(); ++i) {
        const group & g = p.groups[i];
        app(s, "%s{\"id\":%d,\"kind\":%s,\"head\":%d,\"members\":[", i ? "," : "", g.id, esc(g.kind).c_str(), g.head);
        for (size_t k = 0; k < g.members.size(); ++k) app(s, "%s%d", k ? "," : "", g.members[k]);
        s += "]}";
    }
    s += "],\n\"nodes\":[\n";
    for (size_t i = 0; i < p.nodes.size(); ++i) {
        const node & x = p.nodes[i];
        app(s, "%s{\"i\":%d,\"name\":%s,\"op\":%s,\"type\":%s,\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "],\"place\":%s,\"host\":%d,\"group\":%d,\"recs\":[",
            i ? ",\n" : "", x.idx, esc(x.name).c_str(), esc(x.op).c_str(), esc(x.type_name).c_str(), x.ne[0], x.ne[1], x.ne[2], x.ne[3],
            esc(x.place).c_str(), x.host, x.group);
        for (size_t k = 0; k < x.recs.size(); ++k) app(s, "%s[%d,%u]", k ? "," : "", x.recs[k].first, x.recs[k].second);
        s += "],\"src\":[";
        for (size_t k = 0; k < x.srcs.size(); ++k) {
            s += k ? "," : "";
            json_src(s, x.srcs[k]);
        }
        s += "]}";
    }
    s += "\n],\n\"lists\":[\n";
    for (size_t li = 0; li < p.lists.size(); ++li) {
        const list & L = p.lists[li];
        app(s, "%s{\"idx\":%u,\"n_ops\":%u,\"n_joins\":%u,\"blob_bytes\":%u,\"graph_id\":%u,\"pre_entries\":%u,\"pre_lines\":%u,\"end\":%s,\"recs\":[\n",
            li ? ",\n" : "", L.idx, L.n_ops, L.n_joins, L.blob_bytes, L.graph_id, L.pre_entries, L.pre_lines, esc(L.end_reason).c_str());
        for (size_t i = 0; i < L.recs.size(); ++i) {
            const rec & r = L.recs[i];
            app(s, "%s{\"i\":%u,\"kind\":%u,\"tag\":%d,\"node\":%d,\"name\":%s,\"dep\":[%d,%d],\"a\":[", i ? ",\n" : "", r.idx, r.kind, r.tag, r.node,
                esc(r.name).c_str(), r.dep0 == opl::DEP_NONE ? -1 : (int) r.dep0, r.dep1 == opl::DEP_NONE ? -1 : (int) r.dep1);
            for (size_t k = 0; k < r.args.size(); ++k) app(s, "%s%u", k ? "," : "", r.args[k]);
            app(s, "],\"rd\":");
            json_ranges(s, r.rd);
            s += ",\"wr\":";
            json_ranges(s, r.wr);
            app(s, ",\"rd_bytes\":%" PRIu64 ",\"wr_bytes\":%" PRIu64 ",\"w_bytes\":%" PRIu64 ",\"ideal\":%.1f", r.rd_bytes, r.wr_bytes, r.w_bytes,
                r.ideal_cycles);
            if (!r.mvm_passes.empty()) {
                s += ",\"mvm_passes\":[";
                for (size_t k = 0; k < r.mvm_passes.size(); ++k) app(s, "%s[%" PRIu64 ",%.1f]", k ? "," : "", r.mvm_passes[k].first, r.mvm_passes[k].second);
                s += "]";
            }
            app(s, ",\"tiles\":%u,\"tile_rows\":%u,\"scratch_lines\":%u%s}", r.n_tiles, r.tile_rows, r.scratch_lines,
                r.grouped ? ",\"grouped\":1" : "");
        }
        s += "\n]}";
    }
    s += "\n]}\n";
    char name[64];
    snprintf(name, sizeof(name), "plan_%" PRIu64 ".json", p.id);
    write_file(path_in(name), s);
}

void write_launch(const launch & l) {
    if (!on()) return;
    std::string s;
    app(s, "{\"launch\":%" PRIu64 ",\"plan\":%" PRIu64 ",\"list\":%d,\"n_ops\":%u,\"cache_hit\":%s,\"t0_ns\":%" PRIu64 ",\"t1_ns\":%" PRIu64
           ",\"wall_ns\":%" PRIu64 ",\"core_hz\":%.0f,\"rc\":%d",
        l.ordinal, l.plan, l.list, l.n_ops, l.cache_hit ? "true" : "false", l.t0_ns, l.t1_ns, l.wall_ns, l.core_hz, l.rc);
    if (l.have_perf) {
        s += ",\"perf\":[";
        for (int i = 0; i < 64; ++i) app(s, "%s%" PRIu64, i ? "," : "", l.perf[i]);
        s += "]";
    }
    s += "}\n";
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    write_file(path_in("launches.jsonl"), s, "a");
}

// ---------------------------------------------------------------------------------------------
// cpu_fallback.json
// ---------------------------------------------------------------------------------------------
namespace {
struct refusal {
    std::string op, type_name, why;
    int64_t ne[4];
};
std::mutex g_ref_m;
std::map<std::string, refusal> & refusals() {
    static std::map<std::string, refusal> m;
    return m;
}
} // namespace

void note_refusal(const std::string & name, const std::string & op, const int64_t ne[4], const std::string & type_name, const std::string & why) {
    if (!on()) return;
    std::lock_guard<std::mutex> lock(g_ref_m);
    refusal r;
    r.op = op;
    r.type_name = type_name;
    r.why = why;
    for (int i = 0; i < 4; ++i) r.ne[i] = ne[i];
    refusals()[name + "|" + op] = r;   // a node asked again keeps the last reason
}

// ---------------------------------------------------------------------------------------------
// The session: trace writer, PERF, meta.json
// ---------------------------------------------------------------------------------------------
namespace {
#ifdef TSR_PROF_HAVE_TRW
tsr::dbg::trw_ring g_ring;
#endif
bool g_trw_on = false;
std::string g_trw_note;
uint64_t g_trw_lost = 0, g_trw_records = 0;
session_info g_si;
tsr_transport * g_tp = nullptr;
bool g_closed = false;
void at_exit_close() {
    if (g_tp != nullptr && !g_closed) session_close(g_tp);
}
} // namespace

bool perf_read(tsr_transport * t, uint64_t perf[64], double * core_hz) {
    tsr_perf_summary s;
    if (tsr_transport_perf_summary(t, &s) != 0) return false;
    for (int i = 0; i < 64; ++i) perf[i] = s.perf[i];
    if (core_hz != nullptr) *core_hz = s.core_hz;
    return true;
}

void session_open(tsr_transport * t, const session_info & si) {
    if (!on()) return;
    g_si = si;
    g_tp = t;
    g_closed = false;
    write_file(path_in("launches.jsonl"), "");
    static bool registered = false;
    if (!registered) {                 // the files are written even when the process ends without freeing the backend
        registered = true;
        atexit(at_exit_close);
    }
    const char * e = getenv("TSR_PROF_TRW_MIB");
    const uint32_t mib = e != nullptr && e[0] != '\0' ? (uint32_t) atoi(e) : 32u;
    if (mib == 0) {
        g_trw_note = "trace writer off (TSR_PROF_TRW_MIB=0)";
        return;
    }
#ifdef TSR_PROF_HAVE_TRW
    g_ring = tsr::dbg::trw_default_ring(mib);
    if (si.ring_off != 0) g_ring.off = si.ring_off;
    std::string why;
    if (tsr::dbg::trw_start(t, g_ring, &why) != 0) {
        g_trw_note = "trace writer not started: " + why;
        fprintf(stderr, "[tesseract] TSR_DUMP_PLAN: %s\n", g_trw_note.c_str());
        return;
    }
    g_trw_on = true;
    g_trw_note = "trace writer ring " + std::to_string(mib) + " MiB";
#else
    (void) t;
    g_trw_note = "built without libtsr_dbg: no trace writer";
#endif
}

void session_close(tsr_transport * t) {
    if (!on() || g_closed) return;
    g_closed = true;
#ifdef TSR_PROF_HAVE_TRW
    if (g_trw_on) {
        std::vector<tsr::dbg::trw_rec> recs;
        std::string why;
        uint64_t lost = 0;
        tsr_perf_summary s;
        tsr_transport_perf_summary(t, &s);
        if (tsr::dbg::trw_drain(t, g_ring, &recs, &lost, &why) != 0) {
            fprintf(stderr, "[tesseract] TSR_DUMP_PLAN: %s\n", why.c_str());
        }
        tsr::dbg::trw_stop(t);
        g_trw_on = false;
        g_trw_lost = lost + s.perf[TSR_PERF_TRW_DROPS];
        g_trw_records = recs.size();
        tsr::dbg::trw_save_raw(recs, path_in("trw.raw"));
        fprintf(stderr, "[tesseract] TSR_DUMP_PLAN: %zu trace records (%" PRIu64 " dropped or overwritten) -> %s\n", recs.size(), g_trw_lost,
                path_in("trw.raw").c_str());
    }
#else
    (void) t;
#endif
    std::string s;
    app(s, "{\"format\":\"tsr-meta-1\",\"image\":%s,\"image_id\":\"%016" PRIx64 "\",\"description\":%s,\"num_ch\":%u,\"caps\":%u,\"t_max\":%u,"
           "\"mvm_dual\":%d,\"spm_lines\":%u,\"pcap\":%u,\"vl\":%u,\"kind_mask\":\"%016" PRIx64 "\",\"simulated\":%s,\"trw_note\":%s,\"trw_records\":%" PRIu64
           ",\"trw_lost\":%" PRIu64 ",\"regions\":{\"opl\":[%llu,%llu],\"io\":[%llu,%llu],\"aux\":[%llu,%llu],\"weight_off\":%llu},\"kinds\":{",
        esc(g_si.image).c_str(), g_si.image_id, esc(g_si.description).c_str(), g_si.num_ch, g_si.caps, g_si.t_max, mvm_dual(g_si.caps, g_si.t_max) ? 1 : 0,
        g_si.spm_lines, g_si.pcap, g_si.vl, g_si.kind_mask, g_si.simulated ? "true" : "false", esc(g_trw_note).c_str(), g_trw_records, g_trw_lost,
        (unsigned long long) TSR_REGION_OPL_OFF, (unsigned long long) TSR_REGION_OPL_SIZE, (unsigned long long) TSR_REGION_IO_OFF,
        (unsigned long long) TSR_REGION_IO_SIZE, (unsigned long long) TSR_REGION_AUX_OFF, (unsigned long long) TSR_REGION_AUX_SIZE,
        (unsigned long long) TSR_REGION_WEIGHT_OFF);
    bool first = true;
    for (uint32_t k = 0; k < opl::N_KINDS; ++k) {
        if (opl::kind_rev(k) == 0) continue;
        app(s, "%s\"%u\":{\"name\":%s,\"args\":[", first ? "" : ",", k, esc(opl::kind_name(k)).c_str());
        first = false;
        for (const kind_spec & ks : kind_specs()) {
            if (ks.kind != k) continue;
            for (size_t i = 0; i < ks.args.size(); ++i) app(s, "%s%s", i ? "," : "", esc(ks.args[i]).c_str());
        }
        s += "]}";
    }
    s += "}}\n";
    write_file(path_in("meta.json"), s);

    std::string f = "{\"format\":\"tsr-cpu-1\",\"nodes\":[";
    {
        std::lock_guard<std::mutex> lock(g_ref_m);
        bool fi = true;
        for (const auto & kv : refusals()) {
            const std::string name = kv.first.substr(0, kv.first.rfind('|'));
            const refusal & r = kv.second;
            app(f, "%s{\"name\":%s,\"op\":%s,\"type\":%s,\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "],\"why\":%s}", fi ? "" : ",", esc(name).c_str(),
                esc(r.op).c_str(), esc(r.type_name).c_str(), r.ne[0], r.ne[1], r.ne[2], r.ne[3], esc(r.why).c_str());
            fi = false;
        }
    }
    f += "]}\n";
    write_file(path_in("cpu_fallback.json"), f);
}

} // namespace tsr_prof
