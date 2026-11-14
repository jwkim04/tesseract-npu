// test-tsr-backend-units.cpp — unit tests of the ggml-independent parts of ggml-tesseract
// (ctest ggml_tesseract_units, docs/design/
// 06-software-verification.md §10.2): the range allocator (tsr-alloc), the MUL_MAT shape checks,
// SPM tiling and staging decisions (tsr-lower), and the op-list path's host side (tsr-opl): the
// encoding through
// tsr/opl.hpp's writer (read back with its reader and the executor's checks, plan.hpp accepts every
// record), the hazards over DRAM byte ranges (RAW, WAR, WAW, channels, byte boundaries, the addend,
// ACC_DST, the line-padded T = 1 store), the transitive reduction, JOIN folding above two
// predecessors, the op a TILE_ERR answer names (RET - 1 to its tag, with a JOIN in between), the
// 4096-op limit, a random property check (every conflicting pair ordered, every direct dependency a
// conflict), and the OPL cache (hits, misses, LRU eviction by bytes and by entries, a list larger
// than the region, a failing write). No device: tsr-opl-dev.cpp (the launcher) is not linked in.
// The builder (SW §9.3 M7): the DRAM ranges of every M7 kind (row periods, per-row scalars, weights, KV planes,
// the f32 mask copy) and the plane hazards SET_ROWS_KV -> FLASH_ATTN -> SET_ROWS_KV; the preload table
// (deduplication, 64-byte entries, the 256-line limit, NO_ROOM). With ggml-base (TSR_UNITS_BUILD:
// tsr-build.cpp compiled in, graphs on fake device addresses): fusion pairs (MUL_MAT + bias / residual
// ADD, RMS_NORM + MUL) and the cases that must not fuse (a second user, an output, the addend written
// in between, a scalar b, the norm as src1, an addend off the device); records of views (a fused-qkv
// projection's q / k views read by ROPE, a copy into the k part) and KV planes with their
// dependencies, one shared positions entry and FA_NEW naming the K indices (also in a list of its
// own); the graph key (addresses, shapes and parameters, not values); supports_op in probe mode
// against the planners on independently built argument words with the 64-byte rule, over sweeps of
// shapes of every kind (ADD / MUL / SCALE / UNARY / RMS_NORM / GLU / CONT / GET_ROWS / ROPE /
// SET_ROWS / FLASH_ATTN_EXT); TSR_KINDS parsing (tsr_kinds_parse).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <stdexcept>

#include "tsr-alloc.h"
#include "tsr-lower.h"
#include "tsr-opl.h"
#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_ref_at.h"
#if defined(TSR_UNITS_BUILD)
// the M7 builder (tsr-build.cpp, compiled into this test with ggml-base when it exists)
#include <cmath>
#include <functional>
#include <string>
#include "ggml.h"
#include "tsr-build.h"
#include "tsr-kinds.h"
#include "tsr/plan.hpp"
#endif

namespace opl = tsr::opl;
namespace mm  = tsr::opl::mm;

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                            \
        }                                                                        \
    } while (0)

static void test_alloc_basic() {
    tsr_range_alloc a;
    a.init(4096, 1 << 20, 64);
    CHECK(a.check());
    uint64_t o1 = 0, o2 = 0, o3 = 0;
    CHECK(a.alloc(100, &o1) && o1 == 4096);
    CHECK(a.size_of(o1) == 128);
    CHECK(a.alloc(64, &o2) && o2 == 4096 + 128);
    CHECK(a.alloc(1, &o3) && o3 == 4096 + 192);
    CHECK(a.used() == 256);
    a.release(o2);
    CHECK(a.check());
    uint64_t o4 = 0;
    CHECK(a.alloc(64, &o4) && o4 == o2);          // first fit reuses the hole
    a.release(o1);
    a.release(o4);
    a.release(o3);
    CHECK(a.used() == 0 && a.check() && a.largest_free() == (1u << 20));
    uint64_t big = 0;
    CHECK(!a.alloc((1u << 20) + 1, &big));
    CHECK(a.alloc(1u << 20, &big) && big == 4096 && a.free_bytes() == 0);
    a.release(big);
    CHECK(a.check());
    // top-down allocation
    uint64_t t1 = 0, b1 = 0;
    CHECK(a.alloc_top(100, &t1) && t1 == 4096 + (1u << 20) - 128);
    CHECK(a.alloc(64, &b1) && b1 == 4096);
    CHECK(a.check() && a.largest_free() == (1u << 20) - 128 - 64);
    a.release(t1);
    a.release(b1);
    CHECK(a.check() && a.used() == 0);
}

static void test_alloc_random() {
    std::mt19937_64 rng(1234);
    tsr_range_alloc a;
    a.init(0, 64ull << 20, 4096);
    std::map<uint64_t, uint64_t> live;
    for (int it = 0; it < 20000; ++it) {
        if (live.empty() || (rng() % 3) != 0) {
            const uint64_t sz = 1 + rng() % (1u << 20);
            uint64_t off = 0;
            if (a.alloc(sz, &off)) {
                CHECK(off % 4096 == 0);
                // no overlap with any live allocation
                auto nx = live.lower_bound(off);
                if (nx != live.end()) CHECK(off + sz <= nx->first);
                if (nx != live.begin()) {
                    auto pv = std::prev(nx);
                    CHECK(pv->first + pv->second <= off);
                }
                live[off] = sz;
            }
        } else {
            auto it2 = live.begin();
            std::advance(it2, rng() % live.size());
            a.release(it2->first);
            live.erase(it2);
        }
        if (it % 1000 == 0) CHECK(a.check());
    }
    for (auto & kv : live) a.release(kv.first);
    CHECK(a.check() && a.used() == 0);
}

static tsr_config cfg_with_tmax(uint32_t t_max) {
    tsr_config c = TSR_CONFIG_U200;
    c.t_max = t_max;
    return c;
}

static void test_shape_ok() {
    const tsr_config c = cfg_with_tmax(1);
    CHECK(tsr_mm_shape_ok(c, TSR_FMT_Q8_0, 64, 32));
    CHECK(!tsr_mm_shape_ok(c, TSR_FMT_Q8_0, 64, 48));          // K % 32
    CHECK(!tsr_mm_shape_ok(c, TSR_FMT_Q4_K, 64, 32));          // Q4_K needs K % 256
    CHECK(tsr_mm_shape_ok(c, TSR_FMT_Q4_K, 64, 27648));
    CHECK(tsr_mm_shape_ok(c, TSR_FMT_Q8_0, 1, 32768));         // K = K_MAX
    CHECK(!tsr_mm_shape_ok(c, TSR_FMT_Q8_0, 1, 32768 + 32));   // K > K_MAX
    CHECK(!tsr_mm_shape_ok(c, TSR_FMT_Q8_0, 0, 32));           // N = 0
    CHECK(!tsr_mm_shape_ok(c, 2, 64, 256));                    // reserved fmt
    CHECK(tsr_mm_shape_ok(c, TSR_FMT_Q8_0, 152064, 5120));     // Qwen2.5-32B output.weight (converted)
    CHECK(!tsr_mm_shape_ok(c, TSR_FMT_Q8_0, 2000000, 5120));   // one token's y does not fit 4 MiB SPM
}

// The tiling invariants of one MUL_MAT shape.
static void check_tiling(const tsr_config & c, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T) {
    tsr_mm_tiling t;
    CHECK(tsr_mm_plan_tiling(c, fmt, N, K, T, &t));
    const uint64_t spm_lines = c.spm_bytes / 64;
    CHECK(t.t_tile >= 1 && t.t_tile <= std::min<uint32_t>(c.t_max, T));
    CHECK(t.spm_end <= spm_lines);
    CHECK(t.kind == tsr_qa_kind_for_fmt(fmt));
    CHECK(t.x_lines == K * 4 / 64);
    CHECK(t.qa_lines == tsr_qa_lines(tsr_qa_kind_for_fmt(fmt), K));
    CHECK(t.y_lines == (N * 4 + 63) / 64);
    CHECK(t.spm_x == 0 && t.spm_qa == t.spm_x + t.t_tile * t.x_lines && t.spm_y == t.spm_qa + t.t_tile * t.qa_lines);
    CHECK(t.spm_end == t.spm_y + t.t_tile * t.y_lines);
    CHECK(tsr_mm_lines_per_token(fmt, N, K) == (uint64_t) t.x_lines + t.qa_lines + t.y_lines);
    // the largest tile that fits: one more token would not
    if (t.t_tile < std::min<uint32_t>(tsr_mm_t_max(c), T)) {
        CHECK((uint64_t) (t.t_tile + 1) * tsr_mm_lines_per_token(fmt, N, K) > spm_lines);
    }
}

static void test_tiling() {
    for (uint32_t tmax : { 1u, 2u, 8u }) {
        const tsr_config c = cfg_with_tmax(tmax);
        const struct { uint32_t fmt, N, K, T; } cases[] = {
            { TSR_FMT_Q8_0,     64,    32,   1 },
            { TSR_FMT_Q8_0,     33,    96,   7 },
            { TSR_FMT_Q4_K,   5120, 27648,   3 },
            { TSR_FMT_Q4_K,     40,   512,  17 },
            { TSR_FMT_Q8_0, 152064,  5120,   2 },
            { TSR_FMT_Q8_0,      1,    32, 512 },
        };
        for (const auto & d : cases) {
            check_tiling(c, d.fmt, d.N, d.K, d.T);
        }
        CHECK(tsr_mm_t_max(c) == tmax);
    }
    // T tile limited by SPM rather than T_MAX: y of 152064 rows = 9504 lines per token
    {
        const tsr_config c = cfg_with_tmax(8);
        tsr_mm_tiling t;
        CHECK(tsr_mm_plan_tiling(c, TSR_FMT_Q8_0, 152064, 5120, 8, &t));
        CHECK(t.t_tile == 6);   // 65536 / (320 + 90 + 9504) = 6.6
        CHECK(t.spm_end <= 65536);
    }
    // not lowerable: T = 0, a bad shape
    tsr_mm_tiling t;
    CHECK(!tsr_mm_plan_tiling(cfg_with_tmax(1), TSR_FMT_Q8_0, 64, 32, 0, &t));
    CHECK(!tsr_mm_plan_tiling(cfg_with_tmax(1), TSR_FMT_Q8_0, 64, 48, 1, &t));
    CHECK(tsr_mm_lines_per_token(2, 64, 256) == 0 && tsr_mm_lines_per_token(TSR_FMT_Q8_0, 64, 48) == 0);
}

// Staging decisions and stage tiles (tsr_plan_stage, used by the backend's tsr_lower_staged).
static void test_stage_plan() {
    const uint64_t SO = 0x13000000ull, SS = 64 * 1024;
    tsr_stage_plan p;
    // everything in IO with line-aligned rows (N % 16 == 0) or one token: nothing staged
    CHECK(!tsr_plan_stage(SO, SS, 64, 7, 512, 256, true, true, &p) && !p.stage_x && !p.stage_y);
    CHECK(!tsr_plan_stage(SO, SS, 33, 1, 512, 192, true, true, &p) && !p.stage_y);
    // dst rows not on a line with T > 1: only y through the stage
    CHECK(tsr_plan_stage(SO, SS, 33, 5, 512, 192, true, true, &p) && !p.stage_x && p.stage_y && p.per_token == 192);
    CHECK(p.tc_max == 5 && p.n_tiles == 1 && p.y_off == SO);
    // host-memory src1 and dst, many tokens: tiles of the largest token count that fits
    for (uint32_t T : { 1u, 40u, 85u, 86u, 1000u }) {
        CHECK(tsr_plan_stage(SO, SS, 96, T, 512, 384, false, false, &p) && p.stage_x && p.stage_y);
        CHECK(p.per_token == 896 && p.tc_max == std::min<uint64_t>(T, SS / 896) && p.tc_max >= 1);
        CHECK(p.x_off == SO && p.y_off == SO + p.tc_max * 512 && p.y_off + p.tc_max * 384 <= SO + SS);
        CHECK(p.n_tiles * p.tc_max >= T && (p.n_tiles - 1) * p.tc_max < T);
        if (p.tc_max < T) CHECK((p.tc_max + 1) * 896 > SS);
    }
    // one token larger than the stage: refused (the backend aborts with the sizes)
    CHECK(!tsr_plan_stage(SO, 1024, 96, 3, 512, 768, false, false, &p) && p.tc_max == 0 && (p.stage_x || p.stage_y));
}

// ---------------------------------------------------------------------------------------------
// OPL builder
// ---------------------------------------------------------------------------------------------

static constexpr uint64_t IO = 0x4000000ull;   // TSR_REGION_IO_OFF

// MUL_MAT args: x / y at ch0 byte offsets, weight at W_OFF beat w.
static void mm_args(uint32_t * a, uint64_t x, uint64_t y, uint32_t w, uint32_t N, uint32_t K, uint32_t T,
                    uint32_t fmt = TSR_FMT_Q8_0, uint32_t add = opl::HANDLE_NONE, uint32_t add_stride = 0, uint32_t flags = 0) {
    a[mm::X] = tsr_dram_handle(0, x);
    a[mm::X_STRIDE] = K / 16;
    a[mm::Y] = tsr_dram_handle(0, y);
    a[mm::Y_STRIDE] = (N + 15) / 16;
    a[mm::W_OFF] = w;
    a[mm::FMT] = fmt;
    a[mm::N] = N;
    a[mm::K] = K;
    a[mm::T] = T;
    a[mm::ADD] = add;
    a[mm::ADD_STRIDE] = add_stride;
    a[mm::FLAGS] = flags;
}

static uint32_t add_mm(tsr_opl_builder & b, uint64_t x, uint64_t y, uint32_t N, uint32_t K, uint32_t T, int32_t tag,
                       uint32_t add = opl::HANDLE_NONE, uint32_t flags = 0) {
    uint32_t a[mm::NARGS];
    mm_args(a, x, y, 0x800000u + 0x1000u * (uint32_t) tag, N, K, T, TSR_FMT_Q8_0, add, 0, flags);
    return b.mul_mat(a, tag);
}

// The direct dependencies of op i as encoded, through JOIN records (the real predecessors).
static void real_preds(const tsr_opl_builder & b, uint32_t i, std::set<uint32_t> & out) {
    const tsr_opl_builder::op_info & o = b.op(i);
    for (uint32_t d : { o.dep0, o.dep1 }) {
        if (d == opl::DEP_NONE) continue;
        if (b.op(d).kind == opl::K_JOIN) {
            real_preds(b, d, out);
        } else {
            out.insert(d);
        }
    }
}

// Reachability over the encoded deps (of the blob read back with opl::reader).
static std::vector<std::vector<bool>> reach_of_blob(const std::vector<uint8_t> & blob, std::vector<opl::record> * recs) {
    opl::reader r;
    CHECK(r.open(blob.data(), (uint32_t) blob.size(), true) == opl::RET_OK);
    const uint32_t n = r.hdr().n_ops;
    std::vector<std::vector<bool>> reach(n, std::vector<bool>(n, false));
    opl::record rec{};
    for (uint32_t i = 0; i < n; ++i) {
        CHECK(r.next(rec) == opl::RET_OK);
        if (recs) recs->push_back(rec);
        for (uint32_t d : { rec.dep0, rec.dep1 }) {
            if (d == opl::DEP_NONE) continue;
            CHECK(d < i);
            reach[i][d] = true;
            for (uint32_t k = 0; k < d; ++k) {
                if (reach[d][k]) reach[i][k] = true;
            }
        }
    }
    return reach;
}

static void test_opl_encoding() {
    const tsr_config c = cfg_with_tmax(8);
    tsr_opl_builder b(c.num_ch);
    // a 0.5B-like layer: q / k / v from x, o from q, gate / up from h, down from gate
    const uint32_t D = 896, KV = 128, FF = 4864;
    const uint64_t x = IO, q = IO + 0x10000, k = IO + 0x20000, v = IO + 0x30000, h = IO + 0x40000, g = IO + 0x50000,
                   u = IO + 0x60000, o2 = IO + 0x70000;
    const uint32_t iq = add_mm(b, x, q, D, D, 1, 0);
    const uint32_t ik = add_mm(b, x, k, KV, D, 1, 1);
    const uint32_t iv = add_mm(b, x, v, KV, D, 1, 2);
    const uint32_t io = add_mm(b, q, h, D, D, 1, 3);
    const uint32_t ig = add_mm(b, h, g, FF, D, 1, 4);
    const uint32_t iu = add_mm(b, h, u, FF, D, 1, 5);
    const uint32_t id = add_mm(b, g, o2, D, FF, 1, 6);
    CHECK(iq == 0 && ik == 1 && iv == 2 && io == 3 && ig == 4 && iu == 5 && id == 6);
    CHECK(b.op(iq).dep0 == opl::DEP_NONE && b.op(ik).dep0 == opl::DEP_NONE && b.op(iv).dep0 == opl::DEP_NONE);
    CHECK(b.op(io).dep0 == iq && b.op(io).dep1 == opl::DEP_NONE);           // RAW on q only
    CHECK(b.op(ig).dep0 == io && b.op(iu).dep0 == io && b.op(id).dep0 == ig);
    CHECK(b.n_joins() == 0 && b.ok());
    const std::vector<uint8_t> blob = b.finish();
    CHECK(blob.size() == 64 + 7 * 64);
    opl::reader r;
    CHECK(r.open(blob.data(), (uint32_t) blob.size(), true) == opl::RET_OK);
    CHECK(r.hdr().n_ops == 7 && r.hdr().flags == opl::F_STOP_ON_ERROR && r.hdr().graph_id == r.hdr().crc32);
    CHECK(opl::check_header(r.hdr(), (uint32_t) blob.size()) == opl::RET_OK);
    opl::record rec{};
    const tsr::plan::dev_cfg pc = tsr_plan_cfg(c);
    for (uint32_t i = 0; i < 7; ++i) {
        CHECK(r.next(rec) == opl::RET_OK && rec.kind == opl::K_MUL_MAT && rec.n_args == mm::NARGS && rec.n_slots == 1);
        uint32_t a[mm::NARGS];
        for (uint32_t j = 0; j < mm::NARGS; ++j) a[j] = rec.arg(j);
        CHECK(tsr::plan::check_op(pc, rec.kind, a, rec.n_args) == opl::RET_OK);
        CHECK(a[mm::W_OFF] == 0x800000u + 0x1000u * i && a[mm::T] == 1 && a[mm::FLAGS] == 0 && a[mm::ADD] == opl::HANDLE_NONE);
    }
    CHECK(r.next(rec) == opl::RET_BAD_RECORD);       // past the last op
    // equal lists are equal blobs (graph_id = CRC); another list differs
    tsr_opl_builder b2(c.num_ch);
    add_mm(b2, x, q, D, D, 1, 0);
    add_mm(b2, x, k, KV, D, 1, 1);
    add_mm(b2, x, v, KV, D, 1, 2);
    add_mm(b2, q, h, D, D, 1, 3);
    add_mm(b2, h, g, FF, D, 1, 4);
    add_mm(b2, h, u, FF, D, 1, 5);
    add_mm(b2, g, o2, D, FF, 1, 6);
    CHECK(b2.finish() == blob);
    b2.reset();
    CHECK(b2.empty() && b2.finish().empty());
    add_mm(b2, x, q, D, D, 2, 0);
    CHECK(b2.finish() != blob);
    // handles
    CHECK(tsr_dram_handle(0, IO) == (uint32_t) (IO / 64) && tsr_dram_handle(3, 0x40) == 0x30000001u);
    CHECK(tsr_handle_ch(0x30000001u) == 3 && tsr_handle_byte(0x30000001u) == 64);
}

static void test_opl_ranges() {
    tsr_opl_builder b(4);
    uint32_t a[mm::NARGS];
    std::vector<tsr_dram_range> rd, wr;
    // x: 3 rows of 896/16 = 56 lines, stride 56; y: 3 rows of 40/16 -> 3 lines, stride 3
    mm_args(a, IO, IO + 0x100000, 0x800000u, 40, 896, 3);
    b.mm_ranges(a, rd, wr);
    CHECK(rd.size() == 1 + 4 && wr.size() == 1);
    CHECK(rd[0].ch == 0 && rd[0].lo == IO && rd[0].hi == IO + 3 * 56 * 64 && rd[0].row == 0);
    CHECK(wr[0].ch == 0 && wr[0].lo == IO + 0x100000 && wr[0].hi == IO + 0x100000 + 3 * 3 * 64);
    const uint64_t wb = tsr_tensor_ch_beats(TSR_FMT_Q8_0, 40, 896, 4) * 64;
    for (uint32_t ch = 0; ch < 4; ++ch) {
        CHECK(rd[1 + ch].ch == ch && rd[1 + ch].lo == 0x800000ull * 64 && rd[1 + ch].hi == 0x800000ull * 64 + wb);
    }
    // addend with stride 0 (one row), ACC_DST (y read too); a T = 1 row of N = 33: 3 lines written
    mm_args(a, IO, IO + 0x100000, 0x800000u, 33, 64, 1, TSR_FMT_Q8_0, tsr_dram_handle(1, 0x2000), 0, 0);
    b.mm_ranges(a, rd, wr);
    CHECK(rd.size() == 2 + 4 && rd[1].ch == 1 && rd[1].lo == 0x2000 && rd[1].hi == 0x2000 + 3 * 64);
    CHECK(wr[0].hi - wr[0].lo == 3 * 64);
    mm_args(a, IO, IO + 0x100000, 0x800000u, 32, 64, 4, TSR_FMT_Q8_0, opl::HANDLE_NONE, 0, mm::FLAG_ACC_DST);
    b.mm_ranges(a, rd, wr);
    CHECK(rd.size() == 2 + 4 && rd[1].lo == wr[0].lo && rd[1].hi == wr[0].hi && wr[0].hi - wr[0].lo == 4 * 2 * 64);
    // strided rows: x 4 lines every 12 (a K part of a wider row), y 2 lines every 6
    mm_args(a, IO, IO + 0x100000, 0x800000u, 32, 64, 3);
    a[mm::X_STRIDE] = 12;
    a[mm::Y_STRIDE] = 6;
    b.mm_ranges(a, rd, wr);
    CHECK(rd[0].row == 4 * 64 && rd[0].stride == 12 * 64 && rd[0].hi == IO + (2 * 12 + 4) * 64);
    CHECK(wr[0].row == 2 * 64 && wr[0].stride == 6 * 64 && wr[0].hi == IO + 0x100000 + (2 * 6 + 2) * 64);
}

// tsr_ranges_overlap against a byte-set model, and the normalization of tsr_rows_range.
static void test_range_overlap() {
    CHECK(tsr_rows_range(0, 0, 4, 64, 64).row == 0 && tsr_rows_range(0, 0, 4, 64, 64).hi == 256);   // touching rows
    CHECK(tsr_rows_range(0, 0, 1, 64, 640).row == 0 && tsr_rows_range(0, 0, 1, 64, 640).hi == 64);  // one row
    CHECK(tsr_rows_range(0, 0, 5, 64, 0).row == 0 && tsr_rows_range(0, 0, 5, 64, 0).hi == 64);      // the same row
    CHECK(tsr_rows_range(0, 0, 3, 64, 32).row == 0 && tsr_rows_range(0, 0, 3, 64, 32).hi == 128);   // overlapping rows
    const tsr_dram_range s = tsr_rows_range(1, 1000, 3, 10, 100);
    CHECK(s.row == 10 && s.stride == 100 && s.lo == 1000 && s.hi == 1210);
    // interleaved [a | b | c] rows: pairwise disjoint, a reader of the whole span meets each
    const tsr_dram_range ra = tsr_rows_range(0, 0, 3, 256, 768), rb = tsr_rows_range(0, 256, 3, 256, 768),
                         rc = tsr_rows_range(0, 512, 3, 256, 768), all = tsr_rows_range(0, 0, 1, 3 * 768, 0);
    CHECK(!tsr_ranges_overlap(ra, rb) && !tsr_ranges_overlap(rb, rc) && !tsr_ranges_overlap(ra, rc));
    CHECK(tsr_ranges_overlap(ra, all) && tsr_ranges_overlap(all, rb) && tsr_ranges_overlap(rc, all));
    CHECK(!tsr_ranges_overlap(ra, tsr_rows_range(0, 256, 1, 512, 0)) && tsr_ranges_overlap(ra, tsr_rows_range(0, 255, 1, 2, 0)));
    CHECK(!tsr_ranges_overlap(ra, tsr_rows_range(1, 0, 3, 256, 768)));                  // other channel
    // random ranges against bytes
    std::mt19937_64 rng(99);
    auto bytes_of = [](const tsr_dram_range & r) {
        std::set<uint64_t> b;
        if (r.row == 0) {
            for (uint64_t x = r.lo; x < r.hi; ++x) b.insert(x);
        } else {
            for (uint64_t p = r.lo; p + r.row <= r.hi; p += r.stride)
                for (uint64_t x = p; x < p + r.row; ++x) b.insert(x);
        }
        return b;
    };
    for (int it = 0; it < 20000; ++it) {
        auto rnd = [&]() {
            const uint64_t n = 1 + rng() % 6, row = 1 + rng() % 12, stride = rng() % 24, lo = rng() % 64;
            return tsr_rows_range(0, lo, n, row, stride);
        };
        const tsr_dram_range x = rnd(), y = rnd();
        const std::set<uint64_t> bx = bytes_of(x), by = bytes_of(y);
        bool want = false;
        for (uint64_t v : bx) want = want || by.count(v) != 0;
        if (tsr_ranges_overlap(x, y) != want || tsr_ranges_overlap(y, x) != want) {
            CHECK(false);
            fprintf(stderr, "  x [%llu,%llu) row %llu stride %llu, y [%llu,%llu) row %llu stride %llu: want %d\n",
                    (unsigned long long) x.lo, (unsigned long long) x.hi, (unsigned long long) x.row, (unsigned long long) x.stride,
                    (unsigned long long) y.lo, (unsigned long long) y.hi, (unsigned long long) y.row, (unsigned long long) y.stride,
                    (int) want);
            break;
        }
    }
}

static void test_opl_hazards() {
    // RAW / WAR / WAW, independent, channels, byte boundaries
    {
        tsr_opl_builder b(1);
        const uint32_t a0 = add_mm(b, IO, IO + 0x1000, 32, 64, 2, 0);         // reads [IO, +512), writes [+0x1000, +128)
        const uint32_t a1 = add_mm(b, IO, IO + 0x2000, 32, 64, 2, 1);         // same x, other y: independent
        const uint32_t a2 = add_mm(b, IO + 0x1000, IO + 0x3000, 32, 32, 1, 2); // reads a0's y: RAW
        const uint32_t a3 = add_mm(b, IO + 0x4000, IO, 16, 32, 1, 3);          // writes a0/a1's x: WAR on both
        const uint32_t a4 = add_mm(b, IO + 0x5000, IO + 0x2000, 32, 32, 1, 4); // writes a1's y: WAW
        CHECK(b.op(a1).dep0 == opl::DEP_NONE);
        CHECK(b.op(a2).dep0 == a0 && b.op(a2).dep1 == opl::DEP_NONE);
        CHECK(b.op(a3).dep0 == a0 && b.op(a3).dep1 == a1);
        CHECK(b.op(a4).dep0 == a1 && b.op(a4).dep1 == opl::DEP_NONE);
    }
    {
        tsr_opl_builder b(2);
        const uint32_t a0 = add_mm(b, IO, IO + 0x1000, 32, 64, 1, 0);          // y [IO+0x1000, +128)
        const uint32_t a1 = add_mm(b, IO + 0x1000 + 128, IO + 0x9000, 16, 32, 1, 1);   // x right after: no overlap
        const uint32_t a2 = add_mm(b, IO + 0x1000 + 64, IO + 0xA000, 16, 32, 1, 2);    // x inside y's 2nd line
        CHECK(b.op(a1).dep0 == opl::DEP_NONE && b.op(a2).dep0 == a0);
        // the same bytes on channel 1: independent
        uint32_t a[mm::NARGS];
        mm_args(a, IO, IO + 0x1000, 0x800000u, 32, 64, 1);
        a[mm::X] = tsr_dram_handle(1, IO + 0x1000);
        a[mm::Y] = tsr_dram_handle(1, IO);
        const uint32_t a3 = b.mul_mat(a, 3);
        CHECK(b.op(a3).dep0 == opl::DEP_NONE);
        // the line-padded T = 1 store: N = 8 writes 32 bytes, the ST writes the whole line
        const uint32_t a4 = add_mm(b, IO + 0x20000, IO + 0x30000, 8, 32, 1, 4);
        // a reader of bytes 40..44 of that line (past the 32 data bytes, in the tensor's padding)
        const uint32_t a5 = b.add(opl::K_NOP, nullptr, 0, { { 0, IO + 0x30000 + 40, IO + 0x30000 + 44 } }, {}, 5);
        CHECK(b.op(a5).dep0 == a4);
        // touches(): pending writes, and with any_access reads
        CHECK(b.touches({ 0, IO + 0x30000, IO + 0x30001 }, false));
        CHECK(!b.touches({ 0, IO + 0x20000, IO + 0x20001 }, false) && b.touches({ 0, IO + 0x20000, IO + 0x20001 }, true));
    }
    // the addend and ACC_DST are reads of their rows
    {
        tsr_opl_builder b(1);
        const uint32_t a0 = add_mm(b, IO, IO + 0x1000, 32, 64, 1, 0);
        const uint32_t a1 = add_mm(b, IO + 0x2000, IO + 0x3000, 32, 64, 1, 1, tsr_dram_handle(0, IO + 0x1000));
        CHECK(b.op(a1).dep0 == a0);
        const uint32_t a2 = add_mm(b, IO, IO + 0x5000, 32, 64, 1, 2);
        const uint32_t a3 = add_mm(b, IO + 0x6000, IO + 0x5000, 32, 64, 1, 3, opl::HANDLE_NONE, mm::FLAG_ACC_DST);  // K part 2
        CHECK(b.op(a2).dep0 == opl::DEP_NONE && b.op(a3).dep0 == a2 && b.op(a3).dep1 == opl::DEP_NONE);
    }
    // three projections writing interleaved per-token rows [a | b | c] (strided y): independent; a
    // consumer of all three needs a JOIN
    {
        tsr_opl_builder b(1);
        uint32_t a[mm::NARGS], ix[3];
        for (uint32_t i = 0; i < 3; ++i) {
            mm_args(a, IO, IO + 0x10000 + 256 * i, 0x900000u + i, 64, 64, 3);   // y rows 4 lines, stride 12
            a[mm::Y_STRIDE] = 12;
            ix[i] = b.mul_mat(a, (int32_t) i);
            CHECK(b.op(ix[i]).dep0 == opl::DEP_NONE);
        }
        mm_args(a, IO + 0x10000, IO + 0x20000, 0x900010u, 16, 192, 3);     // reads [a | b | c]
        const uint32_t k = b.mul_mat(a, 3);
        CHECK(b.n_joins() == 1 && k == 4 && b.op(3).kind == opl::K_JOIN && b.op(k).dep0 == ix[2] && b.op(k).dep1 == 3);
        // a K part reading x columns [64, 128) of 192 (strided x) does not conflict with a writer of
        // columns [0, 64) only
        mm_args(a, IO + 0x70000, IO + 0x30000, 0x900020u, 64, 64, 3);       // writes 4 lines every 12 from +0
        a[mm::Y_STRIDE] = 12;
        const uint32_t w = b.mul_mat(a, 4);
        mm_args(a, IO + 0x30000 + 256, IO + 0x50000, 0x900030u, 16, 64, 3);  // reads 4 lines every 12 from +256
        a[mm::X_STRIDE] = 12;
        const uint32_t r = b.mul_mat(a, 5);
        CHECK(b.op(w).dep0 == opl::DEP_NONE && b.op(r).dep0 == opl::DEP_NONE);
        mm_args(a, IO + 0x30000 + 192, IO + 0x60000, 0x900040u, 16, 64, 3);  // reads from +192: meets row 0 of w
        a[mm::X_STRIDE] = 12;
        const uint32_t r2 = b.mul_mat(a, 6);
        CHECK(b.op(r2).dep0 == w);
    }
    // transitive reduction: a2 conflicts with a0 and a1, a1 depends on a0 -> only a1 is kept
    {
        tsr_opl_builder b(1);
        const uint32_t a0 = add_mm(b, IO, IO + 0x1000, 16, 64, 1, 0);            // writes [0x1000, +64)
        const uint32_t a1 = add_mm(b, IO + 0x1000, IO + 0x1040, 16, 16, 1, 1);   // reads a0's y, writes right after
        // reads [0x1000, 0x1080): both y's
        uint32_t a[mm::NARGS];
        mm_args(a, IO + 0x1000, IO + 0x9000, 0x900000u, 16, 32, 1);
        const uint32_t a2 = b.mul_mat(a, 2);
        CHECK(b.op(a1).dep0 == a0);
        CHECK(b.op(a2).dep0 == a1 && b.op(a2).dep1 == opl::DEP_NONE);
        CHECK(b.n_dropped() == 1 && b.n_deps() == 2);
    }
}

static void test_opl_joins() {
    for (uint32_t np = 3; np <= 6; ++np) {
        tsr_opl_builder b(1);
        std::vector<uint32_t> preds;
        for (uint32_t i = 0; i < np; ++i) {
            preds.push_back(add_mm(b, IO, IO + 0x10000 + 0x100 * i, 16, 64, 1, (int32_t) i));   // y rows 0x100 apart
        }
        // reads every y: [0x10000, 0x10000 + 0x100 np)
        uint32_t a[mm::NARGS];
        mm_args(a, IO + 0x10000, IO + 0x80000, 0x900000u, 16, 64 * np, 1);   // K = 64 np: x spans np * 256 B
        const uint32_t k = b.mul_mat(a, 100);
        CHECK(b.n_joins() == np - 2 && k == np + (np - 2) && b.op(k).tag == 100);
        std::set<uint32_t> rp;
        real_preds(b, k, rp);
        CHECK(rp.size() == np);
        for (uint32_t p : preds) CHECK(rp.count(p) == 1);
        for (uint32_t j = np; j < k; ++j) CHECK(b.op(j).kind == opl::K_JOIN && b.op(j).tag == -1 && b.tag_of(j) == -1);
        // as the writer folds: JOIN(p0, p1), then pairs from the front
        CHECK(b.op(np).dep0 == preds[0] && b.op(np).dep1 == preds[1]);
        const std::vector<uint8_t> blob = b.finish();
        std::vector<opl::record> recs;
        const auto reach = reach_of_blob(blob, &recs);
        for (uint32_t p : preds) CHECK(reach[k][p]);
        CHECK(recs.size() == k + 1 && recs[np].kind == opl::K_JOIN && recs[np].n_args == 0);
    }
}

// The op a TILE_ERR answer names (SW §9.3 step 5, RET = index + 1): in a list whose indices and
// tags differ (a JOIN before the failing records), so an off-by-one names another node or a JOIN.
static void test_opl_failed_op() {
    tsr_opl_builder b(1);
    for (uint32_t i = 0; i < 3; ++i) {
        add_mm(b, IO, IO + 0x10000 + 0x100 * i, 16, 64, 1, (int32_t) (10 + i));   // ops 0-2, tags 10-12
    }
    uint32_t a[mm::NARGS];
    mm_args(a, IO + 0x10000, IO + 0x80000, 0x900000u, 16, 64 * 3, 1);
    const uint32_t k = b.mul_mat(a, 13);                          // 3 predecessors: JOIN at 3, op 4
    const uint32_t k2 = add_mm(b, IO, IO + 0x90000, 16, 64, 1, 14); // independent: op 5
    CHECK(k == 4 && k2 == 5 && b.op(3).kind == opl::K_JOIN && b.n_ops() == 6);
    const int32_t want_tag[8] = { -1, 10, 11, 12, -1, 13, 14, -1 };   // by RET
    tsr_opl_run_result r;
    r.res.type = TSR_LCQ_R0_TYPE_DONE;
    r.res.status = TSR_LCQ_R0_STATUS_TILE_ERR;
    for (int32_t ret = -1; ret <= 7; ++ret) {
        r.res.ret = ret;
        const int32_t want_op = ret >= 1 && ret <= 6 ? ret - 1 : -1;
        CHECK(tsr_opl_failed_op(r, b.n_ops()) == want_op);
        CHECK(tsr_opl_failed_tag(b, r) == (ret >= 0 ? want_tag[ret] : -1));
    }
    r.res.ret = 6;
    CHECK(tsr_opl_failed_tag(b, r) == 14 && tsr_opl_failed_op(r, 5) == -1);   // RET beyond a shorter list
    r.res.status = TSR_LCQ_R0_STATUS_KERNEL_ERR;                              // executor codes are not ops
    CHECK(tsr_opl_failed_op(r, b.n_ops()) == -1 && tsr_opl_failed_tag(b, r) == -1);
    r.res.status = TSR_LCQ_R0_STATUS_OK;
    CHECK(tsr_opl_failed_op(r, b.n_ops()) == -1);
    r.res.status = TSR_LCQ_R0_STATUS_TILE_ERR;
    r.res.type = TSR_LCQ_R0_TYPE_TRAP;
    CHECK(tsr_opl_failed_op(r, b.n_ops()) == -1);
    r.res.type = TSR_LCQ_R0_TYPE_DONE;
    r.rc = TSR_TE_TIMEOUT;                                                   // not answered
    CHECK(tsr_opl_failed_op(r, b.n_ops()) == -1 && tsr_opl_failed_tag(b, r) == -1);
}

static void test_opl_capacity() {
    tsr_opl_builder b(1);
    uint32_t last = 0;
    for (uint32_t i = 0; i < opl::MAX_OPS; ++i) {
        last = add_mm(b, IO, IO + 0x100000 + 64ull * i, 16, 32, 1, (int32_t) i);    // independent
    }
    CHECK(last == opl::MAX_OPS - 1 && b.n_ops() == opl::MAX_OPS);
    CHECK(add_mm(b, IO, IO + 0x900000, 16, 32, 1, 5000) == tsr_opl_builder::NO_ROOM && b.n_ops() == opl::MAX_OPS);
    const std::vector<uint8_t> blob = b.finish();
    CHECK(blob.size() == 64 + 64ull * opl::MAX_OPS);
    opl::reader r;
    CHECK(r.open(blob.data(), (uint32_t) blob.size(), true) == opl::RET_OK && r.hdr().n_ops == opl::MAX_OPS);
    // NO_ROOM also when the JOINs would not fit: 4094 ops, then one with 4 predecessors (2 JOINs,
    // 4097 records); one with 2 predecessors still fits (4095)
    tsr_opl_builder c(1);
    for (uint32_t i = 0; i < opl::MAX_OPS - 2; ++i) {
        add_mm(c, IO, IO + 0x100000 + 64ull * i, 16, 32, 1, (int32_t) i);
    }
    uint32_t a[mm::NARGS];
    mm_args(a, IO + 0x100000, IO + 0x900000, 0x900000u, 16, 64, 1);    // reads 4 y rows: 4 predecessors
    CHECK(c.mul_mat(a, 9) == tsr_opl_builder::NO_ROOM && c.n_ops() == opl::MAX_OPS - 2);
    mm_args(a, IO + 0x100000, IO + 0x900000, 0x900000u, 16, 32, 1);    // 2 predecessors: fits
    CHECK(c.mul_mat(a, 9) == opl::MAX_OPS - 2 && c.n_joins() == 0);
}


// ---------------------------------------------------------------------------------------------
// MUL_MAT deferral (SW §9.3 "Deferred MUL_MAT"): a plain one-tile MUL_MAT is kept pending, becomes MUL_MAT_BEGIN when the next
// record leaves its y alone and MUL_MAT otherwise; the MUL_MAT_END comes before the first record that touches y, and before
// every cut (settle).
// ---------------------------------------------------------------------------------------------
static uint32_t add_mm_d(tsr_opl_builder & b, uint64_t x, uint64_t y, uint32_t N, uint32_t K, uint32_t T, int32_t tag,
                         uint32_t add = opl::HANDLE_NONE, uint32_t flags = 0, uint32_t fmt = TSR_FMT_Q8_0) {
    uint32_t a[mm::NARGS];
    mm_args(a, x, y, 0x800000u + 0x1000u * (uint32_t) tag, N, K, T, fmt, add, 0, flags);
    return b.mul_mat(a, tag, opl::K_MUL_MAT, true);
}

static std::vector<uint32_t> kinds_of(const tsr_opl_builder & b) {
    std::vector<uint32_t> k;
    for (uint32_t i = 0; i < b.n_ops(); ++i) k.push_back(b.op(i).kind);
    return k;
}

static void test_opl_defer() {
    const uint32_t BEG = opl::K_MUL_MAT_BEGIN, END = opl::K_MUL_MAT_END, MM = opl::K_MUL_MAT;
    const uint64_t X = IO, YA = IO + 0x10000, YB = IO + 0x20000, YC = IO + 0x30000;
    // off by default: the records are the MUL_MATs, in order
    {
        tsr_opl_builder b(1);
        CHECK(!b.defer());
        CHECK(add_mm_d(b, X, YA, 16, 64, 1, 0) == 0 && add_mm_d(b, X, YB, 16, 64, 1, 1) == 1 && b.n_open() == 0 && !b.has_pending());
        CHECK((kinds_of(b) == std::vector<uint32_t>{ MM, MM }));
    }
    // A, B independent; C reads both y: BEGIN(A), MUL_MAT(B), END(A), C (a MUL_MAT is never BEGIN when the next record reads it)
    {
        tsr_opl_builder b(1);
        b.set_defer(true);
        CHECK(add_mm_d(b, X, YA, 16, 64, 1, 0) == tsr_opl_builder::PENDING && b.has_pending() && b.n_ops() == 0 && b.n_records() == 1 && !b.empty());
        CHECK(b.finish().empty());                                           // a pending MUL_MAT is not in the blob
        // touches() sees the pending record: its y as a write, its x with any_access
        CHECK(b.touches({ 0, YA, YA + 1 }, false) && !b.touches({ 0, X, X + 1 }, false) && b.touches({ 0, X, X + 1 }, true) &&
              !b.touches({ 0, YB, YB + 1 }, true));
        CHECK(add_mm_d(b, X, YB, 16, 64, 1, 1) == tsr_opl_builder::PENDING && b.n_ops() == 1 && b.n_open() == 1);
        uint32_t a[mm::NARGS];
        mm_args(a, YA, YC, 0x900000u, 16, 64, 1);
        // a consumer reading A's and B's y: two source rows (T = 2, 0x10000 B apart): the rows at YA and YB
        a[mm::X_STRIDE] = (uint32_t) ((YB - YA) / 64);
        a[mm::T] = 2;
        const uint32_t c = b.mul_mat(a, 2);
        CHECK(c == 3 && b.n_ops() == 4 && b.n_open() == 0 && !b.has_pending());
        CHECK((kinds_of(b) == std::vector<uint32_t>{ BEG, MM, END, MM }));
        // one node, one tag: BEGIN and END of A carry tag 0
        CHECK(b.op(0).tag == 0 && b.op(2).tag == 0 && b.op(1).tag == 1 && b.op(3).tag == 2);
        // END names its BEGIN, depends on it only; BEGIN of A and MUL_MAT of B have no dependencies
        const std::vector<uint8_t> blob = b.finish();
        std::vector<opl::record> recs;
        reach_of_blob(blob, &recs);
        CHECK(recs.size() == 4 && recs[2].kind == END && recs[2].n_args == 1 && recs[2].arg(0) == 0);
        CHECK(recs[2].dep0 == 0 && recs[2].dep1 == opl::DEP_NONE && recs[0].dep0 == opl::DEP_NONE && recs[1].dep0 == opl::DEP_NONE);
        // the consumer depends on END(A) and MUL_MAT(B), not on BEGIN(A) (implied by the END)
        std::set<uint32_t> rp;
        real_preds(b, 3, rp);
        CHECK((rp == std::set<uint32_t>{ 1, 2 }));
        // END's ranges: no read, y written; BEGIN's: the MUL_MAT's (y is its write)
        CHECK(b.op(2).rd.empty() && b.op(2).wr.size() == 1 && b.op(2).wr[0].lo == YA && b.op(0).wr.size() == 1 && b.op(0).wr[0].lo == YA && !b.op(0).rd.empty());
        // the planner accepts every record (BEGIN is one T-tile, END needs no scratch)
        for (const opl::record & r : recs) {
            if (r.kind == MM) continue;                    // plan_kind is the M7 planner; MUL_MAT's is mm_plan
            tsr::plan::kind_plan kp{};
            uint32_t w[opl::mm::NARGS] = {};
            for (uint32_t k = 0; k < r.n_args; ++k) w[k] = r.arg(k);
            CHECK(tsr::plan::plan_kind(tsr::plan::plan_limits_of(tsr::plan::U200_K4), r.kind, w, r.n_args, &kp));
        }
    }
    // the lazy END: C reads only A's y; B stays independent: BEGIN(A), BEGIN(B), END(A), C, then END(B) at the settle
    {
        tsr_opl_builder b(1);
        b.set_defer(true);
        add_mm_d(b, X, YA, 16, 64, 1, 0);
        add_mm_d(b, X, YB, 16, 64, 1, 1);
        const uint32_t c = add_mm(b, YA, YC, 16, 64, 1, 2);                // K = 64: reads y(A)'s line and the gap after it
        CHECK(c == 3 && b.n_open() == 1 && b.n_ops() == 4);
        CHECK((kinds_of(b) == std::vector<uint32_t>{ BEG, BEG, END, MM }));
        CHECK(b.op(2).tag == 0 && b.op(2).dep0 == 0);
        std::set<uint32_t> rp;
        real_preds(b, 3, rp);
        CHECK((rp == std::set<uint32_t>{ 2 }));
        CHECK(b.settle() && b.n_open() == 0 && b.n_ops() == 5 && b.op(4).kind == END && b.op(4).tag == 1);
        std::vector<opl::record> recs;
        reach_of_blob(b.finish(), &recs);
        CHECK(recs.size() == 5 && recs[2].arg(0) == 0 && recs[4].arg(0) == 1 && recs[4].dep0 == 1);
    }
    // the next record reads the result: no BEGIN
    {
        tsr_opl_builder b(1);
        b.set_defer(true);
        add_mm_d(b, X, YA, 16, 64, 1, 0);
        add_mm_d(b, YA, YB, 16, 64, 1, 1);                                  // reads y(A): A is a MUL_MAT; B is pending
        CHECK(b.n_ops() == 1 && b.op(0).kind == MM && b.n_open() == 0);
        CHECK(b.settle() && (kinds_of(b) == std::vector<uint32_t>{ MM, MM }) && b.op(1).dep0 == 0);    // nothing follows B: a MUL_MAT
    }
    // the next record writes the result (WAW), or overwrites what the BEGIN reads (WAR on x)
    {
        tsr_opl_builder b(1);
        b.set_defer(true);
        add_mm_d(b, X, YA, 16, 64, 1, 0);
        add_mm_d(b, X + 0x4000, YA, 16, 64, 1, 1);                          // writes y(A) again: A is a MUL_MAT
        CHECK(b.op(0).kind == MM && b.n_open() == 0);
        add_mm_d(b, X + 0x5000, YB, 16, 64, 1, 2);                          // B pending -> BEGIN(1) at this call, 2 pending
        CHECK(b.op(1).kind == BEG);
        const uint32_t w = add_mm(b, X + 0x6000, X + 0x4000, 16, 64, 1, 3);  // writes B's x (WAR on BEGIN) and nothing of y(B)
        // order: BEGIN(B=1), BEGIN(2) [pending 2 -> BEGIN], then the writer; END only for records touching y
        CHECK(b.op(w).kind == MM);
        std::set<uint32_t> rp;
        real_preds(b, w, rp);
        CHECK(rp.count(1) == 1);                                            // depends on the BEGIN that read x (its ticket is its last LD)
        CHECK(b.n_open() == 2);
        CHECK(b.settle() && b.n_open() == 0);
        const std::vector<uint8_t> blob = b.finish();
        CHECK(!blob.empty());
    }
    // the explicit predecessor: no range conflict, still a dependency; implied ones are reduced like any other
    {
        tsr_opl_builder b(1);
        uint32_t a[mm::NARGS];
        mm_args(a, X, YA, 0x800000u, 16, 64, 1);
        const uint32_t i0 = b.mul_mat(a, 0);
        mm_args(a, X, YB, 0x801000u, 16, 64, 1);
        const uint32_t i1 = b.mul_mat(a, 1);
        const uint32_t w[1] = { i0 };
        const std::vector<tsr_dram_range> none, wr = { tsr_rows_range(0, YC, 1, 64, 0) };
        const uint32_t e = b.add(opl::K_MUL_MAT_END, w, 1, none, wr, 2, (int32_t) i0);
        CHECK(e == 2 && b.op(e).dep0 == i0 && b.op(e).dep1 == opl::DEP_NONE && b.op(i1).dep0 == opl::DEP_NONE);
        // a reader of YC depends on e only; with the pred implied by another conflict (it reads YA too) it is dropped
        const uint32_t w2[1] = { e };
        const std::vector<tsr_dram_range> rdA = { tsr_rows_range(0, YA, 1, 64, 0) };
        const uint32_t e2 = b.add(opl::K_MUL_MAT_END, w2, 1, rdA, wr, 3, (int32_t) e);
        CHECK(b.op(e2).dep0 == e && b.op(e2).dep1 == opl::DEP_NONE && b.n_dropped() >= 1);
    }
    // 8 open at most: the ninth BEGIN comes after the END of the oldest
    {
        tsr_opl_builder b(1);
        b.set_defer(true);
        for (int32_t i = 0; i < 10; ++i) add_mm_d(b, X, IO + 0x40000 + 0x1000ull * i, 16, 64, 1, i);
        // 9 emitted (the tenth is pending): BEGIN x 8, END(0), BEGIN(8); open: 1..8
        CHECK(b.n_open() == 8 && b.n_ops() == 10);
        std::vector<uint32_t> want(8, BEG);
        want.push_back(END);
        want.push_back(BEG);
        CHECK(kinds_of(b) == want);
        CHECK(b.op(8).tag == 0 && b.op(9).tag == 8);
        CHECK(b.settle());
        CHECK(b.n_open() == 0 && b.n_of_kind(BEG) == 9 && b.n_of_kind(END) == 9 && b.n_of_kind(MM) == 1);
        std::vector<opl::record> recs;
        reach_of_blob(b.finish(), &recs);
        int open = 0, peak = 0;
        for (const opl::record & r : recs) {
            open += r.kind == BEG ? 1 : r.kind == END ? -1 : 0;
            peak = std::max(peak, open);
            CHECK(r.kind != END || (r.arg(0) < r.idx && recs[r.arg(0)].kind == BEG));
        }
        CHECK(peak == 8 && open == 0);
    }
    // the reserve at a full list: add answers NO_ROOM early enough that the ENDs of the open BEGINs fit
    for (uint32_t extra = 0; extra < 14; ++extra) {
        tsr_opl_builder b(1);
        b.set_defer(true);
        // independent plain records up to the edge, then deferred ones (each y of its own, so BEGINs stay open)
        const uint32_t plain = opl::MAX_OPS - 4 - extra;
        for (uint32_t i = 0; i < plain; ++i) {
            add_mm(b, IO, IO + 0x1000000 + 64ull * i, 16, 32, 1, (int32_t) i);
        }
        uint32_t last = 0;
        for (uint32_t i = 0; i < 40 && last != tsr_opl_builder::NO_ROOM; ++i) {
            last = add_mm_d(b, X, IO + 0x8000000 + 0x1000ull * i, 16, 64, 1, (int32_t) (plain + i));
            CHECK(b.n_ops() + b.n_open() + (b.has_pending() ? 1u : 0u) <= opl::MAX_OPS);
        }
        CHECK(last == tsr_opl_builder::NO_ROOM || b.n_ops() + b.n_open() <= opl::MAX_OPS);
        CHECK(b.settle() && b.n_ops() <= opl::MAX_OPS && b.n_open() == 0);
        const std::vector<uint8_t> blob = b.finish();
        opl::reader r;
        CHECK(!blob.empty() && r.open(blob.data(), (uint32_t) blob.size(), true) == opl::RET_OK && r.hdr().n_ops == b.n_ops());
        CHECK(b.n_of_kind(BEG) == b.n_of_kind(END));
    }
    // the reserve against plain records: with BEGINs open, records that need no END fill the list up to the ENDs' room
    for (uint32_t n_open = 1; n_open <= 8; ++n_open) {
        tsr_opl_builder b(1);
        b.set_defer(true);
        for (uint32_t i = 0; i <= n_open; ++i) {                                  // n_open BEGINs, the last record pending
            add_mm_d(b, X, IO + 0x8000000 + 0x1000ull * i, 16, 64, 1, (int32_t) i);
        }
        CHECK(b.n_open() == n_open && b.has_pending());
        uint32_t last = 0, i = 0;
        for (; i < opl::MAX_OPS && last != tsr_opl_builder::NO_ROOM; ++i) {
            last = add_mm(b, IO, IO + 0x1000000 + 64ull * i, 16, 32, 1, (int32_t) (100 + i));
        }
        CHECK(last == tsr_opl_builder::NO_ROOM);
        // the pending record became a BEGIN at the first plain record; the open BEGINs' ENDs still fit
        CHECK(b.n_ops() + b.n_open() <= opl::MAX_OPS && b.n_open() == std::min<uint32_t>(n_open + 1, 8));
        CHECK(b.settle() && b.n_ops() <= opl::MAX_OPS && b.n_open() == 0 && b.n_of_kind(END) == n_open + 1);
        CHECK(!b.finish().empty());
    }
    // a fresh builder after reset(): no pending, no open BEGIN
    {
        tsr_opl_builder b(1);
        b.set_defer(true);
        add_mm_d(b, X, YA, 16, 64, 1, 0);
        add_mm_d(b, X, YB, 16, 64, 1, 1);
        b.reset();
        CHECK(b.empty() && !b.has_pending() && b.n_open() == 0 && b.n_of_kind(BEG) == 0 && b.defer());
    }
}

// A random mix of deferred and plain MUL_MATs over a few slots: the deferred list orders every conflict of the program
// order as the eager list does (a reader or writer of y after the END of its BEGIN; a writer of x after the BEGIN), no
// record between a BEGIN and its END touches y, at most 8 BEGINs are open, every END names a BEGIN of the same tag.
static void test_opl_defer_random() {
    const uint32_t BEG = opl::K_MUL_MAT_BEGIN, END = opl::K_MUL_MAT_END, MM = opl::K_MUL_MAT;
    std::mt19937_64 rng(2025);
    uint64_t n_begin = 0, n_cases = 0, n_pairs = 0;
    for (int round = 0; round < 300; ++round) {
        tsr_opl_builder b(1);
        b.set_defer(true);
        const uint32_t nrec = 4 + (uint32_t) (rng() % 40);
        const uint32_t nslot = 3 + (uint32_t) (rng() % 5);
        struct prog { std::vector<tsr_dram_range> rd, wr; };
        std::vector<prog> progs;
        for (uint32_t i = 0; i < nrec; ++i) {
            // x: 4 lines at the start of a 0x200-byte slot; y: one line at the start of a slot or 0x100 into it (the x range
            // of the same slot ends at 0x100, so the two do not touch); an addend (one line) now and then
            const uint32_t xs = (uint32_t) (rng() % nslot), ys = (uint32_t) (rng() % nslot);
            const uint64_t x = IO + 0x200ull * xs, y = IO + 0x200ull * ys + (rng() % 2 ? 0x100 : 0);
            const uint32_t add = rng() % 5 == 0 ? tsr_dram_handle(0, IO + 0x200ull * (rng() % nslot) + (rng() % 2 ? 0x100 : 0)) : opl::HANDLE_NONE;
            uint32_t a[mm::NARGS];
            mm_args(a, x, y, 0x800000u + 0x1000u * i, 16, 64, 1, TSR_FMT_Q8_0, add, 0, 0);
            prog p;
            b.mm_ranges(a, p.rd, p.wr);
            const uint32_t r = b.mul_mat(a, (int32_t) i, MM, rng() % 4 != 0);
            CHECK(r != tsr_opl_builder::NO_ROOM && r != tsr_opl_builder::FAILED);
            progs.push_back(std::move(p));
        }
        CHECK(b.settle() && b.n_open() == 0);
        n_cases++;
        n_begin += b.n_of_kind(BEG);
        std::vector<opl::record> recs;
        const auto reach = reach_of_blob(b.finish(), &recs);
        // the first record of each program record (BEGIN or MUL_MAT) and the last (END or the same)
        std::vector<int64_t> first(nrec, -1), last(nrec, -1);
        int open = 0;
        for (uint32_t i = 0; i < b.n_ops(); ++i) {
            const tsr_opl_builder::op_info & o = b.op(i);
            if (o.kind == opl::K_JOIN) continue;
            CHECK(o.tag >= 0 && (uint32_t) o.tag < nrec);
            if (o.kind == END) {
                const uint32_t bg = recs[i].arg(0);
                CHECK(bg < i && b.op(bg).kind == BEG && b.op(bg).tag == o.tag && first[(size_t) o.tag] == (int64_t) bg);
                last[(size_t) o.tag] = i;
                open--;
                // no record between the BEGIN and the END reads or writes y
                const tsr_dram_range y = b.op(bg).wr[0];
                for (uint32_t k = bg + 1; k < i; ++k) {
                    for (const tsr_dram_range & q : b.op(k).rd) CHECK(!tsr_ranges_overlap(y, q));
                    for (const tsr_dram_range & q : b.op(k).wr) CHECK(!tsr_ranges_overlap(y, q));
                }
            } else {
                CHECK(first[(size_t) o.tag] < 0);
                first[(size_t) o.tag] = last[(size_t) o.tag] = i;
                open += o.kind == BEG ? 1 : 0;
                CHECK(open <= 8);
            }
        }
        CHECK(open == 0);
        for (uint32_t i = 0; i < nrec; ++i) CHECK(first[i] >= 0 && last[i] >= first[i]);
        for (uint32_t j = 0; j < nrec; ++j) {
            for (uint32_t i = 0; i < j; ++i) {
                bool raw = false, war = false;
                for (const tsr_dram_range & q : progs[i].wr) {
                    for (const tsr_dram_range & r : progs[j].rd) raw = raw || tsr_ranges_overlap(q, r);
                    for (const tsr_dram_range & r : progs[j].wr) raw = raw || tsr_ranges_overlap(q, r);
                }
                for (const tsr_dram_range & q : progs[i].rd) {
                    for (const tsr_dram_range & r : progs[j].wr) war = war || tsr_ranges_overlap(q, r);
                }
                if (raw) {        // j reads or writes y(i): after the ST
                    n_pairs++;
                    CHECK(reach[(size_t) first[j]][(size_t) last[i]]);
                }
                if (war) {        // j writes what i reads at its BEGIN: after the BEGIN
                    n_pairs++;
                    CHECK(reach[(size_t) first[j]][(size_t) first[i]]);
                }
            }
        }
    }
    CHECK(n_cases == 300 && n_begin > 100 && n_pairs > 500);
}

// Random ops over random ranges on two channels: every conflicting pair is ordered by the encoded
// DAG, every direct dependency (through JOINs) is a conflict, and no kept dependency is implied by
// another kept one.
static void test_opl_random() {
    std::mt19937_64 rng(77);
    for (int round = 0; round < 40; ++round) {
        tsr_opl_builder b(2);
        const uint32_t n = 20 + (uint32_t) (rng() % 120);
        std::vector<std::vector<tsr_dram_range>> RD, WR;
        auto rnd_range = [&]() {
            const uint32_t ch = (uint32_t) (rng() % 2);
            const uint64_t lo = (rng() % 64) * 16;
            if (rng() % 3 == 0) {
                return tsr_rows_range(ch, lo, 2 + rng() % 3, 16 + (rng() % 2) * 16, 32 + (rng() % 3) * 16);
            }
            return tsr_rows_range(ch, lo, 1, 16 + (rng() % 8) * 16, 0);
        };
        std::vector<uint32_t> idx;
        for (uint32_t i = 0; i < n; ++i) {
            std::vector<tsr_dram_range> rd, wr;
            for (uint32_t k = (uint32_t) (rng() % 3); k > 0; --k) rd.push_back(rnd_range());
            for (uint32_t k = (uint32_t) (rng() % 2) + (rd.empty() ? 1 : 0); k > 0; --k) wr.push_back(rnd_range());
            const uint32_t j = b.add(opl::K_NOP, nullptr, 0, rd, wr, (int32_t) i);
            CHECK(j != tsr_opl_builder::NO_ROOM && j != tsr_opl_builder::FAILED);
            idx.push_back(j);
            RD.push_back(rd);
            WR.push_back(wr);
        }
        const std::vector<uint8_t> blob = b.finish();
        const auto reach = reach_of_blob(blob, nullptr);
        auto conflict = [&](uint32_t i, uint32_t j) {      // op j (later) vs op i
            for (const auto & r : RD[j]) for (const auto & q : WR[i]) if (tsr_ranges_overlap(r, q)) return true;
            for (const auto & r : WR[j]) for (const auto & q : WR[i]) if (tsr_ranges_overlap(r, q)) return true;
            for (const auto & r : WR[j]) for (const auto & q : RD[i]) if (tsr_ranges_overlap(r, q)) return true;
            return false;
        };
        for (uint32_t j = 0; j < n; ++j) {
            for (uint32_t i = 0; i < j; ++i) {
                if (conflict(i, j)) CHECK(reach[idx[j]][idx[i]]);
            }
            std::set<uint32_t> rp;
            real_preds(b, idx[j], rp);
            for (uint32_t p : rp) {
                const int32_t pi = b.tag_of(p);
                CHECK(pi >= 0 && conflict((uint32_t) pi, j));
                for (uint32_t p2 : rp) CHECK(p2 == p || !reach[p2][p]);   // reduced
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// OPL cache
// ---------------------------------------------------------------------------------------------

static std::vector<uint8_t> blob_of(uint8_t fill, size_t lines) { return std::vector<uint8_t>(64 * lines, fill); }

static void test_opl_cache() {
    tsr_opl_cache c;
    c.init(0, 0x1000, 4 * 64 * 4);          // room for 4 lists of 4 lines
    std::map<uint64_t, std::vector<uint8_t>> dram;
    int writes = 0;
    const tsr_dram_write_fn wr = [&](uint32_t ch, uint64_t off, const void * p, size_t n) {
        CHECK(ch == 0);
        dram[off].assign((const uint8_t *) p, (const uint8_t *) p + n);
        writes++;
    };
    uint64_t off[8] = {};
    bool hit = true;
    const auto A = blob_of(1, 4), B = blob_of(2, 4), C = blob_of(3, 4), D = blob_of(4, 4), E = blob_of(5, 4);
    CHECK(c.place(A, wr, &off[0], &hit) && !hit && writes == 1 && dram[off[0]] == A && off[0] == 0x1000);
    CHECK(c.place(A, wr, &off[1], &hit) && hit && writes == 1 && off[1] == off[0]);
    const auto A2 = A;                                    // equal content in another vector: a hit
    CHECK(c.place(A2, wr, &off[1], &hit) && hit && writes == 1);
    CHECK(c.place(B, wr, &off[2], &hit) && !hit && writes == 2 && off[2] != off[0]);
    CHECK(c.place(C, wr, &off[3], &hit) && !hit && c.place(D, wr, &off[4], &hit) && !hit && c.entries() == 4);
    CHECK(c.place(A, wr, &off[1], &hit) && hit);          // A is now the most recently used
    CHECK(c.place(E, wr, &off[5], &hit) && !hit && c.entries() == 4 && off[5] == off[2]);   // evicts B (LRU)
    CHECK(c.get_stats().evictions == 1 && dram[off[5]] == E);
    CHECK(c.place(B, wr, &off[6], &hit) && !hit && off[6] == off[3]);                        // B again: evicts C
    CHECK(c.place(A, wr, &off[1], &hit) && hit && off[1] == off[0]);
    CHECK(c.get_stats().hits == 4 && c.get_stats().misses == 6 && c.get_stats().bytes_written == 6 * 256);
    // a list larger than one slot evicts as many as needed; one larger than the region is refused
    const auto big = blob_of(9, 12);
    CHECK(c.place(big, wr, &off[7], &hit) && !hit && c.used_bytes() == 256 * 4);
    CHECK(!c.place(blob_of(7, 17), wr, &off[7], &hit));
    // a failing write leaves the cache as it was
    const size_t n0 = c.entries();
    const uint64_t used0 = c.used_bytes();
    bool threw = false;
    try {
        c.place(blob_of(6, 4), [](uint32_t, uint64_t, const void *, size_t) { throw std::runtime_error("write"); }, &off[7], &hit);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw && c.entries() <= n0 && c.used_bytes() <= used0);
    CHECK(c.place(blob_of(6, 4), wr, &off[7], &hit) && !hit);
    // the entry bound
    tsr_opl_cache e;
    e.init(0, 0, 1 << 20, 2);
    CHECK(e.place(A, wr, &off[0], &hit) && e.place(B, wr, &off[1], &hit) && e.place(C, wr, &off[2], &hit));
    CHECK(e.entries() == 2 && e.get_stats().evictions == 1);
    CHECK(e.place(A, wr, &off[0], &hit) && !hit);        // A was evicted first
    e.clear();
    CHECK(e.entries() == 0 && e.used_bytes() == 0);
    // FNV-1a 64 test vectors
    CHECK(tsr_opl_cache::hash((const uint8_t *) "", 0) == 0xcbf29ce484222325ull);
    CHECK(tsr_opl_cache::hash((const uint8_t *) "a", 1) == 0xaf63dc4c8601ec8cull);
}

// =============================================================================================
// The builder (SW §9.3 M7): the DRAM ranges of every kind, the preload table, and (with ggml) the
// records of ggml nodes, fusion, dependencies through views and KV planes, FA_NEW, the graph key and
// supports_op against the planners.
// =============================================================================================

static void test_kind_ranges() {
    tsr_opl_builder b(4);
    std::vector<tsr_dram_range> rd, wr;
    const uint64_t a0 = 0x100000, b0 = 0x200000, d0 = 0x300000;
    const uint32_t A = tsr_dram_handle(1, a0), B = tsr_dram_handle(1, b0), D = tsr_dram_handle(2, d0);
    uint32_t w[opl::fa::NARGS];
    // ADD, n = 100 (L = 7 lines), 8 rows, b one row broadcast (stride 0)
    opl::put_args(opl::ew_args{ D, 7, A, 7, B, 0, 100, 8, 1, 0 }, w);
    b.kind_ranges(opl::K_ADD, w, 0, rd, wr);
    CHECK(rd.size() == 2 && wr.size() == 1);
    CHECK(rd[0].ch == 1 && rd[0].lo == a0 && rd[0].hi == a0 + 8 * 7 * 64 && rd[0].row == 0);
    CHECK(rd[1].ch == 1 && rd[1].lo == b0 && rd[1].hi == b0 + 7 * 64);
    CHECK(wr[0].ch == 2 && wr[0].lo == d0 && wr[0].hi == d0 + 8 * 7 * 64);
    // MUL by per-row scalars (FLAG_B_SCALAR, period 40 >= 20 rows: words 0..19 = 2 lines)
    opl::put_args(opl::ew_args{ D, 7, A, 7, B, 0, 100, 20, 40, opl::ew::FLAG_B_SCALAR }, w);
    b.kind_ranges(opl::K_MUL, w, 0, rd, wr);
    CHECK(rd.size() == 2 && rd[1].lo == b0 && rd[1].hi == b0 + 2 * 64);
    // ADD with a row period of 4 of 8 rows (b stride 7 lines): b rows 0..3
    opl::put_args(opl::ew_args{ D, 7, A, 7, B, 7, 100, 8, 4, 0 }, w);
    b.kind_ranges(opl::K_ADD, w, 0, rd, wr);
    CHECK(rd[1].lo == b0 && rd[1].hi == b0 + 4 * 7 * 64);
    // RMS_NORM: a per-row weight of period 3 (dense rows), and no weight
    opl::put_args(opl::rms_norm_args{ D, 7, A, 7, 100, 8, 0, 0, B, 3 }, w);
    b.kind_ranges(opl::K_RMS_NORM, w, 0, rd, wr);
    CHECK(rd.size() == 2 && rd[1].lo == b0 && rd[1].hi == b0 + 3 * 7 * 64 && wr[0].hi == d0 + 8 * 7 * 64);
    opl::put_args(opl::rms_norm_args{ D, 7, A, 7, 100, 8, 0, 0, opl::HANDLE_NONE, 3 }, w);
    b.kind_ranges(opl::K_RMS_NORM, w, 0, rd, wr);
    CHECK(rd.size() == 1);
    // ROPE: 3 tokens of 14 heads x 64 (56 lines) in a fused-qkv view (token stride 60 lines): strided
    // source rows, the freq-factor table (32 values: 2 lines), contiguous result
    opl::put_args(opl::rope_args{ D, 56, A, 60, 0, B, 0, 0, 0, 64, 64, 14, 3, opl::rope::MODE_NEOX }, w);
    b.kind_ranges(opl::K_ROPE, w, 0, rd, wr);
    CHECK(rd.size() == 2 && rd[0].row == 56 * 64 && rd[0].stride == 60 * 64 && rd[0].hi == a0 + 2 * 60 * 64 + 56 * 64);
    CHECK(rd[1].lo == b0 && rd[1].hi == b0 + 2 * 64 && wr[0].row == 0 && wr[0].hi == d0 + 3 * 56 * 64);
    // GET_ROWS: every row of the source (aux = its rows) may be read; 5 ids written
    opl::put_args(opl::get_rows_args{ D, 56, A, 56, 0, 5, 56 }, w);
    b.kind_ranges(opl::K_GET_ROWS, w, 1000, rd, wr);
    CHECK(rd.size() == 1 && rd[0].hi == a0 + 1000 * 56 * 64 && wr[0].hi == d0 + 5 * 56 * 64);
    // SET_ROWS_KV (E = 128, LG = 3): the source rows, and its plane written on every channel (LG·17·E/64 beats)
    const uint32_t geom = opl::geom_word(2, 3);
    opl::put_args(opl::set_rows_kv_args{ 0, 0, A, 8, 0, 2, 128, 0, 0x4000, geom, 0, 0 }, w);
    b.kind_ranges(opl::K_SET_ROWS_KV, w, 0, rd, wr);
    CHECK(rd.size() == 1 && rd[0].hi == a0 + 2 * 8 * 64 && wr.size() == 4);
    tsr_kv_geom g = { 128, 4, 3 };
    for (uint32_t c = 0; c < 4; ++c) {
        CHECK(wr[c].ch == c && wr[c].lo == 0x4000ull * 64 && wr[c].hi == 0x4000ull * 64 + tsr_kv_plane_beats(&g) * 64);
    }
    // FLASH_ATTN: q, the K and V planes on every channel and its f32 mask copy (2 rows of L(300) = 19 lines)
    opl::flash_attn_args f{};
    f.dst = D; f.dst_stride = 8; f.q = A; f.q_head_stride = 4; f.q_token_stride = 8; f.n_kv = 300; f.head_dim = 64;
    f.n_head = 2; f.n_head_kv = 2; f.n_tokens = 2; f.mask = B; f.k_plane = 0x4000; f.v_plane = 0x5000; f.geom = geom; f.d = 64; f.g = 1;
    opl::put_args(f, w);
    b.kind_ranges(opl::K_FLASH_ATTN, w, 0, rd, wr);
    CHECK(rd.size() == 1 + 4 + 4 + 1 && wr.size() == 1);
    CHECK(rd[0].hi == a0 + 2 * 8 * 64 && rd[5].lo == 0x5000ull * 64 && rd[9].lo == b0 && rd[9].hi == b0 + 2 * 19 * 64);
    f.mask = opl::HANDLE_NONE;
    opl::put_args(f, w);
    b.kind_ranges(opl::K_FLASH_ATTN, w, 0, rd, wr);
    CHECK(rd.size() == 9);
    // the plane ranges order a FLASH_ATTN after the SET_ROWS_KV writing its K plane (RAW) and a later
    // SET_ROWS_KV after it (WAR)
    tsr_opl_builder h(4);
    opl::put_args(opl::set_rows_kv_args{ 0, 0, A, 8, 0, 2, 128, 0, 0x4000, geom, 0, 0 }, w);
    std::vector<tsr_dram_range> r2, w2;
    h.kind_ranges(opl::K_SET_ROWS_KV, w, 0, r2, w2);
    const uint32_t s0 = h.add(opl::K_SET_ROWS_KV, w, opl::setrows::NARGS, r2, w2, 0);
    opl::put_args(f, w);
    h.kind_ranges(opl::K_FLASH_ATTN, w, 0, r2, w2);
    const uint32_t f0 = h.add(opl::K_FLASH_ATTN, w, opl::fa::NARGS, r2, w2, 1);
    opl::put_args(opl::set_rows_kv_args{ 0, 0, A, 8, 0, 2, 128, 1, 0x5000, geom, 0, 0 }, w);
    h.kind_ranges(opl::K_SET_ROWS_KV, w, 0, r2, w2);
    const uint32_t s1 = h.add(opl::K_SET_ROWS_KV, w, opl::setrows::NARGS, r2, w2, 2);
    CHECK(h.op(f0).dep0 == s0 && h.op(s1).dep0 == f0 && h.op(s1).dep1 == opl::DEP_NONE);
    // MUL_MAT_BEGIN: the ranges of its MUL_MAT (y is its write); MUL_MAT_END: word 0 is an op index, not a handle: none
    uint32_t m[mm::NARGS];
    mm_args(m, 0x10000, 0x20000, 0x800000u, 32, 64, 2, TSR_FMT_Q8_0, tsr_dram_handle(0, 0x30000), 0, 0);
    std::vector<tsr_dram_range> mr, mw;
    b.kind_ranges(opl::K_MUL_MAT, m, 0, mr, mw);
    b.kind_ranges(opl::K_MUL_MAT_BEGIN, m, 0, rd, wr);
    CHECK(rd.size() == mr.size() && wr.size() == mw.size() && wr.size() == 1 && !rd.empty());
    for (size_t i = 0; i < rd.size(); ++i) CHECK(rd[i].ch == mr[i].ch && rd[i].lo == mr[i].lo && rd[i].hi == mr[i].hi);
    CHECK(wr[0].lo == mw[0].lo && wr[0].hi == mw[0].hi);
    const uint32_t en[1] = { 0x10000u };
    b.kind_ranges(opl::K_MUL_MAT_END, en, 0, rd, wr);
    CHECK(rd.empty() && wr.empty());
}

static void test_builder_preload() {
    tsr_opl_builder b(4);
    const uint32_t P = tsr_dram_handle(1, 0x1000), K = tsr_dram_handle(1, 0x2000);
    CHECK(b.preload(P, 4) == 0 && b.preload(P, 4) == 0 && b.n_pre() == 1);   // one entry per (source, size)
    CHECK(b.preload(K, 8) == 64 && b.n_pre() == 2 && b.pre_lines() == 2);
    CHECK(b.preload(P, 8) == 128 && b.n_pre() == 3);                        // same source, another size
    CHECK(b.preload_lines_with({ { P, 4 }, { K, 8 } }) == 3);               // nothing new
    CHECK(b.preload_lines_with({ { K, 16000 } }) == (192 + 16000 + 63) / 64);
    CHECK(b.preload(tsr_dram_handle(1, 0x9000), 16384) == tsr_opl_builder::NO_ROOM && b.n_pre() == 3);
    CHECK(b.preload(P, 0) == tsr_opl_builder::FAILED);
    CHECK(b.preload(tsr_dram_handle(1, 0xA000), 64 * 253) == 192 && b.pre_lines() == 256);   // exactly full
    CHECK(b.preload(tsr_dram_handle(1, 0xB000), 4) == tsr_opl_builder::NO_ROOM);
    // the blob carries the table: entries at 64-aligned offsets, pre_lines covering the last one
    uint32_t a[mm::NARGS];
    mm_args(a, IO, IO + 0x10000, 0x800000u, 64, 64, 1);
    CHECK(b.mul_mat(a, 0) == 0);
    const std::vector<uint8_t> blob = b.finish();
    opl::reader r;
    CHECK(r.open(blob.data(), (uint32_t) blob.size(), true) == opl::RET_OK && r.hdr().n_pre == 4 && r.hdr().pre_lines == 256);
    CHECK(r.pre(0).src == P && r.pre(0).bytes == 4 && r.pre(1).pre_byte == 64 && r.pre(3).pre_byte == 192);
    b.reset();
    CHECK(b.n_pre() == 0 && b.pre_lines() == 0 && b.preload(K, 8) == 0);
}

#if defined(TSR_UNITS_BUILD)
// ---- ggml graphs on fake device addresses (no device) ----
// Every allocated tensor's data is a fake address FAKE + byte; the fake environment maps it to channel 1
// at that byte. Weights get addresses too (RAW tensors in the real backend). KV-cache tensors resolve by
// name to fake planes.
static const uint64_t FAKE = 0x7e0000000000ull;
static void * fake_at(uint64_t off) { return (void *) (uintptr_t) (FAKE + off); }

struct fake_kv {
    std::map<std::string, tsr_kv_tensor> t;
    void add(const char * name, uint32_t which, uint32_t E, uint32_t kv_size, uint64_t plane_off) {
        tsr_kv_tensor & x = t[name];
        x.name = name;
        x.shape.which = which;
        x.shape.E = E;
        x.shape.kv_size = kv_size;
        x.shape.n_stream = 1;
        x.shape.C = 4;
        x.shape.LG = tsr_kv_lg_min(kv_size, 4);
        x.shape.row_bytes = (size_t) E / 32 * 34;
        tsr_kv_plane p;
        p.which = which;
        p.kv_size = kv_size;
        p.geom = { E, 4, x.shape.LG };
        p.off = plane_off;
        p.bytes = tsr_kv_plane_beats(&p.geom) * 64;
        x.planes.push_back(p);
    }
};

static tsr_build_env fake_env(fake_kv * kv, uint32_t t_max = 8) {
    tsr_build_env e;
    e.cfg = tsr::plan::U200_K4;
    e.cfg.t_max = t_max;
    e.pl = tsr::plan::plan_limits_of(e.cfg);
    e.num_ch = 4;
    e.loc = [](const ggml_tensor * t, tsr_tloc * l) {
        const uint64_t a = (uint64_t) (uintptr_t) t->data;
        if (t->data == nullptr || a < FAKE || a >= FAKE + (1ull << 34)) {
            return false;
        }
        l->ch = 1;
        l->off = a - FAKE;
        return true;
    };
    e.kv_view = [kv](const ggml_tensor * t, tsr_kv_view * v) {
        const ggml_tensor * base = t->view_src != nullptr ? t->view_src : t;
        auto it = kv != nullptr ? kv->t.find(base->name) : std::map<std::string, tsr_kv_tensor>::iterator();
        if (kv == nullptr || it == kv->t.end() || t->data == nullptr || base->data == nullptr) {
            return false;
        }
        const uint64_t rel = (uint64_t) ((const char *) t->data - (const char *) base->data);
        v->t = &it->second;
        v->plane = &it->second.planes[0];
        v->stream = 0;
        v->cell0 = (uint32_t) (rel / it->second.shape.row_bytes);
        v->byte_in_row = (uint32_t) (rel % it->second.shape.row_bytes);
        return true;
    };
    e.fa_mask = [](const ggml_tensor * fa) {
        tsr_msum_ref r;
        r.valid = true;
        r.n_q = (uint32_t) fa->src[0]->ne[1];
        r.n_kv = (uint32_t) fa->src[1]->ne[1];
        r.msum_handle = tsr_dram_handle(1, 0x3F000000);
        r.mask_handle = fa->src[3] != nullptr ? tsr_dram_handle(1, 0x3E000000) : opl::HANDLE_NONE;
        return r;
    };
    e.raw_ok = [](const ggml_tensor * t) {
        return (t->view_src == nullptr || t->view_offs % 64 == 0) && strncmp(t->name, "devw", 4) != 0;
    };
    e.kv_probe = [](const ggml_tensor * t, tsr_kv_shape * s) {
        const ggml_tensor * base = t->view_src != nullptr ? t->view_src : t;
        return tsr_kv_identify(base->name, base->type == GGML_TYPE_Q8_0, base->ne, base->nb, 4, 32768, s, nullptr);
    };
    e.fa_mask_probe = [](const ggml_tensor *) { return true; };
    return e;
}

struct gctx {
    ggml_context * ctx;
    gctx() {
        ggml_init_params ip = { 64u << 20, nullptr, true };
        ctx = ggml_init(ip);
    }
    ~gctx() { ggml_free(ctx); }
    ggml_tensor * f32(int64_t n0, int64_t n1, uint64_t off, const char * name, int64_t n2 = 1) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2);
        t->data = fake_at(off);
        ggml_set_name(t, name);
        return t;
    }
};

static ggml_cgraph * graph_of(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, out);
    return g;
}

static int node_index(const ggml_cgraph * g, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(g)); ++i) {
        if (ggml_graph_node(const_cast<ggml_cgraph *>(g), i) == t) return i;
    }
    return -1;
}

// Pass 3: the pairs of a Qwen-like chain and the cases that must not fuse.
static void test_build_fusion() {
    fake_kv kv;
    const tsr_build_env env = fake_env(&kv);
    {
        gctx c;
        ggml_tensor * x = c.f32(896, 2, 0x0, "x");
        ggml_tensor * wq = ggml_new_tensor_2d(c.ctx, GGML_TYPE_Q8_0, 896, 896);
        wq->data = fake_at(0x10000000);
        ggml_tensor * bq = c.f32(896, 1, 0x100000, "bq");
        ggml_tensor * nw = c.f32(896, 1, 0x101000, "nw");
        ggml_tensor * q = ggml_mul_mat(c.ctx, wq, x);
        q->data = fake_at(0x200000);
        ggml_tensor * qb = ggml_add(c.ctx, q, bq);                 // bias: fuses (in place, as ggml-alloc does)
        qb->data = q->data;
        ggml_tensor * n = ggml_rms_norm(c.ctx, qb, 1e-6f);
        n->data = fake_at(0x210000);
        ggml_tensor * nm = ggml_mul(c.ctx, n, nw);                 // RMS_NORM + MUL(weight): fuses
        nm->data = fake_at(0x220000);
        ggml_tensor * wo = ggml_new_tensor_2d(c.ctx, GGML_TYPE_Q8_0, 896, 896);
        wo->data = fake_at(0x11000000);
        ggml_tensor * o = ggml_mul_mat(c.ctx, wo, nm);
        o->data = fake_at(0x230000);
        ggml_tensor * r = ggml_add(c.ctx, x, o);                   // residual, the MUL_MAT as src1: fuses
        r->data = fake_at(0x240000);
        ggml_cgraph * g = graph_of(c.ctx, r);
        const tsr_fusion f = tsr_build_fusion(env, g);
        CHECK(f.n_pairs == 3);
        CHECK(f.with[node_index(g, q)] == node_index(g, qb) && f.into[node_index(g, qb)] == node_index(g, q));
        CHECK(f.with[node_index(g, n)] == node_index(g, nm) && f.with[node_index(g, o)] == node_index(g, r));
        // the fused MUL_MAT words: y = the ADD's result, addend = x rows (stride 56 lines, 2 tokens)
        uint32_t a[mm::NARGS];
        mm_args(a, 0x220000, 0x230000, 0x800000u, 896, 896, 2);
        std::string why;
        CHECK(tsr_fuse_mm_add(env, o, r, a, &why));
        CHECK(a[mm::Y] == tsr_dram_handle(1, 0x240000) && a[mm::Y_STRIDE] == 56 && a[mm::ADD] == tsr_dram_handle(1, 0) &&
              a[mm::ADD_STRIDE] == 56);
        mm_args(a, 0x0, 0x200000, 0x800000u, 896, 896, 2);
        CHECK(tsr_fuse_mm_add(env, q, qb, a, &why) && a[mm::ADD] == tsr_dram_handle(1, 0x100000) && a[mm::ADD_STRIDE] == 0);
        // the fused RMS_NORM record: the weight word and the MUL's result
        tsr_rec rec;
        CHECK(tsr_lower_m7(env, false, n, nm, &rec, &why));
        CHECK(rec.kind == opl::K_RMS_NORM && rec.args[opl::rms::W] == tsr_dram_handle(1, 0x101000) && rec.args[opl::rms::W_PERIOD] == 1 &&
              rec.args[opl::rms::DST] == tsr_dram_handle(1, 0x220000) && rec.args[opl::rms::SRC] == tsr_dram_handle(1, 0x200000) &&
              rec.args[opl::rms::ROWS] == 2 && rec.args[opl::rms::N] == 896);
    }
    {
        // not fused: a second user, an output, a node writing the addend in between, a scalar b, the
        // MUL's weight as src0, an addend outside device memory
        gctx c;
        ggml_tensor * x = c.f32(64, 1, 0x0, "x");
        ggml_tensor * w = ggml_new_tensor_2d(c.ctx, GGML_TYPE_Q8_0, 64, 64);
        w->data = fake_at(0x10000000);
        ggml_tensor * b = c.f32(64, 1, 0x1000, "b");
        ggml_tensor * s1 = c.f32(1, 1, 0x2000, "s1");
        ggml_tensor * q1 = ggml_mul_mat(c.ctx, w, x);     q1->data = fake_at(0x10000);
        ggml_tensor * a1 = ggml_add(c.ctx, q1, b);        a1->data = fake_at(0x11000);
        ggml_tensor * u1 = ggml_scale(c.ctx, q1, 2.0f);   u1->data = fake_at(0x12000);   // q1's second user
        ggml_tensor * q2 = ggml_mul_mat(c.ctx, w, x);     q2->data = fake_at(0x20000);
        ggml_tensor * t2 = ggml_scale(c.ctx, x, 3.0f);    t2->data = b->data;            // writes b in between
        ggml_tensor * a2 = ggml_add(c.ctx, q2, b);        a2->data = fake_at(0x21000);
        ggml_tensor * q3 = ggml_mul_mat(c.ctx, w, x);     q3->data = fake_at(0x30000);
        ggml_tensor * a3 = ggml_add(c.ctx, q3, s1);       a3->data = fake_at(0x31000);   // b.ne0 = 1
        ggml_tensor * n4 = ggml_rms_norm(c.ctx, x, 1e-6f); n4->data = fake_at(0x40000);
        ggml_tensor * m4 = ggml_mul(c.ctx, b, n4);        m4->data = fake_at(0x41000);   // the norm is src1
        ggml_tensor * q5 = ggml_mul_mat(c.ctx, w, x);     q5->data = fake_at(0x50000);
        ggml_tensor * bh = c.f32(64, 1, 0, "b_host");     bh->data = (void *) (uintptr_t) 0x1000;   // not on the device
        ggml_tensor * a5 = ggml_add(c.ctx, q5, bh);       a5->data = fake_at(0x51000);
        ggml_tensor * q6 = ggml_mul_mat(c.ctx, w, x);     q6->data = fake_at(0x60000);
        ggml_set_output(q6);
        ggml_tensor * a6 = ggml_add(c.ctx, q6, b);        a6->data = fake_at(0x61000);
        ggml_cgraph * g = ggml_new_graph(c.ctx);     // program order: q2, t2 (writes b), a2
        for (ggml_tensor * t : { a1, u1, q2, t2, a2, a3, m4, a5, a6 }) {
            ggml_build_forward_expand(g, t);
        }
        const tsr_fusion f = tsr_build_fusion(env, g);
        CHECK(f.n_pairs == 0);
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            if (f.with[i] >= 0) {
                fprintf(stderr, "  unexpected fusion: %s + %s\n", ggml_graph_node(g, i)->name, ggml_graph_node(g, f.with[i])->name);
            }
        }
    }
    {
        // SW-97: the soft cap behind a MUL_MAT (SCALE, TANH, SCALE) is recorded in cap_* whether or not the image has MUL_MAT_SOFTCAP;
        // softcap_* (the fused kind) only with it. A TANH with a second user is no soft cap.
        gctx c;
        ggml_tensor * x = c.f32(64, 1, 0x0, "x");
        ggml_tensor * w = ggml_new_tensor_2d(c.ctx, GGML_TYPE_Q8_0, 64, 64);
        w->data = fake_at(0x10000000);
        ggml_tensor * h  = ggml_mul_mat(c.ctx, w, x);          h->data  = fake_at(0x10000);
        ggml_tensor * s1 = ggml_scale(c.ctx, h, 1.0f / 30);    s1->data = fake_at(0x11000);
        ggml_tensor * t  = ggml_tanh(c.ctx, s1);               t->data  = fake_at(0x12000);
        ggml_tensor * o  = ggml_scale(c.ctx, t, 30.0f);        o->data  = fake_at(0x13000);
        ggml_tensor * h2 = ggml_mul_mat(c.ctx, w, x);          h2->data = fake_at(0x20000);
        ggml_tensor * u1 = ggml_scale(c.ctx, h2, 1.0f / 30);   u1->data = fake_at(0x21000);
        ggml_tensor * ut = ggml_tanh(c.ctx, u1);               ut->data = fake_at(0x22000);
        ggml_tensor * u2 = ggml_scale(c.ctx, ut, 30.0f);       u2->data = fake_at(0x23000);
        ggml_tensor * u3 = ggml_scale(c.ctx, ut, 2.0f);        u3->data = fake_at(0x24000);   // the TANH's second user
        ggml_cgraph * g = ggml_new_graph(c.ctx);
        for (ggml_tensor * e : { o, u2, u3 }) {
            ggml_build_forward_expand(g, e);
        }
        const int ih = node_index(g, h), ih2 = node_index(g, h2);
        tsr_build_env env_cap = env, env_nokind = env;     // the epilogue needs the capability and the kind
        env_cap.cfg.caps |= tsr::plan::CAPS_MVM_LOCAL_SOFTCAP;
        env_nokind.cfg.caps |= tsr::plan::CAPS_MVM_LOCAL_SOFTCAP;
        env_nokind.kinds &= ~(1ull << opl::K_MUL_MAT_SOFTCAP);
        const tsr_build_env * const envs[2] = { &env_cap, &env_nokind };
        for (const tsr_build_env * e : envs) {
            const tsr_fusion f = tsr_build_fusion(*e, g);
            CHECK(f.cap_first[ih] == node_index(g, s1) && f.cap_mid[ih] == node_index(g, t) && f.cap_tail[ih] == node_index(g, o));
            const bool fused = e == &env_cap;
            CHECK(f.softcap_tail[ih] == (fused ? node_index(g, o) : -1) && f.softcap_mid[ih] == (fused ? node_index(g, t) : -1));
            CHECK(f.cap_first[ih2] == -1 && f.cap_mid[ih2] == -1 && f.cap_tail[ih2] == -1 && f.softcap_tail[ih2] == -1);
        }
    }
    {
        // SW-97: a node between the head and the chain that writes the bytes of the chain's last node blocks the hoist of the chain to
        // the head: no cap_* (the soft cap would run before that node)
        gctx c;
        ggml_tensor * x = c.f32(64, 1, 0x0, "x");
        ggml_tensor * w = ggml_new_tensor_2d(c.ctx, GGML_TYPE_Q8_0, 64, 64);
        w->data = fake_at(0x10000000);
        ggml_tensor * h  = ggml_mul_mat(c.ctx, w, x);          h->data  = fake_at(0x30000);
        ggml_tensor * m  = ggml_scale(c.ctx, x, 2.0f);         m->data  = fake_at(0x33000);   // writes where o is
        ggml_tensor * s1 = ggml_scale(c.ctx, h, 1.0f / 30);    s1->data = fake_at(0x31000);
        ggml_tensor * t  = ggml_tanh(c.ctx, s1);               t->data  = fake_at(0x32000);
        ggml_tensor * o  = ggml_scale(c.ctx, t, 30.0f);        o->data  = fake_at(0x33000);
        ggml_cgraph * g = ggml_new_graph(c.ctx);
        for (ggml_tensor * e : { h, m, o }) {
            ggml_build_forward_expand(g, e);
        }
        CHECK(node_index(g, m) > node_index(g, h) && node_index(g, m) < node_index(g, s1));
        const int ih = node_index(g, h);
        const tsr_fusion f = tsr_build_fusion(env, g);
        CHECK(f.cap_first[ih] == -1 && f.cap_mid[ih] == -1 && f.cap_tail[ih] == -1);
    }
}

// Passes 2 and 4-6: records of views and KV planes, their dependencies, the shared preload entries and
// FA_NEW (SET_ROWS_KV -> FLASH_ATTN), also across a cut list.
static void test_build_deps_views() {
    fake_kv kv;
    kv.add("cache_k_l0", TSR_KV_K, 128, 256, 0x40000000);
    kv.add("cache_v_l0", TSR_KV_V, 128, 256, 0x40100000);
    const tsr_build_env env = fake_env(&kv);
    gctx c;
    const int64_t T = 2, D = 64;
    // a fused qkv projection result [4·64, T] (q: 2 heads, k: 1, v: 1) and its q / k views (ROPE reads views)
    ggml_tensor * qkv = c.f32(4 * D, T, 0x100000, "qkv");
    ggml_tensor * qv = ggml_view_3d(c.ctx, qkv, D, 2, T, D * 4, qkv->nb[1], 0);
    ggml_tensor * kvw = ggml_view_3d(c.ctx, qkv, D, 1, T, D * 4, qkv->nb[1], 2 * D * 4);
    ggml_tensor * pos = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, T);
    pos->data = fake_at(0x200000);
    ggml_tensor * qr = ggml_rope_ext(c.ctx, qv, pos, nullptr, D, GGML_ROPE_TYPE_NEOX, 32768, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    qr->data = fake_at(0x300000);
    ggml_tensor * kr = ggml_rope_ext(c.ctx, kvw, pos, nullptr, D, GGML_ROPE_TYPE_NEOX, 32768, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    kr->data = fake_at(0x310000);
    // a copy into the k part of qkv (bytes [512, 768) of every row): WAR after the K ROPE only
    ggml_tensor * kdst = ggml_view_2d(c.ctx, qkv, D, T, qkv->nb[1], 2 * D * 4);
    ggml_tensor * kc = ggml_cpy(c.ctx, ggml_reshape_2d(c.ctx, kr, D, T), kdst);
    // KV: SET_ROWS K / V, then FLASH_ATTN over the cache views
    ggml_tensor * ck = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, 128, 256, 1);
    ggml_set_name(ck, "cache_k_l0");
    ck->data = fake_at(0x1000000);
    ggml_tensor * cv = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, 128, 256, 1);
    ggml_set_name(cv, "cache_v_l0");
    cv->data = fake_at(0x1100000);
    ggml_tensor * ksrc = c.f32(128, T, 0x400000, "kcur");
    ggml_tensor * vsrc = c.f32(128, T, 0x410000, "vcur");
    ggml_tensor * kidx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, T);
    kidx->data = fake_at(0x500000);
    ggml_tensor * vidx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, T);
    vidx->data = fake_at(0x500100);
    ggml_tensor * sk = ggml_set_rows(c.ctx, ck, ksrc, kidx);
    ggml_tensor * sv = ggml_set_rows(c.ctx, cv, vsrc, vidx);
    ggml_tensor * q = c.f32(D, 2 * 2, 0x600000, "q", T);         // [64, 4 heads, T]
    q->ne[1] = 4; q->ne[2] = T; q->nb[2] = q->nb[1] * 4;
    ggml_tensor * kview = ggml_view_4d(c.ctx, ck, D, 2, 256, 1, ggml_row_size(GGML_TYPE_Q8_0, D), ggml_row_size(GGML_TYPE_Q8_0, 128),
                                       ggml_row_size(GGML_TYPE_Q8_0, 128) * 256, 0);
    ggml_tensor * vview = ggml_view_4d(c.ctx, cv, D, 2, 256, 1, ggml_row_size(GGML_TYPE_Q8_0, D), ggml_row_size(GGML_TYPE_Q8_0, 128),
                                       ggml_row_size(GGML_TYPE_Q8_0, 128) * 256, 0);
    ggml_tensor * mask = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, 256, T);
    ggml_set_name(mask, "attn_inp_kq_mask");
    mask->data = fake_at(0x700000);
    ggml_tensor * fa = ggml_flash_attn_ext(c.ctx, ggml_permute(c.ctx, q, 0, 2, 1, 3), ggml_permute(c.ctx, kview, 0, 2, 1, 3),
                                           ggml_permute(c.ctx, vview, 0, 2, 1, 3), mask, 0.125f, 0.0f, 0.0f);
    fa->data = fake_at(0x800000);
    ggml_tensor * qa = ggml_add(c.ctx, qkv, qkv);
    qa->data = qkv->data;   // in place: writes qkv first

    std::string why;
    tsr_rec r_add, r_qr, r_kr, r_kc, r_sk, r_sv, r_fa;
    CHECK(tsr_lower_m7(env, false, qa, nullptr, &r_add, &why));
    CHECK(tsr_lower_m7(env, false, qr, nullptr, &r_qr, &why));
    CHECK(tsr_lower_m7(env, false, kr, nullptr, &r_kr, &why));
    CHECK(tsr_lower_m7(env, false, kc, nullptr, &r_kc, &why));
    CHECK(tsr_lower_m7(env, false, sk, nullptr, &r_sk, &why));
    CHECK(tsr_lower_m7(env, false, sv, nullptr, &r_sv, &why));
    CHECK(tsr_lower_m7(env, false, fa, nullptr, &r_fa, &why));
    if (!why.empty()) fprintf(stderr, "  lowering: %s\n", why.c_str());
    // the words of SW §7.2: ROPE token strides (qkv row = 16 lines), heads, TS; SET_ROWS_KV plane words;
    // FLASH_ATTN q strides, planes, GEOM, D, G
    CHECK(r_qr.args[opl::rope::SRC] == tsr_dram_handle(1, 0x100000) && r_qr.args[opl::rope::SRC_STRIDE] == 16 &&
          r_qr.args[opl::rope::DST_STRIDE] == 8 && r_qr.args[opl::rope::N_HEADS] == 2 && r_qr.args[opl::rope::N_TOKENS] == 2 &&
          r_qr.args[opl::rope::TS] == opl::writer::f32_bits(powf(1e6f, -2.0f / 64.0f)) && r_qr.args[opl::rope::MODE] == opl::rope::MODE_NEOX);
    CHECK(r_kr.args[opl::rope::SRC] == tsr_dram_handle(1, 0x100000 + 512) && r_kr.args[opl::rope::N_HEADS] == 1);
    CHECK(r_sk.args[opl::setrows::PLANE] == 0x40000000u / 64 && r_sk.args[opl::setrows::WHICH] == 0 &&
          r_sv.args[opl::setrows::WHICH] == 1 && r_sk.args[opl::setrows::GEOM] == opl::geom_word(2, tsr_kv_lg_min(256, 4)) &&
          r_sk.args[opl::setrows::N_ROWS] == 2 && r_sk.args[opl::setrows::ROW_ELEMS] == 128);
    CHECK(r_fa.args[opl::fa::Q_HEAD_STRIDE] == 4 && r_fa.args[opl::fa::Q_TOKEN_STRIDE] == 16 && r_fa.args[opl::fa::DST_STRIDE] == 16 &&
          r_fa.args[opl::fa::K_PLANE] == 0x40000000u / 64 && r_fa.args[opl::fa::V_PLANE] == 0x40100000u / 64 &&
          r_fa.args[opl::fa::D] == 64 && r_fa.args[opl::fa::G] == 2 && r_fa.args[opl::fa::N_HEAD] == 4 &&
          r_fa.args[opl::fa::N_HEAD_KV] == 2 && r_fa.args[opl::fa::N_KV] == 256 && r_fa.args[opl::fa::MASK] == tsr_dram_handle(1, 0x3E000000) &&
          r_fa.args[opl::fa::SCALE] == opl::writer::f32_bits(0.125f));

    // encode in program order: dependencies through the views and the planes, one positions entry,
    // FA_NEW pointing at the K indices' entry
    tsr_opl_builder b(4);
    tsr_fanew_map fanew;
    const uint32_t i_add = tsr_encode(b, r_add, 0, fanew);
    const uint32_t i_qr = tsr_encode(b, r_qr, 1, fanew);
    const uint32_t i_kr = tsr_encode(b, r_kr, 2, fanew);
    const uint32_t i_kc = tsr_encode(b, r_kc, 3, fanew);
    tsr_note_kv_write(r_sk, &fanew);
    const uint32_t i_sk = tsr_encode(b, r_sk, 4, fanew);
    tsr_note_kv_write(r_sv, &fanew);
    const uint32_t i_sv = tsr_encode(b, r_sv, 5, fanew);
    tsr_rec r_fa2 = r_fa;
    const uint32_t i_fa = tsr_encode(b, r_fa, 6, fanew);
    CHECK(i_add == 0 && i_qr == 1 && i_kr == 2 && i_kc == 3 && i_sk == 4 && i_sv == 5 && i_fa == 6 && b.n_joins() == 0);
    CHECK(b.op(i_qr).dep0 == i_add && b.op(i_kr).dep0 == i_add);                 // RAW through the views
    CHECK(b.op(i_kc).dep0 == i_kr && b.op(i_kc).dep1 == opl::DEP_NONE);           // WAR on k; WAW on qkv implied
    CHECK(b.op(i_sk).dep0 == opl::DEP_NONE && b.op(i_sv).dep0 == opl::DEP_NONE);
    CHECK(b.op(i_fa).dep0 == i_sk && b.op(i_fa).dep1 == i_sv);                   // RAW on both planes
    CHECK(r_qr.args[opl::rope::POS] == r_kr.args[opl::rope::POS] && b.n_pre() == 4);   // pos, k idx, v idx, MSUM
    CHECK(r_fa.args[opl::fa::FLAGS] == tsr::plan::attn::fa_flags(r_sk.args[opl::setrows::IDX], 2));
    // the record reads back through the reader with the host's argument rules
    const std::vector<uint8_t> blob = b.finish();
    opl::reader rd;
    CHECK(rd.open(blob.data(), (uint32_t) blob.size(), true) == opl::RET_OK);
    opl::record rr;
    while (rd.next(rr) == opl::RET_OK) {
        CHECK(rd.check_args(rr) == opl::RET_OK);
    }
    // the FLASH_ATTN alone in a new list (as after a cut): FA_NEW gets its own entry for the K indices
    tsr_opl_builder b2(4);
    CHECK(tsr_encode(b2, r_fa2, 0, fanew) == 0 && b2.n_pre() == 2 &&
          r_fa2.args[opl::fa::FLAGS] == tsr::plan::attn::fa_flags(r_fa2.args[opl::fa::MSUM] == 0 ? 64u : 0u, 2));
    // a list whose preload area has no room left: NO_ROOM, nothing added
    tsr_opl_builder b3(4);
    CHECK(b3.preload(tsr_dram_handle(1, 0x900000), 64 * 255) == 0);
    tsr_rec r_fa3 = r_fa;
    CHECK(tsr_encode(b3, r_fa3, 0, fanew) == tsr_opl_builder::NO_ROOM && b3.n_ops() == 0 && b3.n_pre() == 1);

    // KV sharing (SW §9.3, FA_NEW across a cut with the K indices overwritten): the plane a FLASH_ATTN reads, the CPY that copies
    // the K indices into a device area right after the SET_ROWS_KV, and a FLASH_ATTN in another list preloading FA_NEW from the copy
    uint32_t plane = 0;
    CHECK(tsr_fa_k_plane(env, fa, &plane) && plane == r_sk.kv_plane && !tsr_fa_k_plane(env, sk, &plane));
    const uint32_t copy_h = tsr_dram_handle(2, 0x40000);
    tsr_rec r_cp;
    CHECK(tsr_lower_kidx_snapshot(env, r_sk.kv_idx_src, 2, copy_h, &r_cp, &why) && r_cp.kind == opl::K_CPY &&
          r_cp.args[opl::cpy::DST] == copy_h && r_cp.args[opl::cpy::SRC] == r_sk.kv_idx_src && r_cp.args[opl::cpy::ROW_LINES] == 1 &&
          r_cp.args[opl::cpy::ROWS] == 1);
    tsr_rec r_cp680;
    CHECK(tsr_lower_kidx_snapshot(env, r_sk.kv_idx_src, 680, copy_h, &r_cp680, &why) && r_cp680.args[opl::cpy::ROW_LINES] == 85);
    tsr_build_env no_cpy = env;
    no_cpy.kinds &= ~(1ull << opl::K_CPY);
    CHECK(!tsr_lower_kidx_snapshot(no_cpy, r_sk.kv_idx_src, 2, copy_h, &r_cp, &why) && !tsr_lower_kidx_snapshot(env, 0, 0, copy_h, &r_cp, &why));
    // the CPY reads the indices' lines and writes the copy's, so a later writer of the indices waits for it and a reader of the copy
    // waits for the CPY
    tsr_opl_builder b4(4);
    tsr_rec r_sk4 = r_sk, r_cp4 = r_cp680;
    tsr_fanew_map fanew4;
    tsr_note_kv_write(r_sk4, &fanew4);
    const uint32_t i4_sk = tsr_encode(b4, r_sk4, 0, fanew4), i4_cp = tsr_encode(b4, r_cp4, 1, fanew4);
    CHECK(i4_sk == 0 && i4_cp == 1 && b4.op(i4_cp).dep0 == opl::DEP_NONE && b4.op(i4_cp).rd.size() == 1 && b4.op(i4_cp).wr.size() == 1 &&
          b4.op(i4_cp).wr[0].ch == 2 && b4.op(i4_cp).wr[0].lo == 0x40000);
    // the FLASH_ATTN of a later list: its FA_NEW entry comes from the copy (the map the backend hands tsr_encode), not the indices
    const tsr_fanew_map from_copy = { { plane, { copy_h, 2 } } };
    tsr_opl_builder b5(4);
    tsr_rec r_fa5 = r_fa;
    CHECK(tsr_encode(b5, r_fa5, 0, from_copy) == 0 && b5.n_pre() == 2);
    const std::vector<uint8_t> blob5 = b5.finish();
    opl::reader rd5;
    CHECK(rd5.open(blob5.data(), (uint32_t) blob5.size(), true) == opl::RET_OK && rd5.hdr().n_pre == 2);
    bool has_copy = false, has_idx = false;
    for (uint32_t i = 0; i < rd5.hdr().n_pre; ++i) {
        has_copy = has_copy || rd5.pre(i).src == copy_h;
        has_idx = has_idx || rd5.pre(i).src == r_sk.kv_idx_src;
    }
    CHECK(has_copy && !has_idx && r_fa5.args[opl::fa::FLAGS] == r_fa2.args[opl::fa::FLAGS] &&
          (r_fa5.args[opl::fa::FLAGS] >> 16) == 2);
}

// R7: the graph key follows shapes, addresses and parameters, not values.
static void test_build_key() {
    gctx c;
    ggml_tensor * x = c.f32(64, 1, 0x0, "x");
    ggml_tensor * w = c.f32(64, 1, 0x1000, "w");
    ggml_tensor * y = ggml_mul(c.ctx, x, w);
    y->data = fake_at(0x2000);
    ggml_tensor * z = ggml_scale(c.ctx, y, 0.5f);
    z->data = fake_at(0x3000);
    ggml_cgraph * g = graph_of(c.ctx, z);
    const uint64_t k0 = tsr_graph_key(g, 1);
    CHECK(k0 == tsr_graph_key(g, 1) && k0 != tsr_graph_key(g, 2));    // the placement epoch salts it
    float v = 2.0f;
    memcpy(&w->op_params[0], &v, 4);                                 // a leaf's op_params: not a node of the graph
    CHECK(tsr_graph_key(g, 1) == k0);
    x->data = fake_at(0x4000);                                       // a source moved
    CHECK(tsr_graph_key(g, 1) != k0);
    x->data = fake_at(0x0);
    CHECK(tsr_graph_key(g, 1) == k0);
    float s = 0.25f;
    memcpy(&z->op_params[0], &s, 4);                                 // another scale
    CHECK(tsr_graph_key(g, 1) != k0);
    s = 0.5f;
    memcpy(&z->op_params[0], &s, 4);
    CHECK(tsr_graph_key(g, 1) == k0);
    // a node of another shape (or strides) at the same address is another graph, even when no
    // other part of the key (FLASH_ATTN's mask words in the backend) would notice
    y->ne[0] = 32;
    y->ne[1] = 2;
    CHECK(tsr_graph_key(g, 1) != k0);
    y->ne[0] = 64;
    y->ne[1] = 1;
    CHECK(tsr_graph_key(g, 1) == k0);
    y->nb[1] = 512;
    CHECK(tsr_graph_key(g, 1) != k0);
    y->nb[1] = 256;
    CHECK(tsr_graph_key(g, 1) == k0);
    x->ne[1] = 3;                                                    // a source's shape
    CHECK(tsr_graph_key(g, 1) != k0);
    x->ne[1] = 1;
    CHECK(tsr_graph_key(g, 1) == k0);
}

// supports_op (probe mode) against the planners on independently built words (SW-20), with the
// 64-byte rule, over sweeps of shapes per kind; plus the rules that are not the planners'.
static int g_sweep_cases = 0, g_sweep_ok = 0;
static bool plans(uint32_t kind, const uint32_t * a, uint32_t n) {
    tsr::plan::kind_plan kp{};
    const tsr::plan::plan_limits l = tsr::plan::plan_limits_of(tsr::plan::U200_K4);
    return opl::check_args(kind, a, n, tsr::plan::PRE_MAX_LINES) == opl::RET_OK && tsr::plan::plan_kind(l, kind, a, n, &kp);
}
static void sweep_check(bool got, bool want, const char * what, int64_t p0, int64_t p1, int64_t p2) {
    g_sweep_cases++;
    g_sweep_ok += got ? 1 : 0;
    if (got != want) {
        fprintf(stderr, "  supports_op %s (%lld, %lld, %lld): got %d, the planner and the 64-byte rule say %d\n", what,
                (long long) p0, (long long) p1, (long long) p2, (int) got, (int) want);
        g_fail++;
    }
}
static uint32_t L16(int64_t n) { return (uint32_t) ((n + 15) / 16); }

static void test_supports_sweep() {
    fake_kv kv;
    const tsr_build_env env = fake_env(&kv);
    const int64_t ns[] = { 1, 15, 16, 17, 64, 100, 896, 4096, 15360, 262144, 1048560, 1048576 };
    const int64_t rws[] = { 1, 2, 7, 512, 70000 };
    std::string why;
    // element-wise: ADD / MUL with b the same shape, one row, one scalar, per-row scalars; SCALE; UNARY
    for (int64_t n : ns) {
        for (int64_t rows : rws) {
            if (n * rows > (1ll << 31)) continue;
            gctx c;
            ggml_tensor * a = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, n, rows);
            const uint32_t L = L16(n), st = rows > 1 ? (uint32_t) (n * 4 / 64) : L;
            const bool rows_ok = rows == 1 || (n * 4) % 64 == 0;
            for (int bm = 0; bm < 4; ++bm) {
                ggml_tensor * b = bm == 0 ? ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, n, rows)
                                : bm == 1 ? ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n)
                                : bm == 2 ? ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 1)
                                          : ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 1, rows);
                for (int op = 0; op < 2; ++op) {
                    ggml_tensor * t = op == 0 ? ggml_add(c.ctx, a, b) : ggml_mul(c.ctx, a, b);
                    const bool scal = bm >= 2 && n > 1;
                    const uint32_t p = bm == 1 || bm == 2 ? 1u : (uint32_t) rows;
                    const uint32_t w[] = { 0, st, 0, st, 0, scal || p == 1 ? 0u : st, (uint32_t) n, (uint32_t) rows, p,
                                           scal ? opl::ew::FLAG_B_SCALAR : 0u };
                    const bool want = rows_ok && plans(op == 0 ? opl::K_ADD : opl::K_MUL, w, 10);
                    sweep_check(tsr_build_supports(env, t, &why), want, op == 0 ? "ADD" : "MUL", n, rows, bm);
                }
            }
            ggml_tensor * s = ggml_scale_bias(c.ctx, a, 0.5f, 0.25f);
            const uint32_t ws[] = { 0, st, 0, st, (uint32_t) n, (uint32_t) rows, opl::writer::f32_bits(0.5f), opl::writer::f32_bits(0.25f) };
            sweep_check(tsr_build_supports(env, s, &why), rows_ok && plans(opl::K_SCALE, ws, 8), "SCALE", n, rows, 0);
            for (int u = 0; u < 4; ++u) {
                const ggml_unary_op uo[] = { GGML_UNARY_OP_TANH, GGML_UNARY_OP_GELU, GGML_UNARY_OP_SILU, GGML_UNARY_OP_EXP };
                ggml_tensor * t = ggml_unary(c.ctx, a, uo[u]);
                const uint32_t wu[] = { 0, st, 0, st, (uint32_t) n, (uint32_t) rows,
                                        u == 0 ? opl::unary::TANH : u == 1 ? opl::unary::GELU16 : opl::unary::SILU };
                // rows that break the 64-byte row rule: the node is one flat row of n·rows values (contiguous tensors)
                const uint32_t fl = (uint32_t) ((n * rows + 15) / 16);
                const uint32_t wf[] = { 0, fl, 0, fl, (uint32_t) (n * rows), 1u, wu[6] };
                sweep_check(tsr_build_supports(env, t, &why), u < 3 && (rows_ok ? plans(opl::K_UNARY, wu, 7) : plans(opl::K_UNARY, wf, 7)),
                            "UNARY", n, rows, u);
            }
            // RMS_NORM (and fused with a weight row: the probe of the norm alone)
            ggml_tensor * r = ggml_rms_norm(c.ctx, a, 1e-6f);
            const uint32_t wr[] = { 0, st, 0, st, (uint32_t) n, (uint32_t) rows, opl::writer::f32_bits(1e-6f),
                                    opl::writer::f32_bits(1.0f / (float) n), opl::HANDLE_NONE, 1 };
            sweep_check(tsr_build_supports(env, r, &why), rows_ok && plans(opl::K_RMS_NORM, wr, 10), "RMS_NORM", n, rows, 0);
            // GLU split / single form, SWIGLU / GEGLU / REGLU
            for (int v = 0; v < 3; ++v) {
                const ggml_glu_op go[] = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_REGLU };
                ggml_tensor * t = ggml_glu_split(c.ctx, a, ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, n, rows), go[v]);
                const uint32_t wg[] = { 0, st, 0, st, 0, st, (uint32_t) n, (uint32_t) rows, v == 0 ? opl::glu::SWIGLU : opl::glu::GEGLU16 };
                sweep_check(tsr_build_supports(env, t, &why), v < 2 && rows_ok && plans(opl::K_GLU, wg, 9), "GLU split", n, rows, v);
            }
            if (2 * n <= 1048576) {
                ggml_tensor * a2 = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 2 * n, rows);
                ggml_tensor * t = ggml_glu(c.ctx, a2, GGML_GLU_OP_SWIGLU, false);
                const uint32_t st2 = rows > 1 ? (uint32_t) (2 * n * 4 / 64) : L16(2 * n);
                const uint32_t wg[] = { 0, L, 0, st2, 0, st2, (uint32_t) n, (uint32_t) rows, opl::glu::SWIGLU };
                const bool want = (n * 4) % 64 == 0 && (rows == 1 || (2 * n * 4) % 64 == 0) && plans(opl::K_GLU, wg, 9);
                sweep_check(tsr_build_supports(env, t, &why), want, "GLU single", n, rows, 0);
                ggml_tensor * ts = ggml_glu(c.ctx, a2, GGML_GLU_OP_SWIGLU, true);
                sweep_check(tsr_build_supports(env, ts, &why), false, "GLU swapped", n, rows, 0);
            }
            // CONT / CPY (f32 rows), GET_ROWS (f32 source, ids of a leaf)
            ggml_tensor * ct = ggml_cont(c.ctx, a);
            const uint32_t wc[] = { 0, st, 0, st, L, (uint32_t) rows };
            sweep_check(tsr_build_supports(env, ct, &why), rows_ok && plans(opl::K_CPY, wc, 6), "CONT", n, rows, 0);
            ggml_tensor * ids = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, rows);
            ggml_tensor * gr = ggml_get_rows(c.ctx, a, ids);
            const uint32_t wgr[] = { 0, st, 0, st, 0, (uint32_t) rows, L };
            sweep_check(tsr_build_supports(env, gr, &why), rows_ok && plans(opl::K_GET_ROWS, wgr, 7), "GET_ROWS", n, rows, 0);
            ggml_tensor * gh = ggml_get_rows(c.ctx, ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, n, rows), ids);
            sweep_check(tsr_build_supports(env, gh, &why), false, "GET_ROWS f16", n, rows, 0);
        }
    }
    // ROPE: head dims, heads, tokens, modes, rotated dims, freq factors
    for (int64_t hd : { 16, 48, 64, 100, 128, 256, 512, 1024, 1040 }) {
        for (int64_t nh : { 1, 14, 40 }) {
            for (int64_t T : { 1, 3 }) {
                for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_MROPE }) {
                    for (int64_t nd : { hd, hd / 2, (int64_t) 32, (int64_t) 48 }) {
                        for (int ffm = 0; ffm < 2; ++ffm) {
                            gctx c;
                            ggml_tensor * x = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, hd, nh, T);
                            ggml_tensor * pos = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, mode == GGML_ROPE_TYPE_MROPE ? 4 * T : T);
                            ggml_tensor * ff = ffm ? ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, std::max<int64_t>(nd / 2, 1)) : nullptr;
                            int sections[GGML_MROPE_SECTIONS] = { 8, 8, 8, 8 };
                            ggml_tensor * t = mode == GGML_ROPE_TYPE_MROPE
                                ? ggml_rope_multi(c.ctx, x, pos, ff, (int) nd, sections, mode, 4096, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)
                                : ggml_rope_ext(c.ctx, x, pos, ff, (int) nd, mode, 4096, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
                            const uint32_t H = (uint32_t) (hd * nh / 16);
                            const bool lines_ok = (hd * 4) % 64 == 0;     // heads on lines (the token rows then are too)
                            const uint32_t w[] = { 0, H, 0, H, 0, ffm ? 0u : opl::HANDLE_NONE, 0, 0, 0, (uint32_t) hd, (uint32_t) nd,
                                                   (uint32_t) nh, (uint32_t) T, mode == GGML_ROPE_TYPE_NEOX ? opl::rope::MODE_NEOX : 0u };
                            const bool want = mode != GGML_ROPE_TYPE_MROPE && (T == 1 || lines_ok) && plans(opl::K_ROPE, w, 14);
                            sweep_check(tsr_build_supports(env, t, &why), want, "ROPE", hd, nh * 1000 + T, mode * 100000 + nd * 10 + ffm);
                        }
                    }
                }
            }
        }
    }
    // SET_ROWS into KV caches of several E; into a plain tensor (R1: never)
    for (int64_t E : { 64, 96, 128, 576, 1024, 8192, 32704, 32768 }) {
        gctx c;
        ggml_tensor * cache = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, 512, 1);
        ggml_set_name(cache, "cache_k_l3");
        ggml_tensor * src = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, E, 4);
        ggml_tensor * idx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, 4);
        ggml_tensor * t = ggml_set_rows(c.ctx, cache, src, idx);
        tsr_kv_shape s;
        const bool is_kv = tsr_kv_identify("cache_k_l3", true, cache->ne, cache->nb, 4, 32768, &s, nullptr);
        const uint32_t w[] = { 0, 0, 0, (uint32_t) (E * 4 / 64), 0, 4, (uint32_t) E, 0, 0, opl::geom_word((uint32_t) E / 64, s.LG), 0, 0 };
        sweep_check(tsr_build_supports(env, t, &why), is_kv && plans(opl::K_SET_ROWS_KV, w, 12), "SET_ROWS", E, 512, 0);
        ggml_tensor * plain = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, E, 512);
        sweep_check(tsr_build_supports(env, ggml_set_rows(c.ctx, plain, src, idx), &why), false, "SET_ROWS plain", E, 512, 0);
    }
    // FLASH_ATTN_EXT on KV-cache views: D, KV heads, GQA ratio, tokens, cells
    for (int64_t D : { 64, 96, 128, 256, 512 }) {
        for (int64_t nkv : { 1, 2, 8 }) {
            for (int64_t G : { 1, 2, 7, 16, 65 }) {
                for (int64_t T : { 1, 8 }) {
                    for (int64_t cells : { 256, 4096 }) {
                        if (D * nkv > 32768) continue;
                        gctx c;
                        const int64_t E = D * nkv;
                        ggml_tensor * ck = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, cells, 1);
                        ggml_set_name(ck, "cache_k_l0");
                        ggml_tensor * cv = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, cells, 1);
                        ggml_set_name(cv, "cache_v_l0");
                        ggml_tensor * q = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, D, G * nkv, T);
                        const size_t rs = ggml_row_size(GGML_TYPE_Q8_0, D), es = ggml_row_size(GGML_TYPE_Q8_0, E);
                        ggml_tensor * k = ggml_view_4d(c.ctx, ck, D, nkv, cells, 1, rs, es, es * cells, 0);
                        ggml_tensor * v = ggml_view_4d(c.ctx, cv, D, nkv, cells, 1, rs, es, es * cells, 0);
                        ggml_tensor * m = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, cells, T);
                        ggml_tensor * fa = ggml_flash_attn_ext(c.ctx, ggml_permute(c.ctx, q, 0, 2, 1, 3), ggml_permute(c.ctx, k, 0, 2, 1, 3),
                                                               ggml_permute(c.ctx, v, 0, 2, 1, 3), m, 0.125f, 0.0f, 0.0f);
                        tsr_kv_shape s;
                        const bool is_kv = tsr_kv_identify("cache_k_l0", true, ck->ne, ck->nb, 4, 32768, &s, nullptr);
                        const uint32_t K = (uint32_t) (G * nkv * D);
                        opl::flash_attn_args f{};
                        f.dst_stride = K / 16; f.q_head_stride = (uint32_t) D / 16; f.q_token_stride = K / 16; f.n_kv = (uint32_t) cells;
                        f.head_dim = (uint32_t) D; f.n_head = (uint32_t) (G * nkv); f.n_head_kv = (uint32_t) nkv; f.n_tokens = (uint32_t) T;
                        f.mask = opl::HANDLE_NONE; f.scale = opl::writer::f32_bits(0.125f); f.geom = opl::geom_word((uint32_t) E / 64, s.LG);
                        f.d = (uint32_t) D; f.g = (uint32_t) G;
                        uint32_t w[opl::fa::NARGS];
                        opl::put_args(f, w);
                        const bool want = is_kv && (D == 64 || D == 128 || D == 256 || D == 512) && plans(opl::K_FLASH_ATTN, w, 22);
                        sweep_check(tsr_build_supports(env, fa, &why), want, "FLASH_ATTN", D, nkv * 100 + G, T * 100000 + cells);
                    }
                }
            }
        }
    }
    // the no-ops are supported; DUP is not (R1)
    {
        gctx c;
        ggml_tensor * a = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 64, 4);
        CHECK(tsr_build_supports(env, ggml_reshape_1d(c.ctx, a, 256), &why) && tsr_build_supports(env, ggml_transpose(c.ctx, a), &why));
        CHECK(!tsr_build_supports(env, ggml_dup(c.ctx, a), &why));
        // a view at a non-64-byte offset (the 64-byte rule) and a device-format weight as an operand
        ggml_tensor * vv = ggml_view_2d(c.ctx, a, 32, 2, a->nb[1], 32);
        CHECK(!tsr_build_supports(env, ggml_scale(c.ctx, vv, 2.0f), &why));
        ggml_tensor * dw = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 64, 4);
        ggml_set_name(dw, "devw");
        CHECK(!tsr_build_supports(env, ggml_get_rows(c.ctx, dw, ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, 2)), &why));
    }
    printf("  supports_op sweep: %d cases against the planners, %d supported\n", g_sweep_cases, g_sweep_ok);
    CHECK(g_sweep_ok > g_sweep_cases / 4 && g_sweep_ok < g_sweep_cases);
}

// Rules of supports_op that the shape sweep does not reach.
static void test_supports_rules() {
    fake_kv kv;
    const tsr_build_env env = fake_env(&kv);
    std::string why;
    {
        // the stride rule (SW §9.3): rows 80 bytes apart ([16 x 4] of a [20 x 4] f32), as an operand and
        // as the result, are refused although every row is one whole line
        gctx c;
        ggml_tensor * big = c.f32(20, 4, 0x0, "big");
        ggml_tensor * other = c.f32(16, 4, 0x1000, "other");
        ggml_tensor * sv = ggml_view_2d(c.ctx, big, 16, 4, big->nb[1], 0);
        CHECK(!tsr_build_supports(env, ggml_add(c.ctx, sv, other), &why));          // operand a
        CHECK(!tsr_build_supports(env, ggml_add(c.ctx, other, sv), &why));          // operand b
        CHECK(!tsr_build_supports(env, ggml_add_inplace(c.ctx, sv, other), &why));  // result (and a)
        CHECK(!tsr_build_supports(env, ggml_cpy(c.ctx, other, sv), &why));          // CPY into it
        CHECK(!tsr_build_supports(env, ggml_cpy(c.ctx, sv, other), &why));          // CPY from it
        CHECK(tsr_build_supports(env, ggml_add(c.ctx, other, other), &why));        // the control
    }
    {
        // a result that is a view must have rows of whole lines: [17 x 4] of a [32 x 4] f32 (rows 128
        // bytes apart; the whole-line stores of rows of 68 bytes would overwrite columns 17-31)
        gctx c;
        ggml_tensor * A = c.f32(32, 4, 0x0, "A");
        ggml_tensor * S = c.f32(32, 4, 0x1000, "S");
        ggml_tensor * vA = ggml_view_2d(c.ctx, A, 17, 4, A->nb[1], 0);
        ggml_tensor * vS = ggml_view_2d(c.ctx, S, 17, 4, S->nb[1], 0);
        CHECK(!tsr_build_supports(env, ggml_cpy(c.ctx, vS, vA), &why));
        CHECK(!tsr_build_supports(env, ggml_add_inplace(c.ctx, vA, vS), &why));
        CHECK(!tsr_build_supports(env, ggml_scale_inplace(c.ctx, vA, 2.0f), &why));
        CHECK(!tsr_build_supports(env, ggml_tanh_inplace(c.ctx, vA), &why));
        ggml_tensor * vA1 = ggml_view_2d(c.ctx, A, 17, 1, A->nb[1], 0);              // one row of 68 bytes
        CHECK(tsr_build_supports(env, ggml_scale(c.ctx, vA1, 2.0f), &why));        // a view read, a new result
        CHECK(!tsr_build_supports(env, ggml_scale_inplace(c.ctx, vA1, 2.0f), &why));
        // whole-line rows of a view: [16 x 4] of [32 x 4]
        ggml_tensor * wA = ggml_view_2d(c.ctx, A, 16, 4, A->nb[1], 0);
        ggml_tensor * wS = ggml_view_2d(c.ctx, S, 16, 4, S->nb[1], 0);
        CHECK(tsr_build_supports(env, ggml_cpy(c.ctx, wS, wA), &why) && tsr_build_supports(env, ggml_add_inplace(c.ctx, wA, wS), &why));
        // a single row of a tensor of its own may end inside a line (ggml-alloc pads it)
        CHECK(tsr_build_supports(env, ggml_scale(c.ctx, c.f32(17, 1, 0x2000, "r"), 2.0f), &why));
    }
    {
        // KV views the builder cannot encode (SW-41: supports_op refuses them too): K / V starting at
        // cell 16, K / V with other strides, a SET_ROWS into a view of the cache at cell 16 or of every
        // other cell
        gctx c;
        const int64_t D = 64, nkvh = 2, E = D * nkvh, cells = 256, n_kv = 128;
        ggml_tensor * ck = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, cells, 1);
        ggml_set_name(ck, "cache_k_l0");
        ggml_tensor * cv = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, cells, 1);
        ggml_set_name(cv, "cache_v_l0");
        const size_t rs = ggml_row_size(GGML_TYPE_Q8_0, D), es = ggml_row_size(GGML_TYPE_Q8_0, E);
        auto fa_of = [&](size_t off, bool swap, int64_t T) {
            ggml_tensor * q = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, D, 2 * nkvh, T);
            ggml_tensor * k = swap ? ggml_view_4d(c.ctx, ck, D, nkvh, n_kv, 1, es, rs, es * cells, off)
                                   : ggml_view_4d(c.ctx, ck, D, nkvh, n_kv, 1, rs, es, es * cells, off);
            ggml_tensor * v = ggml_view_4d(c.ctx, cv, D, nkvh, n_kv, 1, rs, es, es * cells, off);
            ggml_tensor * m = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, n_kv, T);
            return ggml_flash_attn_ext(c.ctx, ggml_permute(c.ctx, q, 0, 2, 1, 3), ggml_permute(c.ctx, k, 0, 2, 1, 3),
                                       ggml_permute(c.ctx, v, 0, 2, 1, 3), m, 0.125f, 0.0f, 0.0f);
        };
        CHECK(tsr_build_supports(env, fa_of(0, false, 1), &why));
        CHECK(!tsr_build_supports(env, fa_of(16 * es, false, 1), &why));
        CHECK(!tsr_build_supports(env, fa_of(0, true, 1), &why));
        ggml_tensor * src = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, E, 4);
        ggml_tensor * idx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, 4);
        CHECK(tsr_build_supports(env, ggml_set_rows(c.ctx, ck, src, idx), &why));
        CHECK(!tsr_build_supports(env, ggml_set_rows(c.ctx, ggml_view_2d(c.ctx, ck, E, 64, es, 16 * es), src, idx), &why));
        CHECK(!tsr_build_supports(env, ggml_set_rows(c.ctx, ggml_view_2d(c.ctx, ck, E, 64, 2 * es, 0), src, idx), &why));   // every other cell
        // the preload area (SW-43): a FLASH_ATTN of T tokens needs ceil((32 + 16 T) / 64) lines of MSUM
        // and ceil(8 T / 64) of FA_NEW indices in one list: T = 680 fits (171 + 85 = 256), T = 681 not
        ggml_tensor * ck2 = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, 1024, 1);
        ggml_set_name(ck2, "cache_k_l1");
        ggml_tensor * cv2 = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, 1024, 1);
        ggml_set_name(cv2, "cache_v_l1");
        for (int64_t T : { 680, 681, 1022 }) {
            ggml_tensor * q = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, D, 2 * nkvh, T);
            ggml_tensor * k = ggml_view_4d(c.ctx, ck2, D, nkvh, 1024, 1, rs, es, es * 1024, 0);
            ggml_tensor * v = ggml_view_4d(c.ctx, cv2, D, nkvh, 1024, 1, rs, es, es * 1024, 0);
            ggml_tensor * m = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, 1024, T);
            ggml_tensor * fa = ggml_flash_attn_ext(c.ctx, ggml_permute(c.ctx, q, 0, 2, 1, 3), ggml_permute(c.ctx, k, 0, 2, 1, 3),
                                                   ggml_permute(c.ctx, v, 0, 2, 1, 3), m, 0.125f, 0.0f, 0.0f);
            const bool got = tsr_build_supports(env, fa, &why);
            if (got != (T <= 680)) {
                fprintf(stderr, "  FLASH_ATTN T=%lld: supports_op %d (%s)\n", (long long) T, (int) got, why.c_str());
            }
            CHECK(got == (T <= 680));
        }
    }
}

// Gated DeltaNet (SW §7.10): SOFTPLUS / SIGMOID on gate vectors (flat rows), GATED_DELTA_NET at the 2B and 27B shapes,
// the SSM_CONV group (the CONCAT fills a window, the SSM_CONV record reads it, the state CPY is absorbed) and the
// regrouped CPY of the new state into a cache cell.
static void test_gdn_lowering() {
    fake_kv kv;
    const tsr_build_env env = fake_env(&kv);
    std::string why;
    tsr_rec r;
    {
        // gate vectors [1, H_v]: one flat row of H_v values, also below one line
        gctx c;
        for (int64_t hv : { 4, 16, 48 }) {
            ggml_tensor * a = c.f32(1, hv, 0x0, "alpha");
            ggml_tensor * sp = ggml_softplus(c.ctx, a);
            sp->data = fake_at(0x1000);
            CHECK(tsr_lower_m7(env, false, sp, nullptr, &r, &why));
            CHECK(r.kind == opl::K_SOFTPLUS && r.args[opl::softplus::N] == (uint32_t) hv && r.args[opl::softplus::ROWS] == 1u);
            ggml_tensor * sg = ggml_sigmoid(c.ctx, a);
            sg->data = fake_at(0x2000);
            CHECK(tsr_lower_m7(env, false, sg, nullptr, &r, &why));
            CHECK(r.kind == opl::K_UNARY && r.args[opl::unary::FUNC] == opl::unary::SIGMOID);
        }
        // several tokens of [64, 5] rows keep their rows
        ggml_tensor * x = c.f32(64, 5, 0x3000, "x");
        ggml_tensor * sp = ggml_softplus(c.ctx, x);
        sp->data = fake_at(0x4000);
        CHECK(tsr_lower_m7(env, false, sp, nullptr, &r, &why));
        CHECK(r.args[opl::softplus::N] == 64u && r.args[opl::softplus::ROWS] == 5u);
    }
    {
        // GATED_DELTA_NET: 2B (S_v 128, 16 / 16 heads) and 27B (16 key, 48 value heads), decode and a prefill ubatch
        for (int64_t T : { 1, 7 }) {
            for (int64_t hv : { 16, 48 }) {
                gctx c;
                const int64_t S = 128, hk = 16;
                ggml_tensor * q = c.f32(S, hk, 0x100000, "q", T), * k = c.f32(S, hk, 0x200000, "k", T);
                ggml_tensor * v = c.f32(S, hv, 0x300000, "v", T);
                ggml_tensor * g = c.f32(1, hv, 0x400000, "g", T), * b = c.f32(1, hv, 0x410000, "beta", T);
                ggml_tensor * st = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, S, S, hv, 1);
                st->data = fake_at(0x1000000);
                ggml_tensor * n = ggml_gated_delta_net(c.ctx, q, k, v, g, b, st, 1);
                n->data = fake_at(0x8000000);
                CHECK(tsr_lower_m7(env, false, n, nullptr, &r, &why));
                namespace gd = opl::gdn;
                CHECK(r.kind == opl::K_GATED_DELTA_NET && r.args[gd::S_V] == 128u && r.args[gd::H_K] == (uint32_t) hk &&
                      r.args[gd::H_V] == (uint32_t) hv && r.args[gd::N_TOKENS] == (uint32_t) T && r.args[gd::N_SEQS] == 1u);
                CHECK(r.args[gd::Q_HEAD_STRIDE] == 8u && r.args[gd::V_HEAD_STRIDE] == 8u && r.args[gd::G_TOKEN_STRIDE] == (uint32_t) ((hv + 15) / 16));
                // the new state follows the attention rows of the result
                CHECK(r.args[gd::STATE_OUT] == r.args[gd::DST] + (uint32_t) (T * hv * 8));
                CHECK(r.args[gd::STATE_IN] == tsr_dram_handle(1, 0x1000000));
                CHECK(tsr_build_supports(env, n, &why));
            }
        }
        gctx c;
        const int64_t S = 128, hk = 16, hv = 4;
        ggml_tensor * st = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, S, S, hv, 1);
        st->data = fake_at(0x1000000);
        auto build = [&](int64_t T, int64_t K, int64_t gne0) {
            ggml_tensor * q = c.f32(S, hk, 0x100000, "q", T), * k = c.f32(S, hk, 0x200000, "k", T);
            ggml_tensor * v = c.f32(S, hv, 0x300000, "v", T);
            ggml_tensor * g = c.f32(gne0, hv, 0x400000, "g", T), * b = c.f32(1, hv, 0x410000, "beta", T);
            ggml_tensor * n = ggml_gated_delta_net(c.ctx, q, k, v, g, b, st, K);
            n->data = fake_at(0x8000000);
            return n;
        };
        for (int64_t T : { 1, 4 }) {                                       // rollback snapshots (SW-83): K slots
            for (int64_t K : { 2, 4 }) {
                gctx c2;
                const int64_t hv2 = 16, hk2 = 4;
                ggml_tensor * st2 = ggml_new_tensor_4d(c2.ctx, GGML_TYPE_F32, S, S, hv2, 1);
                st2->data = fake_at(0x1000000);
                ggml_tensor * q = c2.f32(S, hk2, 0x100000, "q", T), * k = c2.f32(S, hk2, 0x200000, "k", T);
                ggml_tensor * v = c2.f32(S, hv2, 0x300000, "v", T);
                ggml_tensor * g = c2.f32(1, hv2, 0x400000, "g", T), * b = c2.f32(1, hv2, 0x410000, "beta", T);
                ggml_tensor * n = ggml_gated_delta_net(c2.ctx, q, k, v, g, b, st2, K);
                n->data = fake_at(0x8000000);
                CHECK(tsr_build_supports(env, n, &why));
            }
        }
        CHECK(!tsr_build_supports(env, build(1, 1, S), &why) && why.find("per-row gate") != std::string::npos);   // KDA gate
        CHECK(!tsr_build_supports(env, build(5, 1, 1), &why));            // H_v = 4 rows of 16 bytes between tokens
    }
    {
        // the SSM_CONV group at the 2B shape: CONCAT(state, x^T) -> SSM_CONV -> SILU and the state CPY
        gctx c;
        const int64_t C = 6144, dc = 4;
        for (int64_t T : { 1, 5 }) {
            ggml_tensor * state = c.f32(dc - 1, C, 0x100000, "conv_states");
            ggml_tensor * x = c.f32(C, T, 0x200000, "qkv");
            ggml_tensor * cat = ggml_concat(c.ctx, state, ggml_transpose(c.ctx, x), 0);
            cat->data = fake_at(0x300000);
            ggml_tensor * w = c.f32(dc, C, 0x400000, "taps");
            ggml_tensor * conv = ggml_ssm_conv(c.ctx, cat, w);
            conv->data = fake_at(0x500000);
            ggml_tensor * silu = ggml_silu(c.ctx, conv);
            silu->data = fake_at(0x600000);
            ggml_tensor * last = ggml_view_3d(c.ctx, cat, dc - 1, C, 1, cat->nb[1], cat->nb[2], (size_t) (cat->ne[0] - (dc - 1)) * 4);
            ggml_tensor * cache = c.f32((dc - 1) * C, 1, 0x700000, "cache_r_l0");
            ggml_tensor * cpy = ggml_cpy(c.ctx, last, cache);
            ggml_cgraph * g = ggml_new_graph(c.ctx);
            ggml_build_forward_expand(g, cpy);
            ggml_build_forward_expand(g, silu);
            const int ic = node_index(g, cat), is = node_index(g, conv), icp = node_index(g, cpy), isl = node_index(g, silu);
            const tsr_fusion f = tsr_build_fusion(env, g);
            CHECK(f.conv_concat[is] == ic && f.conv_cpy[is] == icp && f.conv_silu[is] == isl);
            CHECK(tsr_build_supports(env, cat, &why) && tsr_build_supports(env, conv, &why) && tsr_build_supports(env, cpy, &why));
            tsr_rec win[2];
            uint32_t nw = 0;
            CHECK(tsr_lower_conv_window(env, false, cat, win, &nw, &why) && nw == 2);
            const uint32_t Lc = (uint32_t) (C / 16), wh = tsr_dram_handle(1, 0x300000);
            namespace cp = opl::cpy;
            CHECK(win[0].kind == opl::K_CPY && win[0].args[cp::DST] == wh && win[0].args[cp::SRC] == tsr_dram_handle(1, 0x100000) &&
                  win[0].args[cp::ROW_LINES] == Lc && win[0].args[cp::ROWS] == (uint32_t) (dc - 1));
            CHECK(win[1].kind == opl::K_CPY && win[1].args[cp::DST] == wh + (uint32_t) (dc - 1) * Lc &&
                  win[1].args[cp::SRC] == tsr_dram_handle(1, 0x200000) && win[1].args[cp::ROWS] == (uint32_t) T);
            CHECK(tsr_lower_conv_group(env, false, conv, silu, cpy, &r, &why));
            namespace sc = opl::sconv;
            CHECK(r.kind == opl::K_SSM_CONV && r.args[sc::STATE_IN] == wh && r.args[sc::X] == wh + (uint32_t) (dc - 1) * Lc &&
                  r.args[sc::X_TOKEN_STRIDE] == Lc && r.args[sc::STATE_OUT] == tsr_dram_handle(1, 0x700000) &&
                  r.args[sc::DST] == tsr_dram_handle(1, 0x600000) && r.args[sc::FLAGS] == sc::FLAG_SILU &&
                  r.args[sc::N_CH] == (uint32_t) C && r.args[sc::D_CONV] == (uint32_t) dc && r.args[sc::N_TOKENS] == (uint32_t) T);
        }
    }
    {
        // the L2 norm of q (a [128, 16, T] view into the convolution output, token stride of the whole channel row):
        // one record per token, in time linear in T (a copy of the record list inside the loop doubled it per token)
        gctx c;
        const int64_t S = 128, H = 16, C = 6144;
        for (int64_t T : { 2, 26, 64 }) {
            ggml_tensor * conv = c.f32(C, T, 0x100000, "conv_out");
            ggml_tensor * q = ggml_view_3d(c.ctx, conv, S, H, T, S * 4, conv->nb[1], 0);
            ggml_tensor * n = ggml_rms_norm(c.ctx, q, 1e-6f);
            n->data = fake_at(0x4000000);
            CHECK(tsr_lower_m7(env, false, n, nullptr, &r, &why));
            CHECK(r.kind == opl::K_RMS_NORM && r.args[opl::rms::ROWS] == (uint32_t) H && r.more.size() == (size_t) (T - 1));
            if (T > 2) {
                CHECK(r.more.back().args[opl::rms::SRC] == r.args[opl::rms::SRC] + (uint32_t) ((T - 1) * (C / 16)) &&
                      r.more.back().more.empty());
            }
        }
    }
    {
        // IMROPE (Qwen3.5 / 3.8 text positions [p, p, p, 0]) runs as NEOX, but only on llama.cpp's own position input
        gctx c;
        const int64_t D = 256, H = 8, T = 3;
        int sections[GGML_MROPE_SECTIONS] = { 11, 11, 10, 0 };
        ggml_tensor * x = c.f32(D, H, 0x100000, "x", T);
        for (const char * nm : { "inp_pos", "Tesseract0#inp_pos#0", "positions" }) {
            ggml_tensor * pos = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, 4 * T);
            pos->data = fake_at(0x200000);
            ggml_set_name(pos, nm);
            ggml_tensor * rp = ggml_rope_multi(c.ctx, x, pos, nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE, 4096, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            rp->data = fake_at(0x300000);
            const bool ok = tsr_build_supports(env, rp, &why);
            CHECK(ok == (strcmp(nm, "positions") != 0));
            if (ok) {
                CHECK(tsr_lower_m7(env, false, rp, nullptr, &r, &why) && r.kind == opl::K_ROPE &&
                      (r.args[opl::rope::MODE] & opl::rope::MODE_NEOX) != 0);
            }
        }
    }
    {
        // the new state into its cache cell: the GATED_DELTA_NET result's state rows (a view of 128 x 128 x 48), one run
        gctx c;
        const int64_t S = 128, hv = 48;
        ggml_tensor * res = c.f32(S * hv, S + 1, 0x1000000, "gdn_out");
        ggml_tensor * ns = ggml_view_3d(c.ctx, res, S, S, hv, S * 4, S * S * 4, (size_t) res->nb[1]);
        ggml_tensor * cache = c.f32(S * S * hv, 1, 0x4000000, "cache_s_l0");
        ggml_tensor * cpy = ggml_cpy(c.ctx, ns, ggml_view_1d(c.ctx, cache, S * S * hv, 0));
        CHECK(tsr_lower_m7(env, false, cpy, nullptr, &r, &why));
        CHECK(r.kind == opl::K_CPY && r.args[opl::cpy::ROW_LINES] == 8u && r.args[opl::cpy::ROWS] == (uint32_t) (S * hv));
    }
}

// The state cell of a GATED_DELTA_NET (SW-82): llama.cpp's build_rs gathers the layer's state rows by the ids of rs_s_copy
// (GET_ROWS), build_recurrent_attn runs the recurrence and a CPY moves the new state into the sequences' cache cells. When the
// ids are the identity the record reads and writes the cell itself and the gather and the CPY are absorbed. The group is built
// as the pinned graph does (qwen35.cpp, delta-net-base.cpp) and the cases that must keep the three passes are checked.
static void test_gdn_state_cell() {
    fake_kv kv;
    tsr_build_env env = fake_env(&kv);
    std::vector<int32_t> ids_now;
    env.read_i32 = [&](const ggml_tensor *, std::vector<int32_t> * v) {
        *v = ids_now;
        return true;
    };
    std::string why;
    namespace gd = opl::gdn;
    struct shape_t { int64_t S, hk, hv, T, ns, n_rs, head; };
    struct extras_t { bool other_gather_reader, other_state_reader, clash; };
    auto run = [&](const shape_t & sh, const extras_t & ex, std::vector<int32_t> ids, tsr_fusion * fout, int * ig, int * in, int * ic,
                   tsr_rec * rec, ggml_cgraph ** gout, gctx * c) {
        ids_now = ids;
        const int64_t S = sh.S, hk = sh.hk, hv = sh.hv, T = sh.T, ns = sh.ns, D = S * S * hv;
        ggml_tensor * cache = c->f32(D, 8, 0x4000000, "cache_s_l0");
        ggml_tensor * idt = ggml_new_tensor_1d(c->ctx, GGML_TYPE_I32, sh.n_rs);
        idt->data = fake_at(0x500000);
        ggml_set_name(idt, "rs_s_copy");
        ggml_tensor * states = ggml_reshape_2d(c->ctx, cache, D, cache->ne[1]);
        ggml_tensor * gather = ggml_get_rows(c->ctx, states, idt);
        gather->data = fake_at(0x6000000);
        ggml_tensor * out_states = ggml_view_2d(c->ctx, gather, D, ns, gather->nb[1], 0);
        ggml_tensor * st4 = ggml_reshape_4d(c->ctx, out_states, S, S, hv, ns);
        ggml_tensor * q = c->f32(S, hk, 0x100000, "q", T * ns), * k = c->f32(S, hk, 0x200000, "k", T * ns);
        ggml_tensor * v = c->f32(S, hv, 0x300000, "v", T * ns);
        ggml_tensor * g = c->f32(1, hv, 0x400000, "g", T * ns), * b = c->f32(1, hv, 0x410000, "beta", T * ns);
        q = ggml_reshape_4d(c->ctx, q, S, hk, T, ns);
        k = ggml_reshape_4d(c->ctx, k, S, hk, T, ns);
        v = ggml_reshape_4d(c->ctx, v, S, hv, T, ns);
        g = ggml_reshape_4d(c->ctx, g, 1, hv, T, ns);
        b = ggml_reshape_4d(c->ctx, b, 1, hv, T, ns);
        ggml_tensor * gdn = ggml_gated_delta_net(c->ctx, q, k, v, g, b, st4, 1);
        gdn->data = fake_at(0x8000000);
        ggml_tensor * ns_view = ggml_view_4d(c->ctx, gdn, S, S, hv, ns, S * 4, S * S * 4, S * S * hv * 4, (size_t) (S * hv * T * ns * 4));
        ggml_tensor * dst = ggml_view_2d(c->ctx, cache, D, ns, cache->nb[1], (size_t) sh.head * cache->nb[1]);
        ggml_tensor * cpy = ggml_cpy(c->ctx, ns_view, dst);
        ggml_cgraph * graph = ggml_new_graph(c->ctx);
        ggml_tensor * out_view = ggml_view_4d(c->ctx, gdn, S, hv, T, ns, S * 4, S * hv * 4, S * hv * T * 4, 0);
        ggml_build_forward_expand(graph, gather);
        if (ex.other_gather_reader) {
            ggml_build_forward_expand(graph, ggml_scale(c->ctx, out_states, 2.0f));
        }
        ggml_build_forward_expand(graph, gdn);
        ggml_build_forward_expand(graph, ggml_scale(c->ctx, out_view, 1.0f));
        if (ex.other_state_reader) {
            ggml_tensor * sc = ggml_scale(c->ctx, ns_view, 0.5f);
            sc->data = fake_at(0x9000000);
            ggml_build_forward_expand(graph, sc);
        }
        if (ex.clash) {                                   // a node writing the cell between the gather and the copy
            ggml_tensor * w = ggml_scale_inplace(c->ctx, ggml_view_1d(c->ctx, cache, D, (size_t) sh.head * cache->nb[1]), 0.0f);
            ggml_build_forward_expand(graph, w);
        }
        ggml_build_forward_expand(graph, cpy);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_graph_node(graph, i)->flags |= GGML_TENSOR_FLAG_COMPUTE;
        }
        *ig = node_index(graph, gather);
        *in = node_index(graph, gdn);
        *ic = node_index(graph, cpy);
        *fout = tsr_build_fusion(env, graph);
        *gout = graph;
        if (rec != nullptr) {
            const bool ok = tsr_lower_m7(env, false, gdn, fout->gdn_cell[(size_t) *in].in_place ? cpy : nullptr, rec, &why);
            CHECK(ok);
        }
    };
    const shape_t decode{ 64, 2, 4, 1, 1, 1, 3 }, prefill{ 128, 2, 16, 5, 1, 1, 3 }, two{ 64, 2, 16, 2, 2, 2, 3 }, extras{ 64, 2, 4, 1, 1, 3, 3 };
    const extras_t plain{ false, false, false };
    for (const shape_t & sh : { decode, prefill, two }) {
        // the sequences keep their cells: the record works on the cell; the gather and the CPY are absorbed
        {
            gctx c;
            tsr_fusion f;
            int ig = -1, in = -1, ic = -1;
            ggml_cgraph * g = nullptr;
            tsr_rec r;
            std::vector<int32_t> ids;
            for (int64_t j = 0; j < sh.ns; ++j) ids.push_back((int32_t) (sh.head + j));
            run(sh, plain, ids, &f, &ig, &in, &ic, &r, &g, &c);
            const tsr_gdn_cell & gc = f.gdn_cell[(size_t) in];
            CHECK(gc.in_place && gc.gather == ig && gc.cpy == ic && gc.n == (uint32_t) sh.ns && gc.cell0 == (int32_t) sh.head);
            CHECK(f.into[(size_t) ig] == in && f.into[(size_t) ic] == in);
            const uint32_t cell = tsr_dram_handle(1, 0x4000000 + (uint64_t) sh.head * 4ull * (uint64_t) (sh.S * sh.S * sh.hv));
            CHECK(r.kind == opl::K_GATED_DELTA_NET && r.args[gd::STATE_IN] == cell && r.args[gd::STATE_OUT] == cell);
            CHECK(r.args[gd::DST] == tsr_dram_handle(1, 0x8000000) && r.args[gd::N_SEQS] == (uint32_t) sh.ns);
        }
        // ids that move a state to another cell, or a swap: the gather and the copy stay (the group is known, not in place)
        {
            gctx c;
            tsr_fusion f;
            int ig = -1, in = -1, ic = -1;
            ggml_cgraph * g = nullptr;
            std::vector<int32_t> ids;
            for (int64_t j = 0; j < sh.ns; ++j) ids.push_back((int32_t) (sh.head + (sh.ns - 1 - j) + (sh.ns == 1 ? 2 : 0)));
            run(sh, plain, ids, &f, &ig, &in, &ic, nullptr, &g, &c);
            const tsr_gdn_cell & gc = f.gdn_cell[(size_t) in];
            CHECK(gc.ids != nullptr && gc.cpy == ic && !gc.in_place && f.into[(size_t) ig] < 0 && f.into[(size_t) ic] < 0);
        }
    }
    // the plan check the backend makes on reuse: the same ids give the same decision, another id flips it
    {
        gctx c;
        tsr_fusion f;
        int ig = -1, in = -1, ic = -1;
        ggml_cgraph * g = nullptr;
        run(decode, plain, { 3 }, &f, &ig, &in, &ic, nullptr, &g, &c);
        const tsr_gdn_cell & gc = f.gdn_cell[(size_t) in];
        CHECK(gc.in_place && gc.ids != nullptr && gc.n == 1u && gc.cell0 == 3);
    }
    // shapes that must not form a group
    {
        gctx c0;
        tsr_fusion f;
        int ig = -1, in = -1, ic = -1;
        ggml_cgraph * g = nullptr;
        run(extras, plain, { 3, 4, 5 }, &f, &ig, &in, &ic, nullptr, &g, &c0);       // extra states behind the sequence's own
        CHECK(f.gdn_cell[(size_t) in].ids == nullptr && f.into[(size_t) ig] < 0);
        gctx c1;
        run(decode, extras_t{ true, false, false }, { 3 }, &f, &ig, &in, &ic, nullptr, &g, &c1);   // another reader of the gathered copy
        CHECK(f.gdn_cell[(size_t) in].ids == nullptr && f.into[(size_t) ig] < 0);
        gctx c2;
        run(decode, extras_t{ false, true, false }, { 3 }, &f, &ig, &in, &ic, nullptr, &g, &c2);   // another reader of the new state
        CHECK(f.gdn_cell[(size_t) in].ids == nullptr && f.into[(size_t) ic] < 0);
        gctx c3;
        run(decode, extras_t{ false, false, true }, { 3 }, &f, &ig, &in, &ic, nullptr, &g, &c3);   // a node writes the cell in between
        CHECK(f.gdn_cell[(size_t) in].ids == nullptr && f.into[(size_t) ic] < 0);
    }
    // without a reader of the values (a host that cannot read the ids), no group is made
    {
        gctx c;
        tsr_fusion f;
        int ig = -1, in = -1, ic = -1;
        ggml_cgraph * g = nullptr;
        tsr_build_env saved = env;
        env.read_i32 = nullptr;
        run(decode, plain, { 3 }, &f, &ig, &in, &ic, nullptr, &g, &c);
        CHECK(f.gdn_cell[(size_t) in].ids == nullptr);
        env = saved;
    }
    (void) why;
}

// Rollback snapshots (SW-83). With n_rs_seq = K - 1 llama.cpp's graph keeps K snapshot groups per recurrent cache:
//   GATED_DELTA_NET with K slots (the state after each of the last min(T, K) tokens, most recent first) and ONE CPY of the
//   slots, a 3D view of the cache with one group (mem_size cells) between slots, behind the gather of the sequence's state;
//   SSM_CONV: K CPYs, slot s a view of the CONCAT's window that starts at column T - s (clamped to 0).
// The record of a node with several slots is a chain of min(T, K) records, each the K = 1 record of its tokens.
static void test_gdn_snapshots() {
    fake_kv kv;
    tsr_build_env env = fake_env(&kv);
    std::vector<int32_t> ids_now;
    env.read_i32 = [&](const ggml_tensor *, std::vector<int32_t> * v) {
        *v = ids_now;
        return true;
    };
    std::string why;
    namespace gd = opl::gdn;
    const int64_t S = 64, hk = 2, hv = 16, D = S * S * hv, mem = 2, head = 1;
    const uint64_t cache_off = 0x4000000;
    const uint32_t cell_lines = (uint32_t) (D * 4 / 64), group_lines = (uint32_t) mem * cell_lines;
    struct cfg_t { int64_t T, K; };
    for (const cfg_t & cf : { cfg_t{ 4, 4 }, cfg_t{ 2, 4 }, cfg_t{ 6, 4 }, cfg_t{ 1, 4 }, cfg_t{ 3, 3 } }) {
        for (int identity = 0; identity < 2; ++identity) {
            const int64_t T = cf.T, K = cf.K, nw = std::min(T, K);
            gctx c;
            ggml_tensor * cache = c.f32(D, mem * K, cache_off, "cache_s_l0");
            ggml_tensor * idt = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, 1);
            idt->data = fake_at(0x500000);
            ggml_set_name(idt, "rs_s_copy");
            ggml_tensor * states = ggml_reshape_2d(c.ctx, cache, D, cache->ne[1]);
            ggml_tensor * gather = ggml_get_rows(c.ctx, states, idt);
            gather->data = fake_at(0x6000000);
            ggml_tensor * out_states = ggml_view_2d(c.ctx, gather, D, 1, gather->nb[1], 0);
            ggml_tensor * st4 = ggml_reshape_4d(c.ctx, out_states, S, S, hv, 1);
            ggml_tensor * q = c.f32(S, hk, 0x100000, "q", T), * k = c.f32(S, hk, 0x200000, "k", T);
            ggml_tensor * v = c.f32(S, hv, 0x300000, "v", T);
            ggml_tensor * g = c.f32(1, hv, 0x400000, "g", T), * b = c.f32(1, hv, 0x410000, "beta", T);
            q = ggml_reshape_4d(c.ctx, q, S, hk, T, 1);
            k = ggml_reshape_4d(c.ctx, k, S, hk, T, 1);
            v = ggml_reshape_4d(c.ctx, v, S, hv, T, 1);
            g = ggml_reshape_4d(c.ctx, g, 1, hv, T, 1);
            b = ggml_reshape_4d(c.ctx, b, 1, hv, T, 1);
            ggml_tensor * gdn = ggml_gated_delta_net(c.ctx, q, k, v, g, b, st4, K);
            gdn->data = fake_at(0x8000000);
            CHECK(gdn->ne[1] == T + K * S);
            const size_t att = (size_t) (4 * S * hv * T);
            ggml_tensor * src = ggml_view_3d(c.ctx, gdn, D, 1, nw, (size_t) D * 4, (size_t) D * 4, att);
            ggml_tensor * dst = ggml_view_3d(c.ctx, cache, D, 1, nw, cache->nb[1], (size_t) mem * cache->nb[1], (size_t) head * cache->nb[1]);
            ggml_tensor * cpy = ggml_cpy(c.ctx, src, dst);
            ggml_cgraph * graph = ggml_new_graph(c.ctx);
            ggml_tensor * out_view = ggml_view_4d(c.ctx, gdn, S, hv, T, 1, S * 4, S * hv * 4, S * hv * T * 4, 0);
            ggml_build_forward_expand(graph, gather);
            ggml_build_forward_expand(graph, gdn);
            ggml_build_forward_expand(graph, ggml_scale(c.ctx, out_view, 1.0f));
            ggml_build_forward_expand(graph, cpy);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                ggml_graph_node(graph, i)->flags |= GGML_TENSOR_FLAG_COMPUTE;
            }
            ids_now = { identity ? (int32_t) head : (int32_t) (mem * 2 + head) };         // the cell, or a snapshot of group 2
            const int ig = node_index(graph, gather), in = node_index(graph, gdn), ic = node_index(graph, cpy);
            const tsr_fusion f = tsr_build_fusion(env, graph);
            const tsr_gdn_cell & gc = f.gdn_cell[(size_t) in];
            CHECK(tsr_build_supports(env, gdn, &why) && tsr_build_supports(env, cpy, &why));
            CHECK(gc.ids != nullptr && gc.cpy == ic && gc.gather == ig && gc.slots == (uint32_t) nw && gc.in_place == (identity != 0) &&
                  gc.absorb_cpy == (identity != 0 || nw > 1));
            CHECK(f.into[(size_t) ic] == (gc.absorb_cpy ? in : -1) && f.into[(size_t) ig] == (identity ? in : -1));
            if (!gc.absorb_cpy) {
                continue;
            }
            tsr_rec r;
            CHECK(tsr_lower_gdn_group(env, gdn, cpy, gc.in_place, &r, &why));
            CHECK(r.more.size() == (size_t) nw - 1u);
            const uint32_t cache_h = tsr_dram_handle(1, cache_off);
            const uint32_t cell0 = cache_h + (uint32_t) head * cell_lines;                 // slot 0: the cell of the sequence
            const uint32_t gin = identity ? cell0 : tsr_dram_handle(1, 0x6000000);        // the gather's copy when the ids move
            const uint32_t Lg = (uint32_t) ((hv + 15) / 16), L = (uint32_t) (S / 16), tok_stride_q = (uint32_t) (4 * S * hk / 64);
            uint32_t prev = gin, tok0 = 0;
            for (int64_t j = 0; j < nw; ++j) {
                const tsr_rec & rj = j == 0 ? r : r.more[(size_t) j - 1];
                const int64_t t1 = T - nw + j + 1, t0 = j == 0 ? 0 : T - nw + j;
                CHECK(rj.kind == opl::K_GATED_DELTA_NET && rj.args[gd::N_TOKENS] == (uint32_t) (t1 - t0) && rj.args[gd::N_SEQS] == 1u);
                CHECK(rj.args[gd::STATE_IN] == prev);
                CHECK(rj.args[gd::STATE_OUT] == cell0 + (uint32_t) (nw - 1 - j) * group_lines);
                CHECK(rj.args[gd::DST] == tsr_dram_handle(1, 0x8000000) + (uint32_t) t0 * (uint32_t) hv * L);
                CHECK(rj.args[gd::G] == tsr_dram_handle(1, 0x400000) + (uint32_t) t0 * Lg);
                CHECK(rj.args[gd::Q] == tsr_dram_handle(1, 0x100000) + (uint32_t) t0 * rj.args[gd::Q_TOKEN_STRIDE]);
                (void) tok_stride_q;
                CHECK(t0 == tok0);
                prev = rj.args[gd::STATE_OUT];
                tok0 = (uint32_t) t1;
            }
            CHECK(tok0 == (uint32_t) T);                                                   // the chain covers every token once
            // slot 0 is the last record's output: the cell the CPY writes first
            const tsr_rec & last = nw == 1 ? r : r.more.back();
            CHECK(last.args[gd::STATE_OUT] == cell0);
        }
    }
    // a record of several slots for more than one sequence is refused (one sequence only)
    {
        gctx c;
        ggml_tensor * st = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, S, S, hv, 2);
        st->data = fake_at(0x1000000);
        ggml_tensor * q = c.f32(S, hk, 0x100000, "q", 4), * k = c.f32(S, hk, 0x200000, "k", 4), * v = c.f32(S, hv, 0x300000, "v", 4);
        ggml_tensor * g = c.f32(1, hv, 0x400000, "g", 4), * b = c.f32(1, hv, 0x410000, "beta", 4);
        q = ggml_reshape_4d(c.ctx, q, S, hk, 2, 2);
        k = ggml_reshape_4d(c.ctx, k, S, hk, 2, 2);
        v = ggml_reshape_4d(c.ctx, v, S, hv, 2, 2);
        g = ggml_reshape_4d(c.ctx, g, 1, hv, 2, 2);
        b = ggml_reshape_4d(c.ctx, b, 1, hv, 2, 2);
        ggml_tensor * gdn = ggml_gated_delta_net(c.ctx, q, k, v, g, b, st, 2);
        gdn->data = fake_at(0x8000000);
        tsr_rec r;
        CHECK(!tsr_lower_m7(env, false, gdn, nullptr, &r, &why) && why.find("one sequence only") != std::string::npos);
    }
    // SSM_CONV with K state CPYs: slot 0 in the record, the other slots by CPY records of the window rows
    {
        const int64_t C = 6144, dc = 4;
        for (const cfg_t & cf : { cfg_t{ 4, 4 }, cfg_t{ 1, 4 }, cfg_t{ 2, 4 }, cfg_t{ 6, 3 }, cfg_t{ 3, 4 } }) {
            const int64_t T = cf.T, K = cf.K, nw = std::min(T, K);
            gctx c;
            ggml_tensor * state = c.f32(dc - 1, C, 0x100000, "conv_states");
            ggml_tensor * x = c.f32(C, T, 0x200000, "qkv");
            ggml_tensor * cat = ggml_concat(c.ctx, state, ggml_transpose(c.ctx, x), 0);
            cat->data = fake_at(0x300000);
            ggml_tensor * w = c.f32(dc, C, 0x400000, "taps");
            ggml_tensor * conv = ggml_ssm_conv(c.ctx, cat, w);
            conv->data = fake_at(0x500000);
            ggml_tensor * silu = ggml_silu(c.ctx, conv);
            silu->data = fake_at(0x600000);
            const int64_t cmem = 2;
            ggml_tensor * cache = c.f32((dc - 1) * C, cmem * K, 0x700000, "cache_r_l0");
            ggml_cgraph * g = ggml_new_graph(c.ctx);
            std::vector<ggml_tensor *> cpys;
            for (int64_t t = 1; t <= K; ++t) {                          // llama.cpp's order: slot K - t
                const int64_t s_idx = std::max<int64_t>(0, cat->ne[0] - (dc - 1) - K + t), slot = K - t;
                ggml_tensor * last = ggml_view_3d(c.ctx, cat, dc - 1, C, 1, cat->nb[1], cat->nb[2], (size_t) s_idx * 4);
                ggml_tensor * dstv = ggml_view_2d(c.ctx, cache, (dc - 1) * C, 1, cache->nb[1], (size_t) (slot * cmem + 1) * cache->nb[1]);
                cpys.push_back(ggml_cpy(c.ctx, last, dstv));
                ggml_build_forward_expand(g, cpys.back());
            }
            ggml_build_forward_expand(g, silu);
            const int is = node_index(g, conv), ic = node_index(g, cat);
            const tsr_fusion f = tsr_build_fusion(env, g);
            CHECK(f.conv_concat[is] == ic && f.conv_silu[is] == node_index(g, silu) && f.conv_cpys[(size_t) is].size() == (size_t) K);
            CHECK(f.conv_cpy[is] == node_index(g, cpys.back()));            // t = K: slot 0, the last columns
            for (ggml_tensor * cp : cpys) {
                CHECK(tsr_build_supports(env, cp, &why));
            }
            std::vector<const ggml_tensor *> slots;
            for (int32_t i : f.conv_cpys[(size_t) is]) slots.push_back(ggml_graph_node(g, i));
            tsr_rec r;
            CHECK(tsr_lower_conv_group(env, false, conv, silu, cpys.back(), &r, &why, &slots));
            const uint32_t Lc = (uint32_t) (C / 16), wh = tsr_dram_handle(1, 0x300000), cell_lines = (uint32_t) (dc - 1) * Lc;
            namespace cp = opl::cpy;
            CHECK(r.args[opl::sconv::STATE_OUT] == tsr_dram_handle(1, 0x700000) + (uint32_t) (1 * cell_lines));
            // slot s >= 1 (s < nw) copies the window rows from column T - s: state-block rows first (rows < d_conv - 1), then token rows
            size_t at = 0;
            for (int64_t s2 = nw - 1; s2 >= 1; --s2) {                  // llama.cpp expands the slots from K - 1 down to 0
                const uint32_t s_idx = (uint32_t) (T - s2), dst = tsr_dram_handle(1, 0x700000) + (uint32_t) (s2 * cmem + 1) * cell_lines;
                const uint32_t c1 = s_idx < (uint32_t) (dc - 1) ? (uint32_t) (dc - 1) - s_idx : 0u, c2 = (uint32_t) (dc - 1) - c1;
                if (c1 > 0) {
                    CHECK(at < r.more.size() && r.more[at].kind == opl::K_CPY && r.more[at].args[cp::DST] == dst &&
                          r.more[at].args[cp::SRC] == wh + s_idx * Lc && r.more[at].args[cp::ROWS] == c1 && r.more[at].args[cp::ROW_LINES] == Lc);
                    ++at;
                }
                if (c2 > 0) {
                    const uint32_t tok0 = c1 > 0 ? 0u : s_idx - (uint32_t) (dc - 1);
                    CHECK(at < r.more.size() && r.more[at].kind == opl::K_CPY && r.more[at].args[cp::DST] == dst + c1 * Lc &&
                          r.more[at].args[cp::SRC] == wh + ((uint32_t) (dc - 1) + tok0) * Lc && r.more[at].args[cp::ROWS] == c2);
                    ++at;
                }
            }
            CHECK(at == r.more.size());                                  // no record for the slots behind min(T, K)
        }
    }
    // CONCAT of two matrices along the row: two CPY records
    {
        gctx c;
        const int64_t E = 2048;
        for (int64_t T : { 1, 4 }) {
            ggml_tensor * a = c.f32(E, T, 0x100000, "e_norm"), * b = c.f32(E, T, 0x200000, "h_norm");
            ggml_tensor * cat = ggml_concat(c.ctx, a, b, 0);
            cat->data = fake_at(0x300000);
            tsr_rec r;
            CHECK(tsr_build_supports(env, cat, &why));
            CHECK(tsr_lower_m7(env, false, cat, nullptr, &r, &why));
            namespace cp = opl::cpy;
            const uint32_t L = (uint32_t) (E / 16);
            CHECK(r.kind == opl::K_CPY && r.more.size() == 1u && r.args[cp::DST] == tsr_dram_handle(1, 0x300000) &&
                  r.args[cp::SRC] == tsr_dram_handle(1, 0x100000) && r.args[cp::ROW_LINES] == L && r.args[cp::ROWS] == (uint32_t) T &&
                  (T == 1 || r.args[cp::DST_STRIDE] == 2 * L));
            CHECK(r.more[0].args[cp::DST] == tsr_dram_handle(1, 0x300000) + L && r.more[0].args[cp::SRC] == tsr_dram_handle(1, 0x200000) &&
                  r.more[0].args[cp::ROW_LINES] == L && r.more[0].args[cp::ROWS] == (uint32_t) T);
        }
        ggml_tensor * odd = c.f32(40, 2, 0x100000, "odd");
        ggml_tensor * cat = ggml_concat(c.ctx, odd, odd, 0);
        cat->data = fake_at(0x300000);
        CHECK(!tsr_build_supports(env, cat, &why));                    // rows that end inside a line
    }
    (void) why;
}

// ---- drift guard between the two encoders (SW-102) ----
// The ggml backend lowers a node with tsr_lower_m7; the graph runtime (PyTorch) encodes the same op from handles and shapes
// with tsr_k_* (tsr-kinds.cpp). Both must write the same kind and argument words, and the same preload entries. Each case
// builds the ggml node on fake addresses, lowers it, builds the tsr_kd_* descriptor from the same handles and shapes, encodes
// it and compares. Two forms differ by design and are compared where they coincide: ROPE with the ATAB table (the runtime's
// form; the ggml path uses the angle recurrence, TS) is compared in the recurrence form; MUL_MAT is not lowered by
// tsr_lower_m7 (the backend tiles it with tsr-lower and the builder), so its words are not compared here.
static tsr_kenv kenv_of(const tsr_build_env & env) {
    tsr_kenv k;
    k.cfg = env.cfg;
    k.pl = env.pl;
    k.num_ch = env.num_ch;
    return k;
}

static bool same_record(const char * what, const tsr_rec & a, const tsr_rec & b) {
    bool ok = a.kind == b.kind && a.n_args == b.n_args && a.pre.size() == b.pre.size() && a.kv_plane == b.kv_plane &&
              a.kv_idx_src == b.kv_idx_src && a.kv_idx_n == b.kv_idx_n && a.more.size() == b.more.size();
    for (uint32_t i = 0; ok && i < a.n_args; ++i) {
        if (a.args[i] != b.args[i]) {
            fprintf(stderr, "  %s: word %u: ggml %u (0x%x), kinds %u (0x%x)\n", what, i, a.args[i], a.args[i], b.args[i], b.args[i]);
            ok = false;
        }
    }
    for (size_t i = 0; ok && i < a.pre.size(); ++i) {
        if (a.pre[i].word != b.pre[i].word || a.pre[i].src != b.pre[i].src || a.pre[i].bytes != b.pre[i].bytes) {
            fprintf(stderr, "  %s: preload entry %zu differs\n", what, i);
            ok = false;
        }
    }
    if (!ok) {
        fprintf(stderr, "  %s: kind %u / %u, %u / %u words, %zu / %zu preload entries\n", what, a.kind, b.kind, a.n_args, b.n_args,
                a.pre.size(), b.pre.size());
    }
    return ok;
}

static tsr_k_rows rows_at(uint64_t off, uint32_t stride_lines) {
    return { tsr_dram_handle(1, off), stride_lines };
}

static void test_encoders_agree() {
    fake_kv kv;
    kv.add("cache_k_l0", TSR_KV_K, 128, 256, 0x40000000);
    kv.add("cache_v_l0", TSR_KV_V, 128, 256, 0x40100000);
    const tsr_build_env env = fake_env(&kv);
    const tsr_kenv ke = kenv_of(env);
    std::string why;
    tsr_rec g, k;
    const int64_t n = 256, rows = 3;
    const uint32_t L = tsr_k_lines((uint64_t) n);
    auto lower = [&](ggml_tensor * node, ggml_tensor * fused = nullptr) {
        CHECK(tsr_lower_m7(env, false, node, fused, &g, &why));
    };
    // ADD / MUL: b is one row for every row; b with its own rows (period 3)
    for (int mul = 0; mul < 2; ++mul) {
        for (int64_t bp : { 1, 3 }) {
            gctx c;
            ggml_tensor * a = c.f32(n, rows, 0x100000, "a");
            ggml_tensor * b = c.f32(n, bp, 0x110000, "b");
            ggml_tensor * y = mul ? ggml_mul(c.ctx, a, b) : ggml_add(c.ctx, a, b);
            y->data = fake_at(0x120000);
            lower(y);
            tsr_kd_ew d;
            d.kind = mul ? opl::K_MUL : opl::K_ADD;
            d.dst = rows_at(0x120000, L);
            d.a = rows_at(0x100000, L);
            d.b = rows_at(0x110000, L);
            d.n = (uint32_t) n;
            d.rows = (uint32_t) rows;
            d.b_period = (uint32_t) bp;
            CHECK(tsr_k_add(ke, d, &k, &why));
            CHECK(same_record(mul ? "MUL" : "ADD", g, k));
        }
    }
    {   // SCALE, without and with a bias
        for (float bias : { 0.0f, 0.25f }) {
            gctx c;
            ggml_tensor * x = c.f32(n, rows, 0x100000, "x");
            ggml_tensor * y = ggml_scale_bias(c.ctx, x, 0.5f, bias);
            y->data = fake_at(0x120000);
            lower(y);
            tsr_kd_scale d;
            d.dst = rows_at(0x120000, L);
            d.src = rows_at(0x100000, L);
            d.n = (uint32_t) n;
            d.rows = (uint32_t) rows;
            d.s = 0.5f;
            d.b = bias;
            CHECK(tsr_k_scale(ke, d, &k, &why));
            CHECK(same_record("SCALE", g, k));
        }
    }
    {   // RMS_NORM alone and with the weight multiplication folded in
        for (int w = 0; w < 2; ++w) {
            gctx c;
            ggml_tensor * x = c.f32(n, rows, 0x100000, "x");
            ggml_tensor * y = ggml_rms_norm(c.ctx, x, 1e-5f);
            y->data = fake_at(0x120000);
            ggml_tensor * wt = c.f32(n, 1, 0x130000, "w");
            ggml_tensor * ym = ggml_mul(c.ctx, y, wt);
            ym->data = fake_at(0x140000);
            lower(y, w ? ym : nullptr);
            tsr_kd_rms d;
            d.dst = rows_at(w ? 0x140000 : 0x120000, L);
            d.src = rows_at(0x100000, L);
            d.n = (uint32_t) n;
            d.rows = (uint32_t) rows;
            d.eps = 1e-5f;
            d.weight = w ? tsr_dram_handle(1, 0x130000) : opl::HANDLE_NONE;
            d.w_period = 1;
            CHECK(tsr_k_rms_norm(ke, d, &k, &why));
            CHECK(same_record(w ? "RMS_NORM * w" : "RMS_NORM", g, k));
        }
    }
    {   // GLU: both variants
        for (int geglu = 0; geglu < 2; ++geglu) {
            gctx c;
            ggml_tensor * gate = c.f32(n, rows, 0x100000, "gate");
            ggml_tensor * up = c.f32(n, rows, 0x110000, "up");
            ggml_tensor * y = ggml_glu_split(c.ctx, gate, up, geglu ? GGML_GLU_OP_GEGLU : GGML_GLU_OP_SWIGLU);
            y->data = fake_at(0x120000);
            lower(y);
            tsr_kd_glu d;
            d.dst = rows_at(0x120000, L);
            d.gate = rows_at(0x100000, L);
            d.up = rows_at(0x110000, L);
            d.n = (uint32_t) n;
            d.rows = (uint32_t) rows;
            d.variant = geglu ? opl::glu::GEGLU16 : opl::glu::SWIGLU;
            CHECK(tsr_k_glu(ke, d, &k, &why));
            CHECK(same_record("GLU", g, k));
        }
    }
    {   // UNARY
        const std::pair<ggml_unary_op, uint32_t> fns[] = { { GGML_UNARY_OP_SILU, opl::unary::SILU },
                                                           { GGML_UNARY_OP_TANH, opl::unary::TANH },
                                                           { GGML_UNARY_OP_SIGMOID, opl::unary::SIGMOID },
                                                           { GGML_UNARY_OP_GELU, opl::unary::GELU16 } };
        for (const auto & f : fns) {
            gctx c;
            ggml_tensor * x = c.f32(n, rows, 0x100000, "x");
            ggml_tensor * y = ggml_unary(c.ctx, x, f.first);
            y->data = fake_at(0x120000);
            lower(y);
            tsr_kd_unary d;
            d.dst = rows_at(0x120000, L);
            d.src = rows_at(0x100000, L);
            d.n = (uint32_t) n;
            d.rows = (uint32_t) rows;
            d.func = f.second;
            CHECK(tsr_k_unary(ke, d, &k, &why));
            CHECK(same_record("UNARY", g, k));
        }
    }
    {   // ROPE, NEOX, the angle recurrence (no table)
        gctx c;
        const int64_t D = 64, H = 4, T = 3;
        ggml_tensor * x = c.f32(D, H, 0x100000, "x", T);
        ggml_tensor * pos = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, T);
        pos->data = fake_at(0x200000);
        ggml_tensor * y = ggml_rope_ext(c.ctx, x, pos, nullptr, (int) D, GGML_ROPE_TYPE_NEOX, 32768, 1e6f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        y->data = fake_at(0x300000);
        lower(y);
        tsr_kd_rope d;
        d.dst = rows_at(0x300000, (uint32_t) (H * D / 16));
        d.src = rows_at(0x100000, (uint32_t) (H * D / 16));
        d.n_heads = (uint32_t) H;
        d.head_dim = d.n_dims = (uint32_t) D;
        d.T = (uint32_t) T;
        d.pos_src = tsr_dram_handle(1, 0x200000);
        d.neox = true;
        d.freq_base = 1e6f;
        CHECK(tsr_k_rope(ke, d, &k, &why));
        CHECK(same_record("ROPE", g, k));
    }
    {   // CPY, f32 to f32
        gctx c;
        ggml_tensor * x = c.f32(n, rows, 0x100000, "x");
        ggml_tensor * dst = c.f32(n, rows, 0x120000, "dst");
        ggml_tensor * y = ggml_cpy(c.ctx, x, dst);
        y->data = dst->data;
        lower(y);
        tsr_kd_cpy d;
        d.dst = rows_at(0x120000, L);
        d.src = rows_at(0x100000, L);
        d.row_lines = L;
        d.rows = (uint32_t) rows;
        CHECK(tsr_k_cpy(ke, d, &k, &why));
        CHECK(same_record("CPY", g, k));
    }
    {   // SET_ROWS into the K and the V plane, then FLASH_ATTN over them
        gctx c;
        const int64_t T = 2, D = 64, E = 128;
        tsr_k_kv geom;
        CHECK(tsr_k_kv_geom((uint32_t) E, 256, 4, &geom, &why));
        ggml_tensor * ck = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, 256, 1);
        ggml_set_name(ck, "cache_k_l0");
        ck->data = fake_at(0x1000000);
        ggml_tensor * cv = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q8_0, E, 256, 1);
        ggml_set_name(cv, "cache_v_l0");
        cv->data = fake_at(0x1100000);
        ggml_tensor * ksrc = c.f32(E, T, 0x400000, "kcur");
        ggml_tensor * vsrc = c.f32(E, T, 0x410000, "vcur");
        ggml_tensor * kidx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, T);
        kidx->data = fake_at(0x500000);
        ggml_tensor * vidx = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I64, T);
        vidx->data = fake_at(0x500100);
        for (int v = 0; v < 2; ++v) {
            ggml_tensor * sr = ggml_set_rows(c.ctx, v ? cv : ck, v ? vsrc : ksrc, v ? vidx : kidx);
            lower(sr);
            tsr_kd_set_rows d;
            d.src = rows_at(v ? 0x410000 : 0x400000, tsr_k_lines((uint64_t) E));
            d.n_rows = (uint32_t) T;
            d.kv = geom;
            d.which = v ? opl::setrows::WHICH_V : opl::setrows::WHICH_K;
            d.plane = (uint32_t) ((v ? 0x40100000u : 0x40000000u) / 64);
            d.idx_src = tsr_dram_handle(1, v ? 0x500100 : 0x500000);
            CHECK(tsr_k_set_rows_kv(ke, d, &k, &why));
            CHECK(same_record(v ? "SET_ROWS V" : "SET_ROWS K", g, k));
        }
        ggml_tensor * q = c.f32(D, 4, 0x600000, "q", T);
        const size_t rs = ggml_row_size(GGML_TYPE_Q8_0, D), es = ggml_row_size(GGML_TYPE_Q8_0, E);
        ggml_tensor * kview = ggml_view_4d(c.ctx, ck, D, 2, 256, 1, rs, es, es * 256, 0);
        ggml_tensor * vview = ggml_view_4d(c.ctx, cv, D, 2, 256, 1, rs, es, es * 256, 0);
        ggml_tensor * mask = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, 256, T);
        ggml_set_name(mask, "attn_inp_kq_mask");
        mask->data = fake_at(0x700000);
        ggml_tensor * fa = ggml_flash_attn_ext(c.ctx, ggml_permute(c.ctx, q, 0, 2, 1, 3), ggml_permute(c.ctx, kview, 0, 2, 1, 3),
                                               ggml_permute(c.ctx, vview, 0, 2, 1, 3), mask, 0.125f, 0.0f, 0.0f);
        fa->data = fake_at(0x800000);
        lower(fa);
        tsr_kd_flash_attn d;
        d.dst = rows_at(0x800000, (uint32_t) (4 * D / 16));
        d.q = rows_at(0x600000, (uint32_t) (4 * D / 16));
        d.n_kv = 256;
        d.D = (uint32_t) D;
        d.n_head = 4;
        d.n_head_kv = 2;
        d.T = (uint32_t) T;
        d.scale = 0.125f;
        d.kv = geom;
        d.k_plane = 0x40000000u / 64;
        d.v_plane = 0x40100000u / 64;
        d.msum_h = tsr_dram_handle(1, 0x3F000000);       // what fake_env's fa_mask names
        d.mask_h = tsr_dram_handle(1, 0x3E000000);
        CHECK(tsr_k_flash_attn(ke, d, &k, &why));
        CHECK(same_record("FLASH_ATTN", g, k));
    }
}

// TSR_KINDS (tsr_kinds_parse): m6, all, additions and removals from left to right, NOP and JOIN always
// kept, and the words it refuses.
static void test_kinds_parse() {
    const uint64_t NOP = 1ull << opl::K_NOP, JOIN = 1ull << opl::K_JOIN, MM = 1ull << opl::K_MUL_MAT;
    const uint64_t FA = 1ull << opl::K_FLASH_ATTN, RMS = 1ull << opl::K_RMS_NORM, KV = 1ull << opl::K_SET_ROWS_KV;
    uint64_t m = 0;
    std::string why;
    CHECK(tsr_kinds_parse(nullptr, &m, &why) && m == ~0ull);
    CHECK(tsr_kinds_parse("", &m, &why) && m == ~0ull);
    CHECK(tsr_kinds_parse("all", &m, &why) && m == ~0ull);
    CHECK(tsr_kinds_parse("m6", &m, &why) && m == (NOP | JOIN | MM));
    CHECK(tsr_kinds_parse("all,-FLASH_ATTN,-RMS_NORM", &m, &why) && m == (~0ull & ~FA & ~RMS));
    CHECK(tsr_kinds_parse("-FLASH_ATTN,-set_rows_kv", &m, &why) && m == (~0ull & ~FA & ~KV));   // starts from all
    CHECK(tsr_kinds_parse("m6,+RMS_NORM", &m, &why) && m == (NOP | JOIN | MM | RMS));
    CHECK(tsr_kinds_parse("MUL_MAT,FLASH_ATTN", &m, &why) && m == (NOP | JOIN | MM | FA));      // starts from NOP, JOIN
    CHECK(tsr_kinds_parse("MUL_MAT,-ROPE", &m, &why) && m == (NOP | JOIN | MM));                // a later removal alone does not start from all
    CHECK(tsr_kinds_parse("+RMS_NORM", &m, &why) && m == (NOP | JOIN | RMS));
    CHECK(tsr_kinds_parse("-RMS_NORM,+RMS_NORM", &m, &why) && m == ~0ull);
    CHECK(tsr_kinds_parse("m6,-MUL_MAT", &m, &why) && m == (NOP | JOIN));
    m = 7;
    CHECK(!tsr_kinds_parse("m7", &m, &why) && m == 7 && why.find("'m7'") != std::string::npos);
    CHECK(!tsr_kinds_parse("all,-FLASHATTN", &m, &why) && why.find("FLASHATTN") != std::string::npos);
    CHECK(!tsr_kinds_parse("all,", &m, &why));
    CHECK(!tsr_kinds_parse("all,-JOIN", &m, &why) && why.find("cannot be removed") != std::string::npos);
    CHECK(!tsr_kinds_parse("-NOP", &m, &why));
    CHECK(!tsr_kinds_parse("?", &m, &why));          // kind_name of an undefined kind
    CHECK(!tsr_kinds_parse("all, -ROPE", &m, &why));  // no spaces
}
#endif   // TSR_UNITS_BUILD

int main() {
    test_alloc_basic();
    test_alloc_random();
    test_shape_ok();
    test_tiling();
    test_stage_plan();
    test_opl_encoding();
    test_opl_ranges();
    test_range_overlap();
    test_opl_hazards();
    test_opl_joins();
    test_opl_failed_op();
    test_opl_capacity();
    test_opl_defer();
    test_opl_defer_random();
    test_opl_random();
    test_opl_cache();
    test_kind_ranges();
    test_builder_preload();
#if defined(TSR_UNITS_BUILD)
    test_build_fusion();
    test_build_deps_views();
    test_build_key();
    test_supports_sweep();
    test_supports_rules();
    test_gdn_lowering();
    test_gdn_state_cell();
    test_gdn_snapshots();
    test_encoders_agree();
    test_kinds_parse();
#else
    printf("test-tsr-backend-units: built without ggml: the M7 builder's ggml tests are skipped\n");
#endif
    if (g_fail) {
        fprintf(stderr, "test-tsr-backend-units: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-backend-units: OK\n");
    return 0;
}
