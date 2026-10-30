// test-tsr-backend-units.cpp — unit tests of the ggml-independent parts of ggml-tesseract
// (ctest ggml_tesseract_units, rewritten for the OPL builder in P3.2, docs/v2/
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
    tsr_config c = TSR_CONFIG_U200_V1;
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
    opl::record rec;
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

// tsr_ranges_overlap against a byte-set model, and the normalisation of tsr_rows_range.
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
    test_opl_random();
    test_opl_cache();
    if (g_fail) {
        fprintf(stderr, "test-tsr-backend-units: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-backend-units: OK\n");
    return 0;
}
