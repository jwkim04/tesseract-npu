// test-tsr-backend-units.cpp — unit tests of the ggml-independent parts of ggml-tesseract:
// the range allocator (tsr-alloc) and the MUL_MAT SPM tiling (tsr-lower). The emission and
// program-cache tests are deleted with that code (X0); the OPL builder's tests (hazards,
// fusion, encoding) take this ctest's place in P3.2 (docs/v2/06-software-verification.md §10.2).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <random>

#include "tsr-alloc.h"
#include "tsr-lower.h"
#include "tsr_isa.h"
#include "tsr_ref.h"

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

int main() {
    test_alloc_basic();
    test_alloc_random();
    test_shape_ok();
    test_tiling();
    if (g_fail) {
        fprintf(stderr, "test-tsr-backend-units: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-backend-units: OK\n");
    return 0;
}
