// test-tsr-backend-units.cpp — unit tests of the ggml-independent parts of ggml-tesseract:
// the range allocator (tsr-alloc), the MUL_MAT lowering / SPM tiling and the program cache
// (tsr-lower).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <vector>

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

// Decode the program and check the tiling invariants and the fields of every instruction.
static void check_program(const tsr_config & c, const tsr_mm_desc & d) {
    tsr_mm_tiling t;
    CHECK(tsr_mm_plan_tiling(c, d.fmt, d.N, d.K, d.T, &t));
    const uint64_t spm_lines = c.spm_bytes / 64;
    CHECK(t.t_tile >= 1 && t.t_tile <= std::min<uint32_t>(c.t_max, d.T));
    CHECK(t.spm_end <= spm_lines);
    CHECK(t.x_lines == d.K * 4 / 64);
    CHECK(t.qa_lines == tsr_qa_lines(tsr_qa_kind_for_fmt(d.fmt), d.K));
    CHECK(t.y_lines == (d.N * 4 + 63) / 64);
    CHECK(t.spm_qa == t.spm_x + t.t_tile * t.x_lines && t.spm_y == t.spm_qa + t.t_tile * t.qa_lines);

    std::vector<tsr_insn> in;
    std::vector<size_t> cuts;
    tsr_mm_emit(c, d, in, cuts);
    const uint32_t tiles = (d.T + t.t_tile - 1) / t.t_tile;
    CHECK(in.size() == 4u * tiles && cuts.size() == tiles);
    for (uint32_t i = 0; i < tiles; ++i) {
        const uint32_t t0 = i * t.t_tile;
        const uint32_t tt = std::min(t.t_tile, d.T - t0);
        const tsr_insn & ld = in[4 * i + 0];
        const tsr_insn & qa = in[4 * i + 1];
        const tsr_insn & mv = in[4 * i + 2];
        const tsr_insn & st = in[4 * i + 3];
        CHECK(cuts[i] == 4u * (i + 1));
        CHECK(tsr_w0_opcode(ld.w[0]) == TSR_OP_LD && tsr_w0_sub(ld.w[0]) == 0);
        CHECK(ld.w[1] == t.spm_x && ld.w[2] == d.x_beat + (uint64_t) t0 * t.x_lines && ld.w[3] == tt * t.x_lines);
        CHECK(tsr_w0_opcode(qa.w[0]) == TSR_OP_QACT && tsr_w0_sub(qa.w[0]) == t.kind);
        CHECK(qa.w[1] == t.spm_qa && qa.w[2] == t.spm_x && qa.w[3] == d.K && qa.w[4] == tt && qa.w[5] == t.x_lines);
        CHECK(qa.w[6] == 0 && qa.w[7] == 0);
        CHECK(tsr_w0_opcode(mv.w[0]) == TSR_OP_MVM && tsr_w0_sub(mv.w[0]) == d.fmt && tsr_w0_flags(mv.w[0]) == 0);
        CHECK(mv.w[1] == t.spm_y && mv.w[2] == t.spm_qa && mv.w[3] == d.w_off_beat && mv.w[4] == d.N);
        CHECK((mv.w[5] & 0xFFFF) == d.K && ((mv.w[5] >> 16) & 0xF) == tt && (mv.w[5] >> 20) == 0);
        CHECK(mv.w[6] == t.y_lines && mv.w[7] == 0);
        CHECK(tsr_w0_opcode(st.w[0]) == TSR_OP_ST && tsr_w0_sub(st.w[0]) == 0);
        CHECK(st.w[1] == t.spm_y && st.w[2] == d.y_beat + (uint64_t) t0 * t.y_lines && st.w[3] == tt * t.y_lines);
        // registers are R0 everywhere (absolute addresses)
        for (const tsr_insn * p : { &ld, &qa, &mv, &st }) {
            CHECK(tsr_w0_ra(p->w[0]) == 0 && tsr_w0_rb(p->w[0]) == 0 && tsr_w0_rc(p->w[0]) == 0 && tsr_w0_rd(p->w[0]) == 0);
        }
    }
}

static void test_lowering() {
    for (uint32_t tmax : { 1u, 2u, 8u }) {
        const tsr_config c = cfg_with_tmax(tmax);
        const tsr_mm_desc cases[] = {
            { TSR_FMT_Q8_0,     64,    32,   1, 0x800000, 0x100000, 0x200000 },
            { TSR_FMT_Q8_0,     33,    96,   7, 0x800040, 0x100010, 0x200020 },
            { TSR_FMT_Q4_K,   5120, 27648,   3, 0x900000, 0x100000, 0x300000 },
            { TSR_FMT_Q4_K,     40,   512,  17, 0x900000, 0x100000, 0x300000 },
            { TSR_FMT_Q8_0, 152064,  5120,   2, 0xA00000, 0x100000, 0x300000 },
            { TSR_FMT_Q8_0,      1,    32, 512, 0xA00000, 0x100000, 0x300000 },
        };
        for (const tsr_mm_desc & d : cases) {
            check_program(c, d);
        }
    }
    // T tile limited by SPM rather than T_MAX: y of 152064 rows = 9504 lines per token
    {
        const tsr_config c = cfg_with_tmax(8);
        tsr_mm_tiling t;
        CHECK(tsr_mm_plan_tiling(c, TSR_FMT_Q8_0, 152064, 5120, 8, &t));
        CHECK(t.t_tile == 6);   // 65536 / (320 + 90 + 9504) = 6.6
        CHECK(t.spm_end <= 65536);
    }
    // disassembly smoke test
    std::vector<tsr_insn> in;
    std::vector<size_t> cuts;
    tsr_mm_emit(cfg_with_tmax(1), { TSR_FMT_Q4_K, 16, 256, 1, 0x10, 0x20, 0x30 }, in, cuts);
    for (const tsr_insn & i : in) {
        CHECK(!tsr_insn_str(i).empty());
    }
    CHECK(tsr_insn_str(in[2]).find("MVM") == 0);
}

// Program cache (backend review #5): bounded by bytes and entries, LRU eviction, programs above
// the per-entry limit are not cached.
static tsr_prog make_prog(const tsr_config & c, uint32_t T, uint64_t x_beat) {
    tsr_prog p;
    tsr_mm_emit(c, { TSR_FMT_Q8_0, 64, 256, T, 0x800000, x_beat, 0x200000 }, p.insns, p.cuts);
    return p;
}

static std::string key_of(uint32_t T, uint64_t x_beat) {
    std::string k(16, '\0');
    memcpy(&k[0], &T, sizeof(T));
    memcpy(&k[8], &x_beat, sizeof(x_beat));
    return k;
}

static void test_prog_cache() {
    const tsr_config c = cfg_with_tmax(1);
    // one T=16 program: 64 insns * 32 B + 16 cuts * 8 B + key = ~2.4 KiB
    const size_t one = tsr_prog_cache::cost(key_of(16, 0), make_prog(c, 16, 0));
    CHECK(one > 2048 && one < 4096);
    tsr_prog_cache pc(10 * one, 4 * one, 1000);
    for (uint64_t i = 0; i < 10; ++i) {
        tsr_prog p = make_prog(c, 16, 0x100000 + i * 64);
        CHECK(pc.cacheable(key_of(16, i), p));
        pc.insert(key_of(16, i), std::move(p));
    }
    CHECK(pc.size() == 10 && pc.bytes() <= 10 * one && pc.evictions() == 0);
    CHECK(pc.find(key_of(16, 0)) != nullptr);              // 0 becomes most recently used
    pc.insert(key_of(16, 10), make_prog(c, 16, 0x200000)); // evicts the LRU entry: 1
    CHECK(pc.evictions() == 1 && pc.size() == 10 && pc.bytes() <= 10 * one);
    CHECK(pc.find(key_of(16, 1)) == nullptr);
    CHECK(pc.find(key_of(16, 0)) != nullptr && pc.find(key_of(16, 10)) != nullptr);
    // a bigger program evicts as many LRU entries as needed to stay within the byte bound
    tsr_prog big = make_prog(c, 48, 0x300000);
    CHECK(pc.cacheable(key_of(48, 0), big));
    pc.insert(key_of(48, 0), std::move(big));
    CHECK(pc.bytes() <= 10 * one && pc.evictions() >= 3);
    CHECK(pc.find(key_of(48, 0)) != nullptr && pc.find(key_of(16, 10)) != nullptr);
    // above the per-entry limit: not cacheable (prefill programs grow with T)
    CHECK(!pc.cacheable(key_of(512, 0), make_prog(c, 512, 0x300000)));
    // re-inserting a key replaces it without double accounting
    const size_t b0 = pc.bytes();
    pc.insert(key_of(48, 0), make_prog(c, 48, 0x300000));
    CHECK(pc.bytes() == b0);
    // entry bound
    tsr_prog_cache small(1u << 30, 1u << 20, 3);
    for (uint64_t i = 0; i < 5; ++i) {
        small.insert(key_of(1, i), make_prog(c, 1, i * 64));
    }
    CHECK(small.size() == 3 && small.evictions() == 2 && small.find(key_of(1, 0)) == nullptr && small.find(key_of(1, 4)) != nullptr);
    small.clear();
    CHECK(small.size() == 0 && small.bytes() == 0);
    // the backend's bounds: 64 MiB, 4 MiB per program (a T=1 decode segment of 2 MUL_MATs is
    // ~300 B; a 32B-head prefill of 512 tokens with T_MAX = 1 is 2048 instructions = 64 KiB)
    CHECK(tsr_prog_cache::cost(key_of(512, 0), make_prog(c, 512, 0)) < (4u << 20));
}

int main() {
    test_alloc_basic();
    test_alloc_random();
    test_shape_ok();
    test_lowering();
    test_prog_cache();
    if (g_fail) {
        fprintf(stderr, "test-tsr-backend-units: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-backend-units: OK\n");
    return 0;
}
