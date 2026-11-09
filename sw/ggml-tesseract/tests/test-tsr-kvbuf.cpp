// test-tsr-kvbuf.cpp — unit tests of the KV-cache buffers (tsr-kvbuf) and the mask-summary producer
// (tsr-msum) of ggml-tesseract, without ggml and without a device (ctest ggml_tesseract_kvbuf; docs/design/06-software-verification.md §9.3 M7, §10.2; docs/design/03-attention-kv.md §4.6, §7.3).
// The device is a host image of C channels behind the read / write callbacks.
//
// KV buffers:
//   - names (cache_<tag>k_l<n> / cache_<tag>v_l<n>) and the identification rules (AT §4.6, §12);
//   - placement: one plane per (tensor, stream), symmetric, top-down from the heap's end, 4 KiB
//     aligned, the first plane highest, no overlap; a heap without room refuses and takes nothing;
//     release() gives every byte back;
//   - conversion: after every operation of a random sequence (set / get / memset of random byte
//     ranges: partial blocks, partial rows, rows across window, stream and tensor boundaries, whole
//     tensors, the state save / load pattern of row ranges; clear) the device planes equal an
//     independent image (tsr_kv_pack_row of every row of a host mirror into planes of the full
//     geometry; padding cells = the last clear value) and get() of random ranges equals the mirror,
//     with C in {1, 3, 4}, E in {64, 128, 192, 1024}, K and V, n_stream 1 and 2, small windows (many
//     windows per access) and the default; a write of whole groups reads nothing from the device,
//     a partial one reads its windows;
//   - resolve(): stream, first cell and byte in row of addresses in every stream.
// Mask summaries:
//   - the layout (get_alloc_size) and the name / shape rules;
//   - random F16 masks (causal, SWA, a wrapped SWA ring, holes, finite values, -0, empty rows,
//     row strides above n_kv): the MSUM written to the device equals tsr_msum_build_f16 word for
//     word (the f32-copy handle = the reservation), with no reader no copy is written;
//   - registered readers (attach): a copy is written exactly when the MSUM is GENERAL or the shared
//     planner (tsr_fa_plan) puts a query tile on the GENERAL path, and it equals tsr_msum_mask_f32
//     with the reader's scale; a tile the planner refuses (-3, rule 5 / 6; found by a seeded search at
//     a small PCAP) makes the MSUM GENERAL (promotion) with the copy written; a reader attached after
//     the mask was written re-produces it (the scheduler's set_tensor comes before graph_compute);
//   - two scales: the second copy is a side allocation, released by forget(); a +-0 / -inf mask gives
//     the same copy for every scale; nomask(): one run [0, n_kv) per row.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "tsr-alloc.h"
#include "tsr-kvbuf.h"
#include "tsr-msum.h"
#include "tsr_isa.h"
#include "tsr_ref.h"
#include "tsr_ref_at.h"

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                            \
        }                                                                        \
    } while (0)

namespace {

// ---------------------------------------------------------------------------------------------
// A device: C channels of host memory
// ---------------------------------------------------------------------------------------------
struct fake_dev {
    uint32_t C;
    uint64_t ch_bytes;
    std::vector<std::vector<uint8_t>> mem;
    uint64_t rd_bytes = 0, wr_bytes = 0;

    fake_dev(uint32_t c, uint64_t bytes, uint8_t init = 0xA5) : C(c), ch_bytes(bytes), mem(c, std::vector<uint8_t>(bytes, init)) {}

    tsr_dram_read_fn reader() {
        return [this](uint32_t ch, uint64_t off, void * dst, size_t n) {
            if (ch >= C || off + n > ch_bytes) {
                fprintf(stderr, "fake_dev: read outside (ch %u, 0x%llx + %zu)\n", ch, (unsigned long long) off, n);
                abort();
            }
            memcpy(dst, mem[ch].data() + off, n);
            rd_bytes += n;
        };
    }
    tsr_dram_write_fn writer() {
        return [this](uint32_t ch, uint64_t off, const void * src, size_t n) {
            if (ch >= C || off + n > ch_bytes) {
                fprintf(stderr, "fake_dev: write outside (ch %u, 0x%llx + %zu)\n", ch, (unsigned long long) off, n);
                abort();
            }
            memcpy(mem[ch].data() + off, src, n);
            wr_bytes += n;
        };
    }
};

// ---------------------------------------------------------------------------------------------
// KV: names and identification
// ---------------------------------------------------------------------------------------------
void test_kv_names() {
    uint32_t w = 9;
    int l = -2;
    CHECK(tsr_kv_parse_name("cache_k_l0", &w, &l) && w == TSR_KV_K && l == 0);
    CHECK(tsr_kv_parse_name("cache_v_l23", &w, &l) && w == TSR_KV_V && l == 23);
    CHECK(tsr_kv_parse_name("cache_swa_k_l7", &w, &l) && w == TSR_KV_K && l == 7);   // a name_tag
    CHECK(tsr_kv_parse_name("cache_xv_l12", &w, &l) && w == TSR_KV_V && l == 12);
    CHECK(!tsr_kv_parse_name("cache_k_l", &w, &l));
    CHECK(!tsr_kv_parse_name("cache_k_lx", &w, &l));
    CHECK(!tsr_kv_parse_name("cache_q_l0", &w, &l));
    CHECK(!tsr_kv_parse_name("cachek_l0", &w, &l));
    CHECK(!tsr_kv_parse_name("cache_k_l0 (view)", &w, &l));
    CHECK(!tsr_kv_parse_name("k_l0", &w, &l));
    CHECK(!tsr_kv_parse_name("cache_k_l1234567890", &w, &l));
    CHECK(!tsr_kv_parse_name(nullptr, &w, &l));

    auto nb_of = [](int64_t E, int64_t kv, int64_t ns, size_t nb[4]) {
        nb[0] = 34;
        nb[1] = (size_t) (E / 32) * 34;
        nb[2] = nb[1] * (size_t) kv;
        nb[3] = nb[2] * (size_t) ns;
    };
    int64_t ne[4] = { 128, 1000, 1, 1 };
    size_t nb[4];
    nb_of(128, 1000, 1, nb);
    tsr_kv_shape s;
    std::string why;
    CHECK(tsr_kv_identify("cache_k_l3", true, ne, nb, 4, 32768, &s, &why));
    CHECK(s.which == TSR_KV_K && s.layer == 3 && s.E == 128 && s.kv_size == 1000 && s.n_stream == 1 && s.C == 4);
    CHECK(s.LG == 16 && s.row_bytes == 136);                                    // ceil(1000 / 64)
    CHECK(s.plane_bytes == 16ull * 34 * 64);                                    // LG * 17 E / 64 beats
    CHECK(!tsr_kv_identify("cache_k_l3", false, ne, nb, 4, 32768, &s, &why));  // not q8_0
    CHECK(!tsr_kv_identify("attn_k", true, ne, nb, 4, 32768, &s, &why));
    int64_t ne2[4] = { 96, 1000, 1, 1 };                                        // E % 64 != 0
    nb_of(96, 1000, 1, nb);
    CHECK(!tsr_kv_identify("cache_k_l3", true, ne2, nb, 4, 32768, &s, &why));
    int64_t ne3[4] = { 512 * 64, 16, 1, 1 };                                    // E/64 = 512 > 511
    nb_of(512 * 64, 16, 1, nb);
    CHECK(!tsr_kv_identify("cache_k_l3", true, ne3, nb, 4, 0, &s, &why));
    int64_t ne4[4] = { 2048, 16, 1, 1 };                                        // E > K_MAX
    nb_of(2048, 16, 1, nb);
    CHECK(!tsr_kv_identify("cache_v_l0", true, ne4, nb, 4, 1024, &s, &why));
    CHECK(tsr_kv_identify("cache_v_l0", true, ne4, nb, 4, 0, &s, &why));
    int64_t ne5[4] = { 128, 1000, 2, 1 };                                       // two streams
    nb_of(128, 1000, 2, nb);
    CHECK(tsr_kv_identify("cache_v_l1", true, ne5, nb, 3, 32768, &s, &why) && s.n_stream == 2 && s.LG == 21);
    nb[2] += 34;                                                                // not contiguous
    CHECK(!tsr_kv_identify("cache_v_l1", true, ne5, nb, 3, 32768, &s, &why));
    int64_t ne6[4] = { 64, 1000, 1, 2 };                                        // 4D
    nb_of(64, 1000, 1, nb);
    nb[3] = nb[2];
    CHECK(!tsr_kv_identify("cache_v_l1", true, ne6, nb, 3, 32768, &s, &why));
    int64_t ne7[4] = { 64, 16ll * 4 * 65536, 1, 1 };                            // LG = 2^16
    nb_of(64, ne7[1], 1, nb);
    CHECK(!tsr_kv_identify("cache_v_l1", true, ne7, nb, 4, 32768, &s, &why));
}

// ---------------------------------------------------------------------------------------------
// KV: placement
// ---------------------------------------------------------------------------------------------
tsr_kv_shape shape_of(const char * name, uint32_t E, uint32_t kv, uint32_t ns, uint32_t C) {
    int64_t ne[4] = { E, kv, ns, 1 };
    size_t nb[4] = { 34, (size_t) (E / 32) * 34, 0, 0 };
    nb[2] = nb[1] * kv;
    nb[3] = nb[2] * ns;
    tsr_kv_shape s;
    std::string why;
    if (!tsr_kv_identify(name, true, ne, nb, C, 0, &s, &why)) {
        fprintf(stderr, "shape_of(%s): %s\n", name, why.c_str());
        abort();
    }
    return s;
}

void test_kv_place() {
    const uint32_t C = 4;
    const uint64_t HEAP0 = 1u << 20, CH = 8u << 20;
    fake_dev dev(C, CH);
    tsr_range_alloc heap;
    heap.init(HEAP0, CH - HEAP0, 4096);
    uint64_t low = 0;
    CHECK(heap.alloc(64 * 1024, &low) && low == HEAP0);             // a weight at the bottom
    {
        tsr_kvbuf kb(C, &heap, dev.reader(), dev.writer());
        uint64_t va = 0x4000000000000000ull;
        std::vector<std::string> names = { "cache_k_l0", "cache_v_l0", "cache_k_l1", "cache_v_l1" };
        for (const std::string & n : names) {
            const tsr_kv_shape s = shape_of(n.c_str(), 128, 300, 2, C);
            kb.add(n, va, (size_t) s.n_stream * s.kv_size * s.row_bytes, s);
            va += 1 << 20;
        }
        CHECK(kb.place(nullptr) && kb.placed());
        uint64_t prev_hi = HEAP0 + (CH - HEAP0);    // the heap's end
        std::vector<std::pair<uint64_t, uint64_t>> spans;
        for (const tsr_kv_tensor & t : kb.tensors()) {
            CHECK(t.planes.size() == 2);
            for (const tsr_kv_plane & p : t.planes) {
                CHECK(p.off % 4096 == 0);
                CHECK(p.off + p.bytes <= prev_hi);                     // top-down, in order
                CHECK(p.bytes == t.shape.plane_bytes && p.geom.LG == 5 && p.geom.C == C && p.geom.E == 128);
                CHECK(p.base_beat() == p.off / 64 && p.handle() == (uint32_t) (p.off / 64));
                CHECK(tsr::opl::geom_e64(p.geom_word()) == 2 && tsr::opl::geom_lg(p.geom_word()) == 5);
                std::vector<tsr_dram_range> r;
                p.ranges(r);
                CHECK(r.size() == C && r[2].ch == 2 && r[2].lo == p.off && r[2].hi == p.off + p.bytes);
                spans.push_back({ p.off, p.off + ((p.bytes + 4095) / 4096) * 4096 });
                prev_hi = p.off;
            }
        }
        CHECK(kb.tensors()[0].planes[0].off + 4096 * ((kb.tensors()[0].planes[0].bytes + 4095) / 4096) == CH);  // highest
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); ++i) {
            CHECK(spans[i - 1].second <= spans[i].first);
        }
        CHECK(kb.device_bytes_per_ch() == 8 * kb.tensors()[0].planes[0].bytes);
        // a buffer that does not fit: refused, nothing taken
        const uint64_t used = heap.used();
        tsr_kvbuf big(C, &heap, dev.reader(), dev.writer());
        const tsr_kv_shape s1 = shape_of("cache_k_l0", 1024, 16384, 1, C);  // 4.25 MiB per channel: one fits, two do not
        big.add("cache_k_l0", 0x5000000000000000ull, (size_t) s1.kv_size * s1.row_bytes, s1);
        big.add("cache_v_l0", 0x5000000010000000ull, (size_t) s1.kv_size * s1.row_bytes, s1);
        std::string err;
        CHECK(!big.place(&err) && !err.empty() && heap.used() == used && heap.check());
    }
    CHECK(heap.used() == 64 * 1024 && heap.check());                     // released with the buffer
    heap.release(low);
}

// ---------------------------------------------------------------------------------------------
// KV: conversion against an independent image
// ---------------------------------------------------------------------------------------------
struct kv_case {
    uint32_t C, E, kv, ns, ntens;
    uint64_t window;    // 0: default
};

// The planes the mirror must produce: every row packed with tsr_kv_pack_row (full geometry),
// padding cells = the clear value; compared with the device bytes of every channel.
bool planes_match(const fake_dev & dev, const tsr_kvbuf & kb, const std::vector<std::vector<uint8_t>> & mirror, uint8_t clear) {
    bool ok = true;
    for (size_t ti = 0; ti < kb.tensors().size(); ++ti) {
        const tsr_kv_tensor & t = kb.tensors()[ti];
        for (const tsr_kv_plane & p : t.planes) {
            std::vector<std::vector<uint8_t>> img(dev.C, std::vector<uint8_t>((size_t) p.bytes, clear));
            std::vector<uint8_t *> ptr(dev.C);
            for (uint32_t c = 0; c < dev.C; ++c) {
                ptr[c] = img[c].data();
            }
            const size_t sb = (size_t) t.shape.kv_size * t.shape.row_bytes;
            for (uint32_t cell = 0; cell < t.shape.kv_size; ++cell) {
                const uint8_t * row = mirror[ti].data() + p.stream * sb + (size_t) cell * t.shape.row_bytes;
                tsr_kv_pack_row(p.which, &p.geom, ptr.data(), cell, (const tsr_block_q8_0 *) row);
            }
            for (uint32_t c = 0; c < dev.C; ++c) {
                if (memcmp(img[c].data(), dev.mem[c].data() + p.off, (size_t) p.bytes) != 0) {
                    fprintf(stderr, "  plane %s stream %u ch%u differs\n", t.name.c_str(), p.stream, c);
                    ok = false;
                }
            }
        }
    }
    return ok;
}

void test_kv_convert(const kv_case & kc, uint32_t seed) {
    std::mt19937_64 rng(seed);
    const uint64_t HEAP0 = 1u << 20;
    tsr_kv_shape s0 = shape_of("cache_k_l0", kc.E, kc.kv, kc.ns, kc.C);
    const uint64_t need = HEAP0 + (uint64_t) kc.ntens * kc.ns * ((s0.plane_bytes + 4095) / 4096 * 4096) + (1u << 20);
    fake_dev dev(kc.C, need);
    tsr_range_alloc heap;
    heap.init(HEAP0, need - HEAP0, 4096);
    tsr_kvbuf kb(kc.C, &heap, dev.reader(), dev.writer());
    if (kc.window != 0) {
        kb.set_window_bytes(kc.window);
    }
    // tensors with gaps between them in the virtual space (as ggml-alloc's 64-byte padding)
    std::vector<uint64_t> va;
    std::vector<std::vector<uint8_t>> mirror;
    uint64_t v = 0x4000000000000000ull;
    for (uint32_t i = 0; i < kc.ntens; ++i) {
        char name[32];
        snprintf(name, sizeof name, "cache_%c_l%u", i % 2 ? 'v' : 'k', i / 2);
        const tsr_kv_shape s = shape_of(name, kc.E, kc.kv, kc.ns, kc.C);
        const size_t nbytes = (size_t) s.n_stream * s.kv_size * s.row_bytes;
        kb.add(name, v, nbytes, s);
        va.push_back(v);
        mirror.emplace_back(nbytes, 0);
        v += (i % 2) ? nbytes : (nbytes + 63) / 64 * 64 + 64;     // odd -> next adjacent, even -> a gap
    }
    CHECK(kb.place(nullptr));
    kb.clear(0);
    uint8_t clear = 0;
    CHECK(planes_match(dev, kb, mirror, clear));

    auto rnd = [&](uint64_t n) { return n == 0 ? 0 : rng() % n; };
    const size_t rb = s0.row_bytes;
    for (int op = 0; op < 60; ++op) {
        const uint32_t ti = (uint32_t) rnd(kc.ntens);
        const size_t nbytes = mirror[ti].size();
        size_t off = 0, n = 0;
        switch (rnd(6)) {
            case 0: off = rnd(nbytes); n = 1 + rnd(std::min<size_t>(nbytes - off, 3 * rb)); break;     // bytes
            case 1: off = rnd(nbytes / rb) * rb; n = std::min(nbytes - off, (1 + rnd(40)) * rb); break;  // rows (state)
            case 2: off = 0; n = nbytes; break;                                                         // whole tensor
            case 3: off = rnd(nbytes / (16 * rb) + 1) * 16 * rb; off = std::min(off, nbytes - rb);       // groups
                    n = std::min(nbytes - off, (1 + rnd(8)) * 16 * kc.C * rb); break;
            case 4: off = rnd(nbytes); n = nbytes - off; break;                                         // to the end
            default: off = rnd(nbytes); n = 1 + rnd(std::min<size_t>(nbytes - off, 64)); break;
        }
        const uint64_t a = va[ti] + off;
        const int kind = (int) rnd(10);
        if (kind < 5) {                                   // set
            std::vector<uint8_t> data(n);
            for (auto & b : data) b = (uint8_t) rng();
            const uint64_t reads0 = kb.st_windows_read;
            kb.set(a, data.data(), n);
            memcpy(mirror[ti].data() + off, data.data(), n);
            // whole groups of every cell < kv_size: nothing read
            const bool aligned = off % (16 * kc.C * rb) == 0 && n % (16 * kc.C * rb) == 0 &&
                                 (off / (kc.kv * rb)) == ((off + n - 1) / (kc.kv * rb)) && (off % (kc.kv * rb)) == 0 &&
                                 n == (size_t) kc.kv * rb;
            if (aligned) {
                CHECK(kb.st_windows_read == reads0);
            }
        } else if (kind < 7) {                            // memset
            const uint8_t val = (uint8_t) rng();
            kb.fill(a, val, n);
            memset(mirror[ti].data() + off, val, n);
        } else if (kind < 9) {                            // get
            std::vector<uint8_t> got(n, 0xEE);
            kb.get(a, got.data(), n);
            CHECK(memcmp(got.data(), mirror[ti].data() + off, n) == 0);
        } else {                                          // clear
            clear = (uint8_t) (rnd(4) == 0 ? rng() : 0);
            kb.clear(clear);
            for (auto & m : mirror) std::fill(m.begin(), m.end(), clear);
        }
        if (op % 6 == 5) {
            CHECK(planes_match(dev, kb, mirror, clear));
        }
    }
    // an access spanning two tensors (the gap between them is not part of a tensor: the span is
    // tensor 0's tail only when they are adjacent)
    bool spanned = kc.ntens < 3;
    for (uint32_t ti = 0; ti + 1 < kc.ntens; ++ti) {
        if (va[ti + 1] != va[ti] + mirror[ti].size()) {
            continue;
        }
        const size_t tail = std::min<size_t>(mirror[ti].size(), 100), head = std::min<size_t>(mirror[ti + 1].size(), 77);
        std::vector<uint8_t> data(tail + head);
        for (auto & b : data) b = (uint8_t) rng();
        kb.set(va[ti + 1] - tail, data.data(), data.size());
        memcpy(mirror[ti].data() + mirror[ti].size() - tail, data.data(), tail);
        memcpy(mirror[ti + 1].data(), data.data() + tail, head);
        std::vector<uint8_t> got(data.size());
        kb.get(va[ti + 1] - tail, got.data(), got.size());
        CHECK(got == data);
        spanned = true;
    }
    CHECK(spanned);
    CHECK(planes_match(dev, kb, mirror, clear));
    // state save / load (llama.cpp's state_write_data / state_read_data: row ranges per tensor):
    // save every tensor, overwrite with garbage, load: the planes are as before, bit for bit
    std::vector<std::vector<uint8_t>> saved(kc.ntens);
    for (uint32_t ti = 0; ti < kc.ntens; ++ti) {
        saved[ti].resize(mirror[ti].size());
        const size_t rows = mirror[ti].size() / rb;
        for (size_t r0 = 0; r0 < rows;) {
            const size_t k = std::min(rows - r0, (size_t) 1 + rnd(97));
            kb.get(va[ti] + r0 * rb, saved[ti].data() + r0 * rb, k * rb);
            r0 += k;
        }
        CHECK(saved[ti] == mirror[ti]);
    }
    std::vector<std::vector<uint8_t>> before(kc.C);
    for (uint32_t c = 0; c < kc.C; ++c) before[c] = dev.mem[c];
    kb.clear(0x5A);
    for (uint32_t ti = 0; ti < kc.ntens; ++ti) {
        const size_t rows = mirror[ti].size() / rb;
        for (size_t r0 = 0; r0 < rows;) {
            const size_t k = std::min(rows - r0, (size_t) 1 + rnd(113));
            kb.set(va[ti] + r0 * rb, saved[ti].data() + r0 * rb, k * rb);
            r0 += k;
        }
    }
    CHECK(planes_match(dev, kb, mirror, 0x5A));
    // resolve(): every stream's first cell, a cell in the middle, a byte inside a row
    for (uint32_t ti = 0; ti < kc.ntens; ++ti) {
        for (uint32_t st = 0; st < kc.ns; ++st) {
            tsr_kv_view vw;
            const uint64_t sb = (uint64_t) kc.kv * rb;
            CHECK(kb.resolve(va[ti] + st * sb, &vw) && vw.stream == st && vw.cell0 == 0 && vw.byte_in_row == 0 &&
                  vw.plane == &kb.tensors()[ti].planes[st] && vw.t == &kb.tensors()[ti]);
            const uint32_t cell = (uint32_t) rnd(kc.kv);
            CHECK(kb.resolve(va[ti] + st * sb + cell * rb + 5, &vw) && vw.stream == st && vw.cell0 == cell && vw.byte_in_row == 5);
        }
    }
    tsr_kv_view vw;
    CHECK(!kb.resolve(va[0] - 1, &vw));
    printf("  kv C=%u E=%u kv_size=%u n_stream=%u tensors=%u window=%llu: %llu windows (%llu read), %.2f MiB in, %.2f MiB out\n",
           kc.C, kc.E, kc.kv, kc.ns, kc.ntens, (unsigned long long) kc.window, (unsigned long long) kb.st_windows,
           (unsigned long long) kb.st_windows_read, kb.st_bytes_in / 1048576.0, kb.st_bytes_out / 1048576.0);
}

// ---------------------------------------------------------------------------------------------
// Mask summaries
// ---------------------------------------------------------------------------------------------
void test_msum_layout() {
    CHECK(tsr_msum_is_mask_name("attn_inp_kq_mask"));
    CHECK(tsr_msum_is_mask_name("Tesseract0#attn_inp_kq_mask#0"));
    CHECK(!tsr_msum_is_mask_name("attn_inp_pos"));
    int64_t ne[4] = { 300, 5, 1, 1 };
    size_t nb[4] = { 2, 600, 3000, 3000 };
    tsr_msum_layout L;
    CHECK(tsr_msum_layout_of("kq_mask", true, ne, nb, 3000, &L));
    CHECK(L.n_kv == 300 && L.n_q == 5 && L.row_stride == 300);
    CHECK(L.mask_bytes == 3008 && L.msum_off == 3008);
    CHECK(L.msum_bytes == 128);                       // 4 * (8 + 4 * 5) = 112 -> 128
    CHECK(L.copy_off == 3136 && L.copy_bytes == 5ull * 64 * 19 && L.total == 3136 + 5ull * 64 * 19);
    CHECK(!tsr_msum_layout_of("kq_mask", false, ne, nb, 3000, &L));     // F32
    CHECK(!tsr_msum_layout_of("mask", true, ne, nb, 3000, &L));         // name
    int64_t ne2[4] = { 300, 5, 1, 2 };                                  // n_stream 2
    CHECK(!tsr_msum_layout_of("kq_mask", true, ne2, nb, 6000, &L));
    size_t nb3[4] = { 2, 640, 3200, 3200 };                             // padded rows
    CHECK(tsr_msum_layout_of("kq_mask", true, ne, nb3, 3200, &L) && L.row_stride == 320);
}

constexpr uint16_t NINF = 0xFC00u;

// A random F16 mask of n_q rows (row stride rs >= n_kv elements; padding elements random).
std::vector<uint16_t> random_mask(std::mt19937_64 & rng, uint32_t n_q, uint32_t n_kv, size_t rs, int mode) {
    std::vector<uint16_t> m((size_t) n_q * rs);
    for (auto & h : m) h = (uint16_t) rng();                             // padding: garbage
    // mode 2 (a wrapped SWA ring): positions run past n_kv and cell j holds the latest position
    // P' <= P (the last query's) with P' = j mod n_kv; other modes: cell j holds position j
    const uint32_t p0 = (uint32_t) (rng() % (mode == 2 ? 3u * n_kv + 1u : n_kv + 1u));
    const uint32_t swa = 1 + (uint32_t) (rng() % 300);
    const int64_t P = (int64_t) p0 + n_q - 1;
    for (uint32_t q = 0; q < n_q; ++q) {
        uint16_t * row = m.data() + (size_t) q * rs;
        const int64_t pq = (int64_t) p0 + q;
        for (uint32_t j = 0; j < n_kv; ++j) {
            const int64_t pj = mode == 2 ? P - ((P - (int64_t) j) % (int64_t) n_kv + (int64_t) n_kv) % (int64_t) n_kv : (int64_t) j;
            bool vis = pj >= 0 && pj <= pq;
            if (mode == 1 || mode == 2) vis = vis && pq - pj < swa;
            row[j] = vis ? ((rng() % 7 == 0) ? 0x8000u : 0x0000u) : NINF;
            if (mode == 3 && vis && rng() % 9 == 0) row[j] = NINF;       // holes
            if (mode == 4 && vis && rng() % 5 == 0) {
                const uint16_t mag = (uint16_t) (0x3000u + rng() % 0x0C00u);
                row[j] = (uint16_t) (mag | (rng() % 2 ? 0x8000u : 0u));
            }
            if (mode == 5) row[j] = NINF;                                // empty rows
        }
    }
    return m;
}

struct msum_env {
    fake_dev dev;
    tsr_range_alloc side;               // side allocations (ch0)
    tsr_msum_store st;
    int allocs = 0, frees = 0;
    // device of `bytes` (ch0); its top quarter holds the side allocations
    explicit msum_env(const tsr_att_cfg & cfg, uint64_t bytes = 64u << 20) : dev(1, bytes, 0xCD) {
        side.init(bytes / 4 * 3, bytes / 4, 64);
        st.init(cfg, dev.writer(),
                [this](uint64_t bytes, uint32_t * ch, uint64_t * off) { *ch = 0; ++allocs; return side.alloc(bytes, off); },
                [this](uint32_t, uint64_t off) { ++frees; side.release(off); });
    }
};

// Expected decision of the producer for a mask and readers (the same reference functions it runs,
// tile by tile): refused (promotion), and per scale whether a copy is read.
void expect_decision(const tsr_att_cfg & cfg, const std::vector<uint32_t> & words, const std::vector<tsr_msum_reader> & rd,
                     bool * refused, std::set<uint32_t> * need) {
    *refused = false;
    need->clear();
    tsr_msum ms;
    tsr_msum_view(words.data(), words.size(), &ms);
    if (ms.flags & TSR_MSUM_GENERAL) {
        for (const auto & r : rd) need->insert(r.scale_bits);
        return;
    }
    static std::vector<tsr_fa_chunk> ch(256);
    for (const auto & r : rd) {
        uint32_t GS, Tq;
        tsr_fa_pass_rule(&cfg, r.G, &GS, &Tq);
        for (uint32_t q0 = 0; q0 < ms.n_q; q0 += Tq) {
            const uint32_t tq = std::min(Tq, ms.n_q - q0);
            long n = tsr_fa_plan(&cfg, r.E, r.D, &ms, q0, tq, nullptr, 0, ch.data(), ch.size());
            if (n > (long) ch.size()) {
                ch.resize((size_t) n);
                n = tsr_fa_plan(&cfg, r.E, r.D, &ms, q0, tq, nullptr, 0, ch.data(), ch.size());
            }
            if (n == -3) *refused = true;
            for (long k = 0; k < n; ++k) {
                if (ch[(size_t) k].general) need->insert(r.scale_bits);
            }
        }
    }
    if (*refused) {
        for (const auto & r : rd) need->insert(r.scale_bits);
    }
}

uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

void test_msum_random(uint32_t seed) {
    std::mt19937_64 rng(seed);
    const tsr_att_cfg cfg = { 4, 8, 9344, 32768 };
    msum_env env(cfg);
    long n_general = 0, n_copies = 0, n_masks = 0;
    for (int it = 0; it < 300; ++it) {
        const uint32_t q_max = rng() % 3 == 0 ? 40 : 4;          // (one rng() per statement: a fixed sequence)
        const uint32_t n_q = 1 + (uint32_t) (rng() % q_max);
        const uint32_t n_kv = 1 + (uint32_t) (rng() % 700);
        const size_t rs = n_kv + (rng() % 3 == 0 ? (size_t) (rng() % 40) : 0);
        const int mode = (int) (rng() % 6);
        const std::vector<uint16_t> m = random_mask(rng, n_q, n_kv, rs, mode);
        int64_t ne[4] = { n_kv, n_q, 1, 1 };
        size_t nb[4] = { 2, rs * 2, rs * 2 * n_q, rs * 2 * n_q };
        tsr_msum_layout L;
        CHECK(tsr_msum_layout_of("attn_inp_kq_mask", true, ne, nb, rs * 2 * n_q, &L));
        const uint64_t key = 0x4000000000000000ull + (uint64_t) (it % 7) * 0x100000;   // keys reused
        const uint64_t off = 0x10000 + (uint64_t) (it % 7) * (6u << 20);
        if (L.total > (6u << 20)) continue;
        ++n_masks;
        // the reservation: a sentinel the producer must not touch unless it writes a copy
        memset(env.dev.mem[0].data() + off + L.copy_off, 0xCD, (size_t) L.copy_bytes);
        env.st.on_set(key, 0, off, L, m.data());
        std::vector<uint32_t> want(tsr_msum_words(n_q));
        const uint32_t fl = tsr_msum_build_f16(m.data(), rs, n_kv, n_q, tsr_dram_handle(0, off + L.copy_off), want.data());
        n_general += (fl & TSR_MSUM_GENERAL) != 0;
        const uint8_t * dev_words = env.dev.mem[0].data() + off + L.msum_off;
        // readers registered earlier at this key were dropped only if the layout changed
        std::vector<uint32_t> last;
        CHECK(env.st.last_msum(key, &last));
        if (memcmp(last.data(), want.data(), want.size() * 4) != 0) {
            // only a promotion (a reader of an earlier mask at this key with the same layout) may differ
            CHECK((last[3] & TSR_MSUM_GENERAL) && !(want[3] & TSR_MSUM_GENERAL));
        } else {
            CHECK(memcmp(dev_words, want.data(), want.size() * 4) == 0);
        }
        // attach 1-2 readers (decode or prefill shapes), compare the copies
        std::vector<tsr_msum_reader> rd;
        const uint32_t nr = 1 + (uint32_t) (rng() % 2);
        for (uint32_t i = 0; i < nr; ++i) {
            tsr_msum_reader r;
            r.D = 64u << (rng() % 3);
            r.E = r.D * (1 + (uint32_t) (rng() % 4));
            r.G = 1 + (uint32_t) (rng() % 8);
            r.scale_bits = fbits(i == 0 ? 0.125f : 1.0f);
            rd.push_back(r);
        }
        tsr_msum_ref ref;
        for (const auto & r : rd) {
            ref = env.st.attach(key, r);
            CHECK(ref.valid && ref.n_q == n_q && ref.n_kv == n_kv);
            CHECK(ref.msum_handle == tsr_dram_handle(0, off + L.msum_off));
            CHECK(ref.msum_range.lo == off + L.msum_off && ref.msum_range.hi == off + L.msum_off + want.size() * 4);
        }
        // all readers registered at this key (also earlier masks' with the same layout)
        std::vector<uint32_t> cur;
        env.st.last_msum(key, &cur);
        bool refused = false;
        std::set<uint32_t> need;
        // the expected copies follow from the readers of this production; check the copies of the
        // readers just attached against the reference when their scale needs one
        expect_decision(cfg, want, rd, &refused, &need);
        if (refused) CHECK(cur[3] & TSR_MSUM_GENERAL);
        for (const auto & r : rd) {
            const tsr_msum_ref rr = env.st.attach(key, r);   // known reader: a lookup
            const uint8_t * cp = env.dev.mem[0].data() + rr.copy_range.lo;
            if (need.count(r.scale_bits) || (cur[3] & TSR_MSUM_GENERAL)) {
                std::vector<float> wantc((size_t) (L.copy_bytes / 4));
                float sc;
                memcpy(&sc, &r.scale_bits, 4);
                tsr_msum_mask_f32(m.data(), rs, n_kv, n_q, sc, wantc.data());
                CHECK(memcmp(cp, wantc.data(), (size_t) L.copy_bytes) == 0);
                ++n_copies;
            } else if (rr.copy_range.lo == off + L.copy_off) {
                // not needed: the reservation is untouched unless an earlier reader needed it
                (void) cp;
            }
        }
    }
    printf("  msum random: %ld masks, %ld GENERAL, %ld copies checked; producer: %llu produced, %llu copies, %llu promoted, "
           "%llu re-produced, side allocations %d (released %d)\n", n_masks, n_general, n_copies,
           (unsigned long long) env.st.st_produced, (unsigned long long) env.st.st_copies,
           (unsigned long long) env.st.st_promoted, (unsigned long long) env.st.st_reattach, env.allocs, env.frees);
    CHECK(n_general > 10 && n_copies > 10);
    env.st.forget(0, ~0ull);
    CHECK(env.st.n_masks() == 0 && env.frees == env.allocs);
}

// Exact behaviour on fresh keys: no reader -> no copy; reader with FAST tiles -> no copy; GENERAL
// -> copy; refusal -> promotion; two scales; ordering; nomask.
void test_msum_rules() {
    const tsr_att_cfg cfg = { 4, 8, 9344, 32768 };
    msum_env env(cfg);
    auto layout = [](uint32_t n_q, uint32_t n_kv) {
        int64_t ne[4] = { n_kv, n_q, 1, 1 };
        size_t nb[4] = { 2, (size_t) n_kv * 2, (size_t) n_kv * 2 * n_q, (size_t) n_kv * 2 * n_q };
        tsr_msum_layout L;
        tsr_msum_layout_of("kq_mask", true, ne, nb, (size_t) n_kv * 2 * n_q, &L);
        return L;
    };
    const tsr_msum_reader qwen = { fbits(0.125f), 128, 64, 7 };   // E = 2 x 64, G = 7
    // 1. decode causal mask: FAST, no copy, even with a reader
    {
        const uint32_t n_kv = 256;
        std::vector<uint16_t> m(n_kv, NINF);
        for (uint32_t j = 0; j <= 40; ++j) m[j] = 0;
        const tsr_msum_layout L = layout(1, n_kv);
        const uint64_t key = 1000, off = 0x100000;
        CHECK(!env.st.attach(key, qwen).valid);                      // before the first set_tensor
        env.st.on_set(key, 0, off, L, m.data());
        const uint64_t copies = env.st.st_copies;
        const tsr_msum_ref r = env.st.attach(key, qwen);
        CHECK(r.valid && r.flags == 0 && env.st.st_copies == copies && env.st.st_reattach == 1);
        CHECK(r.mask_handle == tsr_dram_handle(0, off + L.copy_off));
        const uint32_t * w = (const uint32_t *) (env.dev.mem[0].data() + off + L.msum_off);
        CHECK(w[0] == TSR_MSUM_MAGIC && w[1] == 1 && w[2] == n_kv && w[3] == 0 && w[8] == 0 && w[9] == 41 && w[10] == 0 && w[11] == 0);
        // a reattach of the same reader is a lookup
        env.st.attach(key, qwen);
        CHECK(env.st.st_reattach == 1);
        // next token: the reader is known, set_tensor alone produces
        m[41] = 0;
        const uint64_t prod = env.st.st_produced;
        env.st.on_set(key, 0, off, L, m.data());
        CHECK(env.st.st_produced == prod + 1 && w[9] == 42);
    }
    // 2. wrapped SWA ring in decode: two runs, FAST; in prefill (rows with two runs, T_q > 1): GENERAL
    //    tiles -> the copy is written although the MSUM is not GENERAL
    {
        const uint32_t n_kv = 512, n_q = 4;
        std::vector<uint16_t> m((size_t) n_q * n_kv, NINF);
        for (uint32_t q = 0; q < n_q; ++q) {
            for (uint32_t j = 0; j < 30 + q; ++j) m[(size_t) q * n_kv + j] = 0;          // new cells at the ring start
            for (uint32_t j = 400 + q; j < n_kv; ++j) m[(size_t) q * n_kv + j] = 0;      // older cells at its end
        }
        const tsr_msum_layout L = layout(n_q, n_kv);
        const uint64_t key = 2000, off = 0x200000;
        const tsr_msum_reader gemma = { fbits(1.0f), 2048, 256, 2 };                    // T_q = 4 at T_MAX 8
        env.st.on_set(key, 0, off, L, m.data());
        const uint64_t copies = env.st.st_copies;
        const tsr_msum_ref r = env.st.attach(key, gemma);
        CHECK(r.valid && (r.flags & TSR_MSUM_GENERAL) == 0 && env.st.st_copies == copies + 1);
        std::vector<float> want((size_t) (L.copy_bytes / 4));
        tsr_msum_mask_f32(m.data(), n_kv, n_kv, n_q, 1.0f, want.data());
        CHECK(memcmp(env.dev.mem[0].data() + off + L.copy_off, want.data(), (size_t) L.copy_bytes) == 0);
    }
    // 3. GENERAL by values, two scales: slot 0 the reservation, slot 1 a side allocation; each copy
    //    with its own scale; forget() releases the side copy
    {
        const uint32_t n_kv = 100, n_q = 2;
        std::vector<uint16_t> m((size_t) n_q * n_kv, 0);
        m[5] = 0x3C00;            // 1.0
        m[n_kv + 7] = 0xC000;     // -2.0
        m[n_kv + 8] = NINF;
        const tsr_msum_layout L = layout(n_q, n_kv);
        const uint64_t key = 3000, off = 0x300000;
        env.st.on_set(key, 0, off, L, m.data());
        const tsr_msum_reader a = { fbits(0.125f), 128, 64, 1 }, b = { fbits(0.5f), 128, 64, 1 };
        const tsr_msum_ref ra = env.st.attach(key, a);
        const int allocs = env.allocs;
        const tsr_msum_ref rb = env.st.attach(key, b);
        CHECK(env.allocs == allocs + 1);
        CHECK(ra.mask_handle == tsr_dram_handle(0, off + L.copy_off) && rb.mask_handle != ra.mask_handle);
        CHECK((ra.flags & TSR_MSUM_GENERAL) && (rb.flags & TSR_MSUM_GENERAL));
        const uint32_t * w = (const uint32_t *) (env.dev.mem[0].data() + off + L.msum_off);
        CHECK(w[3] == TSR_MSUM_GENERAL && w[4] == ra.mask_handle);
        std::vector<float> wa((size_t) (L.copy_bytes / 4)), wb(wa.size());
        tsr_msum_mask_f32(m.data(), n_kv, n_kv, n_q, 0.125f, wa.data());
        tsr_msum_mask_f32(m.data(), n_kv, n_kv, n_q, 0.5f, wb.data());
        CHECK(memcmp(env.dev.mem[0].data() + ra.copy_range.lo, wa.data(), (size_t) L.copy_bytes) == 0);
        CHECK(memcmp(env.dev.mem[0].data() + rb.copy_range.lo, wb.data(), (size_t) L.copy_bytes) == 0);
        CHECK(wa[5] == 8.0f && wb[5] == 2.0f);                                          // fdiv(1, 0.125), fdiv(1, 0.5)
        // a +-0 / -inf mask with holes (GENERAL by runs): the same image for both scales
        std::vector<uint16_t> h((size_t) n_q * n_kv, 0);
        for (uint32_t j = 3; j < n_kv; j += 10) h[j] = NINF;
        h[n_kv + 2] = 0x8000;
        env.st.on_set(key, 0, off, L, h.data());
        std::vector<float> wh((size_t) (L.copy_bytes / 4));
        tsr_msum_mask_f32(h.data(), n_kv, n_kv, n_q, 1.0f, wh.data());
        CHECK(memcmp(env.dev.mem[0].data() + ra.copy_range.lo, wh.data(), (size_t) L.copy_bytes) == 0);
        CHECK(memcmp(env.dev.mem[0].data() + rb.copy_range.lo, wh.data(), (size_t) L.copy_bytes) == 0);
        const int frees = env.frees;
        env.st.forget(key, key + 1);
        CHECK(env.frees == frees + 1 && !env.st.attach(key, a).valid);
    }
    // 4. a new layout at a key drops its readers and side copies
    {
        const uint64_t key = 4000, off = 0x400000;
        std::vector<uint16_t> m(64, 0);
        const tsr_msum_layout L1 = layout(1, 64), L2 = layout(2, 32);
        env.st.on_set(key, 0, off, L1, m.data());
        env.st.attach(key, qwen);
        const tsr_msum_reader other = { fbits(0.5f), 128, 64, 7 };
        env.st.attach(key, other);
        const int frees = env.frees;
        env.st.on_set(key, 0, off, L2, m.data());
        CHECK(env.frees == frees + 1);
        const uint64_t re = env.st.st_reattach;
        env.st.attach(key, qwen);
        CHECK(env.st.st_reattach == re + 1);                         // a new reader again
    }
    // 5. nomask
    {
        const tsr_msum_ref r = env.st.nomask(3, 200), r2 = env.st.nomask(3, 200);
        CHECK(r.valid && r.mask_handle == tsr::opl::HANDLE_NONE && r.msum_handle == r2.msum_handle);
        const uint32_t * w = (const uint32_t *) (env.dev.mem[0].data() + r.msum_range.lo);
        CHECK(w[0] == TSR_MSUM_MAGIC && w[1] == 3 && w[2] == 200 && w[3] == 0);
        for (uint32_t q = 0; q < 3; ++q) {
            CHECK(w[8 + 4 * q] == 0 && w[9 + 4 * q] == 200 && w[10 + 4 * q] == 0 && w[11 + 4 * q] == 0);
        }
    }
    env.st.clear();
    CHECK(env.allocs == env.frees);
}

// Promotion: masks whose prefill tiles the planner refuses (rule 5 / 6; tiny SWA windows next to the
// causal edge at a small PCAP, found by a seeded search). The producer must mark them GENERAL and
// write the copy; masks it does not refuse keep their flags.
void test_msum_promotion() {
    std::mt19937_64 rng(99);
    long refused = 0, kept = 0;
    for (int it = 0; it < 100000 && refused < 8; ++it) {
        const uint32_t C = 1 + (uint32_t) (rng() % 4), E = 64u * (1 + (uint32_t) (rng() % 2)), D = 64, G = 1 + (uint32_t) (rng() % 2);
        const tsr_att_cfg cfg = { C, 8, 16u * (E / D) * (1 + (uint32_t) (rng() % 3)), 32768 };
        const uint32_t n_q = 2 + (uint32_t) (rng() % 7), n_kv = 16 * (8 + (uint32_t) (rng() % 120));
        uint32_t p0 = (uint32_t) (rng() % n_kv);
        if (p0 + n_q > n_kv) p0 = n_kv - n_q;
        const uint32_t swa = 1 + (uint32_t) (rng() % 12);
        std::vector<uint16_t> m((size_t) n_q * n_kv, NINF);
        for (uint32_t q = 0; q < n_q; ++q) {
            const int64_t pq = (int64_t) p0 + q;
            for (uint32_t j = 0; j < n_kv; ++j) {
                if ((int64_t) j <= pq && pq - (int64_t) j < (int64_t) swa) m[(size_t) q * n_kv + j] = 0;
            }
        }
        std::vector<uint32_t> want(tsr_msum_words(n_q));
        tsr_msum_build_f16(m.data(), n_kv, n_kv, n_q, 0, want.data());
        const tsr_msum_reader r = { fbits(0.125f), E, D, G };
        bool will_refuse = false;
        std::set<uint32_t> need;
        expect_decision(cfg, want, { r }, &will_refuse, &need);
        if (!will_refuse && kept >= 50) continue;
        msum_env env(cfg, 1u << 20);
        int64_t ne[4] = { n_kv, n_q, 1, 1 };
        size_t nb[4] = { 2, (size_t) n_kv * 2, (size_t) n_kv * 2 * n_q, (size_t) n_kv * 2 * n_q };
        tsr_msum_layout L;
        tsr_msum_layout_of("kq_mask", true, ne, nb, (size_t) n_kv * 2 * n_q, &L);
        const uint64_t off = 0x10000;
        env.st.on_set(77, 0, off, L, m.data());
        const tsr_msum_ref ref = env.st.attach(77, r);
        const uint32_t * w = (const uint32_t *) (env.dev.mem[0].data() + off + L.msum_off);
        if (will_refuse) {
            ++refused;
            CHECK((ref.flags & TSR_MSUM_GENERAL) && w[3] == TSR_MSUM_GENERAL && w[4] == ref.mask_handle && env.st.st_promoted == 1);
            CHECK(memcmp(w + 8, want.data() + 8, (want.size() - 8) * 4) == 0);     // the rows keep the runs
            std::vector<float> wc((size_t) (L.copy_bytes / 4));
            tsr_msum_mask_f32(m.data(), n_kv, n_kv, n_q, 0.125f, wc.data());
            CHECK(memcmp(env.dev.mem[0].data() + off + L.copy_off, wc.data(), (size_t) L.copy_bytes) == 0);
            // with the promoted MSUM no tile of this reader is refused any more
            tsr_msum ms;
            std::vector<uint32_t> cur(w, w + want.size());
            tsr_msum_view(cur.data(), cur.size(), &ms);
            uint32_t GS, Tq;
            tsr_fa_pass_rule(&cfg, G, &GS, &Tq);
            std::vector<tsr_fa_chunk> ch(4096);
            for (uint32_t q0 = 0; q0 < n_q; q0 += Tq) {
                CHECK(tsr_fa_plan(&cfg, E, D, &ms, q0, std::min(Tq, n_q - q0), nullptr, 0, ch.data(), ch.size()) > 0);
            }
        } else {
            ++kept;
            CHECK(ref.flags == want[3] && env.st.st_promoted == 0);
        }
    }
    printf("  msum promotion: %ld refused masks promoted, %ld others kept\n", refused, kept);
    CHECK(refused >= 5);
}

}  // namespace

int main() {
    test_kv_names();
    test_kv_place();
    const kv_case cases[] = {
        { 4, 128, 1000, 1, 4, 0 },          // Qwen-like, default window
        { 4, 128, 1000, 1, 2, 9000 },       // many small windows
        { 3, 192, 777, 2, 3, 20000 },       // C = 3, two streams, E not a power of two
        { 1, 64, 300, 1, 2, 3000 },         // C = 1
        { 4, 1024, 200, 1, 2, 0 },          // E = 1024 (one chunk per K row)
        { 4, 64, 4100, 2, 2, 50000 },       // E = 64 (16 rows per K chunk), kv_size not a multiple of 64
    };
    uint32_t seed = 1;
    for (const kv_case & kc : cases) {
        test_kv_convert(kc, seed++);
    }
    test_msum_layout();
    test_msum_rules();
    test_msum_random(5);
    test_msum_promotion();
    if (g_fail) {
        fprintf(stderr, "test-tsr-kvbuf: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-tsr-kvbuf: OK\n");
    return 0;
}
