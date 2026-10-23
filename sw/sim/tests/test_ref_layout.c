// test_ref_layout.c — ROW_BEATS, repack/unrepack, symmetric allocation, QA line counts.
// Spec: "디바이스 가중치 레이아웃", "메모리 모델 / 대칭 할당", "양자화 활성값(QA) 레이아웃".
// Independent of ggml: every expectation is recomputed here from the spec text, not from
// tsr_ref internals.

#include "test_util.h"
#include "tsr_ref.h"

// ---------------------------------------------------------------------------
// Independent ROW_BEATS: walk the row block by block like the chunk parser would.
// ---------------------------------------------------------------------------
static uint32_t row_beats_walk(uint32_t block_size, uint32_t S, uint32_t QB, uint32_t K) {
    uint32_t nb = K / block_size, beats = 0, in_chunk = 0, qbytes = 0;
    for (uint32_t i = 0; i < nb; ++i) {
        if (in_chunk == 0) beats += 1;            // meta beat opens a chunk
        qbytes += QB;
        if (++in_chunk == S || i == nb - 1) {     // chunk closes: quant beats, beat aligned
            beats += (qbytes + 63) / 64;
            in_chunk = 0;
            qbytes = 0;
        }
    }
    return beats;
}

static void test_row_beats(void) {
    // Spec examples.
    CHECK(tsr_row_beats(TSR_FMT_Q4_K, 27648) == 243, "Q4_K K=27648 -> %u", tsr_row_beats(TSR_FMT_Q4_K, 27648));
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 5120) == 85, "Q8_0 K=5120 -> %u", tsr_row_beats(TSR_FMT_Q8_0, 5120));
    // Partial-chunk cases (hand-computed from the formula in the spec).
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 32 * 33) == 17 + 2, "Q8_0 nb=33");
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 32 * 1) == 2, "Q8_0 nb=1");
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 32 * 2) == 2, "Q8_0 nb=2");
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 32 * 3) == 3, "Q8_0 nb=3 (odd rem)");
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 32 * 69) == 2 * 17 + 1 + 3, "Q8_0 nb=69 (full=2, rem=5)");
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 32 * 63) == 17 + 1 + 16, "Q8_0 nb=63 (rem=31)");
    CHECK(tsr_row_beats(TSR_FMT_Q4_K, 256 * 5) == 9 + 3, "Q4_K nb=5");
    CHECK(tsr_row_beats(TSR_FMT_Q4_K, 256 * 1) == 3, "Q4_K nb=1");
    CHECK(tsr_row_beats(TSR_FMT_Q4_K, 256 * 3) == 7, "Q4_K nb=3");
    CHECK(tsr_row_beats(TSR_FMT_Q4_K, 256 * 4) == 9, "Q4_K nb=4");
    // Invalid inputs give 0.
    CHECK(tsr_row_beats(2, 256) == 0, "reserved fmt");
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 33) == 0, "K not multiple of 32");
    CHECK(tsr_row_beats(TSR_FMT_Q4_K, 512 + 32) == 0, "K not multiple of 256");
    CHECK(tsr_row_beats(TSR_FMT_Q8_0, 0) == 0, "K=0");

    // Every legal K up to K_MAX against the independent walker; 0% overhead when rem == 0.
    for (uint32_t K = 32; K <= 32768; K += 32) {
        uint32_t rb = tsr_row_beats(TSR_FMT_Q8_0, K);
        CHECK(rb == row_beats_walk(32, 32, 32, K), "Q8_0 K=%u: %u vs walk %u", K, rb, row_beats_walk(32, 32, 32, K));
        if ((K / 32) % 32 == 0) CHECK(rb * 64u == (K / 32) * 34u, "Q8_0 K=%u not 0%% overhead", K);
    }
    for (uint32_t K = 256; K <= 32768; K += 256) {
        uint32_t rb = tsr_row_beats(TSR_FMT_Q4_K, K);
        CHECK(rb == row_beats_walk(256, 4, 128, K), "Q4_K K=%u: %u vs walk %u", K, rb, row_beats_walk(256, 4, 128, K));
        if ((K / 256) % 4 == 0) CHECK(rb * 64u == (K / 256) * 144u, "Q4_K K=%u not 0%% overhead", K);
    }

    // Format tables.
    CHECK(tsr_fmt_block_size(TSR_FMT_Q8_0) == 32 && tsr_fmt_block_size(TSR_FMT_Q4_K) == 256, "block size");
    CHECK(tsr_fmt_chunk_blocks(TSR_FMT_Q8_0) == 32 && tsr_fmt_chunk_blocks(TSR_FMT_Q4_K) == 4, "S");
    CHECK(tsr_fmt_quant_bytes(TSR_FMT_Q8_0) == 32 && tsr_fmt_quant_bytes(TSR_FMT_Q4_K) == 128, "QB");
    CHECK(tsr_fmt_ggml_block_bytes(TSR_FMT_Q8_0) == 34 && tsr_fmt_ggml_block_bytes(TSR_FMT_Q4_K) == 144, "ggml bytes");
    CHECK(tsr_fmt_valid(0) && tsr_fmt_valid(1) && !tsr_fmt_valid(2) && !tsr_fmt_valid(15), "fmt_valid");
    CHECK(tsr_fmt_k_valid(TSR_FMT_Q8_0, 32, 32768), "k_valid 32");
    CHECK(!tsr_fmt_k_valid(TSR_FMT_Q8_0, 0, 32768), "k_valid 0");
    CHECK(!tsr_fmt_k_valid(TSR_FMT_Q8_0, 48, 32768), "k_valid 48");
    CHECK(tsr_fmt_k_valid(TSR_FMT_Q4_K, 32768, 32768), "k_valid kmax");
    CHECK(!tsr_fmt_k_valid(TSR_FMT_Q4_K, 32768 + 256, 32768), "k_valid > kmax");
    CHECK(!tsr_fmt_k_valid(TSR_FMT_Q4_K, 128, 32768), "k_valid Q4_K 128");
    CHECK(!tsr_fmt_k_valid(3, 256, 32768), "k_valid bad fmt");
}

// ---------------------------------------------------------------------------
// Repack / unrepack
// ---------------------------------------------------------------------------
// Spec byte placement, written out independently of tsr_ref.
static size_t spec_chunk_start(uint32_t fmt, uint32_t i) {
    return fmt == TSR_FMT_Q8_0 ? (size_t)(i / 32) * 17 * 64 : (size_t)(i / 4) * 9 * 64;
}

static void check_repacked_row(uint32_t fmt, uint32_t K, const uint8_t * blocks, const uint8_t * dst,
                               size_t row_bytes) {
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    uint8_t * covered = (uint8_t *)calloc(row_bytes, 1);
    if (fmt == TSR_FMT_Q8_0) {
        for (uint32_t i = 0; i < nb; ++i) {
            const uint8_t * blk = blocks + (size_t)i * 34;
            const size_t cs = spec_chunk_start(fmt, i);
            const uint32_t b = i % 32;
            // meta bytes [2b, 2b+2) = d[b]
            CHECK(dst[cs + 2 * b] == blk[0] && dst[cs + 2 * b + 1] == blk[1], "Q8_0 K=%u blk %u meta", K, i);
            covered[cs + 2 * b] = covered[cs + 2 * b + 1] = 1;
            // qs[32] at quant beat floor(b/2), bytes [32*(b%2), +32)
            const size_t qo = cs + 64 + 64 * (b / 2) + 32 * (b % 2);
            CHECK(memcmp(dst + qo, blk + 2, 32) == 0, "Q8_0 K=%u blk %u quant", K, i);
            memset(covered + qo, 1, 32);
        }
        // "quant beat j holds chunk weights [64j, 64j+64)"
        for (uint32_t c = 0; c * 32 < nb; ++c) {
            const size_t cs = (size_t)c * 17 * 64;
            for (uint32_t w = 0; w < 1024 && c * 32 + w / 32 < nb; ++w) {
                const uint32_t j = w / 64, o = w % 64;
                const uint8_t * blk = blocks + (size_t)(c * 32 + w / 32) * 34;
                CHECK(dst[cs + 64 + 64 * j + o] == blk[2 + w % 32], "Q8_0 K=%u chunk %u weight %u", K, c, w);
            }
        }
    } else {
        for (uint32_t i = 0; i < nb; ++i) {
            const uint8_t * blk = blocks + (size_t)i * 144;
            const size_t cs = spec_chunk_start(fmt, i);
            const uint32_t s = i % 4;
            // meta bytes [16s, 16s+16) = first 16 bytes of the block (d, dmin, scales[12])
            CHECK(memcmp(dst + cs + 16 * s, blk, 16) == 0, "Q4_K K=%u sb %u meta", K, i);
            memset(covered + cs + 16 * s, 1, 16);
            // qs[128] unchanged in quant beats 2s, 2s+1
            const size_t qo = cs + 64 + 128 * s;
            CHECK(memcmp(dst + qo, blk + 16, 128) == 0, "Q4_K K=%u sb %u quant", K, i);
            memset(covered + qo, 1, 128);
            // beat 2s holds sb weights [0,128), beat 2s+1 holds [128,256): decode the nibble of
            // weight k as ggml's dequantize_row_q4_K loop lays it out and find it in the beat.
            for (uint32_t k = 0; k < 256; ++k) {
                const uint32_t g = k / 64, hi = (k % 64) / 32, l = k % 32;          // ggml order
                const uint8_t want = (uint8_t)((blk[16 + 32 * g + l] >> (4 * hi)) & 0xF);
                const uint32_t beat = 2 * s + k / 128, kk = k % 128;
                const uint8_t byte = dst[cs + 64 + 64 * beat + 32 * (kk / 64) + (kk % 32)];
                const uint8_t got = (uint8_t)((byte >> (4 * ((kk % 64) / 32))) & 0xF);
                CHECK(got == want, "Q4_K K=%u sb %u weight %u", K, i, k);
            }
        }
    }
    // Everything else (unused meta entries, tail of the last quant beat) must be zero.
    for (size_t o = 0; o < row_bytes; ++o) {
        if (!covered[o]) CHECK(dst[o] == 0, "fmt %u K=%u byte %zu not zero (0x%02x)", fmt, K, o, dst[o]);
    }
    free(covered);
}

static void roundtrip_one(rng_t * rng, uint32_t fmt, uint32_t K) {
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const size_t bb = tsr_fmt_ggml_block_bytes(fmt);
    const size_t row_bytes = (size_t)tsr_row_beats(fmt, K) * 64;
    uint8_t * blocks = (uint8_t *)malloc(nb * bb);
    uint8_t * dst = (uint8_t *)malloc(row_bytes + 128);
    uint8_t * back = (uint8_t *)malloc(nb * bb);
    rng_bytes(rng, blocks, nb * bb);
    memset(dst, 0xCC, row_bytes + 128);
    tsr_repack_row(fmt, blocks, K, dst);
    for (size_t o = row_bytes; o < row_bytes + 128; ++o) CHECK(dst[o] == 0xCC, "fmt %u K=%u wrote past row end", fmt, K);
    check_repacked_row(fmt, K, blocks, dst, row_bytes);
    memset(back, 0x5A, nb * bb);
    tsr_unrepack_row(fmt, dst, K, back);
    CHECK(memcmp(back, blocks, nb * bb) == 0, "fmt %u K=%u round trip", fmt, K);
    free(blocks); free(dst); free(back);
}

static void test_repack(void) {
    rng_t rng = { 0x7E55E7AC7ull };
    for (uint32_t nb = 1; nb <= 160; ++nb) roundtrip_one(&rng, TSR_FMT_Q8_0, 32 * nb);
    for (uint32_t nb = 1; nb <= 40; ++nb) roundtrip_one(&rng, TSR_FMT_Q4_K, 256 * nb);
    for (int it = 0; it < 60; ++it) {
        roundtrip_one(&rng, TSR_FMT_Q8_0, 32 * (1 + rng_below(&rng, 1024)));
        roundtrip_one(&rng, TSR_FMT_Q4_K, 256 * (1 + rng_below(&rng, 128)));
    }
    roundtrip_one(&rng, TSR_FMT_Q8_0, 32768);
    roundtrip_one(&rng, TSR_FMT_Q4_K, 32768);
    roundtrip_one(&rng, TSR_FMT_Q4_K, 27648);
}

// ---------------------------------------------------------------------------
// Symmetric allocation
// ---------------------------------------------------------------------------
static void test_alloc(void) {
    const uint32_t Ns[] = { 1, 2, 3, 4, 5, 7, 10, 64, 65, 127, 896, 5120, 27648, 152064 };
    const uint32_t Cs[] = { 1, 2, 3, 4, 8 };
    for (size_t a = 0; a < sizeof Ns / sizeof Ns[0]; ++a) {
        for (size_t b = 0; b < sizeof Cs / sizeof Cs[0]; ++b) {
            const uint32_t N = Ns[a], C = Cs[b], K = 1024;
            const uint32_t R = (N + C - 1) / C;
            CHECK(tsr_rows_per_ch(N, C) == R, "R N=%u C=%u", N, C);
            for (uint32_t fmt = 0; fmt < 2; ++fmt) {
                const uint32_t rb = tsr_row_beats(fmt, K);
                CHECK(tsr_tensor_ch_beats(fmt, N, K, C) == (uint64_t)R * rb, "ch beats N=%u C=%u", N, C);
                uint32_t prev_ch = 0;
                uint64_t prev_off = 0;
                for (uint32_t r = 0; r < N; ++r) {
                    uint32_t ch; uint64_t off;
                    tsr_row_location(fmt, N, K, C, r, &ch, &off);
                    // channel c owns rows [cR, min((c+1)R, N))
                    CHECK(ch < C && r >= ch * R && r < (ch + 1) * R, "row %u -> ch %u (N=%u C=%u)", r, ch, N, C);
                    CHECK(off == (uint64_t)(r - ch * R) * rb, "row %u off", r);
                    if (r > 0) {
                        // rows are contiguous inside a channel
                        if (ch == prev_ch) CHECK(off == prev_off + rb, "row %u not contiguous", r);
                        else CHECK(ch == prev_ch + 1 && off == 0, "row %u channel switch", r);
                    }
                    prev_ch = ch; prev_off = off;
                }
            }
        }
    }
    // Explicit: N=2, C=4 -> channels 0,1 hold one row each, 2,3 empty.
    uint32_t ch; uint64_t off;
    tsr_row_location(TSR_FMT_Q8_0, 2, 64, 4, 1, &ch, &off);
    CHECK(ch == 1 && off == 0, "N=2 C=4 row 1");
    CHECK(tsr_rows_per_ch(0, 4) == 0, "N=0");
}

// ---------------------------------------------------------------------------
// QA line counts
// ---------------------------------------------------------------------------
static void test_qa_lines(void) {
    CHECK(tsr_qa_block_size(TSR_QA_A32) == 32 && tsr_qa_block_size(TSR_QA_A256) == 256, "qa block");
    CHECK(tsr_qa_kind_for_fmt(TSR_FMT_Q8_0) == TSR_QA_A32 && tsr_qa_kind_for_fmt(TSR_FMT_Q4_K) == TSR_QA_A256, "kind");
    // Spec: K = 27648 broadcast is "about 500 lines": 432 + 7 + 54 = 493.
    CHECK(tsr_qa_qs_lines(TSR_QA_A256, 27648) == 432, "qs");
    CHECK(tsr_qa_d_lines(TSR_QA_A256, 27648) == 7, "d");
    CHECK(tsr_qa_bs_lines(TSR_QA_A256, 27648) == 54, "bs");
    CHECK(tsr_qa_lines(TSR_QA_A256, 27648) == 493, "total");
    CHECK(tsr_qa_lines(TSR_QA_A32, 5120) == 80 + 10, "A32 5120");
    CHECK(tsr_qa_bs_lines(TSR_QA_A32, 5120) == 0, "A32 no bs");
    CHECK(tsr_qa_lines(TSR_QA_A32, 32) == 1 + 1, "A32 32");
    CHECK(tsr_qa_lines(TSR_QA_A256, 256) == 4 + 1 + 1, "A256 256");
    for (uint32_t K = 32; K <= 32768; K += 32) {
        const uint32_t e = (K + 63) / 64 + (4 * (K / 32) + 63) / 64;
        CHECK(tsr_qa_lines(TSR_QA_A32, K) == e, "A32 K=%u", K);
        if (K % 256 == 0) {
            const uint32_t e2 = (K + 63) / 64 + (4 * (K / 256) + 63) / 64 + (2 * (K / 16) + 63) / 64;
            CHECK(tsr_qa_lines(TSR_QA_A256, K) == e2, "A256 K=%u", K);
        }
    }
    // Spec 구성 파라미터: the cluster activation buffer holds a whole QA vector per token,
    // QA_LINES(A256, K_MAX) x 64 x T_MAX bytes (QS + D + BS), not K_MAX x T_MAX.
    CHECK(tsr_qa_lines(TSR_QA_A256, 32768) == 584, "A256 K_MAX: 512 + 8 + 64 lines");
    CHECK(tsr_qa_lines(TSR_QA_A32, 32768) == 576, "A32 K_MAX: 512 + 64 lines");
    CHECK(tsr_qa_lines(TSR_QA_A256, 28672) * 64 <= 32768 && tsr_qa_lines(TSR_QA_A256, 28928) * 64 > 32768,
          "A256 outgrows K_MAX bytes above K = 28672");
    CHECK(tsr_qa_lines(TSR_QA_A32, 29120) * 64 <= 32768 && tsr_qa_lines(TSR_QA_A32, 29152) * 64 > 32768,
          "A32 outgrows K_MAX bytes above K = 29120");
    CHECK(tsr_qa_lines(TSR_QA_A32, 29568) == 520, "Qwen2.5-72B FFN (K = 29568, A32)");
}

// tsr_ref.h "Error policy": the functions that return a value instead of aborting.
static void test_error_policy(void) {
    // invalid fmt / kind -> 0
    CHECK(tsr_fmt_block_size(2) == 0 && tsr_fmt_chunk_blocks(2) == 0 && tsr_fmt_quant_bytes(2) == 0 &&
          tsr_fmt_ggml_block_bytes(2) == 0, "fmt tables: invalid fmt -> 0");
    CHECK(!tsr_fmt_valid(2) && !tsr_fmt_k_valid(2, 256, 32768), "fmt predicates");
    CHECK(tsr_qa_block_size(2) == 0, "qa block: invalid kind -> 0");
    CHECK(tsr_qa_qs_lines(2, 256) == 0 && tsr_qa_d_lines(2, 256) == 0 && tsr_qa_bs_lines(2, 256) == 0 &&
          tsr_qa_lines(2, 256) == 0, "qa lines: invalid kind -> 0");
    // tsr_row_beats also validates K
    CHECK(tsr_row_beats(2, 256) == 0 && tsr_row_beats(TSR_FMT_Q8_0, 48) == 0 && tsr_row_beats(TSR_FMT_Q4_K, 300) == 0,
          "row_beats: invalid fmt or K -> 0");
    CHECK(!tsr_fmt_k_valid(TSR_FMT_Q4_K, 300, 32768) && !tsr_fmt_k_valid(TSR_FMT_Q8_0, 32800, 32768),
          "k_valid: K not a multiple / above k_max");
    // the QA line counts do NOT validate K: they count floor(K / block) blocks (documented)
    CHECK(tsr_qa_lines(TSR_QA_A32, 48) == 2 && tsr_qa_lines(TSR_QA_A256, 300) == 7,
          "qa lines with an unvalidated K (A32 48 -> %u, A256 300 -> %u)",
          tsr_qa_lines(TSR_QA_A32, 48), tsr_qa_lines(TSR_QA_A256, 300));
}

int main(void) {
    test_row_beats();
    test_repack();
    test_alloc();
    test_qa_lines();
    test_error_policy();
    return test_finish("test_ref_layout");
}
