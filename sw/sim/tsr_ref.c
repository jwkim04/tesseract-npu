// tsr_ref.c — the single reference implementation of Tesseract numerics and layouts.
//
// Normative sources: docs/spec.md sections "디바이스 가중치 레이아웃", "양자화 활성값(QA)
// 레이아웃", "수치 규칙" and "구현 명확화 (v0.2.1)". The ISA simulator, the ggml backend and
// the RTL testbenches (ctypes on libtsr_ref.so) all use these functions; the RTL must match
// them bit for bit.
//
// Implementation rules
//   - Every f32 operation goes through tsr_fadd/fsub/fmul/fdiv/fsqrt/frecip, which apply DAZ
//     to the inputs with explicit bit tests, perform one native IEEE binary32 operation, and
//     then apply FTZ and NaN canonicalisation with explicit bit tests. MXCSR is never read or
//     written (the process also runs ggml CPU code).
//   - The native operation is assumed to run with the default IEEE environment
//     (round-to-nearest-even, no hardware FTZ/DAZ), which is what every hosted C process
//     starts with and what ggml leaves alone.
//   - Conversions (int -> f32, f32 -> int, fp16 <-> f32) are done purely on integer bits.
//   - Every multi-byte value read from or written to a device image (repacked rows, QA
//     images) is accessed byte-wise as little-endian, independent of host byte order.

#include "tsr_ref.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "tsr_ref.c needs FLT_EVAL_METHOD == 0 (f32 arithmetic without excess precision)"
#endif
#if defined(__FAST_MATH__)
#error "tsr_ref.c must not be compiled with -ffast-math"
#endif

_Static_assert(sizeof(float) == 4, "binary32 float required");
_Static_assert(sizeof(tsr_block_q8_0) == 34, "tsr_block_q8_0 must match ggml block_q8_0");
_Static_assert(sizeof(tsr_block_q4_K) == 144, "tsr_block_q4_K must match ggml block_q4_K");

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
static void tsr_fatal(const char * fn, const char * msg, uint32_t a, uint32_t b) {
    fprintf(stderr, "tsr_ref: %s: %s (%u, %u)\n", fn, msg, a, b);
    abort();
}

#define F32_SIGN 0x80000000u
#define F32_EXP  0x7F800000u
#define F32_MANT 0x007FFFFFu

uint32_t tsr_f32_bits(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }
float    tsr_bits_f32(uint32_t u) { float f; memcpy(&f, &u, sizeof f); return f; }

static inline uint32_t daz_bits(uint32_t u) {
    return ((u & F32_EXP) == 0) ? (u & F32_SIGN) : u;   // zero or subnormal -> signed zero
}
static inline float daz(float x) { return tsr_bits_f32(daz_bits(tsr_f32_bits(x))); }

// Result post-processing: NaN -> canonical NaN, subnormal -> signed zero. r is the native
// (gradual-underflow, bounded-exponent) RNE result, so this flushes iff |exact| < 2^-126 -
// 2^-150; not IEEE "tininess after rounding" (see tsr_ref.h and spec 구현 명확화).
static inline float fin(float r) {
    const uint32_t u = tsr_f32_bits(r);
    if ((u & F32_EXP) == F32_EXP) {
        return (u & F32_MANT) ? tsr_bits_f32(TSR_CANON_NAN_BITS) : r;
    }
    if ((u & F32_EXP) == 0) {
        return tsr_bits_f32(u & F32_SIGN);
    }
    return r;
}

static inline uint16_t rd_u16le(const uint8_t * p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline int16_t  rd_i16le(const uint8_t * p) { return (int16_t)rd_u16le(p); }
static inline uint32_t rd_u32le(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline float rd_f32le(const uint8_t * p) { return tsr_bits_f32(rd_u32le(p)); }
static inline void wr_u16le(uint8_t * p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void wr_u32le(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static inline int msb_index_u32(uint32_t v) {   // v != 0
#if defined(__GNUC__) || defined(__clang__)
    return 31 - __builtin_clz(v);
#else
    int i = 0;
    while (v >>= 1) ++i;
    return i;
#endif
}

// ---------------------------------------------------------------------------
// f32 arithmetic with the spec's fp rules
// ---------------------------------------------------------------------------
float tsr_f32_daz(float x) { return daz(x); }

float tsr_fadd(float a, float b) { return fin(daz(a) + daz(b)); }
float tsr_fsub(float a, float b) { return fin(daz(a) - daz(b)); }
float tsr_fmul(float a, float b) { return fin(daz(a) * daz(b)); }
float tsr_fdiv(float a, float b) { return fin(daz(a) / daz(b)); }
float tsr_fsqrt(float a)         { return fin(sqrtf(daz(a))); }
float tsr_frecip(float a)        { return fin(1.0f / daz(a)); }

// int32 -> f32, round-to-nearest-even, integer-only.
float tsr_i32_to_f32(int32_t v) {
    if (v == 0) {
        return 0.0f;
    }
    const uint32_t sign = (v < 0) ? F32_SIGN : 0u;
    uint32_t mag = (v < 0) ? (0u - (uint32_t)v) : (uint32_t)v;  // INT32_MIN -> 2^31
    const int msb = msb_index_u32(mag);
    uint32_t exp = 127u + (uint32_t)msb;
    uint32_t sig;                                                // 24-bit significand
    if (msb <= 23) {
        sig = mag << (23 - msb);
    } else {
        const int sh = msb - 23;
        const uint32_t rem  = mag & ((1u << sh) - 1u);
        const uint32_t half = 1u << (sh - 1);
        sig = mag >> sh;
        if (rem > half || (rem == half && (sig & 1u))) {
            sig += 1u;
            if (sig == (1u << 24)) {
                sig >>= 1;
                exp += 1u;
            }
        }
    }
    return tsr_bits_f32(sign | (exp << 23) | (sig & F32_MANT));
}

// fp16 -> f32, exact (fp16 subnormals included). NaN: sign and payload kept, quiet bit set
// (bit-identical to ggml's GGML_FP16_TO_FP32 for all 65536 inputs).
float tsr_fp16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t e = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = sign;
        } else {                     // subnormal fp16: m * 2^-24, normal in f32
            uint32_t sh = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++sh;
            }
            u = sign | ((113u - sh) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (e == 31) {
        u = sign | F32_EXP | (m ? (0x00400000u | (m << 13)) : 0u);
    } else {
        u = sign | ((e + 112u) << 23) | (m << 13);
    }
    return tsr_bits_f32(u);
}

// f32 -> fp16, round-to-nearest-even, overflow -> Inf, NaN -> sign | 0x7E00
// (bit-identical to ggml's GGML_FP32_TO_FP16 for all 2^32 inputs).
uint16_t tsr_f32_to_fp16(float f) {
    const uint32_t u = tsr_f32_bits(f);
    const uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    const uint32_t a = u & ~F32_SIGN;
    if (a > F32_EXP) {
        return (uint16_t)(sign | 0x7E00u);
    }
    if (a == F32_EXP) {
        return (uint16_t)(sign | 0x7C00u);
    }
    const int32_t E = (int32_t)(a >> 23);       // biased f32 exponent, 0 for zero/subnormal
    const int32_t e = E - 127;
    if (E == 0 || e < -25) {                     // |f| < 2^-25 (or f32 subnormal) -> +-0
        return sign;
    }
    if (e > 15) {
        return (uint16_t)(sign | 0x7C00u);
    }
    const uint32_t sig = 0x00800000u | (a & F32_MANT);  // 24-bit significand
    uint32_t out;
    if (e >= -14) {                               // fp16 normal range (may round up into Inf)
        const uint32_t rem = sig & 0x1FFFu;
        out = ((uint32_t)(e + 15) << 10) + ((sig >> 13) & 0x3FFu);
        if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) {
            out += 1u;                            // carries into the exponent / Inf naturally
        }
    } else {                                      // fp16 subnormal: units of 2^-24
        const uint32_t sh = (uint32_t)(-e - 1);   // 14 .. 24
        const uint32_t rem  = sig & ((1u << sh) - 1u);
        const uint32_t half = 1u << (sh - 1u);
        out = sig >> sh;
        if (rem > half || (rem == half && (out & 1u))) {
            out += 1u;                            // may become the smallest normal 0x0400
        }
    }
    return (uint16_t)(sign | out);
}

// f32 -> integer rounding used by QACT. NaN -> 0; the rounded value is saturated to
// [-128, 127]. half_even = 1: round-half-even (ggml nearest_int), 0: round-half-away (roundf).
static int32_t f32_round_sat8(float x, int half_even) {
    const uint32_t u = tsr_f32_bits(x);
    const int neg = (u >> 31) != 0;
    const uint32_t a = u & ~F32_SIGN;
    if (a > F32_EXP) {
        return 0;
    }
    if (a >= 0x43800000u) {                       // |x| >= 256 (incl. Inf): saturate
        return neg ? -128 : 127;
    }
    const uint32_t E = a >> 23;
    if (E < 126) {                                // |x| < 0.5 (incl. zero/subnormal) -> 0
        return 0;
    }
    // |x| in [0.5, 256): value = sig * 2^(E-150), sh = 150 - E in [16, 24]
    const uint32_t sig  = 0x00800000u | (a & F32_MANT);
    const uint32_t sh   = 150u - E;
    const uint32_t rem  = sig & ((1u << sh) - 1u);
    const uint32_t half = 1u << (sh - 1u);
    int32_t ip = (int32_t)(sig >> sh);
    if (half_even ? (rem > half || (rem == half && (ip & 1))) : (rem >= half)) {
        ip += 1;
    }
    if (neg) {
        ip = -ip;
    }
    return ip < -128 ? -128 : (ip > 127 ? 127 : ip);
}

int8_t tsr_q8_round_away(float x) { return (int8_t)f32_round_sat8(x, 0); }
int8_t tsr_q8_round_even(float x) { return (int8_t)f32_round_sat8(x, 1); }

// ---------------------------------------------------------------------------
// Weight formats and device layout
// ---------------------------------------------------------------------------
int tsr_fmt_valid(uint32_t fmt) { return fmt == TSR_FMT_Q8_0 || fmt == TSR_FMT_Q4_K; }

uint32_t tsr_fmt_block_size(uint32_t fmt) {
    return fmt == TSR_FMT_Q8_0 ? TSR_QK8_0 : fmt == TSR_FMT_Q4_K ? TSR_QK_K : 0u;
}
uint32_t tsr_fmt_chunk_blocks(uint32_t fmt) {
    return fmt == TSR_FMT_Q8_0 ? 32u : fmt == TSR_FMT_Q4_K ? 4u : 0u;
}
uint32_t tsr_fmt_quant_bytes(uint32_t fmt) {
    return fmt == TSR_FMT_Q8_0 ? 32u : fmt == TSR_FMT_Q4_K ? 128u : 0u;
}
size_t tsr_fmt_ggml_block_bytes(uint32_t fmt) {
    return fmt == TSR_FMT_Q8_0 ? sizeof(tsr_block_q8_0) : fmt == TSR_FMT_Q4_K ? sizeof(tsr_block_q4_K) : 0u;
}
int tsr_fmt_k_valid(uint32_t fmt, uint32_t K, uint32_t k_max) {
    return tsr_fmt_valid(fmt) && K > 0 && K % tsr_fmt_block_size(fmt) == 0 && K <= k_max;
}

// Beats of one chunk holding nblk blocks: meta beat + ceil(nblk*QB/64) quant beats.
static inline uint32_t chunk_beats(uint32_t fmt, uint32_t nblk) {
    return 1u + tsr_ceil_div_u32(nblk * tsr_fmt_quant_bytes(fmt), TSR_BEAT_BYTES);
}

uint32_t tsr_row_beats(uint32_t fmt, uint32_t K) {
    if (!tsr_fmt_valid(fmt) || K % tsr_fmt_block_size(fmt) != 0) {
        return 0;
    }
    const uint32_t nb   = K / tsr_fmt_block_size(fmt);
    const uint32_t S    = tsr_fmt_chunk_blocks(fmt);
    const uint32_t full = nb / S;
    const uint32_t rem  = nb % S;
    return full * chunk_beats(fmt, S) + (rem ? chunk_beats(fmt, rem) : 0u);
}

static void check_fmt_k(const char * fn, uint32_t fmt, uint32_t K) {
    if (!tsr_fmt_valid(fmt)) {
        tsr_fatal(fn, "invalid fmt", fmt, K);
    }
    if (K % tsr_fmt_block_size(fmt) != 0) {
        tsr_fatal(fn, "K is not a multiple of the block size", fmt, K);
    }
}

// Byte offsets of block (Q8_0) / superblock (Q4_K) i inside a repacked row.
//   meta: offset of its meta entry (2 B for Q8_0, 16 B for Q4_K)
//   quant: offset of its quant bytes (32 B for Q8_0, 128 B for Q4_K)
static inline void block_offsets(uint32_t fmt, uint32_t i, size_t * meta, size_t * quant) {
    const uint32_t S = tsr_fmt_chunk_blocks(fmt);
    const size_t base = (size_t)(i / S) * chunk_beats(fmt, S) * TSR_BEAT_BYTES;
    const uint32_t b = i % S;
    if (fmt == TSR_FMT_Q8_0) {
        *meta  = base + 2u * b;
        *quant = base + TSR_BEAT_BYTES + (size_t)(b / 2u) * TSR_BEAT_BYTES + 32u * (b % 2u);
    } else {
        *meta  = base + 16u * b;
        *quant = base + TSR_BEAT_BYTES + (size_t)(2u * b) * TSR_BEAT_BYTES;
    }
}

void tsr_repack_row(uint32_t fmt, const void * ggml_blocks, uint32_t K, uint8_t * dst) {
    check_fmt_k(__func__, fmt, K);
    const uint8_t * src = (const uint8_t *)ggml_blocks;
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const size_t bsz = tsr_fmt_ggml_block_bytes(fmt);
    memset(dst, 0, (size_t)tsr_row_beats(fmt, K) * TSR_BEAT_BYTES);
    for (uint32_t i = 0; i < nb; ++i) {
        const uint8_t * blk = src + (size_t)i * bsz;
        size_t meta, quant;
        block_offsets(fmt, i, &meta, &quant);
        if (fmt == TSR_FMT_Q8_0) {
            memcpy(dst + meta, blk, 2);              // d (fp16, LE as stored by ggml/GGUF)
            memcpy(dst + quant, blk + 2, 32);        // qs[32]
        } else {
            memcpy(dst + meta, blk, 16);             // d, dmin, scales[12]
            memcpy(dst + quant, blk + 16, 128);      // qs[128] -> beats 2s, 2s+1
        }
    }
}

void tsr_unrepack_row(uint32_t fmt, const uint8_t * src, uint32_t K, void * ggml_blocks) {
    check_fmt_k(__func__, fmt, K);
    uint8_t * dst = (uint8_t *)ggml_blocks;
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const size_t bsz = tsr_fmt_ggml_block_bytes(fmt);
    for (uint32_t i = 0; i < nb; ++i) {
        uint8_t * blk = dst + (size_t)i * bsz;
        size_t meta, quant;
        block_offsets(fmt, i, &meta, &quant);
        if (fmt == TSR_FMT_Q8_0) {
            memcpy(blk, src + meta, 2);
            memcpy(blk + 2, src + quant, 32);
        } else {
            memcpy(blk, src + meta, 16);
            memcpy(blk + 16, src + quant, 128);
        }
    }
}

uint32_t tsr_rows_per_ch(uint32_t N, uint32_t num_ch) {
    if (num_ch == 0) {
        tsr_fatal(__func__, "num_ch must be > 0", N, num_ch);
    }
    return tsr_ceil_div_u32(N, num_ch);
}

void tsr_row_location(uint32_t fmt, uint32_t N, uint32_t K, uint32_t num_ch, uint32_t r,
                      uint32_t * ch, uint64_t * beat_off) {
    check_fmt_k(__func__, fmt, K);
    if (r >= N) {
        tsr_fatal(__func__, "row out of range", r, N);
    }
    const uint32_t R = tsr_rows_per_ch(N, num_ch);
    const uint32_t c = r / R;
    *ch = c;
    *beat_off = (uint64_t)(r - c * R) * tsr_row_beats(fmt, K);
}

uint64_t tsr_tensor_ch_beats(uint32_t fmt, uint32_t N, uint32_t K, uint32_t num_ch) {
    check_fmt_k(__func__, fmt, K);
    return (uint64_t)tsr_rows_per_ch(N, num_ch) * tsr_row_beats(fmt, K);
}

void tsr_get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

void tsr_dequant_row(uint32_t fmt, const uint8_t * row, uint32_t K, float * dst) {
    check_fmt_k(__func__, fmt, K);
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    for (uint32_t i = 0; i < nb; ++i) {
        size_t meta, quant;
        block_offsets(fmt, i, &meta, &quant);
        if (fmt == TSR_FMT_Q8_0) {
            const float d = tsr_fp16_to_f32(rd_u16le(row + meta));
            const int8_t * qs = (const int8_t *)(row + quant);
            float * y = dst + (size_t)i * TSR_QK8_0;
            for (uint32_t l = 0; l < TSR_QK8_0; ++l) {
                y[l] = tsr_fmul(d, tsr_i32_to_f32(qs[l]));
            }
        } else {
            const uint8_t * mt = row + meta;
            const float d    = tsr_fp16_to_f32(rd_u16le(mt));
            const float dmin = tsr_fp16_to_f32(rd_u16le(mt + 2));
            const uint8_t * qs = row + quant;
            float * y = dst + (size_t)i * TSR_QK_K;
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, m;
                tsr_get_scale_min_k4(j, mt + 4, &sc, &m);
                const float d1 = tsr_fmul(d, tsr_i32_to_f32(sc));
                const float m1 = tsr_fmul(dmin, tsr_i32_to_f32(m));
                const uint8_t * q = qs + 32 * (j / 2);
                const int shift = 4 * (j % 2);
                for (int l = 0; l < 32; ++l) {
                    const int32_t qv = (q[l] >> shift) & 0xF;
                    y[32 * j + l] = tsr_fsub(tsr_fmul(d1, tsr_i32_to_f32(qv)), m1);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized activations
// ---------------------------------------------------------------------------
static int qa_kind_valid(uint32_t kind) { return kind == TSR_QA_A32 || kind == TSR_QA_A256; }

uint32_t tsr_qa_block_size(uint32_t kind) {
    return kind == TSR_QA_A32 ? 32u : kind == TSR_QA_A256 ? 256u : 0u;
}
uint32_t tsr_qa_kind_for_fmt(uint32_t fmt) {
    if (!tsr_fmt_valid(fmt)) {
        tsr_fatal(__func__, "invalid fmt", fmt, 0);
    }
    return fmt == TSR_FMT_Q8_0 ? TSR_QA_A32 : TSR_QA_A256;
}
uint32_t tsr_qa_qs_lines(uint32_t kind, uint32_t K) {
    return qa_kind_valid(kind) ? tsr_ceil_div_u32(K, TSR_LINE_BYTES) : 0u;
}
uint32_t tsr_qa_d_lines(uint32_t kind, uint32_t K) {
    return qa_kind_valid(kind) ? tsr_ceil_div_u32(4u * (K / tsr_qa_block_size(kind)), TSR_LINE_BYTES) : 0u;
}
uint32_t tsr_qa_bs_lines(uint32_t kind, uint32_t K) {
    return kind == TSR_QA_A256 ? tsr_ceil_div_u32(2u * (K / 16u), TSR_LINE_BYTES) : 0u;
}
uint32_t tsr_qa_lines(uint32_t kind, uint32_t K) {
    return tsr_qa_qs_lines(kind, K) + tsr_qa_d_lines(kind, K) + tsr_qa_bs_lines(kind, K);
}

static inline size_t qa_d_off(uint32_t kind, uint32_t K) {
    return (size_t)tsr_qa_qs_lines(kind, K) * TSR_LINE_BYTES;
}
static inline size_t qa_bs_off(uint32_t kind, uint32_t K) {
    return (size_t)(tsr_qa_qs_lines(kind, K) + tsr_qa_d_lines(kind, K)) * TSR_LINE_BYTES;
}

const int8_t * tsr_qa_qs(uint32_t kind, uint32_t K, const uint8_t * qa) {
    (void)kind; (void)K;
    return (const int8_t *)qa;
}
const float * tsr_qa_d(uint32_t kind, uint32_t K, const uint8_t * qa) {
    return (const float *)(const void *)(qa + qa_d_off(kind, K));
}
const int16_t * tsr_qa_bs(uint32_t kind, uint32_t K, const uint8_t * qa) {
    return kind == TSR_QA_A256 ? (const int16_t *)(const void *)(qa + qa_bs_off(kind, K)) : NULL;
}

// Magnitude bits of DAZ(x) and whether x takes part in the amax search (NaN does not).
static inline int amax_candidate(float x, uint32_t * mag) {
    const uint32_t a = daz_bits(tsr_f32_bits(x)) & ~F32_SIGN;
    *mag = a;
    return a <= F32_EXP;     // non-NaN; for these, unsigned compare == float compare of |x|
}

void tsr_qact(uint32_t kind, const float * x, uint32_t K, uint8_t * qa) {
    if (!qa_kind_valid(kind)) {
        tsr_fatal(__func__, "invalid QA kind", kind, K);
    }
    const uint32_t bs = tsr_qa_block_size(kind);
    if (K % bs != 0) {
        tsr_fatal(__func__, "K is not a multiple of the QA block size", kind, K);
    }
    memset(qa, 0, (size_t)tsr_qa_lines(kind, K) * TSR_LINE_BYTES);
    int8_t * qs = (int8_t *)qa;
    uint8_t * dreg = qa + qa_d_off(kind, K);
    const uint32_t nb = K / bs;

    for (uint32_t i = 0; i < nb; ++i) {
        const float * xb = x + (size_t)i * bs;
        int8_t * qb = qs + (size_t)i * bs;
        if (kind == TSR_QA_A32) {
            // quantize_row_q8_0_ref: amax = max|x|, d = amax/127, id = d ? 1/d : 0,
            // q = roundf(x*id) (half away from zero), D = f32(fp16(d)).
            uint32_t amax = 0;
            for (uint32_t j = 0; j < bs; ++j) {
                uint32_t a;
                if (amax_candidate(xb[j], &a) && a > amax) {
                    amax = a;
                }
            }
            const float d  = tsr_fdiv(tsr_bits_f32(amax), 127.0f);
            const float id = (tsr_f32_bits(d) & ~F32_SIGN) != 0 ? tsr_frecip(d) : 0.0f;
            for (uint32_t j = 0; j < bs; ++j) {
                qb[j] = (int8_t)f32_round_sat8(tsr_fmul(xb[j], id), 0);
            }
            wr_u32le(dreg + 4u * i, tsr_f32_bits(tsr_fp16_to_f32(tsr_f32_to_fp16(d))));
        } else {
            // quantize_row_q8_K_ref (pin e3797b18): max = signed DAZ(x) of the first largest
            // |x|; amax == 0 -> d = 0, q = 0; else iscale = -127/max,
            // q = min(127, nearest_int(iscale*x)) (half even), d = 1/iscale.
            uint32_t amax = 0;
            float max = 0.0f;
            for (uint32_t j = 0; j < bs; ++j) {
                uint32_t a;
                if (amax_candidate(xb[j], &a) && a > amax) {
                    amax = a;
                    max = daz(xb[j]);
                }
            }
            float d = 0.0f;
            if (amax != 0) {
                const float iscale = tsr_fdiv(-127.0f, max);
                for (uint32_t j = 0; j < bs; ++j) {
                    qb[j] = (int8_t)f32_round_sat8(tsr_fmul(iscale, xb[j]), 1);
                }
                d = tsr_frecip(iscale);
            }   // else: q already zero
            wr_u32le(dreg + 4u * i, tsr_f32_bits(d));
            uint8_t * bsreg = qa + qa_bs_off(kind, K);
            for (uint32_t g = 0; g < bs / 16u; ++g) {
                int32_t sum = 0;
                for (uint32_t l = 0; l < 16u; ++l) {
                    sum += qb[16u * g + l];
                }
                wr_u16le(bsreg + 2u * (i * (bs / 16u) + g), (uint16_t)(int16_t)sum);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dot products
// ---------------------------------------------------------------------------
static inline int32_t dot32_i8(const int8_t * w, const int8_t * x) {
    int32_t s = 0;
    for (int l = 0; l < 32; ++l) {
        s += (int32_t)w[l] * (int32_t)x[l];
    }
    return s;
}

// t = (d * Dx) * f32(isum) for one Q8_0 block: meta2 = its 2 meta bytes, qw = its 32 quants.
static inline float term_q8_0_ptr(const uint8_t * meta2, const int8_t * qw, const int8_t * qx, float Dx) {
    const float d = tsr_fp16_to_f32(rd_u16le(meta2));
    const int32_t isum = dot32_i8(qw, qx);
    return tsr_fmul(tsr_fmul(d, Dx), tsr_i32_to_f32(isum));
}

// t = (d * Dx) * f32(isum) - (dmin * Dx) * f32(msum) for one Q4_K superblock:
// mt = its 16 meta bytes (d, dmin, scales[12]), qs = its 128 quant bytes, qx = 256 activation
// int8, bs16 = its 16 little-endian int16 bsums. Integer sums are exact, so their order is free.
static inline float term_q4_K_ptr(const uint8_t * mt, const uint8_t * qs, const int8_t * qx,
                                  const uint8_t * bs16, float Dx) {
    const float d    = tsr_fp16_to_f32(rd_u16le(mt));
    const float dmin = tsr_fp16_to_f32(rd_u16le(mt + 2));
    int32_t isum = 0, msum = 0;
    for (int g = 0; g < 4; ++g) {            // 64 weights: sub-block 2g = low nibbles, 2g+1 = high
        const uint8_t * q = qs + 32 * g;
        const int8_t * x0 = qx + 64 * g;
        const int8_t * x1 = x0 + 32;
        int32_t s0 = 0, s1 = 0;
        for (int l = 0; l < 32; ++l) {
            s0 += (int32_t)(q[l] & 0xF) * (int32_t)x0[l];
            s1 += (int32_t)(q[l] >> 4) * (int32_t)x1[l];
        }
        uint8_t sc0, m0, sc1, m1;
        tsr_get_scale_min_k4(2 * g, mt + 4, &sc0, &m0);
        tsr_get_scale_min_k4(2 * g + 1, mt + 4, &sc1, &m1);
        isum += (int32_t)sc0 * s0 + (int32_t)sc1 * s1;
        msum += (int32_t)m0 * ((int32_t)rd_i16le(bs16 + 8 * g) + (int32_t)rd_i16le(bs16 + 8 * g + 2)) +
                (int32_t)m1 * ((int32_t)rd_i16le(bs16 + 8 * g + 4) + (int32_t)rd_i16le(bs16 + 8 * g + 6));
    }
    const float t1 = tsr_fmul(tsr_fmul(d, Dx), tsr_i32_to_f32(isum));
    const float t2 = tsr_fmul(tsr_fmul(dmin, Dx), tsr_i32_to_f32(msum));
    return tsr_fsub(t1, t2);
}

float tsr_term_q8_0(const uint8_t * row, uint32_t K, uint32_t b, const uint8_t * qa) {
    check_fmt_k(__func__, TSR_FMT_Q8_0, K);
    if (b >= K / TSR_QK8_0) {
        tsr_fatal(__func__, "block index out of range", b, K);
    }
    size_t meta, quant;
    block_offsets(TSR_FMT_Q8_0, b, &meta, &quant);
    const float Dx = rd_f32le(qa + qa_d_off(TSR_QA_A32, K) + 4u * b);
    return term_q8_0_ptr(row + meta, (const int8_t *)(row + quant), (const int8_t *)qa + (size_t)b * TSR_QK8_0, Dx);
}

float tsr_term_q4_K(const uint8_t * row, uint32_t K, uint32_t s, const uint8_t * qa) {
    check_fmt_k(__func__, TSR_FMT_Q4_K, K);
    if (s >= K / TSR_QK_K) {
        tsr_fatal(__func__, "superblock index out of range", s, K);
    }
    size_t meta, quant;
    block_offsets(TSR_FMT_Q4_K, s, &meta, &quant);
    const float Dx = rd_f32le(qa + qa_d_off(TSR_QA_A256, K) + 4u * s);
    const uint8_t * bs16 = qa + qa_bs_off(TSR_QA_A256, K) + 2u * 16u * s;
    return term_q4_K_ptr(row + meta, row + quant, (const int8_t *)qa + (size_t)s * TSR_QK_K, bs16, Dx);
}

float tsr_tree16(const float p[TSR_ROW_PARTIALS]) {
    float l1[8], l2[4], l3[2];
    for (int k = 0; k < 8; ++k) l1[k] = tsr_fadd(p[2 * k], p[2 * k + 1]);
    for (int k = 0; k < 4; ++k) l2[k] = tsr_fadd(l1[2 * k], l1[2 * k + 1]);
    for (int k = 0; k < 2; ++k) l3[k] = tsr_fadd(l2[2 * k], l2[2 * k + 1]);
    return tsr_fadd(l3[0], l3[1]);
}

float tsr_dot_row(uint32_t fmt, const uint8_t * row, uint32_t K, const uint8_t * qa) {
    check_fmt_k(__func__, fmt, K);
    float p[TSR_ROW_PARTIALS];
    for (uint32_t k = 0; k < TSR_ROW_PARTIALS; ++k) {
        p[k] = 0.0f;   // +0.0
    }
    // Walk the row chunk by chunk (meta beat, then quant beats); term i -> p[i % 16].
    const uint32_t nb = K / tsr_fmt_block_size(fmt);
    const uint32_t S = tsr_fmt_chunk_blocks(fmt);
    const size_t chunk_bytes = (size_t)chunk_beats(fmt, S) * TSR_BEAT_BYTES;
    const int8_t * qx = (const int8_t *)qa;
    uint32_t i = 0;
    if (fmt == TSR_FMT_Q8_0) {
        const uint8_t * dreg = qa + qa_d_off(TSR_QA_A32, K);
        for (const uint8_t * chunk = row; i < nb; chunk += chunk_bytes) {
            const int8_t * quant = (const int8_t *)(chunk + TSR_BEAT_BYTES);   // block b at 32*b
            for (uint32_t b = 0; b < S && i < nb; ++b, ++i) {
                const float t = term_q8_0_ptr(chunk + 2u * b, quant + 32u * b, qx + (size_t)i * TSR_QK8_0,
                                              rd_f32le(dreg + 4u * i));
                p[i % TSR_ROW_PARTIALS] = tsr_fadd(p[i % TSR_ROW_PARTIALS], t);
            }
        }
    } else {
        const uint8_t * dreg  = qa + qa_d_off(TSR_QA_A256, K);
        const uint8_t * bsreg = qa + qa_bs_off(TSR_QA_A256, K);
        for (const uint8_t * chunk = row; i < nb; chunk += chunk_bytes) {
            const uint8_t * quant = chunk + TSR_BEAT_BYTES;                     // sb s at 128*s
            for (uint32_t s = 0; s < S && i < nb; ++s, ++i) {
                const float t = term_q4_K_ptr(chunk + 16u * s, quant + 128u * s, qx + (size_t)i * TSR_QK_K,
                                              bsreg + 32u * i, rd_f32le(dreg + 4u * i));
                p[i % TSR_ROW_PARTIALS] = tsr_fadd(p[i % TSR_ROW_PARTIALS], t);
            }
        }
    }
    return tsr_tree16(p);
}
