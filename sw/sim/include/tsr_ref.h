// tsr_ref.h — the single reference implementation of Tesseract numerics and layouts.
//
// Spec rule 3: repacking, QA quantization, dot products and approximate functions exist
// exactly once, here. The ISA simulator, the ggml backend and the cocotb tests (via ctypes
// on libtsr_ref.so) all call these functions. Nothing in this library depends on ggml;
// ggml-layout blocks are described by the plain structs below (byte-identical to ggml's).
//
// Build with -ffp-contract=off and without -ffast-math: every f32 operation below is a
// separately rounded IEEE binary32 operation (spec "Numeric rules", no FMA fusion).
//
// Error policy (callers — the simulator, the backend, tests — validate fmt/kind/K first and
// raise the ISA error codes; tsr_fmt_valid / tsr_fmt_k_valid are the checks to use):
//   - return 0 for an invalid fmt or QA kind: tsr_fmt_block_size, tsr_fmt_chunk_blocks,
//     tsr_fmt_quant_bytes, tsr_fmt_ggml_block_bytes, tsr_qa_block_size, tsr_qa_qs_lines,
//     tsr_qa_d_lines, tsr_qa_bs_lines, tsr_qa_lines (and tsr_fmt_valid / tsr_fmt_k_valid
//     return false);
//   - tsr_row_beats also returns 0 for a K that is not a multiple of the block size;
//   - the tsr_qa_*_lines counts and the QA views (tsr_qa_qs / _d / _bs) do NOT check K: for a
//     K that is not a multiple of the QA block size they count floor(K / block) scale blocks
//     and are meaningless;
//   - abort() with a message: tsr_qa_kind_for_fmt (invalid fmt), tsr_tensor_ch_beats (invalid
//     fmt or K), tsr_rows_per_ch (num_ch == 0), and every function that processes data
//     (repack, unrepack, dequant, qact, terms, dot, row_location).
// All functions are pure and thread-safe.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "tsr_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// ggml-layout blocks (byte-identical to ggml-common.h)
// ---------------------------------------------------------------------------
#define TSR_QK8_0 32
#define TSR_QK_K  256
#define TSR_K_SCALE_SIZE 12

typedef struct { uint16_t d; int8_t qs[TSR_QK8_0]; } tsr_block_q8_0;                        // 34 B
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[TSR_K_SCALE_SIZE];
                 uint8_t qs[TSR_QK_K / 2]; } tsr_block_q4_K;                                  // 144 B

// ---------------------------------------------------------------------------
// f32 arithmetic with the spec's fp rules
//   - inputs: subnormal -> signed zero (DAZ)
//   - result: IEEE binary32 round-to-nearest-even with the binary32 exponent range (gradual
//     underflow); if that rounded result is subnormal, it becomes signed zero (FTZ).
//     Equivalently: |exact| < 2^-126 - 2^-150 -> signed zero, otherwise the RNE result.
//     This is NOT IEEE 754 "tininess after rounding" (unbounded exponent, x86 MXCSR.FTZ)
//     nor "before rounding": an exact mul/div result in [2^-126 - 2^-150, 2^-126) rounds
//     to +-2^-126 and is kept, e.g. 0x3F7FFFFF * 0x00800000 = 0x00800000 (x86 FTZ: 0).
//   - any NaN result -> canonical NaN 0x7FC00000; Inf propagates per IEEE
// These definitions are normative for the simulator and RTL (bit-exact).
// ---------------------------------------------------------------------------
float    tsr_f32_daz(float x);                 // flush a subnormal input to signed zero
float    tsr_fadd(float a, float b);
float    tsr_fsub(float a, float b);
float    tsr_fmul(float a, float b);
float    tsr_fdiv(float a, float b);
float    tsr_fsqrt(float a);                   // correctly rounded
float    tsr_frecip(float a);                  // correctly rounded 1/a
float    tsr_i32_to_f32(int32_t v);            // RNE
float    tsr_fp16_to_f32(uint16_t h);          // exact, fp16 subnormals included
uint16_t tsr_f32_to_fp16(float f);             // RNE, same as ggml's GGML_FP32_TO_FP16
uint32_t tsr_f32_bits(float f);
float    tsr_bits_f32(uint32_t u);

// QACT f32 -> int8 conversions (spec §21 (QACT special values)): NaN -> 0, the rounded value
// saturates to [-128, 127]. round_away = roundf (A32), round_even = ggml nearest_int (A256).
// Exposed so the RTL QACT unit can be checked against them directly.
int8_t   tsr_q8_round_away(float x);
int8_t   tsr_q8_round_even(float x);

// ---------------------------------------------------------------------------
// Weight formats and device layout (spec "Device weight layout")
// ---------------------------------------------------------------------------
uint32_t tsr_fmt_block_size(uint32_t fmt);      // weights per block: Q8_0 32, Q4_K 256
uint32_t tsr_fmt_chunk_blocks(uint32_t fmt);    // S: Q8_0 32, Q4_K 4
uint32_t tsr_fmt_quant_bytes(uint32_t fmt);     // QB: Q8_0 32, Q4_K 128
size_t   tsr_fmt_ggml_block_bytes(uint32_t fmt);// Q8_0 34, Q4_K 144
int      tsr_fmt_valid(uint32_t fmt);
int      tsr_fmt_k_valid(uint32_t fmt, uint32_t K, uint32_t k_max); // K>0, K%block==0, K<=k_max

uint32_t tsr_row_beats(uint32_t fmt, uint32_t K);           // ROW_BEATS(fmt, K)

// Repack one row of K weights from ggml blocks to the device layout.
// dst receives exactly tsr_row_beats(fmt,K)*64 bytes; unused meta entries and the unused
// tail of the last quant beat are zero.
void tsr_repack_row(uint32_t fmt, const void * ggml_blocks, uint32_t K, uint8_t * dst);
// Inverse of tsr_repack_row (debug get_tensor, tests).
void tsr_unrepack_row(uint32_t fmt, const uint8_t * src, uint32_t K, void * ggml_blocks);

// Symmetric allocation (spec "Symmetric allocation"): R = ceil(N/C), channel c owns rows [cR, min((c+1)R,N)).
uint32_t tsr_rows_per_ch(uint32_t N, uint32_t num_ch);
// Row r -> channel and beat offset relative to W_OFF.
void     tsr_row_location(uint32_t fmt, uint32_t N, uint32_t K, uint32_t num_ch, uint32_t r,
                          uint32_t * ch, uint64_t * beat_off);
// Beats a tensor occupies in each channel = R * ROW_BEATS (= EXP_STRIDE for expert tensors).
uint64_t tsr_tensor_ch_beats(uint32_t fmt, uint32_t N, uint32_t K, uint32_t num_ch);

// Q4_K 6-bit scale/min unpack, identical to ggml get_scale_min_k4.
void tsr_get_scale_min_k4(int j, const uint8_t * scales, uint8_t * sc, uint8_t * m);

// Dequantize one repacked row to f32 (DQROW semantics: d*q, Q4_K (d*sc)*q - (dmin*m),
// each product/subtraction separately rounded with the fp rules above).
void tsr_dequant_row(uint32_t fmt, const uint8_t * row, uint32_t K, float * dst);

// ---------------------------------------------------------------------------
// Quantized activations (spec "Quantized activation (QA) layout")
// ---------------------------------------------------------------------------
uint32_t tsr_qa_block_size(uint32_t kind);      // A32 32, A256 256
uint32_t tsr_qa_kind_for_fmt(uint32_t fmt);     // Q8_0 -> A32, Q4_K -> A256
uint32_t tsr_qa_qs_lines(uint32_t kind, uint32_t K);
uint32_t tsr_qa_d_lines(uint32_t kind, uint32_t K);
uint32_t tsr_qa_bs_lines(uint32_t kind, uint32_t K);   // 0 for A32
uint32_t tsr_qa_lines(uint32_t kind, uint32_t K);      // QA_LINES(kind, K)

// Quantize one token vector x[K] into a QA image of tsr_qa_lines(kind,K)*64 bytes
// (QS region, then D region, then BS region, each line-aligned; tail bytes zero).
// Bit-exact definition of QACT; matches ggml quantize_row_q8_0_ref / quantize_row_q8_K_ref
// (A256: iscale = -127/max, as at the llama.cpp pin) for inputs without subnormals whose
// block amax is >= 2^-119 (so that d stays normal) and without NaN. All-zero A256 blocks get
// BS = 0 (ggml leaves bsums unwritten there). K must be a multiple of the QA block size.
void tsr_qact(uint32_t kind, const float * x, uint32_t K, uint8_t * qa);

// Views into a QA image (host byte order = little-endian assumed; qa must be 4-byte aligned).
const int8_t  * tsr_qa_qs(uint32_t kind, uint32_t K, const uint8_t * qa);
const float   * tsr_qa_d(uint32_t kind, uint32_t K, const uint8_t * qa);
const int16_t * tsr_qa_bs(uint32_t kind, uint32_t K, const uint8_t * qa);

// ---------------------------------------------------------------------------
// Dot products (spec "Numeric rules")
// ---------------------------------------------------------------------------
// Term of one Q8_0 block b of a repacked row against an A32 QA image.
float tsr_term_q8_0(const uint8_t * row, uint32_t K, uint32_t b, const uint8_t * qa);
// Term of one Q4_K superblock s of a repacked row against an A256 QA image.
float tsr_term_q4_K(const uint8_t * row, uint32_t K, uint32_t s, const uint8_t * qa);
// Full row: terms accumulated into 16 partials (term i -> p[i % 16]), then the fixed
// binary tree ((p0+p1)+(p2+p3))+... . Returns y (without ACC).
float tsr_dot_row(uint32_t fmt, const uint8_t * row, uint32_t K, const uint8_t * qa);
// Fixed 16-way partial-sum tree used above (exposed for RTL block tests).
float tsr_tree16(const float p[TSR_ROW_PARTIALS]);

#ifdef __cplusplus
}
#endif
