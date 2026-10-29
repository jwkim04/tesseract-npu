// tsr_isa.h — Tesseract ISA encoding, CSR offsets, error codes, config parameters.
//
// This header is the C mirror of docs/spec.md ("ISA", "구성 파라미터", "메모리 모델") as amended by
// the v2 execution model (docs/v2/01-execution-model.md, "EX": ISA v2 §7, CSR map §5, error codes
// §8, DRAM map §11). If they disagree, fix the documents first, then this file. Shared by the ISA
// simulator, the reference library, the ggml backend, the runtime and the tools.
//
// v1's host command-buffer path is deleted (X0, EX §12, D14): no run registers, no R0-R15, no
// LI / ADDI / END, no register-relative addressing. What remains is what the tile ops LD, ST,
// QACT, MVM, NOP and FENCE need; their encodings are unchanged from v1 with x16 = 0 (EX §7.1).
// The v2 tables (opcodes, units, error codes, the whole CSR map, PERF v2, DRAM map, the VU / AT
// encodings) are generated from the single table source sw/tables/tsr_tables.py (V2-EX-30, P1.1)
// into tsr_tables.h, included here; this header keeps only what the table source does not hold
// (configuration struct, partial-sum counts, the tile-op encoders).
#pragma once

#include <stdint.h>

#include "tsr_tables.h"   // GENERATED (sw/tables/gen_tables.py): TSR_OP_*, TSR_ERR_*, TSR_CSR_*, ...

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Fixed architecture constants
// ---------------------------------------------------------------------------
// From tsr_tables.h: TSR_INSN_BYTES / TSR_INSN_WORDS, TSR_ID_MAGIC, TSR_ISA_MAJOR / MINOR
// (VERSION = 0x0200, EX §7.5: a v1 core or tool fails the major check), TSR_NUM_CH_MAX.
#define TSR_BEAT_BYTES      64u      // DRAM beat (U200 BEAT_BITS = 512)
#define TSR_LINE_BYTES      64u      // SPM line
#define TSR_ROW_PARTIALS    16u      // MVM row accumulation partial sums (spec 수치 규칙)
#define TSR_RED_PARTIALS    128u     // VU reduction partial sums
#define TSR_MAX_BURST_BEATS 64u      // no burst crosses a 4 KiB boundary
#define TSR_CANON_NAN_BITS  0x7FC00000u

// Channel DRAM map (byte offsets inside a channel; EX §11): TSR_REGION_{OPL,IO,AUX}_OFF / _SIZE
// and TSR_REGION_WEIGHT_OFF come from tsr_tables.h. OPL (was CMD), IO and AUX are on ch0. The
// core fetches nothing from DRAM: OPL holds op lists, replay push streams and argument blocks
// that RISC-V kernels read. The backend splits IO into compute-buffer windows (bottom) and a
// staging area (top, max(16 MiB, 2 x SPM)) for activations kept in host memory (spec "v1 lowering
// 제약"); that split is a software convention, the hardware sees one region.
#define TSR_MiB             (1024ull * 1024ull)

// ---------------------------------------------------------------------------
// Elaboration parameters (spec "구성 파라미터"); software reads them from CFG* CSRs.
// ---------------------------------------------------------------------------
typedef struct tsr_config {
    uint32_t num_ch;          // NUM_CH (C)
    uint32_t beat_bits;       // BEAT_BITS (v1 software assumes 512)
    uint32_t lanes;           // LANES
    uint32_t t_max;           // T_MAX (synthesized value; architecture upper bound 8)
    uint32_t k_max;           // K_MAX
    uint32_t spm_bytes;       // SPM_BYTES
    uint32_t vl;              // VL
    uint32_t nsreg;           // NSREG
    uint32_t rd_outstanding;  // RD_OUTSTANDING
    uint64_t ch_bytes;        // CH_BYTES
    uint32_t addr_beat_w;     // ADDR_BEAT_W
    uint32_t caps;            // CAPS register value (TSR_CAPS_*)
} tsr_config;

// U200 values (the tile core of the M4 bitstreams; NIREG and LOOP_DEPTH have no v2 meaning).
#define TSR_CONFIG_U200_V1 { \
    /*num_ch*/ 4, /*beat_bits*/ 512, /*lanes*/ 128, /*t_max*/ 1, /*k_max*/ 32768, \
    /*spm_bytes*/ 4u * 1024u * 1024u, /*vl*/ 16, /*nsreg*/ 16,                     \
    /*rd_outstanding*/ 16, /*ch_bytes*/ 16ull << 30, /*addr_beat_w*/ 28, /*caps*/ 0 }

#define TSR_T_MAX_ARCH 8u
// TSR_NUM_CH_MAX (tsr_tables.h, 16; spec 구성 파라미터, v0.2.1): LD/ST name the channel in the
// 4-bit w0 sub field, so channels >= 16 would be unreachable. CFG0 NUM_CH[7:0] is wider than needed.

// ---------------------------------------------------------------------------
// CSRs, PERF slots, error codes, opcodes, formats: tsr_tables.h (EX §5-§8, VU §5, AT §5).
// This software uses the identity, configuration and PERF CSRs, which keep their v1 offsets in
// the v2 map (EX §5.1). PERF counter i is at TSR_CSR_PERF_LO(i) (LO) and + 4 (HI); fixed slots
// TSR_PERF_CYCLES .. TSR_PERF_VU_CYCLES (4 reserved), per-channel TSR_PERF_CH_RD_BEATS(c) and
// TSR_PERF_CH_WAIT(C, c), then the extension slots X + k (TSR_PERF_XOFF_*). The v1 run registers,
// END / LI / ADDI and v1's VOP / RED / ROPE / SOP sub-codes are gone (the v2 VU encodings are
// TSR_VOP_FUNC_*, TSR_RED_*, TSR_ROPE_*, TSR_SOP_SUB_*).
// ---------------------------------------------------------------------------
#define TSR_CSR_PERF_BASE   TSR_CSR_PERF_LO_BASE   // counter i: LO at + 8i, HI at + 8i + 4

#define TSR_MVM_FLAG_ACC 0x1u    // MVM flags bit 0 (w0[12], TSR_MVM_ACC in place): y += W.x

// ---------------------------------------------------------------------------
// Instruction word w0 and encode/decode helpers
// ---------------------------------------------------------------------------
typedef struct tsr_insn {
    uint32_t w[TSR_INSN_WORDS];   // little-endian in memory
} tsr_insn;

// w0 = {x16[31:16], flags[15:12], sub[11:8], opcode[7:0]} (EX §7.1). x16 is an op-specific field
// that must be 0 for every op that does not define it, which includes every op of this header.
// Address fields A1, A2, A3 are the immediate words themselves (there is no base register).
static inline uint32_t tsr_w0(uint32_t opcode, uint32_t sub, uint32_t flags, uint32_t x16) {
    return (opcode & 0xFFu) | ((sub & 0xFu) << 8) | ((flags & 0xFu) << 12) | ((x16 & 0xFFFFu) << 16);
}
static inline uint32_t tsr_w0_opcode(uint32_t w0) { return w0 & 0xFFu; }
static inline uint32_t tsr_w0_sub(uint32_t w0)    { return (w0 >> 8) & 0xFu; }
static inline uint32_t tsr_w0_flags(uint32_t w0)  { return (w0 >> 12) & 0xFu; }
static inline uint32_t tsr_w0_x16(uint32_t w0)    { return w0 >> 16; }

// MVM w5 = K[15:0] | T[19:16]; bits [31:20] reserved.
static inline uint32_t tsr_mvm_w5(uint32_t K, uint32_t T) { return (K & 0xFFFFu) | ((T & 0xFu) << 16); }

// Convenience encoders for the tile ops (absolute addresses). Each sets every unused field,
// x16 included, to zero, as the ISA requires.
static inline tsr_insn tsr_enc_nop(void)   { tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_NOP, 0, 0, 0); return i; }
static inline tsr_insn tsr_enc_fence(void) { tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_FENCE, 0, 0, 0); return i; }
// LD: DRAM(ch, A2 beat) -> SPM(A1 line), n_lines lines.
static inline tsr_insn tsr_enc_ld(uint32_t ch, uint32_t spm_dst_line, uint32_t dram_src_beat, uint32_t n_lines) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_LD, ch, 0, 0);
    i.w[1] = spm_dst_line; i.w[2] = dram_src_beat; i.w[3] = n_lines; return i;
}
// ST: SPM(A1 line) -> DRAM(ch, A2 beat), n_lines lines.
static inline tsr_insn tsr_enc_st(uint32_t ch, uint32_t spm_src_line, uint32_t dram_dst_beat, uint32_t n_lines) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_ST, ch, 0, 0);
    i.w[1] = spm_src_line; i.w[2] = dram_dst_beat; i.w[3] = n_lines; return i;
}
// QACT: T f32 vectors (length K, src_stride lines apart; 0 = ceil(4K/64)) -> T QA vectors.
static inline tsr_insn tsr_enc_qact(uint32_t kind, uint32_t qa_dst_line, uint32_t f32_src_line,
                                    uint32_t K, uint32_t T, uint32_t src_stride_lines) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_QACT, kind, 0, 0);
    i.w[1] = qa_dst_line; i.w[2] = f32_src_line; i.w[3] = K; i.w[4] = T; i.w[5] = src_stride_lines; return i;
}
// MVM: dst(A1 line) = W(fmt, A3 = W_OFF beat, N x K) * QA x (A2 line), T tokens.
static inline tsr_insn tsr_enc_mvm(uint32_t fmt, uint32_t flags, uint32_t dst_line, uint32_t x_line,
                                   uint32_t w_off_beat, uint32_t N, uint32_t K, uint32_t T,
                                   uint32_t dst_stride_lines) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_MVM, fmt, flags, 0);
    i.w[1] = dst_line; i.w[2] = x_line; i.w[3] = w_off_beat; i.w[4] = N;
    i.w[5] = tsr_mvm_w5(K, T); i.w[6] = dst_stride_lines; return i;
}

// PERF slot helpers (EX §6): per-channel read beats, per-channel DRAM wait cycles, and the
// number of 64-bit counters the op-level core keeps for num_ch channels: the fixed and
// per-channel slots, i.e. up to the extension index X = 5 + 2C (the X + k slots come with P1.2).
static inline uint32_t tsr_perf_ch_rd_beats(uint32_t ch) { return TSR_PERF_CH_RD_BEATS(ch); }
static inline uint32_t tsr_perf_ch_wait(uint32_t num_ch, uint32_t ch) { return TSR_PERF_CH_WAIT(num_ch, ch); }
static inline uint32_t tsr_perf_count(uint32_t num_ch) { return TSR_PERF_CH_BASE + 2u * num_ch; }

static inline uint32_t tsr_ceil_div_u32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
static inline uint64_t tsr_ceil_div_u64(uint64_t a, uint64_t b) { return (a + b - 1u) / b; }

#ifdef __cplusplus
}
#endif
