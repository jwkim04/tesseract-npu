// tsr_isa.h — Tesseract ISA encoding, CSR map, error codes, config parameters.
//
// This header is the C mirror of docs/spec.md ("ISA", "실행 모델과 제어 인터페이스",
// "구성 파라미터", "메모리 모델"). If the two disagree, fix the spec first, then this file.
// Shared by the ISA simulator, the reference library, the ggml backend and the trace tools.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Fixed architecture constants
// ---------------------------------------------------------------------------
#define TSR_BEAT_BYTES      64u      // DRAM beat (U200 BEAT_BITS = 512)
#define TSR_LINE_BYTES      64u      // SPM line
#define TSR_INSN_BYTES      32u      // one instruction = w0..w7
#define TSR_INSN_WORDS      8u
#define TSR_ROW_PARTIALS    16u      // MVM row accumulation partial sums (spec 수치 규칙)
#define TSR_RED_PARTIALS    128u     // VU reduction partial sums
#define TSR_MAX_BURST_BEATS 64u      // no burst crosses a 4 KiB boundary
#define TSR_ID_MAGIC        0x54535231u  // "TSR1"
#define TSR_ISA_MAJOR       1u
#define TSR_ISA_MINOR       0u
#define TSR_CANON_NAN_BITS  0x7FC00000u

// Channel DRAM map (byte offsets inside a channel). CMD/IO/AUX are used on ch0 only in v1.
// The v1 backend splits IO into compute-buffer windows (bottom) and a staging area (top,
// max(16 MiB, 2 x SPM)) for activations kept in host memory (spec "v1 lowering 제약"); that
// split is a software convention, the hardware sees one region.
#define TSR_MiB             (1024ull * 1024ull)
#define TSR_REGION_CMD_OFF     (0ull)
#define TSR_REGION_CMD_SIZE    (64ull  * TSR_MiB)
#define TSR_REGION_IO_OFF      (64ull  * TSR_MiB)
#define TSR_REGION_IO_SIZE     (256ull * TSR_MiB)
#define TSR_REGION_AUX_OFF     (320ull * TSR_MiB)
#define TSR_REGION_AUX_SIZE    (192ull * TSR_MiB)
#define TSR_REGION_WEIGHT_OFF  (512ull * TSR_MiB)

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
    uint32_t nireg;           // NIREG
    uint32_t nsreg;           // NSREG
    uint32_t loop_depth;      // LOOP_DEPTH
    uint32_t rd_outstanding;  // RD_OUTSTANDING
    uint64_t ch_bytes;        // CH_BYTES
    uint32_t addr_beat_w;     // ADDR_BEAT_W
    uint32_t caps;            // CAPS register value (TSR_CAPS_*)
} tsr_config;

// U200 v1 values.
#define TSR_CONFIG_U200_V1 { \
    /*num_ch*/ 4, /*beat_bits*/ 512, /*lanes*/ 128, /*t_max*/ 1, /*k_max*/ 32768, \
    /*spm_bytes*/ 4u * 1024u * 1024u, /*vl*/ 16, /*nireg*/ 16, /*nsreg*/ 16,       \
    /*loop_depth*/ 4, /*rd_outstanding*/ 16, /*ch_bytes*/ 16ull << 30,             \
    /*addr_beat_w*/ 28, /*caps*/ 0 }

#define TSR_T_MAX_ARCH 8u
// NUM_CH upper bound (spec 구성 파라미터, v0.2.1): LD/ST name the channel in the 4-bit w0 sub
// field, so channels >= 16 would be unreachable. CFG0 NUM_CH[7:0] is wider than needed.
#define TSR_NUM_CH_MAX 16u

// ---------------------------------------------------------------------------
// CSR map (32-bit registers, byte offsets)
// ---------------------------------------------------------------------------
enum tsr_csr {
    TSR_CSR_ID          = 0x000,
    TSR_CSR_VERSION     = 0x004,  // major[15:8] minor[7:0]
    TSR_CSR_CFG0        = 0x008,  // NUM_CH[7:0] LANES[19:8] T_MAX[23:20]
    TSR_CSR_CFG1        = 0x00C,  // SPM size in lines
    TSR_CSR_CFG2        = 0x010,  // VL[7:0] NSREG[15:8] K_MAX/256[31:16]
    TSR_CSR_CAPS        = 0x014,
    TSR_CSR_CTRL        = 0x020,  // W: bit0 START, bit1 SOFT_RESET, bit2 ABORT
    TSR_CSR_STATUS      = 0x024,  // R/W1C: bit0 BUSY, bit1 DONE, bit2 ERR, bit3 WAITING
    TSR_CSR_CMD_ADDR_LO = 0x028,
    TSR_CSR_CMD_ADDR_HI = 0x02C,
    TSR_CSR_CMD_COUNT   = 0x030,
    TSR_CSR_ERR_PC      = 0x040,
    TSR_CSR_ERR_CODE    = 0x044,
    TSR_CSR_ERR_INFO    = 0x048,
    TSR_CSR_PC          = 0x050,
    TSR_CSR_ARG0        = 0x060,  // ARG0..ARG7 at 0x060..0x07C -> R8..R15 at start
    TSR_CSR_PERF_BASE   = 0x100,  // 64-bit counters, LO at +0, HI at +4 (see tsr_perf_idx)
};

#define TSR_CTRL_START      (1u << 0)
#define TSR_CTRL_SOFT_RESET (1u << 1)
#define TSR_CTRL_ABORT      (1u << 2)

#define TSR_STATUS_BUSY     (1u << 0)
#define TSR_STATUS_DONE     (1u << 1)
#define TSR_STATUS_ERR      (1u << 2)
#define TSR_STATUS_WAITING  (1u << 3)

#define TSR_CAPS_VU         (1u << 0)
#define TSR_CAPS_RLD_LOOP   (1u << 2)   // RLD, MULI, LOOP
#define TSR_CAPS_TOPK       (1u << 3)
#define TSR_CAPS_WAITM_JUMP (1u << 4)

// PERF_* counter order. Counter i is at TSR_CSR_PERF_BASE + 8*i (LO), +8*i+4 (HI).
// Per-channel counters take NUM_CH consecutive slots, channel 0 first.
enum tsr_perf_idx {
    TSR_PERF_CYCLES = 0,          // total cycles (sim: modelled cycles)
    TSR_PERF_RETIRED = 1,         // retired instructions
    TSR_PERF_MVM_CYCLES = 2,
    TSR_PERF_VU_CYCLES = 3,
    TSR_PERF_WAITM_CYCLES = 4,
    TSR_PERF_CH_RD_BEATS = 5,     // + ch, NUM_CH slots
    // TSR_PERF_CH_WAIT = TSR_PERF_CH_RD_BEATS + NUM_CH  (+ ch)
};

// ---------------------------------------------------------------------------
// Error codes (ERR_CODE)
// ---------------------------------------------------------------------------
enum tsr_err {
    TSR_ERR_NONE         = 0,
    TSR_ERR_ILLEGAL_OP   = 1,
    TSR_ERR_BAD_FMT      = 2,
    TSR_ERR_QA_MISMATCH  = 3,
    TSR_ERR_SPM_RANGE    = 4,
    TSR_ERR_DRAM_RANGE   = 5,
    TSR_ERR_K_RANGE      = 6,
    TSR_ERR_T_RANGE      = 7,
    TSR_ERR_ABORTED      = 8,
    TSR_ERR_LOOP_NEST    = 9,
    TSR_ERR_WAIT_TIMEOUT = 10,
};

// ---------------------------------------------------------------------------
// Opcodes and sub-codes
// ---------------------------------------------------------------------------
enum tsr_opcode {
    TSR_OP_NOP   = 0x00,
    TSR_OP_END   = 0x01,
    TSR_OP_LI    = 0x02,
    TSR_OP_ADDI  = 0x03,
    TSR_OP_MULI  = 0x04,
    TSR_OP_RLD   = 0x05,
    TSR_OP_LOOP  = 0x06,
    TSR_OP_WAITM = 0x07,
    TSR_OP_JUMP  = 0x08,
    TSR_OP_LD    = 0x10,
    TSR_OP_ST    = 0x11,
    TSR_OP_DQROW = 0x12,
    TSR_OP_QACT  = 0x18,
    TSR_OP_MVM   = 0x20,
    TSR_OP_VOP   = 0x30,
    TSR_OP_RED   = 0x31,
    TSR_OP_ROPE  = 0x32,
    TSR_OP_TOPK  = 0x33,
    TSR_OP_SOP   = 0x40,
    TSR_OP_FENCE = 0x7F,
};

// Weight formats (MVM/DQROW sub). 2..15 reserved.
enum tsr_fmt {
    TSR_FMT_Q8_0 = 0,
    TSR_FMT_Q4_K = 1,
};

// QA kinds (QACT sub).
enum tsr_qa_kind {
    TSR_QA_A32  = 0,   // block_q8_0 rules, block 32
    TSR_QA_A256 = 1,   // block_q8_K rules, block 256, with bsums
};

#define TSR_MVM_FLAG_ACC 0x1u

enum tsr_vop_func {
    TSR_VOP_ADD = 0, TSR_VOP_SUB = 1, TSR_VOP_MUL = 2, TSR_VOP_MAX = 3,
    TSR_VOP_MULS = 4, TSR_VOP_ADDS = 5, TSR_VOP_SILU = 6, TSR_VOP_SWIGLU = 7,
    TSR_VOP_EXP = 8, TSR_VOP_COPY = 9,
};
enum tsr_red_sub  { TSR_RED_SUM = 0, TSR_RED_SUMSQ = 1, TSR_RED_MAX = 2 };
enum tsr_rope_sub { TSR_ROPE_NORM = 0, TSR_ROPE_NEOX = 1 };
enum tsr_sop_sub  {
    TSR_SOP_MOVI = 0, TSR_SOP_ADD = 1, TSR_SOP_MUL = 2, TSR_SOP_ADDI = 3, TSR_SOP_MULI = 4,
    TSR_SOP_SQRT = 5, TSR_SOP_RECIP = 6, TSR_SOP_LDS = 7, TSR_SOP_STS = 8,
};
enum tsr_rld_sub   { TSR_RLD_I32 = 0, TSR_RLD_F32 = 1 };
enum tsr_waitm_sub { TSR_WAITM_EQ = 0, TSR_WAITM_NE = 1, TSR_WAITM_GE = 2 };

// ---------------------------------------------------------------------------
// Instruction word w0 and encode/decode helpers
// ---------------------------------------------------------------------------
typedef struct tsr_insn {
    uint32_t w[TSR_INSN_WORDS];   // little-endian in memory
} tsr_insn;

static inline uint32_t tsr_w0(uint32_t opcode, uint32_t sub, uint32_t flags,
                              uint32_t ra, uint32_t rb, uint32_t rc, uint32_t rd) {
    return (opcode & 0xFFu) | ((sub & 0xFu) << 8) | ((flags & 0xFu) << 12) |
           ((ra & 0xFu) << 16) | ((rb & 0xFu) << 20) | ((rc & 0xFu) << 24) | ((rd & 0xFu) << 28);
}
static inline uint32_t tsr_w0_opcode(uint32_t w0) { return w0 & 0xFFu; }
static inline uint32_t tsr_w0_sub(uint32_t w0)    { return (w0 >> 8) & 0xFu; }
static inline uint32_t tsr_w0_flags(uint32_t w0)  { return (w0 >> 12) & 0xFu; }
static inline uint32_t tsr_w0_ra(uint32_t w0)     { return (w0 >> 16) & 0xFu; }
static inline uint32_t tsr_w0_rb(uint32_t w0)     { return (w0 >> 20) & 0xFu; }
static inline uint32_t tsr_w0_rc(uint32_t w0)     { return (w0 >> 24) & 0xFu; }
static inline uint32_t tsr_w0_rd(uint32_t w0)     { return (w0 >> 28) & 0xFu; }

// MVM w5 = K[15:0] | T[19:16]; bits [31:20] reserved.
static inline uint32_t tsr_mvm_w5(uint32_t K, uint32_t T) { return (K & 0xFFFFu) | ((T & 0xFu) << 16); }

// Convenience encoders for the M1 instruction subset (base registers R0 = absolute).
// Each sets every unused field to zero, as the spec requires.
static inline tsr_insn tsr_enc_nop(void)   { tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_NOP, 0,0,0,0,0,0); return i; }
static inline tsr_insn tsr_enc_end(void)   { tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_END, 0,0,0,0,0,0); return i; }
static inline tsr_insn tsr_enc_fence(void) { tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_FENCE, 0,0,0,0,0,0); return i; }
static inline tsr_insn tsr_enc_li(uint32_t rd, uint32_t imm) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_LI, 0,0,0,0,0, rd); i.w[1] = imm; return i;
}
static inline tsr_insn tsr_enc_addi(uint32_t rd, uint32_t ra, uint32_t imm) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_ADDI, 0,0, ra,0,0, rd); i.w[1] = imm; return i;
}
// LD: DRAM(ch, A2 beat) -> SPM(A1 line), n_lines lines.
static inline tsr_insn tsr_enc_ld(uint32_t ch, uint32_t spm_dst_line, uint32_t dram_src_beat, uint32_t n_lines) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_LD, ch,0,0,0,0,0);
    i.w[1] = spm_dst_line; i.w[2] = dram_src_beat; i.w[3] = n_lines; return i;
}
// ST: SPM(A1 line) -> DRAM(ch, A2 beat), n_lines lines.
static inline tsr_insn tsr_enc_st(uint32_t ch, uint32_t spm_src_line, uint32_t dram_dst_beat, uint32_t n_lines) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_ST, ch,0,0,0,0,0);
    i.w[1] = spm_src_line; i.w[2] = dram_dst_beat; i.w[3] = n_lines; return i;
}
// QACT: T f32 vectors (length K, src_stride lines apart; 0 = ceil(4K/64)) -> T QA vectors.
static inline tsr_insn tsr_enc_qact(uint32_t kind, uint32_t qa_dst_line, uint32_t f32_src_line,
                                    uint32_t K, uint32_t T, uint32_t src_stride_lines) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_QACT, kind,0,0,0,0,0);
    i.w[1] = qa_dst_line; i.w[2] = f32_src_line; i.w[3] = K; i.w[4] = T; i.w[5] = src_stride_lines; return i;
}
// MVM: dst(A1 line) = W(fmt, A3 = W_OFF beat, N x K) * QA x (A2 line), T tokens.
static inline tsr_insn tsr_enc_mvm(uint32_t fmt, uint32_t flags, uint32_t dst_line, uint32_t x_line,
                                   uint32_t w_off_beat, uint32_t N, uint32_t K, uint32_t T,
                                   uint32_t dst_stride_lines) {
    tsr_insn i = {{0}}; i.w[0] = tsr_w0(TSR_OP_MVM, fmt, flags,0,0,0,0);
    i.w[1] = dst_line; i.w[2] = x_line; i.w[3] = w_off_beat; i.w[4] = N;
    i.w[5] = tsr_mvm_w5(K, T); i.w[6] = dst_stride_lines; return i;
}

// Set the base-register fields ra/rb/rc of an encoded instruction (register-relative
// addressing: A1 = R[ra] + w1, A2 = R[rb] + w2, A3 = R[rc] + w3). The encoders above use R0.
static inline void tsr_insn_set_base(tsr_insn * i, uint32_t ra, uint32_t rb, uint32_t rc) {
    i->w[0] = (i->w[0] & ~0x0FFF0000u) | ((ra & 0xFu) << 16) | ((rb & 0xFu) << 20) | ((rc & 0xFu) << 24);
}

// PERF_* slot helpers (see enum tsr_perf_idx): per-channel read beats, per-channel DRAM wait
// cycles, and the total number of 64-bit counters for a core with num_ch channels.
static inline uint32_t tsr_perf_ch_rd_beats(uint32_t ch) { return (uint32_t)TSR_PERF_CH_RD_BEATS + ch; }
static inline uint32_t tsr_perf_ch_wait(uint32_t num_ch, uint32_t ch) {
    return (uint32_t)TSR_PERF_CH_RD_BEATS + num_ch + ch;
}
static inline uint32_t tsr_perf_count(uint32_t num_ch) { return (uint32_t)TSR_PERF_CH_RD_BEATS + 2u * num_ch; }

static inline uint32_t tsr_ceil_div_u32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
static inline uint64_t tsr_ceil_div_u64(uint64_t a, uint64_t b) { return (a + b - 1u) / b; }

#ifdef __cplusplus
}
#endif
