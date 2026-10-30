// selftest_abi.hpp — the `selftest` kernel's interface (docs/design/06-software-verification.md "SW"
// §7.1, §7.6, SW-28; kernel sw/ctrl/kernels/selftest.cpp): argument words, the parameter line,
// the data-area layout of every test id, and the return codes. Shared by the kernel (freestanding
// C++20), the sim tests (sw/ctrl/tests/test_fw_selftest_sim.c, C) and later tsr-selftest (P3.3), so it
// is written in the common subset of C11 and C++ (macros, plain structs, static inline functions).
//
// Kernel arguments (SW §7.1):
//   a0 test id (1-10)    a1 data handle {ch[31:28], beat[27:0]}: the test's data area
//   a2 area lines        (the kernel refuses a layout that does not fit: RET AREA)
//   a3 seed              (patterns the device generates: ids 1, 8, 9)
//   a4 flags             (id-specific; 0 everywhere today)
// The data area starts with a parameter line P (16 words, written by the host):
//   P[0] = TSR_ST_MAGIC, P[1] = the test id, P[2..15] = the id's parameters (below).
// Inputs follow at line 1; outputs follow the inputs. The kernel LDs the whole input part, works in SPM
// and STs its outputs back; the host compares them with references (tsr_ref*) and, on the card, with
// the simulator running the same image (tsr-selftest, SW §7.6).
// RET: 0 = passed; > 0 = a device-side check failed (code per id, below); < 0: the arguments
// (KERNEL_ERR, SW §6.4).
#ifndef TSR_SELFTEST_ABI_HPP
#define TSR_SELFTEST_ABI_HPP

#include <stdint.h>

#define TSR_ST_MAGIC 0x54535354u            // "TSST" little-endian

#define TSR_ST_ID_LDST    1u                // LD / ST round trips on every channel, cross-channel copies
#define TSR_ST_ID_QACT    2u                // QACT A32 / A256 on host data
#define TSR_ST_ID_MVM     3u                // MVM Q8_0 / Q4_K, T = 1, T = T_MAX, ACC
#define TSR_ST_ID_VU      4u                // every VU function and reduction (VOP, RED, SOP, ROPE, TOPK, DQROW)
#define TSR_ST_ID_AT      5u                // every AT op (KVST, ATTQK, ASMAX, AMASK, ATTPV, AMERGE, ANORM)
#define TSR_ST_ID_CONC    6u                // a matvec_batch pipeline on every unit
#define TSR_ST_ID_ERRORS  7u                // DRAM_RANGE, ILLEGAL_OP, BAD_DEP, DEP_FAILED chain, QABORT + QCLEAR
#define TSR_ST_ID_SPMWIN  8u                // SPM window byte / half / word, store-before-push ordering
#define TSR_ST_ID_LOG     9u                // log records and a kernel event (PING is the runtime's)
#define TSR_ST_ID_CREDITS 10u               // fill every queue (QHOLD), credits
#define TSR_ST_N_IDS      10u

#define TSR_ST_RET_BAD_ID     0x80000201u   // a0 not 1..10
#define TSR_ST_RET_BAD_PARAMS 0x80000202u   // bad magic / id in P, or a parameter out of range
#define TSR_ST_RET_AREA       0x80000203u   // the layout needs more than a2 lines
#define TSR_ST_RET_CAPS       0x80000204u   // the device lacks a CAPS bit the test needs (VU, ATT)

static inline uint32_t tsr_st_lines(uint32_t words) { return (words + 15u) / 16u; }   // f32 words -> lines

// ---------------------------------------------------------------------------------------------
// 1 LDST. P[2] = Lmax lines (1..4096), P[3] = scratch beat S (the same on every channel; the kernel
// writes [S, S + 2 SUM) there, SUM = the sum of the lengths). Input: src at line 1, Lmax lines.
// Lengths: 1, 63, 64, 65 (those below Lmax), then Lmax. For each length k and channel c: LD src ->
// SPM A; ST A -> (c, S + off_k); LD back -> SPM B; the RISC-V compares A and B. Then the cross copies:
// LD (c, S + off_last) -> B; ST B -> ((c + 1) mod C, S + SUM). Outputs: those DRAM ranges (the host
// checks every one against src). Device-side RET: 0x01000000 | k << 8 | c (round trip k, channel c),
// 0x02000000 | c (cross copy into channel c).
#define TSR_ST1_LMAX_MAX 4096u
static inline uint32_t tsr_st1_lens(uint32_t lmax, uint32_t lens[5]) {
    static const uint32_t base[4] = { 1u, 63u, 64u, 65u };
    uint32_t n = 0;
    for (uint32_t i = 0; i < 4u; ++i)
        if (base[i] < lmax) lens[n++] = base[i];
    lens[n++] = lmax;
    return n;
}

// ---------------------------------------------------------------------------------------------
// 2 QACT. P[2] = K (a multiple of 256, 256..8192), P[3] = T (1..T_MAX). Input x: T rows of K f32 at
// line 1 (K/16 lines each). Outputs: QA(A32) of each row (T x QA_LINES(A32, K)), then QA(A256).
typedef struct tsr_st2_layout { uint32_t x, qa32, qa256, end; } tsr_st2_layout;
static inline uint32_t tsr_st_qa_lines(uint32_t kind, uint32_t K) {
    const uint32_t b = kind ? 256u : 32u;
    return (K + 63u) / 64u + (4u * (K / b) + 63u) / 64u + (kind ? (2u * (K / 16u) + 63u) / 64u : 0u);
}
static inline tsr_st2_layout tsr_st2(uint32_t K, uint32_t T) {
    tsr_st2_layout l;
    l.x = 1u;
    l.qa32 = l.x + T * (K / 16u);
    l.qa256 = l.qa32 + T * tsr_st_qa_lines(0, K);
    l.end = l.qa256 + T * tsr_st_qa_lines(1, K);
    return l;
}

// ---------------------------------------------------------------------------------------------
// 3 MVM. P[2] = N (1..8192), P[3] = K (a multiple of 256, 256..8192), P[4] = W_OFF of an N x K Q8_0
// tensor, P[5] = W_OFF of an N x K Q4_K tensor (symmetric layout, every channel), TM = CFG0.T_MAX.
// Input x: TM rows of K f32 at line 1. Outputs (ceil(N/16) lines per token): y8_1 (Q8_0, T = 1),
// y8_T (Q8_0, T = TM), y4_1, y4_T (Q4_K), yacc (TM tokens: y8_T += W4 x by MVM ACC).
typedef struct tsr_st3_layout { uint32_t x, y8_1, y8_t, y4_1, y4_t, yacc, end; } tsr_st3_layout;
static inline tsr_st3_layout tsr_st3(uint32_t N, uint32_t K, uint32_t TM) {
    const uint32_t yl = (N + 15u) / 16u;
    tsr_st3_layout l;
    l.x = 1u;
    l.y8_1 = l.x + TM * (K / 16u);
    l.y8_t = l.y8_1 + yl;
    l.y4_1 = l.y8_t + TM * yl;
    l.y4_t = l.y4_1 + yl;
    l.yacc = l.y4_t + TM * yl;
    l.end = l.yacc + TM * yl;
    return l;
}

// ---------------------------------------------------------------------------------------------
// 4 VU. P[2] = M (1..64 elements per row), P[3] = R (1..16 rows), P[4] = ROPE TS bits, P[5] = ROPE M
// bits, P[6] = ROPE freq_scale bits, P[7] = RMS scale bits (sk), P[8] = RMS epsilon bits (se).
// Inputs (lines from 1): a and b (R rows of M f32, L(M) lines apart), c (R words: the per-row VEC
// scalars), sv (16 words: s0..s15 by SOP LDS; words 10, 11 are SOP's sa / sb, word 12 its imm),
// rope_x (2 tokens x 2 heads x 64 f32), rope_pos (2 i32), rope_ff (16 f32 freq factors), dq8 / dq4 (2
// repacked K = 256 rows each, Q8_0 / Q4_K, ROW_BEATS lines apart).
// Outputs: vop[f] for f = 0..24 (VOP func f on a, b, c with cmode VEC when it reads c, else NONE),
// vop_sreg (MULC with cmode SREG, s3), red[7] (one line each: SUM, SUM+RCP, SUM+RMS, SUMSQ, SUMSQ+RCP,
// SUMSQ+RMS, MAX; RMS with s1 = P[7], s2 = P[8]), sop (one line: words 0..11 = SOP ADD, MUL, ADDI,
// MULI, SQRT, RECIP, SUB, DIV, RSQRT, MAX, MIN, EXP of sa = s10, sb = s11, imm = sv[12] into s8, each
// stored by STS; word 12 = RED SUM row 0 TO_S s7; word 13 = s0 after LDS; word 14 = s9 after MOVI of
// P[7]; word 15 = 0), rope_norm (NORM, n_dims 64, s4 = M, s5 = TS), rope_neox (NEOX, n_dims 32, FF table
// rope_ff, FS with s6 = freq_scale), topk (k = 4 of each a row, one line per row; with CAPS.TOPK, else
// left as the host wrote it), dq8_out / dq4_out (2 rows of 256 f32 each; with CAPS.DQROW).
#define TSR_ST4_N_FUNC 25u
#define TSR_ST4_N_RED  7u
#define TSR_ST4_ROPE_LINES 16u
typedef struct tsr_st4_layout {
    uint32_t L;                                      // lines per row: L(M)
    uint32_t a, b, c, sv, rope_x, rope_pos, rope_ff, dq8, dq4, in_end;
    uint32_t vop, vop_sreg, red, sop, rope_norm, rope_neox, topk, dq8_out, dq4_out, end;
    uint32_t dq8_beats, dq4_beats;                  // ROW_BEATS(fmt, 256)
} tsr_st4_layout;
static inline tsr_st4_layout tsr_st4(uint32_t M, uint32_t R) {
    tsr_st4_layout l;
    l.L = tsr_st_lines(M);
    l.dq8_beats = 1u + (8u * 32u + 63u) / 64u;      // Q8_0, K = 256: one chunk of 8 blocks
    l.dq4_beats = 1u + (128u + 63u) / 64u;          // Q4_K, K = 256: one superblock
    l.a = 1u;
    l.b = l.a + R * l.L;
    l.c = l.b + R * l.L;
    l.sv = l.c + 1u;
    l.rope_x = l.sv + 1u;
    l.rope_pos = l.rope_x + TSR_ST4_ROPE_LINES;
    l.rope_ff = l.rope_pos + 1u;
    l.dq8 = l.rope_ff + 1u;
    l.dq4 = l.dq8 + 2u * l.dq8_beats;
    l.in_end = l.dq4 + 2u * l.dq4_beats;
    l.vop = l.in_end;
    l.vop_sreg = l.vop + TSR_ST4_N_FUNC * R * l.L;
    l.red = l.vop_sreg + R * l.L;
    l.sop = l.red + TSR_ST4_N_RED;
    l.rope_norm = l.sop + 1u;
    l.rope_neox = l.rope_norm + TSR_ST4_ROPE_LINES;
    l.topk = l.rope_neox + TSR_ST4_ROPE_LINES;
    l.dq8_out = l.topk + R;
    l.dq4_out = l.dq8_out + 2u * 16u;
    l.end = l.dq4_out + 2u * 16u;
    return l;
}

// ---------------------------------------------------------------------------------------------
// 5 AT. P[2] = n_kv (1..16 C cells), P[3] = T_q (1..T_MAX), P[4] = softmax scale bits (finite, > 0),
// P[5] = K plane base beat, P[6] = V plane base beat (every channel; LG = 1: 17 beats each).
// Geometry: E = D = 64, G = 1 (one head), R = T_q rows, n_groups = ceil(n_kv / 16) (<= C).
// Inputs (lines from 1): k (n_kv rows of 64 f32), v (n_kv rows), q (T_q rows), mask (T_q rows of
// 16 n_groups f32, n_groups lines apart).
// The kernel: QACT A32 of k, v, q (outputs kqa, vqa, qqa: QA_LINES(A32, 64) = 2 lines per row); KVST K
// and V of cells 0 .. n_kv - 1 (the planes are outputs in DRAM); ATTQK MAXOUT (u0 0, c_lo 0, c_hi n_kv)
// -> scores, mvec; ATTQK CAUSAL MAXOUT -> scores_c, mvec_c; ASMAX(scores, mvec) -> p; AMASK(scores,
// mask) -> sm, mvec_m; ATTPV LSUM(p) -> parts (C channel partials, then C row-sum lines); AMERGE INIT
// (parts, mvec) -> orun1, st1; AMERGE INIT then AMERGE merge with the same parts and mvec_m as the
// second chunk's maxima -> orun2, st2; ANORM(orun2, st2 l_run) -> out.
typedef struct tsr_st5_layout {
    uint32_t n_groups, R, qal;                       // groups, rows, QA_LINES(A32, 64)
    uint32_t k, v, q, mask, in_end;
    uint32_t kqa, vqa, qqa, scores, mvec, scores_c, mvec_c, p, sm, mvec_m, parts, orun1, st1, orun2, st2, out, end;
} tsr_st5_layout;
static inline tsr_st5_layout tsr_st5(uint32_t n_kv, uint32_t Tq, uint32_t C) {
    tsr_st5_layout l;
    l.n_groups = (n_kv + 15u) / 16u;
    l.R = Tq;
    l.qal = 2u;
    l.k = 1u;
    l.v = l.k + 4u * n_kv;
    l.q = l.v + 4u * n_kv;
    l.mask = l.q + 4u * Tq;
    l.in_end = l.mask + Tq * l.n_groups;
    l.kqa = l.in_end;
    l.vqa = l.kqa + l.qal * n_kv;
    l.qqa = l.vqa + l.qal * n_kv;
    l.scores = l.qqa + l.qal * Tq;
    l.mvec = l.scores + l.R * l.n_groups;
    l.scores_c = l.mvec + 1u;
    l.mvec_c = l.scores_c + l.R * l.n_groups;
    l.p = l.mvec_c + 1u;
    l.sm = l.p + l.R * l.n_groups;
    l.mvec_m = l.sm + l.R * l.n_groups;
    l.parts = l.mvec_m + 1u;
    l.orun1 = l.parts + C * 4u * l.R + C;            // C x R x D/16 partials, then C row-sum lines
    l.st1 = l.orun1 + 4u * l.R;
    l.orun2 = l.st1 + 2u;
    l.st2 = l.orun2 + 4u * l.R;
    l.out = l.st2 + 2u;
    l.end = l.out + 4u * l.R;
    return l;
}

// ---------------------------------------------------------------------------------------------
// 6 CONC. P[2] = N (16..4096), P[3] = K (a multiple of 256, 256..4096), P[4] = W_OFF, P[5] = fmt
// (0 Q8_0, 1 Q4_K), P[6] = J jobs (1..64). Input: J rows of K f32 at line 1. Per job j, the
// docs/ctrl-riscv.md §8.7 pipeline (double-buffered, deps only, no joins): LD x_j, QACT, MVM, VOP
// SILU, RED SUM, ST. Outputs: y_j = SILU(W x_j) (J x ceil(N/16) lines), then s (J lines: word 0 of
// line j = the RED SUM of y_j).
typedef struct tsr_st6_layout { uint32_t x, y, s, end; } tsr_st6_layout;
static inline tsr_st6_layout tsr_st6(uint32_t N, uint32_t K, uint32_t J) {
    tsr_st6_layout l;
    l.x = 1u;
    l.y = l.x + J * (K / 16u);
    l.s = l.y + J * ((N + 15u) / 16u);
    l.end = l.s + J;
    return l;
}

// ---------------------------------------------------------------------------------------------
// 7 ERRORS (device self-checking). P[2] = a DRAM beat beyond the channel limit (DRAM_RANGE), P[3] =
// W_OFF of a 16 x 32 Q8_0 tensor (the DEP_FAILED chain's MVM). Input: 2 lines at line 1; output:
// 2 lines (the recovery round trip after QABORT + QCLEAR). RET 0, or the number of the first failed
// check (1-based, the order of selftest.cpp); every step ends with a QCLEAR, so DONE is OK.
typedef struct tsr_st7_layout { uint32_t in, out, end; } tsr_st7_layout;
static inline tsr_st7_layout tsr_st7(void) { tsr_st7_layout l; l.in = 1u; l.out = 3u; l.end = 5u; return l; }

// ---------------------------------------------------------------------------------------------
// 8 SPMWIN (device self-checking). P[2] = an SPM line L0 (L0 .. L0 + 3 are used). Output: 2 lines at
// line 1: the pattern the RISC-V wrote into L0, L0 + 1 with byte, half, word and misaligned stores,
// stored by an ST pushed right after the stores. RET 0, or 1 + the first failed check.
typedef struct tsr_st8_layout { uint32_t out, end; } tsr_st8_layout;
static inline tsr_st8_layout tsr_st8(void) { tsr_st8_layout l; l.out = 1u; l.end = 3u; return l; }
// The expected pattern (128 bytes) for seed: what selftest.cpp's sequence leaves in L0, L0 + 1.
static inline void tsr_st8_pattern(uint32_t seed, uint8_t b[128]) {
    for (uint32_t i = 0; i < 64u; ++i) b[i] = (uint8_t)(seed + 37u * i);                  // byte stores, line 0
    for (uint32_t k = 0; k < 32u; ++k) {                                                    // half stores, line 1
        const uint32_t h = ((seed >> 8) ^ (k * 0x0101u)) & 0xFFFFu;
        b[64u + 2u * k] = (uint8_t)h;
        b[64u + 2u * k + 1u] = (uint8_t)(h >> 8);
    }
    const uint32_t w = seed * 2654435761u;                                                  // word at byte 8
    for (uint32_t i = 0; i < 4u; ++i) b[8u + i] = (uint8_t)(w >> (8u * i));
    const uint32_t m = ~seed;                                                               // misaligned word at 61
    for (uint32_t i = 0; i < 4u; ++i) b[61u + i] = (uint8_t)(m >> (8u * i));
    const uint32_t mh = (seed ^ 0x5A5Au) & 0xFFFFu;                                         // misaligned half at 99
    b[99] = (uint8_t)mh;
    b[100] = (uint8_t)(mh >> 8);
    b[32] = (uint8_t)(seed ^ 0xC3u);                                                        // byte after an alias fill
}

// ---------------------------------------------------------------------------------------------
// 9 LOG. No data (a2 may be 1). The kernel writes two FMT log records (SOURCE kernel): INFO
// "selftest 9: seed %x, id %u" (seed, 9) and WARN "selftest 9: %u %x" (0xFFFFFFFF ^ seed, seed + 1),
// and posts a kernel event type TSR_ST9_EVENT, info = seed[15:0], r2 = seed, r3 = ~seed. RET 0.
#define TSR_ST9_EVENT 0xC9u

// ---------------------------------------------------------------------------------------------
// 10 CREDITS (device self-checking). P[2] = W_OFF of a 16 x 32 Q8_0 tensor. Input: 2 lines (32 f32) at
// line 1. Under a QHOLD every built unit's queue is filled with Q_DEPTH entries (SYNC: NOPs waiting
// for a held op); its credit must then read 0. RET 0, or 0x100 (u + 1) for unit u.
typedef struct tsr_st10_layout { uint32_t x, end; } tsr_st10_layout;
static inline tsr_st10_layout tsr_st10(void) { tsr_st10_layout l; l.x = 1u; l.end = 3u; return l; }

#endif  // TSR_SELFTEST_ABI_HPP
