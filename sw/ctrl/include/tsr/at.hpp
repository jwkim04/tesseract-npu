// GENERATED FILE, DO NOT EDIT: produced by sw/tables/gen_tables.py from sw/tables/tsr_tables.py.
//
// tsr/at.hpp — Tesseract v2 tables for the firmware SDK (V2-EX-30, SW §6):
// the AT encodings (AT §5) and their constexpr builders.
// Freestanding C++20: no libstdc++ headers, no heap, everything constexpr in namespace tsr.
// Names as tsr_tables.h without the TSR_ prefix (see sw/tables/gen_tables.py).
// Builders take a field struct in encoding order (C++20 designated initializers; omitted
// fields are 0, as every unused field must be), mask each value to its field width and
// mark every nonzero word in op::used:
//     ASMAX({.dst = 100, .src = 100, .mvec = 300, .n_lines = 4, .r = 14, .scale = 0x3E000000u})
#ifndef TSR_AT_HPP
#define TSR_AT_HPP

#include <stdint.h>

#include "mmio.hpp"

namespace tsr {

// -----------------------------------------------------------------------------------------------
// AT encodings (AT §5; VU §5.11 restates the vector-op encodings)
// -----------------------------------------------------------------------------------------------
// KVST 0x14 (AT §5.1, owner AT)
constexpr uint32_t KVST_KV              = 0x00000F00u;  // w0[11:8] plane
constexpr uint32_t KVST_KV_LSB          = 8u;
constexpr uint32_t KVST_KV_W            = 4u;
constexpr uint32_t KVST_KV_WORD         = 0u;
constexpr uint32_t KVST_KV_BIT          = 8u;
constexpr uint32_t KVST_N_ROWS_M1       = 0x003F0000u;  // x16[5:0] n_rows - 1 (n_rows 1..64)
constexpr uint32_t KVST_N_ROWS_M1_LSB   = 16u;
constexpr uint32_t KVST_N_ROWS_M1_W     = 6u;
constexpr uint32_t KVST_N_ROWS_M1_WORD  = 0u;
constexpr uint32_t KVST_N_ROWS_M1_BIT   = 16u;
constexpr uint32_t KVST_SRC             = 0xFFFFFFFFu;  // w1 first QA(A32, E) vector (SPM line)
constexpr uint32_t KVST_SRC_LSB         = 0u;
constexpr uint32_t KVST_SRC_W           = 32u;
constexpr uint32_t KVST_SRC_WORD        = 1u;
constexpr uint32_t KVST_SRC_BIT         = 32u;
constexpr uint32_t KVST_BASE            = 0xFFFFFFFFu;  // w2 plane base B (beat, every channel)
constexpr uint32_t KVST_BASE_LSB        = 0u;
constexpr uint32_t KVST_BASE_W          = 32u;
constexpr uint32_t KVST_BASE_WORD       = 2u;
constexpr uint32_t KVST_BASE_BIT        = 64u;
constexpr uint32_t KVST_E64             = 0x000001FFu;  // w3[8:0] E / 64
constexpr uint32_t KVST_E64_LSB         = 0u;
constexpr uint32_t KVST_E64_W           = 9u;
constexpr uint32_t KVST_E64_WORD        = 3u;
constexpr uint32_t KVST_E64_BIT         = 96u;
constexpr uint32_t KVST_LG              = 0xFFFF0000u;  // w3[31:16] local groups per channel
constexpr uint32_t KVST_LG_LSB          = 16u;
constexpr uint32_t KVST_LG_W            = 16u;
constexpr uint32_t KVST_LG_WORD         = 3u;
constexpr uint32_t KVST_LG_BIT          = 112u;
constexpr uint32_t KVST_C0              = 0xFFFFFFFFu;  // w4 first cell
constexpr uint32_t KVST_C0_LSB          = 0u;
constexpr uint32_t KVST_C0_W            = 32u;
constexpr uint32_t KVST_C0_WORD         = 4u;
constexpr uint32_t KVST_C0_BIT          = 128u;
constexpr uint32_t KVST_SRC_STRIDE      = 0xFFFFFFFFu;  // w5 lines; 0 = QA_LINES(A32, E)
constexpr uint32_t KVST_SRC_STRIDE_LSB  = 0u;
constexpr uint32_t KVST_SRC_STRIDE_W    = 32u;
constexpr uint32_t KVST_SRC_STRIDE_WORD = 5u;
constexpr uint32_t KVST_SRC_STRIDE_BIT  = 160u;
constexpr uint32_t KVST_KV_K            = 0u;           // K plane
constexpr uint32_t KVST_KV_V            = 1u;           // V plane
constexpr uint32_t KVST_RSV_W0          = 0xFFC0F000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t KVST_RSV_W1          = 0x00000000u;
constexpr uint32_t KVST_RSV_W2          = 0x00000000u;
constexpr uint32_t KVST_RSV_W3          = 0x0000FE00u;
constexpr uint32_t KVST_RSV_W4          = 0x00000000u;
constexpr uint32_t KVST_RSV_W5          = 0x00000000u;
constexpr uint32_t KVST_RSV_W6          = 0xFFFFFFFFu;
constexpr uint32_t KVST_RSV_W7          = 0xFFFFFFFFu;
constexpr uint32_t KVST_WORDS           = 0x3Fu;        // words holding a field (w0 always)
struct kvst_f {
    uint32_t kv = 0;          // w0[11:8] plane
    uint32_t n_rows_m1 = 0;   // x16[5:0] n_rows - 1 (n_rows 1..64)
    uint32_t src = 0;         // w1 first QA(A32, E) vector (SPM line)
    uint32_t base = 0;        // w2 plane base B (beat, every channel)
    uint32_t e64 = 0;         // w3[8:0] E / 64
    uint32_t lg = 0;          // w3[31:16] local groups per channel
    uint32_t c0 = 0;          // w4 first cell
    uint32_t src_stride = 0;  // w5 lines; 0 = QA_LINES(A32, E)
};
constexpr op KVST(const kvst_f& f) {
    op o{};
    o.w[0] = OP_KVST;
    o.w[0] |= (f.kv & 0xFu) << 8;
    o.w[0] |= (f.n_rows_m1 & 0x3Fu) << 16;
    o.w[1] |= f.src;
    o.w[2] |= f.base;
    o.w[3] |= (f.e64 & 0x1FFu) << 0;
    o.w[3] |= (f.lg & 0xFFFFu) << 16;
    o.w[4] |= f.c0;
    o.w[5] |= f.src_stride;
    return op_finish(o);
}
// ATTQK 0x21 (AT §5.2, owner AT)
constexpr uint32_t ATTQK_FMT           = 0x00000F00u;  // w0[11:8] KV format
constexpr uint32_t ATTQK_FMT_LSB       = 8u;
constexpr uint32_t ATTQK_FMT_W         = 4u;
constexpr uint32_t ATTQK_FMT_WORD      = 0u;
constexpr uint32_t ATTQK_FMT_BIT       = 8u;
constexpr uint32_t ATTQK_CAUSAL        = 0x00001000u;  // w0[12]
constexpr uint32_t ATTQK_CAUSAL_LSB    = 12u;
constexpr uint32_t ATTQK_CAUSAL_W      = 1u;
constexpr uint32_t ATTQK_CAUSAL_WORD   = 0u;
constexpr uint32_t ATTQK_CAUSAL_BIT    = 12u;
constexpr uint32_t ATTQK_SLIDE         = 0x00002000u;  // w0[13]
constexpr uint32_t ATTQK_SLIDE_LSB     = 13u;
constexpr uint32_t ATTQK_SLIDE_W       = 1u;
constexpr uint32_t ATTQK_SLIDE_WORD    = 0u;
constexpr uint32_t ATTQK_SLIDE_BIT     = 13u;
constexpr uint32_t ATTQK_MAXOUT        = 0x00004000u;  // w0[14] write row maxima at MDST
constexpr uint32_t ATTQK_MAXOUT_LSB    = 14u;
constexpr uint32_t ATTQK_MAXOUT_W      = 1u;
constexpr uint32_t ATTQK_MAXOUT_WORD   = 0u;
constexpr uint32_t ATTQK_MAXOUT_BIT    = 14u;
constexpr uint32_t ATTQK_G_M1          = 0x003F0000u;  // x16[5:0] G - 1 (heads per KV head)
constexpr uint32_t ATTQK_G_M1_LSB      = 16u;
constexpr uint32_t ATTQK_G_M1_W        = 6u;
constexpr uint32_t ATTQK_G_M1_WORD     = 0u;
constexpr uint32_t ATTQK_G_M1_BIT      = 16u;
constexpr uint32_t ATTQK_HG0           = 0x0FC00000u;  // x16[11:6] first head of the pass within each KV-head group
constexpr uint32_t ATTQK_HG0_LSB       = 22u;
constexpr uint32_t ATTQK_HG0_W         = 6u;
constexpr uint32_t ATTQK_HG0_WORD      = 0u;
constexpr uint32_t ATTQK_HG0_BIT       = 22u;
constexpr uint32_t ATTQK_DST           = 0xFFFFFFFFu;  // w1 score rows, row r at DST + r x RSTRIDE
constexpr uint32_t ATTQK_DST_LSB       = 0u;
constexpr uint32_t ATTQK_DST_W         = 32u;
constexpr uint32_t ATTQK_DST_WORD      = 1u;
constexpr uint32_t ATTQK_DST_BIT       = 32u;
constexpr uint32_t ATTQK_X             = 0xFFFFFFFFu;  // w2 T_q QA(A32, n_head x D) vectors
constexpr uint32_t ATTQK_X_LSB         = 0u;
constexpr uint32_t ATTQK_X_W           = 32u;
constexpr uint32_t ATTQK_X_WORD        = 2u;
constexpr uint32_t ATTQK_X_BIT         = 64u;
constexpr uint32_t ATTQK_BASE          = 0xFFFFFFFFu;  // w3 K plane base B
constexpr uint32_t ATTQK_BASE_LSB      = 0u;
constexpr uint32_t ATTQK_BASE_W        = 32u;
constexpr uint32_t ATTQK_BASE_WORD     = 3u;
constexpr uint32_t ATTQK_BASE_BIT      = 96u;
constexpr uint32_t ATTQK_E64           = 0x000001FFu;  // w4[8:0] E / 64 (1..512)
constexpr uint32_t ATTQK_E64_LSB       = 0u;
constexpr uint32_t ATTQK_E64_W         = 9u;
constexpr uint32_t ATTQK_E64_WORD      = 4u;
constexpr uint32_t ATTQK_E64_BIT       = 128u;
constexpr uint32_t ATTQK_DCODE         = 0x00000600u;  // w4[10:9] D = 64 x 2^dcode
constexpr uint32_t ATTQK_DCODE_LSB     = 9u;
constexpr uint32_t ATTQK_DCODE_W       = 2u;
constexpr uint32_t ATTQK_DCODE_WORD    = 4u;
constexpr uint32_t ATTQK_DCODE_BIT     = 137u;
constexpr uint32_t ATTQK_TQ_M1         = 0x00003800u;  // w4[13:11] T_q - 1
constexpr uint32_t ATTQK_TQ_M1_LSB     = 11u;
constexpr uint32_t ATTQK_TQ_M1_W       = 3u;
constexpr uint32_t ATTQK_TQ_M1_WORD    = 4u;
constexpr uint32_t ATTQK_TQ_M1_BIT     = 139u;
constexpr uint32_t ATTQK_GS_M1         = 0x0001C000u;  // w4[16:14] GS - 1
constexpr uint32_t ATTQK_GS_M1_LSB     = 14u;
constexpr uint32_t ATTQK_GS_M1_W       = 3u;
constexpr uint32_t ATTQK_GS_M1_WORD    = 4u;
constexpr uint32_t ATTQK_GS_M1_BIT     = 142u;
constexpr uint32_t ATTQK_RSTRIDE       = 0xFFFE0000u;  // w4[31:17] lines; 0 = n_groups
constexpr uint32_t ATTQK_RSTRIDE_LSB   = 17u;
constexpr uint32_t ATTQK_RSTRIDE_W     = 15u;
constexpr uint32_t ATTQK_RSTRIDE_WORD  = 4u;
constexpr uint32_t ATTQK_RSTRIDE_BIT   = 145u;
constexpr uint32_t ATTQK_U0            = 0x0000FFFFu;  // w5[15:0] first group
constexpr uint32_t ATTQK_U0_LSB        = 0u;
constexpr uint32_t ATTQK_U0_W          = 16u;
constexpr uint32_t ATTQK_U0_WORD       = 5u;
constexpr uint32_t ATTQK_U0_BIT        = 160u;
constexpr uint32_t ATTQK_N_GROUPS      = 0xFFFF0000u;  // w5[31:16] 1..4095
constexpr uint32_t ATTQK_N_GROUPS_LSB  = 16u;
constexpr uint32_t ATTQK_N_GROUPS_W    = 16u;
constexpr uint32_t ATTQK_N_GROUPS_WORD = 5u;
constexpr uint32_t ATTQK_N_GROUPS_BIT  = 176u;
constexpr uint32_t ATTQK_C_LO          = 0x0000FFFFu;  // w6[15:0] cell bound relative to 16 u0
constexpr uint32_t ATTQK_C_LO_LSB      = 0u;
constexpr uint32_t ATTQK_C_LO_W        = 16u;
constexpr uint32_t ATTQK_C_LO_WORD     = 6u;
constexpr uint32_t ATTQK_C_LO_BIT      = 192u;
constexpr uint32_t ATTQK_C_HI          = 0xFFFF0000u;  // w6[31:16] cell bound relative to 16 u0
constexpr uint32_t ATTQK_C_HI_LSB      = 16u;
constexpr uint32_t ATTQK_C_HI_W        = 16u;
constexpr uint32_t ATTQK_C_HI_WORD     = 6u;
constexpr uint32_t ATTQK_C_HI_BIT      = 208u;
constexpr uint32_t ATTQK_MDST          = 0xFFFFFFFFu;  // w7 row-maximum vector line; 0 unless MAXOUT
constexpr uint32_t ATTQK_MDST_LSB      = 0u;
constexpr uint32_t ATTQK_MDST_W        = 32u;
constexpr uint32_t ATTQK_MDST_WORD     = 7u;
constexpr uint32_t ATTQK_MDST_BIT      = 224u;
// ATTQK_FMT takes the KVFMT_* values
// ATTQK_DCODE takes the DCODE_* values
constexpr uint32_t ATTQK_RSV_W0 = 0xF0008000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t ATTQK_RSV_W1 = 0x00000000u;
constexpr uint32_t ATTQK_RSV_W2 = 0x00000000u;
constexpr uint32_t ATTQK_RSV_W3 = 0x00000000u;
constexpr uint32_t ATTQK_RSV_W4 = 0x00000000u;
constexpr uint32_t ATTQK_RSV_W5 = 0x00000000u;
constexpr uint32_t ATTQK_RSV_W6 = 0x00000000u;
constexpr uint32_t ATTQK_RSV_W7 = 0x00000000u;
constexpr uint32_t ATTQK_WORDS  = 0xFFu;        // words holding a field (w0 always)
struct attqk_f {
    uint32_t fmt = 0;       // w0[11:8] KV format
    uint32_t causal = 0;    // w0[12]
    uint32_t slide = 0;     // w0[13]
    uint32_t maxout = 0;    // w0[14] write row maxima at MDST
    uint32_t g_m1 = 0;      // x16[5:0] G - 1 (heads per KV head)
    uint32_t hg0 = 0;       // x16[11:6] first head of the pass within each KV-head group
    uint32_t dst = 0;       // w1 score rows, row r at DST + r x RSTRIDE
    uint32_t x = 0;         // w2 T_q QA(A32, n_head x D) vectors
    uint32_t base = 0;      // w3 K plane base B
    uint32_t e64 = 0;       // w4[8:0] E / 64 (1..512)
    uint32_t dcode = 0;     // w4[10:9] D = 64 x 2^dcode
    uint32_t tq_m1 = 0;     // w4[13:11] T_q - 1
    uint32_t gs_m1 = 0;     // w4[16:14] GS - 1
    uint32_t rstride = 0;   // w4[31:17] lines; 0 = n_groups
    uint32_t u0 = 0;        // w5[15:0] first group
    uint32_t n_groups = 0;  // w5[31:16] 1..4095
    uint32_t c_lo = 0;      // w6[15:0] cell bound relative to 16 u0
    uint32_t c_hi = 0;      // w6[31:16] cell bound relative to 16 u0
    uint32_t mdst = 0;      // w7 row-maximum vector line; 0 unless MAXOUT
};
constexpr op ATTQK(const attqk_f& f) {
    op o{};
    o.w[0] = OP_ATTQK;
    o.w[0] |= (f.fmt & 0xFu) << 8;
    o.w[0] |= (f.causal & 0x1u) << 12;
    o.w[0] |= (f.slide & 0x1u) << 13;
    o.w[0] |= (f.maxout & 0x1u) << 14;
    o.w[0] |= (f.g_m1 & 0x3Fu) << 16;
    o.w[0] |= (f.hg0 & 0x3Fu) << 22;
    o.w[1] |= f.dst;
    o.w[2] |= f.x;
    o.w[3] |= f.base;
    o.w[4] |= (f.e64 & 0x1FFu) << 0;
    o.w[4] |= (f.dcode & 0x3u) << 9;
    o.w[4] |= (f.tq_m1 & 0x7u) << 11;
    o.w[4] |= (f.gs_m1 & 0x7u) << 14;
    o.w[4] |= (f.rstride & 0x7FFFu) << 17;
    o.w[5] |= (f.u0 & 0xFFFFu) << 0;
    o.w[5] |= (f.n_groups & 0xFFFFu) << 16;
    o.w[6] |= (f.c_lo & 0xFFFFu) << 0;
    o.w[6] |= (f.c_hi & 0xFFFFu) << 16;
    o.w[7] |= f.mdst;
    return op_finish(o);
}
// ATTPV 0x22 (AT §5.3, owner AT)
constexpr uint32_t ATTPV_FMT           = 0x00000F00u;  // w0[11:8] KV format
constexpr uint32_t ATTPV_FMT_LSB       = 8u;
constexpr uint32_t ATTPV_FMT_W         = 4u;
constexpr uint32_t ATTPV_FMT_WORD      = 0u;
constexpr uint32_t ATTPV_FMT_BIT       = 8u;
constexpr uint32_t ATTPV_LSUM          = 0x00001000u;  // w0[12] also row sums
constexpr uint32_t ATTPV_LSUM_LSB      = 12u;
constexpr uint32_t ATTPV_LSUM_W        = 1u;
constexpr uint32_t ATTPV_LSUM_WORD     = 0u;
constexpr uint32_t ATTPV_LSUM_BIT      = 12u;
constexpr uint32_t ATTPV_G_M1          = 0x003F0000u;  // x16[5:0] G - 1
constexpr uint32_t ATTPV_G_M1_LSB      = 16u;
constexpr uint32_t ATTPV_G_M1_W        = 6u;
constexpr uint32_t ATTPV_G_M1_WORD     = 0u;
constexpr uint32_t ATTPV_G_M1_BIT      = 16u;
constexpr uint32_t ATTPV_HG0           = 0x0FC00000u;  // x16[11:6] first head of the pass
constexpr uint32_t ATTPV_HG0_LSB       = 22u;
constexpr uint32_t ATTPV_HG0_W         = 6u;
constexpr uint32_t ATTPV_HG0_WORD      = 0u;
constexpr uint32_t ATTPV_HG0_BIT       = 22u;
constexpr uint32_t ATTPV_DST           = 0xFFFFFFFFu;  // w1 channel partials
constexpr uint32_t ATTPV_DST_LSB       = 0u;
constexpr uint32_t ATTPV_DST_W         = 32u;
constexpr uint32_t ATTPV_DST_WORD      = 1u;
constexpr uint32_t ATTPV_DST_BIT       = 32u;
constexpr uint32_t ATTPV_P             = 0xFFFFFFFFu;  // w2 weight rows, row r at P + r x PSTRIDE
constexpr uint32_t ATTPV_P_LSB         = 0u;
constexpr uint32_t ATTPV_P_W           = 32u;
constexpr uint32_t ATTPV_P_WORD        = 2u;
constexpr uint32_t ATTPV_P_BIT         = 64u;
constexpr uint32_t ATTPV_BASE          = 0xFFFFFFFFu;  // w3 V plane base B
constexpr uint32_t ATTPV_BASE_LSB      = 0u;
constexpr uint32_t ATTPV_BASE_W        = 32u;
constexpr uint32_t ATTPV_BASE_WORD     = 3u;
constexpr uint32_t ATTPV_BASE_BIT      = 96u;
constexpr uint32_t ATTPV_E64           = 0x000001FFu;  // w4[8:0] E / 64
constexpr uint32_t ATTPV_E64_LSB       = 0u;
constexpr uint32_t ATTPV_E64_W         = 9u;
constexpr uint32_t ATTPV_E64_WORD      = 4u;
constexpr uint32_t ATTPV_E64_BIT       = 128u;
constexpr uint32_t ATTPV_DCODE         = 0x00000600u;  // w4[10:9]
constexpr uint32_t ATTPV_DCODE_LSB     = 9u;
constexpr uint32_t ATTPV_DCODE_W       = 2u;
constexpr uint32_t ATTPV_DCODE_WORD    = 4u;
constexpr uint32_t ATTPV_DCODE_BIT     = 137u;
constexpr uint32_t ATTPV_TQ_M1         = 0x00003800u;  // w4[13:11] T_q - 1
constexpr uint32_t ATTPV_TQ_M1_LSB     = 11u;
constexpr uint32_t ATTPV_TQ_M1_W       = 3u;
constexpr uint32_t ATTPV_TQ_M1_WORD    = 4u;
constexpr uint32_t ATTPV_TQ_M1_BIT     = 139u;
constexpr uint32_t ATTPV_GS_M1         = 0x0001C000u;  // w4[16:14] GS - 1
constexpr uint32_t ATTPV_GS_M1_LSB     = 14u;
constexpr uint32_t ATTPV_GS_M1_W       = 3u;
constexpr uint32_t ATTPV_GS_M1_WORD    = 4u;
constexpr uint32_t ATTPV_GS_M1_BIT     = 142u;
constexpr uint32_t ATTPV_PSTRIDE       = 0xFFFE0000u;  // w4[31:17] lines; 0 = n_groups
constexpr uint32_t ATTPV_PSTRIDE_LSB   = 17u;
constexpr uint32_t ATTPV_PSTRIDE_W     = 15u;
constexpr uint32_t ATTPV_PSTRIDE_WORD  = 4u;
constexpr uint32_t ATTPV_PSTRIDE_BIT   = 145u;
constexpr uint32_t ATTPV_U0            = 0x0000FFFFu;  // w5[15:0] first group
constexpr uint32_t ATTPV_U0_LSB        = 0u;
constexpr uint32_t ATTPV_U0_W          = 16u;
constexpr uint32_t ATTPV_U0_WORD       = 5u;
constexpr uint32_t ATTPV_U0_BIT        = 160u;
constexpr uint32_t ATTPV_N_GROUPS      = 0xFFFF0000u;  // w5[31:16]
constexpr uint32_t ATTPV_N_GROUPS_LSB  = 16u;
constexpr uint32_t ATTPV_N_GROUPS_W    = 16u;
constexpr uint32_t ATTPV_N_GROUPS_WORD = 5u;
constexpr uint32_t ATTPV_N_GROUPS_BIT  = 176u;
constexpr uint32_t ATTPV_LG            = 0x0000FFFFu;  // w6[15:0] local groups per channel
constexpr uint32_t ATTPV_LG_LSB        = 0u;
constexpr uint32_t ATTPV_LG_W          = 16u;
constexpr uint32_t ATTPV_LG_WORD       = 6u;
constexpr uint32_t ATTPV_LG_BIT        = 192u;
constexpr uint32_t ATTPV_CSTRIDE       = 0xFFFFFFFFu;  // w7 lines; 0 = R x D / 16
constexpr uint32_t ATTPV_CSTRIDE_LSB   = 0u;
constexpr uint32_t ATTPV_CSTRIDE_W     = 32u;
constexpr uint32_t ATTPV_CSTRIDE_WORD  = 7u;
constexpr uint32_t ATTPV_CSTRIDE_BIT   = 224u;
// ATTPV_FMT takes the KVFMT_* values
// ATTPV_DCODE takes the DCODE_* values
constexpr uint32_t ATTPV_RSV_W0 = 0xF000E000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t ATTPV_RSV_W1 = 0x00000000u;
constexpr uint32_t ATTPV_RSV_W2 = 0x00000000u;
constexpr uint32_t ATTPV_RSV_W3 = 0x00000000u;
constexpr uint32_t ATTPV_RSV_W4 = 0x00000000u;
constexpr uint32_t ATTPV_RSV_W5 = 0x00000000u;
constexpr uint32_t ATTPV_RSV_W6 = 0xFFFF0000u;
constexpr uint32_t ATTPV_RSV_W7 = 0x00000000u;
constexpr uint32_t ATTPV_WORDS  = 0xFFu;        // words holding a field (w0 always)
struct attpv_f {
    uint32_t fmt = 0;       // w0[11:8] KV format
    uint32_t lsum = 0;      // w0[12] also row sums
    uint32_t g_m1 = 0;      // x16[5:0] G - 1
    uint32_t hg0 = 0;       // x16[11:6] first head of the pass
    uint32_t dst = 0;       // w1 channel partials
    uint32_t p = 0;         // w2 weight rows, row r at P + r x PSTRIDE
    uint32_t base = 0;      // w3 V plane base B
    uint32_t e64 = 0;       // w4[8:0] E / 64
    uint32_t dcode = 0;     // w4[10:9]
    uint32_t tq_m1 = 0;     // w4[13:11] T_q - 1
    uint32_t gs_m1 = 0;     // w4[16:14] GS - 1
    uint32_t pstride = 0;   // w4[31:17] lines; 0 = n_groups
    uint32_t u0 = 0;        // w5[15:0] first group
    uint32_t n_groups = 0;  // w5[31:16]
    uint32_t lg = 0;        // w6[15:0] local groups per channel
    uint32_t cstride = 0;   // w7 lines; 0 = R x D / 16
};
constexpr op ATTPV(const attpv_f& f) {
    op o{};
    o.w[0] = OP_ATTPV;
    o.w[0] |= (f.fmt & 0xFu) << 8;
    o.w[0] |= (f.lsum & 0x1u) << 12;
    o.w[0] |= (f.g_m1 & 0x3Fu) << 16;
    o.w[0] |= (f.hg0 & 0x3Fu) << 22;
    o.w[1] |= f.dst;
    o.w[2] |= f.p;
    o.w[3] |= f.base;
    o.w[4] |= (f.e64 & 0x1FFu) << 0;
    o.w[4] |= (f.dcode & 0x3u) << 9;
    o.w[4] |= (f.tq_m1 & 0x7u) << 11;
    o.w[4] |= (f.gs_m1 & 0x7u) << 14;
    o.w[4] |= (f.pstride & 0x7FFFu) << 17;
    o.w[5] |= (f.u0 & 0xFFFFu) << 0;
    o.w[5] |= (f.n_groups & 0xFFFFu) << 16;
    o.w[6] |= (f.lg & 0xFFFFu) << 0;
    o.w[7] |= f.cstride;
    return op_finish(o);
}
// ASMAX 0x34 (AT §5.4, VU §5.11, owner AT)
constexpr uint32_t ASMAX_DST          = 0xFFFFFFFFu;  // w1
constexpr uint32_t ASMAX_DST_LSB      = 0u;
constexpr uint32_t ASMAX_DST_W        = 32u;
constexpr uint32_t ASMAX_DST_WORD     = 1u;
constexpr uint32_t ASMAX_DST_BIT      = 32u;
constexpr uint32_t ASMAX_SRC          = 0xFFFFFFFFu;  // w2 may equal DST
constexpr uint32_t ASMAX_SRC_LSB      = 0u;
constexpr uint32_t ASMAX_SRC_W        = 32u;
constexpr uint32_t ASMAX_SRC_WORD     = 2u;
constexpr uint32_t ASMAX_SRC_BIT      = 64u;
constexpr uint32_t ASMAX_MVEC         = 0xFFFFFFFFu;  // w3 row maxima, word r
constexpr uint32_t ASMAX_MVEC_LSB     = 0u;
constexpr uint32_t ASMAX_MVEC_W       = 32u;
constexpr uint32_t ASMAX_MVEC_WORD    = 3u;
constexpr uint32_t ASMAX_MVEC_BIT     = 96u;
constexpr uint32_t ASMAX_N_LINES      = 0x0000FFFFu;  // w4[15:0] 1..4095
constexpr uint32_t ASMAX_N_LINES_LSB  = 0u;
constexpr uint32_t ASMAX_N_LINES_W    = 16u;
constexpr uint32_t ASMAX_N_LINES_WORD = 4u;
constexpr uint32_t ASMAX_N_LINES_BIT  = 128u;
constexpr uint32_t ASMAX_STRIDE       = 0xFFFF0000u;  // w4[31:16] 0 = n_lines
constexpr uint32_t ASMAX_STRIDE_LSB   = 16u;
constexpr uint32_t ASMAX_STRIDE_W     = 16u;
constexpr uint32_t ASMAX_STRIDE_WORD  = 4u;
constexpr uint32_t ASMAX_STRIDE_BIT   = 144u;
constexpr uint32_t ASMAX_R            = 0xFFFFFFFFu;  // w5 1..1024
constexpr uint32_t ASMAX_R_LSB        = 0u;
constexpr uint32_t ASMAX_R_W          = 32u;
constexpr uint32_t ASMAX_R_WORD       = 5u;
constexpr uint32_t ASMAX_R_BIT        = 160u;
constexpr uint32_t ASMAX_SCALE        = 0xFFFFFFFFu;  // w6 f32, finite, normal, > 0
constexpr uint32_t ASMAX_SCALE_LSB    = 0u;
constexpr uint32_t ASMAX_SCALE_W      = 32u;
constexpr uint32_t ASMAX_SCALE_WORD   = 6u;
constexpr uint32_t ASMAX_SCALE_BIT    = 192u;
constexpr uint32_t ASMAX_RSV_W0       = 0xFFFFFF00u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t ASMAX_RSV_W1       = 0x00000000u;
constexpr uint32_t ASMAX_RSV_W2       = 0x00000000u;
constexpr uint32_t ASMAX_RSV_W3       = 0x00000000u;
constexpr uint32_t ASMAX_RSV_W4       = 0x00000000u;
constexpr uint32_t ASMAX_RSV_W5       = 0x00000000u;
constexpr uint32_t ASMAX_RSV_W6       = 0x00000000u;
constexpr uint32_t ASMAX_RSV_W7       = 0xFFFFFFFFu;
constexpr uint32_t ASMAX_WORDS        = 0x7Fu;        // words holding a field (w0 always)
struct asmax_f {
    uint32_t dst = 0;      // w1
    uint32_t src = 0;      // w2 may equal DST
    uint32_t mvec = 0;     // w3 row maxima, word r
    uint32_t n_lines = 0;  // w4[15:0] 1..4095
    uint32_t stride = 0;   // w4[31:16] 0 = n_lines
    uint32_t r = 0;        // w5 1..1024
    uint32_t scale = 0;    // w6 f32, finite, normal, > 0
};
constexpr op ASMAX(const asmax_f& f) {
    op o{};
    o.w[0] = OP_ASMAX;
    o.w[1] |= f.dst;
    o.w[2] |= f.src;
    o.w[3] |= f.mvec;
    o.w[4] |= (f.n_lines & 0xFFFFu) << 0;
    o.w[4] |= (f.stride & 0xFFFFu) << 16;
    o.w[5] |= f.r;
    o.w[6] |= f.scale;
    return op_finish(o);
}
// AMASK 0x35 (AT §5.5, VU §5.11, owner AT)
constexpr uint32_t AMASK_DST          = 0xFFFFFFFFu;  // w1
constexpr uint32_t AMASK_DST_LSB      = 0u;
constexpr uint32_t AMASK_DST_W        = 32u;
constexpr uint32_t AMASK_DST_WORD     = 1u;
constexpr uint32_t AMASK_DST_BIT      = 32u;
constexpr uint32_t AMASK_SRC          = 0xFFFFFFFFu;  // w2 rows at STRIDE
constexpr uint32_t AMASK_SRC_LSB      = 0u;
constexpr uint32_t AMASK_SRC_W        = 32u;
constexpr uint32_t AMASK_SRC_WORD     = 2u;
constexpr uint32_t AMASK_SRC_BIT      = 64u;
constexpr uint32_t AMASK_MASK         = 0xFFFFFFFFu;  // w3 f32 mask rows
constexpr uint32_t AMASK_MASK_LSB     = 0u;
constexpr uint32_t AMASK_MASK_W       = 32u;
constexpr uint32_t AMASK_MASK_WORD    = 3u;
constexpr uint32_t AMASK_MASK_BIT     = 96u;
constexpr uint32_t AMASK_N_LINES      = 0x0000FFFFu;  // w4[15:0]
constexpr uint32_t AMASK_N_LINES_LSB  = 0u;
constexpr uint32_t AMASK_N_LINES_W    = 16u;
constexpr uint32_t AMASK_N_LINES_WORD = 4u;
constexpr uint32_t AMASK_N_LINES_BIT  = 128u;
constexpr uint32_t AMASK_STRIDE       = 0xFFFF0000u;  // w4[31:16]
constexpr uint32_t AMASK_STRIDE_LSB   = 16u;
constexpr uint32_t AMASK_STRIDE_W     = 16u;
constexpr uint32_t AMASK_STRIDE_WORD  = 4u;
constexpr uint32_t AMASK_STRIDE_BIT   = 144u;
constexpr uint32_t AMASK_R            = 0x0000FFFFu;  // w5[15:0]
constexpr uint32_t AMASK_R_LSB        = 0u;
constexpr uint32_t AMASK_R_W          = 16u;
constexpr uint32_t AMASK_R_WORD       = 5u;
constexpr uint32_t AMASK_R_BIT        = 160u;
constexpr uint32_t AMASK_RPQ          = 0xFFFF0000u;  // w5[31:16] rows per query (= n_head)
constexpr uint32_t AMASK_RPQ_LSB      = 16u;
constexpr uint32_t AMASK_RPQ_W        = 16u;
constexpr uint32_t AMASK_RPQ_WORD     = 5u;
constexpr uint32_t AMASK_RPQ_BIT      = 176u;
constexpr uint32_t AMASK_MSTRIDE      = 0xFFFFFFFFu;  // w6 lines between mask rows
constexpr uint32_t AMASK_MSTRIDE_LSB  = 0u;
constexpr uint32_t AMASK_MSTRIDE_W    = 32u;
constexpr uint32_t AMASK_MSTRIDE_WORD = 6u;
constexpr uint32_t AMASK_MSTRIDE_BIT  = 192u;
constexpr uint32_t AMASK_MDST         = 0xFFFFFFFFu;  // w7 row maxima out
constexpr uint32_t AMASK_MDST_LSB     = 0u;
constexpr uint32_t AMASK_MDST_W       = 32u;
constexpr uint32_t AMASK_MDST_WORD    = 7u;
constexpr uint32_t AMASK_MDST_BIT     = 224u;
constexpr uint32_t AMASK_RSV_W0       = 0xFFFFFF00u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t AMASK_RSV_W1       = 0x00000000u;
constexpr uint32_t AMASK_RSV_W2       = 0x00000000u;
constexpr uint32_t AMASK_RSV_W3       = 0x00000000u;
constexpr uint32_t AMASK_RSV_W4       = 0x00000000u;
constexpr uint32_t AMASK_RSV_W5       = 0x00000000u;
constexpr uint32_t AMASK_RSV_W6       = 0x00000000u;
constexpr uint32_t AMASK_RSV_W7       = 0x00000000u;
constexpr uint32_t AMASK_WORDS        = 0xFFu;        // words holding a field (w0 always)
struct amask_f {
    uint32_t dst = 0;      // w1
    uint32_t src = 0;      // w2 rows at STRIDE
    uint32_t mask = 0;     // w3 f32 mask rows
    uint32_t n_lines = 0;  // w4[15:0]
    uint32_t stride = 0;   // w4[31:16]
    uint32_t r = 0;        // w5[15:0]
    uint32_t rpq = 0;      // w5[31:16] rows per query (= n_head)
    uint32_t mstride = 0;  // w6 lines between mask rows
    uint32_t mdst = 0;     // w7 row maxima out
};
constexpr op AMASK(const amask_f& f) {
    op o{};
    o.w[0] = OP_AMASK;
    o.w[1] |= f.dst;
    o.w[2] |= f.src;
    o.w[3] |= f.mask;
    o.w[4] |= (f.n_lines & 0xFFFFu) << 0;
    o.w[4] |= (f.stride & 0xFFFFu) << 16;
    o.w[5] |= (f.r & 0xFFFFu) << 0;
    o.w[5] |= (f.rpq & 0xFFFFu) << 16;
    o.w[6] |= f.mstride;
    o.w[7] |= f.mdst;
    return op_finish(o);
}
// AMERGE 0x36 (AT §5.6, VU §5.11, owner AT)
constexpr uint32_t AMERGE_INIT         = 0x00001000u;  // w0[12] copy instead of merge
constexpr uint32_t AMERGE_INIT_LSB     = 12u;
constexpr uint32_t AMERGE_INIT_W       = 1u;
constexpr uint32_t AMERGE_INIT_WORD    = 0u;
constexpr uint32_t AMERGE_INIT_BIT     = 12u;
constexpr uint32_t AMERGE_ORUN         = 0xFFFFFFFFu;  // w1 R rows of D/16 lines
constexpr uint32_t AMERGE_ORUN_LSB     = 0u;
constexpr uint32_t AMERGE_ORUN_W       = 32u;
constexpr uint32_t AMERGE_ORUN_WORD    = 1u;
constexpr uint32_t AMERGE_ORUN_BIT     = 32u;
constexpr uint32_t AMERGE_PARTS        = 0xFFFFFFFFu;  // w2 an ATTPV destination written with LSUM
constexpr uint32_t AMERGE_PARTS_LSB    = 0u;
constexpr uint32_t AMERGE_PARTS_W      = 32u;
constexpr uint32_t AMERGE_PARTS_WORD   = 2u;
constexpr uint32_t AMERGE_PARTS_BIT    = 64u;
constexpr uint32_t AMERGE_MK           = 0xFFFFFFFFu;  // w3 chunk row maxima
constexpr uint32_t AMERGE_MK_LSB       = 0u;
constexpr uint32_t AMERGE_MK_W         = 32u;
constexpr uint32_t AMERGE_MK_WORD      = 3u;
constexpr uint32_t AMERGE_MK_BIT       = 96u;
constexpr uint32_t AMERGE_ST           = 0xFFFFFFFFu;  // w4 m_run at ST, l_run at ST + ceil(R/16)
constexpr uint32_t AMERGE_ST_LSB       = 0u;
constexpr uint32_t AMERGE_ST_W         = 32u;
constexpr uint32_t AMERGE_ST_WORD      = 4u;
constexpr uint32_t AMERGE_ST_BIT       = 128u;
constexpr uint32_t AMERGE_R            = 0x0000FFFFu;  // w5[15:0]
constexpr uint32_t AMERGE_R_LSB        = 0u;
constexpr uint32_t AMERGE_R_W          = 16u;
constexpr uint32_t AMERGE_R_WORD       = 5u;
constexpr uint32_t AMERGE_R_BIT        = 160u;
constexpr uint32_t AMERGE_DCODE        = 0x00030000u;  // w5[17:16]
constexpr uint32_t AMERGE_DCODE_LSB    = 16u;
constexpr uint32_t AMERGE_DCODE_W      = 2u;
constexpr uint32_t AMERGE_DCODE_WORD   = 5u;
constexpr uint32_t AMERGE_DCODE_BIT    = 176u;
constexpr uint32_t AMERGE_SCALE        = 0xFFFFFFFFu;  // w6 f32, as ASMAX
constexpr uint32_t AMERGE_SCALE_LSB    = 0u;
constexpr uint32_t AMERGE_SCALE_W      = 32u;
constexpr uint32_t AMERGE_SCALE_WORD   = 6u;
constexpr uint32_t AMERGE_SCALE_BIT    = 192u;
constexpr uint32_t AMERGE_CSTRIDE      = 0xFFFFFFFFu;  // w7 as the producing ATTPV
constexpr uint32_t AMERGE_CSTRIDE_LSB  = 0u;
constexpr uint32_t AMERGE_CSTRIDE_W    = 32u;
constexpr uint32_t AMERGE_CSTRIDE_WORD = 7u;
constexpr uint32_t AMERGE_CSTRIDE_BIT  = 224u;
// AMERGE_DCODE takes the DCODE_* values
constexpr uint32_t AMERGE_RSV_W0 = 0xFFFFEF00u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t AMERGE_RSV_W1 = 0x00000000u;
constexpr uint32_t AMERGE_RSV_W2 = 0x00000000u;
constexpr uint32_t AMERGE_RSV_W3 = 0x00000000u;
constexpr uint32_t AMERGE_RSV_W4 = 0x00000000u;
constexpr uint32_t AMERGE_RSV_W5 = 0xFFFC0000u;
constexpr uint32_t AMERGE_RSV_W6 = 0x00000000u;
constexpr uint32_t AMERGE_RSV_W7 = 0x00000000u;
constexpr uint32_t AMERGE_WORDS  = 0xFFu;        // words holding a field (w0 always)
struct amerge_f {
    uint32_t init = 0;     // w0[12] copy instead of merge
    uint32_t orun = 0;     // w1 R rows of D/16 lines
    uint32_t parts = 0;    // w2 an ATTPV destination written with LSUM
    uint32_t mk = 0;       // w3 chunk row maxima
    uint32_t st = 0;       // w4 m_run at ST, l_run at ST + ceil(R/16)
    uint32_t r = 0;        // w5[15:0]
    uint32_t dcode = 0;    // w5[17:16]
    uint32_t scale = 0;    // w6 f32, as ASMAX
    uint32_t cstride = 0;  // w7 as the producing ATTPV
};
constexpr op AMERGE(const amerge_f& f) {
    op o{};
    o.w[0] = OP_AMERGE;
    o.w[0] |= (f.init & 0x1u) << 12;
    o.w[1] |= f.orun;
    o.w[2] |= f.parts;
    o.w[3] |= f.mk;
    o.w[4] |= f.st;
    o.w[5] |= (f.r & 0xFFFFu) << 0;
    o.w[5] |= (f.dcode & 0x3u) << 16;
    o.w[6] |= f.scale;
    o.w[7] |= f.cstride;
    return op_finish(o);
}
// ANORM 0x37 (AT §5.7, VU §5.11, owner AT)
constexpr uint32_t ANORM_OUT        = 0xFFFFFFFFu;  // w1 row r at OUT + r x D/16
constexpr uint32_t ANORM_OUT_LSB    = 0u;
constexpr uint32_t ANORM_OUT_W      = 32u;
constexpr uint32_t ANORM_OUT_WORD   = 1u;
constexpr uint32_t ANORM_OUT_BIT    = 32u;
constexpr uint32_t ANORM_ORUN       = 0xFFFFFFFFu;  // w2
constexpr uint32_t ANORM_ORUN_LSB   = 0u;
constexpr uint32_t ANORM_ORUN_W     = 32u;
constexpr uint32_t ANORM_ORUN_WORD  = 2u;
constexpr uint32_t ANORM_ORUN_BIT   = 64u;
constexpr uint32_t ANORM_L_RUN      = 0xFFFFFFFFu;  // w3
constexpr uint32_t ANORM_L_RUN_LSB  = 0u;
constexpr uint32_t ANORM_L_RUN_W    = 32u;
constexpr uint32_t ANORM_L_RUN_WORD = 3u;
constexpr uint32_t ANORM_L_RUN_BIT  = 96u;
constexpr uint32_t ANORM_R          = 0x0000FFFFu;  // w4[15:0]
constexpr uint32_t ANORM_R_LSB      = 0u;
constexpr uint32_t ANORM_R_W        = 16u;
constexpr uint32_t ANORM_R_WORD     = 4u;
constexpr uint32_t ANORM_R_BIT      = 128u;
constexpr uint32_t ANORM_DCODE      = 0x00030000u;  // w4[17:16]
constexpr uint32_t ANORM_DCODE_LSB  = 16u;
constexpr uint32_t ANORM_DCODE_W    = 2u;
constexpr uint32_t ANORM_DCODE_WORD = 4u;
constexpr uint32_t ANORM_DCODE_BIT  = 144u;
// ANORM_DCODE takes the DCODE_* values
constexpr uint32_t ANORM_RSV_W0 = 0xFFFFFF00u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t ANORM_RSV_W1 = 0x00000000u;
constexpr uint32_t ANORM_RSV_W2 = 0x00000000u;
constexpr uint32_t ANORM_RSV_W3 = 0x00000000u;
constexpr uint32_t ANORM_RSV_W4 = 0xFFFC0000u;
constexpr uint32_t ANORM_RSV_W5 = 0xFFFFFFFFu;
constexpr uint32_t ANORM_RSV_W6 = 0xFFFFFFFFu;
constexpr uint32_t ANORM_RSV_W7 = 0xFFFFFFFFu;
constexpr uint32_t ANORM_WORDS  = 0x1Fu;        // words holding a field (w0 always)
struct anorm_f {
    uint32_t out = 0;    // w1 row r at OUT + r x D/16
    uint32_t orun = 0;   // w2
    uint32_t l_run = 0;  // w3
    uint32_t r = 0;      // w4[15:0]
    uint32_t dcode = 0;  // w4[17:16]
};
constexpr op ANORM(const anorm_f& f) {
    op o{};
    o.w[0] = OP_ANORM;
    o.w[1] |= f.out;
    o.w[2] |= f.orun;
    o.w[3] |= f.l_run;
    o.w[4] |= (f.r & 0xFFFFu) << 0;
    o.w[4] |= (f.dcode & 0x3u) << 16;
    return op_finish(o);
}

}  // namespace tsr

#endif  // TSR_AT_HPP
