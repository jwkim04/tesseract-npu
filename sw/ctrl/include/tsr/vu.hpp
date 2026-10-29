// GENERATED FILE, DO NOT EDIT: produced by sw/tables/gen_tables.py from sw/tables/tsr_tables.py.
//
// tsr/vu.hpp — Tesseract v2 tables for the firmware SDK (V2-EX-30, SW §6):
// the VU encodings (VU §5) and their constexpr builders.
// Freestanding C++20: no libstdc++ headers, no heap, everything constexpr in namespace tsr.
// Names as tsr_tables.h without the TSR_ prefix (see sw/tables/gen_tables.py).
// Builders take a field struct in encoding order (C++20 designated initializers; omitted
// fields are 0, as every unused field must be), mask each value to its field width and
// mark every nonzero word in op::used:
//     VOP({.cmode = VOP_CMODE_SREG, .func = VOP_FUNC_MULC, .si = 3, .d = 100, .a = 200, .m = 896, .rows = 1})
#ifndef TSR_VU_HPP
#define TSR_VU_HPP

#include <stdint.h>

#include "mmio.hpp"

namespace tsr {

// -----------------------------------------------------------------------------------------------
// VU encodings (VU §5)
// -----------------------------------------------------------------------------------------------
// VOP 0x30 (VU §5.2, owner VU)
constexpr uint32_t VOP_CMODE         = 0x00000F00u;  // w0[11:8] per-row scalar c
constexpr uint32_t VOP_CMODE_LSB     = 8u;
constexpr uint32_t VOP_CMODE_W       = 4u;
constexpr uint32_t VOP_CMODE_WORD    = 0u;
constexpr uint32_t VOP_CMODE_BIT     = 8u;
constexpr uint32_t VOP_FUNC          = 0x00FF0000u;  // x16[7:0] function (VOP_FUNCS)
constexpr uint32_t VOP_FUNC_LSB      = 16u;
constexpr uint32_t VOP_FUNC_W        = 8u;
constexpr uint32_t VOP_FUNC_WORD     = 0u;
constexpr uint32_t VOP_FUNC_BIT      = 16u;
constexpr uint32_t VOP_SI            = 0x0F000000u;  // x16[11:8] s register for CMODE SREG, else 0
constexpr uint32_t VOP_SI_LSB        = 24u;
constexpr uint32_t VOP_SI_W          = 4u;
constexpr uint32_t VOP_SI_WORD       = 0u;
constexpr uint32_t VOP_SI_BIT        = 24u;
constexpr uint32_t VOP_D             = 0x00FFFFFFu;  // w1[23:0] destination line
constexpr uint32_t VOP_D_LSB         = 0u;
constexpr uint32_t VOP_D_W           = 24u;
constexpr uint32_t VOP_D_WORD        = 1u;
constexpr uint32_t VOP_D_BIT         = 32u;
constexpr uint32_t VOP_A             = 0x00FFFFFFu;  // w2[23:0] source a line (0 if the function reads no a)
constexpr uint32_t VOP_A_LSB         = 0u;
constexpr uint32_t VOP_A_W           = 24u;
constexpr uint32_t VOP_A_WORD        = 2u;
constexpr uint32_t VOP_A_BIT         = 64u;
constexpr uint32_t VOP_B             = 0x00FFFFFFu;  // w3[23:0] source b line (0 if unused)
constexpr uint32_t VOP_B_LSB         = 0u;
constexpr uint32_t VOP_B_W           = 24u;
constexpr uint32_t VOP_B_WORD        = 3u;
constexpr uint32_t VOP_B_BIT         = 96u;
constexpr uint32_t VOP_C             = 0x00FFFFFFu;  // w4[23:0] row-vector line for CMODE VEC, else 0
constexpr uint32_t VOP_C_LSB         = 0u;
constexpr uint32_t VOP_C_W           = 24u;
constexpr uint32_t VOP_C_WORD        = 4u;
constexpr uint32_t VOP_C_BIT         = 128u;
constexpr uint32_t VOP_M             = 0x00FFFFFFu;  // w5[23:0] elements per row
constexpr uint32_t VOP_M_LSB         = 0u;
constexpr uint32_t VOP_M_W           = 24u;
constexpr uint32_t VOP_M_WORD        = 5u;
constexpr uint32_t VOP_M_BIT         = 160u;
constexpr uint32_t VOP_ROWS          = 0x0000FFFFu;  // w6[15:0] rows
constexpr uint32_t VOP_ROWS_LSB      = 0u;
constexpr uint32_t VOP_ROWS_W        = 16u;
constexpr uint32_t VOP_ROWS_WORD     = 6u;
constexpr uint32_t VOP_ROWS_BIT      = 192u;
constexpr uint32_t VOP_D_STRIDE      = 0xFFFF0000u;  // w6[31:16] lines
constexpr uint32_t VOP_D_STRIDE_LSB  = 16u;
constexpr uint32_t VOP_D_STRIDE_W    = 16u;
constexpr uint32_t VOP_D_STRIDE_WORD = 6u;
constexpr uint32_t VOP_D_STRIDE_BIT  = 208u;
constexpr uint32_t VOP_A_STRIDE      = 0x0000FFFFu;  // w7[15:0] lines (0 if a is unused)
constexpr uint32_t VOP_A_STRIDE_LSB  = 0u;
constexpr uint32_t VOP_A_STRIDE_W    = 16u;
constexpr uint32_t VOP_A_STRIDE_WORD = 7u;
constexpr uint32_t VOP_A_STRIDE_BIT  = 224u;
constexpr uint32_t VOP_B_STRIDE      = 0xFFFF0000u;  // w7[31:16] lines (0 if b is unused)
constexpr uint32_t VOP_B_STRIDE_LSB  = 16u;
constexpr uint32_t VOP_B_STRIDE_W    = 16u;
constexpr uint32_t VOP_B_STRIDE_WORD = 7u;
constexpr uint32_t VOP_B_STRIDE_BIT  = 240u;
constexpr uint32_t VOP_CMODE_NONE    = 0u;           // no c
constexpr uint32_t VOP_CMODE_SREG    = 1u;           // c = s[SI]
constexpr uint32_t VOP_CMODE_VEC     = 2u;           // c_r = word r of the vector at line C
constexpr uint32_t VOP_FUNC_ADD      = 0u;           // reads ab, rate F: FA(a, b)
constexpr uint32_t VOP_FUNC_SUB      = 1u;           // reads ab, rate F: FS(a, b)
constexpr uint32_t VOP_FUNC_MUL      = 2u;           // reads ab, rate F: FM(a, b)
constexpr uint32_t VOP_FUNC_MAX      = 3u;           // reads ab, rate F: MAXT(a, b)
constexpr uint32_t VOP_FUNC_MULC     = 4u;           // reads ac, rate F: FM(a, c)
constexpr uint32_t VOP_FUNC_ADDC     = 5u;           // reads ac, rate F: FA(a, c)
constexpr uint32_t VOP_FUNC_SILU     = 6u;           // reads a, rate H: SILU(a)
constexpr uint32_t VOP_FUNC_SWIGLU   = 7u;           // reads ab, rate H: FM(SILU(a), b)
constexpr uint32_t VOP_FUNC_EXP      = 8u;           // reads a, rate F: EXPK(a, P1)
constexpr uint32_t VOP_FUNC_COPY     = 9u;           // reads a, rate F: a (raw)
constexpr uint32_t VOP_FUNC_MIN      = 10u;          // reads ab, rate F: MINT(a, b)
constexpr uint32_t VOP_FUNC_SUBC     = 11u;          // reads ac, rate F: FS(a, c)
constexpr uint32_t VOP_FUNC_MULCB    = 12u;          // reads abc, rate F: FM(FM(a, c), b)
constexpr uint32_t VOP_FUNC_MULCADD  = 13u;          // reads abc, rate F: FA(FM(a, c), b)
constexpr uint32_t VOP_FUNC_ADDMULC  = 14u;          // reads abc, rate F: FM(FA(a, b), c)
constexpr uint32_t VOP_FUNC_EXPSUB   = 15u;          // reads ab, rate F: EXPSUB(a, b)
constexpr uint32_t VOP_FUNC_EXPSUBC  = 0x10u;        // reads ac, rate F: EXPSUB(a, c)
constexpr uint32_t VOP_FUNC_SIGMOID  = 0x11u;        // reads a, rate H: SIG(a)
constexpr uint32_t VOP_FUNC_GELU     = 0x12u;        // reads a, rate H: GELU(a), exact tanh form
constexpr uint32_t VOP_FUNC_GEGLU    = 0x13u;        // reads ab, rate H: FM(GELU(a), b)
constexpr uint32_t VOP_FUNC_TANH     = 0x14u;        // reads a, rate H: TANH(a)
constexpr uint32_t VOP_FUNC_RECIP    = 0x15u;        // reads a, rate H: RC(a), correctly rounded
constexpr uint32_t VOP_FUNC_SETC     = 0x16u;        // reads c, rate F: c (raw)
constexpr uint32_t VOP_FUNC_GELU16   = 0x17u;        // reads a, rate H: ggml-CPU fp16 form (VU §6.5)
constexpr uint32_t VOP_FUNC_GEGLU16  = 0x18u;        // reads ab, rate H: GEGLU16(a, b) (VU §6.5)
constexpr uint32_t VOP_RSV_W0        = 0xF000F000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t VOP_RSV_W1        = 0xFF000000u;
constexpr uint32_t VOP_RSV_W2        = 0xFF000000u;
constexpr uint32_t VOP_RSV_W3        = 0xFF000000u;
constexpr uint32_t VOP_RSV_W4        = 0xFF000000u;
constexpr uint32_t VOP_RSV_W5        = 0xFF000000u;
constexpr uint32_t VOP_RSV_W6        = 0x00000000u;
constexpr uint32_t VOP_RSV_W7        = 0x00000000u;
constexpr uint32_t VOP_WORDS         = 0xFFu;        // words holding a field (w0 always)
struct vop_f {
    uint32_t cmode = 0;     // w0[11:8] per-row scalar c
    uint32_t func = 0;      // x16[7:0] function (VOP_FUNCS)
    uint32_t si = 0;        // x16[11:8] s register for CMODE SREG, else 0
    uint32_t d = 0;         // w1[23:0] destination line
    uint32_t a = 0;         // w2[23:0] source a line (0 if the function reads no a)
    uint32_t b = 0;         // w3[23:0] source b line (0 if unused)
    uint32_t c = 0;         // w4[23:0] row-vector line for CMODE VEC, else 0
    uint32_t m = 0;         // w5[23:0] elements per row
    uint32_t rows = 0;      // w6[15:0] rows
    uint32_t d_stride = 0;  // w6[31:16] lines
    uint32_t a_stride = 0;  // w7[15:0] lines (0 if a is unused)
    uint32_t b_stride = 0;  // w7[31:16] lines (0 if b is unused)
};
constexpr op VOP(const vop_f& f) {
    op o{};
    o.w[0] = OP_VOP;
    o.w[0] |= (f.cmode & 0xFu) << 8;
    o.w[0] |= (f.func & 0xFFu) << 16;
    o.w[0] |= (f.si & 0xFu) << 24;
    o.w[1] |= (f.d & 0xFFFFFFu) << 0;
    o.w[2] |= (f.a & 0xFFFFFFu) << 0;
    o.w[3] |= (f.b & 0xFFFFFFu) << 0;
    o.w[4] |= (f.c & 0xFFFFFFu) << 0;
    o.w[5] |= (f.m & 0xFFFFFFu) << 0;
    o.w[6] |= (f.rows & 0xFFFFu) << 0;
    o.w[6] |= (f.d_stride & 0xFFFFu) << 16;
    o.w[7] |= (f.a_stride & 0xFFFFu) << 0;
    o.w[7] |= (f.b_stride & 0xFFFFu) << 16;
    return op_finish(o);
}
// RED 0x31 (VU §5.3, owner VU)
constexpr uint32_t RED_KIND          = 0x00000F00u;  // w0[11:8]
constexpr uint32_t RED_KIND_LSB      = 8u;
constexpr uint32_t RED_KIND_W        = 4u;
constexpr uint32_t RED_KIND_WORD     = 0u;
constexpr uint32_t RED_KIND_BIT      = 8u;
constexpr uint32_t RED_TO_S          = 0x00001000u;  // w0[12] 1: result to s[SD] (ROWS must be 1); 0: per-row vector at D
constexpr uint32_t RED_TO_S_LSB      = 12u;
constexpr uint32_t RED_TO_S_W        = 1u;
constexpr uint32_t RED_TO_S_WORD     = 0u;
constexpr uint32_t RED_TO_S_BIT      = 12u;
constexpr uint32_t RED_SD            = 0x000F0000u;  // x16[3:0] destination s register if TO_S, else 0
constexpr uint32_t RED_SD_LSB        = 16u;
constexpr uint32_t RED_SD_W          = 4u;
constexpr uint32_t RED_SD_WORD       = 0u;
constexpr uint32_t RED_SD_BIT        = 16u;
constexpr uint32_t RED_POST          = 0x00300000u;  // x16[5:4] post-op; MAX requires NONE
constexpr uint32_t RED_POST_LSB      = 20u;
constexpr uint32_t RED_POST_W        = 2u;
constexpr uint32_t RED_POST_WORD     = 0u;
constexpr uint32_t RED_POST_BIT      = 20u;
constexpr uint32_t RED_SK            = 0x0F000000u;  // x16[11:8] RMS scale register (typically 1/n), else 0
constexpr uint32_t RED_SK_LSB        = 24u;
constexpr uint32_t RED_SK_W          = 4u;
constexpr uint32_t RED_SK_WORD       = 0u;
constexpr uint32_t RED_SK_BIT        = 24u;
constexpr uint32_t RED_SE            = 0xF0000000u;  // x16[15:12] RMS epsilon register, else 0
constexpr uint32_t RED_SE_LSB        = 28u;
constexpr uint32_t RED_SE_W          = 4u;
constexpr uint32_t RED_SE_WORD       = 0u;
constexpr uint32_t RED_SE_BIT        = 28u;
constexpr uint32_t RED_D             = 0x00FFFFFFu;  // w1[23:0] result vector line (rows words); 0 if TO_S
constexpr uint32_t RED_D_LSB         = 0u;
constexpr uint32_t RED_D_W           = 24u;
constexpr uint32_t RED_D_WORD        = 1u;
constexpr uint32_t RED_D_BIT         = 32u;
constexpr uint32_t RED_A             = 0x00FFFFFFu;  // w2[23:0] source line
constexpr uint32_t RED_A_LSB         = 0u;
constexpr uint32_t RED_A_W           = 24u;
constexpr uint32_t RED_A_WORD        = 2u;
constexpr uint32_t RED_A_BIT         = 64u;
constexpr uint32_t RED_M             = 0x00FFFFFFu;  // w5[23:0] elements per row
constexpr uint32_t RED_M_LSB         = 0u;
constexpr uint32_t RED_M_W           = 24u;
constexpr uint32_t RED_M_WORD        = 5u;
constexpr uint32_t RED_M_BIT         = 160u;
constexpr uint32_t RED_ROWS          = 0x0000FFFFu;  // w6[15:0] rows
constexpr uint32_t RED_ROWS_LSB      = 0u;
constexpr uint32_t RED_ROWS_W        = 16u;
constexpr uint32_t RED_ROWS_WORD     = 6u;
constexpr uint32_t RED_ROWS_BIT      = 192u;
constexpr uint32_t RED_A_STRIDE      = 0x0000FFFFu;  // w7[15:0] lines
constexpr uint32_t RED_A_STRIDE_LSB  = 0u;
constexpr uint32_t RED_A_STRIDE_W    = 16u;
constexpr uint32_t RED_A_STRIDE_WORD = 7u;
constexpr uint32_t RED_A_STRIDE_BIT  = 224u;
constexpr uint32_t RED_KIND_SUM      = 0u;
constexpr uint32_t RED_KIND_SUMSQ    = 1u;
constexpr uint32_t RED_KIND_MAX      = 2u;
constexpr uint32_t RED_POST_NONE     = 0u;
constexpr uint32_t RED_POST_RCP      = 1u;           // reciprocal
constexpr uint32_t RED_POST_RMS      = 2u;           // RC(SQ(FA(FM(y, s[SK]), s[SE]))) (VU §6.2)
constexpr uint32_t RED_RSV_W0        = 0x00C0E000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t RED_RSV_W1        = 0xFF000000u;
constexpr uint32_t RED_RSV_W2        = 0xFF000000u;
constexpr uint32_t RED_RSV_W3        = 0xFFFFFFFFu;
constexpr uint32_t RED_RSV_W4        = 0xFFFFFFFFu;
constexpr uint32_t RED_RSV_W5        = 0xFF000000u;
constexpr uint32_t RED_RSV_W6        = 0xFFFF0000u;
constexpr uint32_t RED_RSV_W7        = 0xFFFF0000u;
constexpr uint32_t RED_WORDS         = 0xE7u;        // words holding a field (w0 always)
struct red_f {
    uint32_t kind = 0;      // w0[11:8]
    uint32_t to_s = 0;      // w0[12] 1: result to s[SD] (ROWS must be 1); 0: per-row vector at D
    uint32_t sd = 0;        // x16[3:0] destination s register if TO_S, else 0
    uint32_t post = 0;      // x16[5:4] post-op; MAX requires NONE
    uint32_t sk = 0;        // x16[11:8] RMS scale register (typically 1/n), else 0
    uint32_t se = 0;        // x16[15:12] RMS epsilon register, else 0
    uint32_t d = 0;         // w1[23:0] result vector line (rows words); 0 if TO_S
    uint32_t a = 0;         // w2[23:0] source line
    uint32_t m = 0;         // w5[23:0] elements per row
    uint32_t rows = 0;      // w6[15:0] rows
    uint32_t a_stride = 0;  // w7[15:0] lines
};
constexpr op RED(const red_f& f) {
    op o{};
    o.w[0] = OP_RED;
    o.w[0] |= (f.kind & 0xFu) << 8;
    o.w[0] |= (f.to_s & 0x1u) << 12;
    o.w[0] |= (f.sd & 0xFu) << 16;
    o.w[0] |= (f.post & 0x3u) << 20;
    o.w[0] |= (f.sk & 0xFu) << 24;
    o.w[0] |= (f.se & 0xFu) << 28;
    o.w[1] |= (f.d & 0xFFFFFFu) << 0;
    o.w[2] |= (f.a & 0xFFFFFFu) << 0;
    o.w[5] |= (f.m & 0xFFFFFFu) << 0;
    o.w[6] |= (f.rows & 0xFFFFu) << 0;
    o.w[7] |= (f.a_stride & 0xFFFFu) << 0;
    return op_finish(o);
}
// ROPE 0x32 (VU §5.4, owner VU)
constexpr uint32_t ROPE_MODE            = 0x00000F00u;  // w0[11:8]
constexpr uint32_t ROPE_MODE_LSB        = 8u;
constexpr uint32_t ROPE_MODE_W          = 4u;
constexpr uint32_t ROPE_MODE_WORD       = 0u;
constexpr uint32_t ROPE_MODE_BIT        = 8u;
constexpr uint32_t ROPE_FF              = 0x00001000u;  // w0[12] recurrence form: the table holds freq factors
constexpr uint32_t ROPE_FF_LSB          = 12u;
constexpr uint32_t ROPE_FF_W            = 1u;
constexpr uint32_t ROPE_FF_WORD         = 0u;
constexpr uint32_t ROPE_FF_BIT          = 12u;
constexpr uint32_t ROPE_FS              = 0x00002000u;  // w0[13] recurrence form: multiply by freq_scale in s[SFS]
constexpr uint32_t ROPE_FS_LSB          = 13u;
constexpr uint32_t ROPE_FS_W            = 1u;
constexpr uint32_t ROPE_FS_WORD         = 0u;
constexpr uint32_t ROPE_FS_BIT          = 13u;
constexpr uint32_t ROPE_ATAB            = 0x00004000u;  // w0[14] A-table form (YaRN): FF, FS and STS must be 0
constexpr uint32_t ROPE_ATAB_LSB        = 14u;
constexpr uint32_t ROPE_ATAB_W          = 1u;
constexpr uint32_t ROPE_ATAB_WORD       = 0u;
constexpr uint32_t ROPE_ATAB_BIT        = 14u;
constexpr uint32_t ROPE_SM              = 0x000F0000u;  // x16[3:0] s register holding M
constexpr uint32_t ROPE_SM_LSB          = 16u;
constexpr uint32_t ROPE_SM_W            = 4u;
constexpr uint32_t ROPE_SM_WORD         = 0u;
constexpr uint32_t ROPE_SM_BIT          = 16u;
constexpr uint32_t ROPE_STS             = 0x00F00000u;  // x16[7:4] s register holding TS (recurrence form; 0 with ATAB)
constexpr uint32_t ROPE_STS_LSB         = 20u;
constexpr uint32_t ROPE_STS_W           = 4u;
constexpr uint32_t ROPE_STS_WORD        = 0u;
constexpr uint32_t ROPE_STS_BIT         = 20u;
constexpr uint32_t ROPE_SFS             = 0x0F000000u;  // x16[11:8] s register holding freq_scale when FS, else 0
constexpr uint32_t ROPE_SFS_LSB         = 24u;
constexpr uint32_t ROPE_SFS_W           = 4u;
constexpr uint32_t ROPE_SFS_WORD        = 0u;
constexpr uint32_t ROPE_SFS_BIT         = 24u;
constexpr uint32_t ROPE_D               = 0x00FFFFFFu;  // w1[23:0] destination line of token 0
constexpr uint32_t ROPE_D_LSB           = 0u;
constexpr uint32_t ROPE_D_W             = 24u;
constexpr uint32_t ROPE_D_WORD          = 1u;
constexpr uint32_t ROPE_D_BIT           = 32u;
constexpr uint32_t ROPE_SRC             = 0x00FFFFFFu;  // w2[23:0] source line of token 0
constexpr uint32_t ROPE_SRC_LSB         = 0u;
constexpr uint32_t ROPE_SRC_W           = 24u;
constexpr uint32_t ROPE_SRC_WORD        = 2u;
constexpr uint32_t ROPE_SRC_BIT         = 64u;
constexpr uint32_t ROPE_POS             = 0x00FFFFFFu;  // w3[23:0] line of the I32 position vector
constexpr uint32_t ROPE_POS_LSB         = 0u;
constexpr uint32_t ROPE_POS_W           = 24u;
constexpr uint32_t ROPE_POS_WORD        = 3u;
constexpr uint32_t ROPE_POS_BIT         = 96u;
constexpr uint32_t ROPE_TABLE           = 0x00FFFFFFu;  // w4[23:0] line of the f32 table of n_dims/2 entries; 0 when neither FF nor ATAB
constexpr uint32_t ROPE_TABLE_LSB       = 0u;
constexpr uint32_t ROPE_TABLE_W         = 24u;
constexpr uint32_t ROPE_TABLE_WORD      = 4u;
constexpr uint32_t ROPE_TABLE_BIT       = 128u;
constexpr uint32_t ROPE_HEAD_DIM        = 0x0000FFFFu;  // w5[15:0] elements per head
constexpr uint32_t ROPE_HEAD_DIM_LSB    = 0u;
constexpr uint32_t ROPE_HEAD_DIM_W      = 16u;
constexpr uint32_t ROPE_HEAD_DIM_WORD   = 5u;
constexpr uint32_t ROPE_HEAD_DIM_BIT    = 160u;
constexpr uint32_t ROPE_N_DIMS          = 0xFFFF0000u;  // w5[31:16] rotated elements per head
constexpr uint32_t ROPE_N_DIMS_LSB      = 16u;
constexpr uint32_t ROPE_N_DIMS_W        = 16u;
constexpr uint32_t ROPE_N_DIMS_WORD     = 5u;
constexpr uint32_t ROPE_N_DIMS_BIT      = 176u;
constexpr uint32_t ROPE_N_HEADS         = 0x0000FFFFu;  // w6[15:0] heads per token
constexpr uint32_t ROPE_N_HEADS_LSB     = 0u;
constexpr uint32_t ROPE_N_HEADS_W       = 16u;
constexpr uint32_t ROPE_N_HEADS_WORD    = 6u;
constexpr uint32_t ROPE_N_HEADS_BIT     = 192u;
constexpr uint32_t ROPE_T               = 0xFFFF0000u;  // w6[31:16] tokens
constexpr uint32_t ROPE_T_LSB           = 16u;
constexpr uint32_t ROPE_T_W             = 16u;
constexpr uint32_t ROPE_T_WORD          = 6u;
constexpr uint32_t ROPE_T_BIT           = 208u;
constexpr uint32_t ROPE_SRC_STRIDE      = 0x0000FFFFu;  // w7[15:0] lines between tokens; 0 = H
constexpr uint32_t ROPE_SRC_STRIDE_LSB  = 0u;
constexpr uint32_t ROPE_SRC_STRIDE_W    = 16u;
constexpr uint32_t ROPE_SRC_STRIDE_WORD = 7u;
constexpr uint32_t ROPE_SRC_STRIDE_BIT  = 224u;
constexpr uint32_t ROPE_D_STRIDE        = 0xFFFF0000u;  // w7[31:16] lines between tokens; 0 = H
constexpr uint32_t ROPE_D_STRIDE_LSB    = 16u;
constexpr uint32_t ROPE_D_STRIDE_W      = 16u;
constexpr uint32_t ROPE_D_STRIDE_WORD   = 7u;
constexpr uint32_t ROPE_D_STRIDE_BIT    = 240u;
constexpr uint32_t ROPE_MODE_NORM       = 0u;           // adjacent pairs
constexpr uint32_t ROPE_MODE_NEOX       = 1u;           // half-split pairs
constexpr uint32_t ROPE_RSV_W0          = 0xF0008000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t ROPE_RSV_W1          = 0xFF000000u;
constexpr uint32_t ROPE_RSV_W2          = 0xFF000000u;
constexpr uint32_t ROPE_RSV_W3          = 0xFF000000u;
constexpr uint32_t ROPE_RSV_W4          = 0xFF000000u;
constexpr uint32_t ROPE_RSV_W5          = 0x00000000u;
constexpr uint32_t ROPE_RSV_W6          = 0x00000000u;
constexpr uint32_t ROPE_RSV_W7          = 0x00000000u;
constexpr uint32_t ROPE_WORDS           = 0xFFu;        // words holding a field (w0 always)
struct rope_f {
    uint32_t mode = 0;        // w0[11:8]
    uint32_t ff = 0;          // w0[12] recurrence form: the table holds freq factors
    uint32_t fs = 0;          // w0[13] recurrence form: multiply by freq_scale in s[SFS]
    uint32_t atab = 0;        // w0[14] A-table form (YaRN): FF, FS and STS must be 0
    uint32_t sm = 0;          // x16[3:0] s register holding M
    uint32_t sts = 0;         // x16[7:4] s register holding TS (recurrence form; 0 with ATAB)
    uint32_t sfs = 0;         // x16[11:8] s register holding freq_scale when FS, else 0
    uint32_t d = 0;           // w1[23:0] destination line of token 0
    uint32_t src = 0;         // w2[23:0] source line of token 0
    uint32_t pos = 0;         // w3[23:0] line of the I32 position vector
    uint32_t table = 0;       // w4[23:0] line of the f32 table of n_dims/2 entries; 0 when neither FF nor ATAB
    uint32_t head_dim = 0;    // w5[15:0] elements per head
    uint32_t n_dims = 0;      // w5[31:16] rotated elements per head
    uint32_t n_heads = 0;     // w6[15:0] heads per token
    uint32_t t = 0;           // w6[31:16] tokens
    uint32_t src_stride = 0;  // w7[15:0] lines between tokens; 0 = H
    uint32_t d_stride = 0;    // w7[31:16] lines between tokens; 0 = H
};
constexpr op ROPE(const rope_f& f) {
    op o{};
    o.w[0] = OP_ROPE;
    o.w[0] |= (f.mode & 0xFu) << 8;
    o.w[0] |= (f.ff & 0x1u) << 12;
    o.w[0] |= (f.fs & 0x1u) << 13;
    o.w[0] |= (f.atab & 0x1u) << 14;
    o.w[0] |= (f.sm & 0xFu) << 16;
    o.w[0] |= (f.sts & 0xFu) << 20;
    o.w[0] |= (f.sfs & 0xFu) << 24;
    o.w[1] |= (f.d & 0xFFFFFFu) << 0;
    o.w[2] |= (f.src & 0xFFFFFFu) << 0;
    o.w[3] |= (f.pos & 0xFFFFFFu) << 0;
    o.w[4] |= (f.table & 0xFFFFFFu) << 0;
    o.w[5] |= (f.head_dim & 0xFFFFu) << 0;
    o.w[5] |= (f.n_dims & 0xFFFFu) << 16;
    o.w[6] |= (f.n_heads & 0xFFFFu) << 0;
    o.w[6] |= (f.t & 0xFFFFu) << 16;
    o.w[7] |= (f.src_stride & 0xFFFFu) << 0;
    o.w[7] |= (f.d_stride & 0xFFFFu) << 16;
    return op_finish(o);
}
// TOPK 0x33 (VU §5.5, owner VU)
constexpr uint32_t TOPK_K             = 0x001F0000u;  // x16[4:0] 1..16
constexpr uint32_t TOPK_K_LSB         = 16u;
constexpr uint32_t TOPK_K_W           = 5u;
constexpr uint32_t TOPK_K_WORD        = 0u;
constexpr uint32_t TOPK_K_BIT         = 16u;
constexpr uint32_t TOPK_D             = 0x00FFFFFFu;  // w1[23:0] output line of row 0: k (index, value) pairs
constexpr uint32_t TOPK_D_LSB         = 0u;
constexpr uint32_t TOPK_D_W           = 24u;
constexpr uint32_t TOPK_D_WORD        = 1u;
constexpr uint32_t TOPK_D_BIT         = 32u;
constexpr uint32_t TOPK_A             = 0x00FFFFFFu;  // w2[23:0] source line
constexpr uint32_t TOPK_A_LSB         = 0u;
constexpr uint32_t TOPK_A_W           = 24u;
constexpr uint32_t TOPK_A_WORD        = 2u;
constexpr uint32_t TOPK_A_BIT         = 64u;
constexpr uint32_t TOPK_M             = 0x00FFFFFFu;  // w5[23:0] elements per row
constexpr uint32_t TOPK_M_LSB         = 0u;
constexpr uint32_t TOPK_M_W           = 24u;
constexpr uint32_t TOPK_M_WORD        = 5u;
constexpr uint32_t TOPK_M_BIT         = 160u;
constexpr uint32_t TOPK_ROWS          = 0x0000FFFFu;  // w6[15:0] rows
constexpr uint32_t TOPK_ROWS_LSB      = 0u;
constexpr uint32_t TOPK_ROWS_W        = 16u;
constexpr uint32_t TOPK_ROWS_WORD     = 6u;
constexpr uint32_t TOPK_ROWS_BIT      = 192u;
constexpr uint32_t TOPK_D_STRIDE      = 0xFFFF0000u;  // w6[31:16] lines; 0 = ceil(k/8)
constexpr uint32_t TOPK_D_STRIDE_LSB  = 16u;
constexpr uint32_t TOPK_D_STRIDE_W    = 16u;
constexpr uint32_t TOPK_D_STRIDE_WORD = 6u;
constexpr uint32_t TOPK_D_STRIDE_BIT  = 208u;
constexpr uint32_t TOPK_A_STRIDE      = 0x0000FFFFu;  // w7[15:0] lines
constexpr uint32_t TOPK_A_STRIDE_LSB  = 0u;
constexpr uint32_t TOPK_A_STRIDE_W    = 16u;
constexpr uint32_t TOPK_A_STRIDE_WORD = 7u;
constexpr uint32_t TOPK_A_STRIDE_BIT  = 224u;
constexpr uint32_t TOPK_RSV_W0        = 0xFFE0FF00u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t TOPK_RSV_W1        = 0xFF000000u;
constexpr uint32_t TOPK_RSV_W2        = 0xFF000000u;
constexpr uint32_t TOPK_RSV_W3        = 0xFFFFFFFFu;
constexpr uint32_t TOPK_RSV_W4        = 0xFFFFFFFFu;
constexpr uint32_t TOPK_RSV_W5        = 0xFF000000u;
constexpr uint32_t TOPK_RSV_W6        = 0x00000000u;
constexpr uint32_t TOPK_RSV_W7        = 0xFFFF0000u;
constexpr uint32_t TOPK_WORDS         = 0xE7u;        // words holding a field (w0 always)
struct topk_f {
    uint32_t k = 0;         // x16[4:0] 1..16
    uint32_t d = 0;         // w1[23:0] output line of row 0: k (index, value) pairs
    uint32_t a = 0;         // w2[23:0] source line
    uint32_t m = 0;         // w5[23:0] elements per row
    uint32_t rows = 0;      // w6[15:0] rows
    uint32_t d_stride = 0;  // w6[31:16] lines; 0 = ceil(k/8)
    uint32_t a_stride = 0;  // w7[15:0] lines
};
constexpr op TOPK(const topk_f& f) {
    op o{};
    o.w[0] = OP_TOPK;
    o.w[0] |= (f.k & 0x1Fu) << 16;
    o.w[1] |= (f.d & 0xFFFFFFu) << 0;
    o.w[2] |= (f.a & 0xFFFFFFu) << 0;
    o.w[5] |= (f.m & 0xFFFFFFu) << 0;
    o.w[6] |= (f.rows & 0xFFFFu) << 0;
    o.w[6] |= (f.d_stride & 0xFFFFu) << 16;
    o.w[7] |= (f.a_stride & 0xFFFFu) << 0;
    return op_finish(o);
}
// DQROW 0x12 (VU §5.6, owner VU)
constexpr uint32_t DQROW_FMT             = 0x00000F00u;  // w0[11:8] 2..15 BAD_FMT
constexpr uint32_t DQROW_FMT_LSB         = 8u;
constexpr uint32_t DQROW_FMT_W           = 4u;
constexpr uint32_t DQROW_FMT_WORD        = 0u;
constexpr uint32_t DQROW_FMT_BIT         = 8u;
constexpr uint32_t DQROW_D               = 0x00FFFFFFu;  // w1[23:0] f32 output line of row 0
constexpr uint32_t DQROW_D_LSB           = 0u;
constexpr uint32_t DQROW_D_W             = 24u;
constexpr uint32_t DQROW_D_WORD          = 1u;
constexpr uint32_t DQROW_D_BIT           = 32u;
constexpr uint32_t DQROW_SRC             = 0x00FFFFFFu;  // w2[23:0] SPM line of row 0's repacked bytes
constexpr uint32_t DQROW_SRC_LSB         = 0u;
constexpr uint32_t DQROW_SRC_W           = 24u;
constexpr uint32_t DQROW_SRC_WORD        = 2u;
constexpr uint32_t DQROW_SRC_BIT         = 64u;
constexpr uint32_t DQROW_K               = 0xFFFFFFFFu;  // w5 elements per row (K_RANGE checks)
constexpr uint32_t DQROW_K_LSB           = 0u;
constexpr uint32_t DQROW_K_W             = 32u;
constexpr uint32_t DQROW_K_WORD          = 5u;
constexpr uint32_t DQROW_K_BIT           = 160u;
constexpr uint32_t DQROW_ROWS            = 0x0000FFFFu;  // w6[15:0] rows
constexpr uint32_t DQROW_ROWS_LSB        = 0u;
constexpr uint32_t DQROW_ROWS_W          = 16u;
constexpr uint32_t DQROW_ROWS_WORD       = 6u;
constexpr uint32_t DQROW_ROWS_BIT        = 192u;
constexpr uint32_t DQROW_SRC_STRIDE      = 0x0000FFFFu;  // w7[15:0] lines; 0 = ROW_BEATS(fmt, K)
constexpr uint32_t DQROW_SRC_STRIDE_LSB  = 0u;
constexpr uint32_t DQROW_SRC_STRIDE_W    = 16u;
constexpr uint32_t DQROW_SRC_STRIDE_WORD = 7u;
constexpr uint32_t DQROW_SRC_STRIDE_BIT  = 224u;
constexpr uint32_t DQROW_D_STRIDE        = 0xFFFF0000u;  // w7[31:16] lines; 0 = L(K)
constexpr uint32_t DQROW_D_STRIDE_LSB    = 16u;
constexpr uint32_t DQROW_D_STRIDE_W      = 16u;
constexpr uint32_t DQROW_D_STRIDE_WORD   = 7u;
constexpr uint32_t DQROW_D_STRIDE_BIT    = 240u;
// DQROW_FMT takes the FMT_* values
constexpr uint32_t DQROW_RSV_W0 = 0xFFFFF000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t DQROW_RSV_W1 = 0xFF000000u;
constexpr uint32_t DQROW_RSV_W2 = 0xFF000000u;
constexpr uint32_t DQROW_RSV_W3 = 0xFFFFFFFFu;
constexpr uint32_t DQROW_RSV_W4 = 0xFFFFFFFFu;
constexpr uint32_t DQROW_RSV_W5 = 0x00000000u;
constexpr uint32_t DQROW_RSV_W6 = 0xFFFF0000u;
constexpr uint32_t DQROW_RSV_W7 = 0x00000000u;
constexpr uint32_t DQROW_WORDS  = 0xE7u;        // words holding a field (w0 always)
struct dqrow_f {
    uint32_t fmt = 0;         // w0[11:8] 2..15 BAD_FMT
    uint32_t d = 0;           // w1[23:0] f32 output line of row 0
    uint32_t src = 0;         // w2[23:0] SPM line of row 0's repacked bytes
    uint32_t k = 0;           // w5 elements per row (K_RANGE checks)
    uint32_t rows = 0;        // w6[15:0] rows
    uint32_t src_stride = 0;  // w7[15:0] lines; 0 = ROW_BEATS(fmt, K)
    uint32_t d_stride = 0;    // w7[31:16] lines; 0 = L(K)
};
constexpr op DQROW(const dqrow_f& f) {
    op o{};
    o.w[0] = OP_DQROW;
    o.w[0] |= (f.fmt & 0xFu) << 8;
    o.w[1] |= (f.d & 0xFFFFFFu) << 0;
    o.w[2] |= (f.src & 0xFFFFFFu) << 0;
    o.w[5] |= f.k;
    o.w[6] |= (f.rows & 0xFFFFu) << 0;
    o.w[7] |= (f.src_stride & 0xFFFFu) << 0;
    o.w[7] |= (f.d_stride & 0xFFFFu) << 16;
    return op_finish(o);
}
// VMASK 0x38 (VU §5.7, owner VU)
constexpr uint32_t VMASK_SUB           = 0x00000F00u;  // w0[11:8]
constexpr uint32_t VMASK_SUB_LSB       = 8u;
constexpr uint32_t VMASK_SUB_W         = 4u;
constexpr uint32_t VMASK_SUB_WORD      = 0u;
constexpr uint32_t VMASK_SUB_BIT       = 8u;
constexpr uint32_t VMASK_X             = 0x00FFFFFFu;  // w1[23:0] operand line (in place)
constexpr uint32_t VMASK_X_LSB         = 0u;
constexpr uint32_t VMASK_X_W           = 24u;
constexpr uint32_t VMASK_X_WORD        = 1u;
constexpr uint32_t VMASK_X_BIT         = 32u;
constexpr uint32_t VMASK_WIN           = 0xFFFFFFFFu;  // w2 window size W, u32; 0 = no window
constexpr uint32_t VMASK_WIN_LSB       = 0u;
constexpr uint32_t VMASK_WIN_W         = 32u;
constexpr uint32_t VMASK_WIN_WORD      = 2u;
constexpr uint32_t VMASK_WIN_BIT       = 64u;
constexpr uint32_t VMASK_K0            = 0xFFFFFFFFu;  // w3 i32 position of column 0
constexpr uint32_t VMASK_K0_LSB        = 0u;
constexpr uint32_t VMASK_K0_W          = 32u;
constexpr uint32_t VMASK_K0_WORD       = 3u;
constexpr uint32_t VMASK_K0_BIT        = 96u;
constexpr uint32_t VMASK_Q0            = 0xFFFFFFFFu;  // w4 i32 query position of row group 0
constexpr uint32_t VMASK_Q0_LSB        = 0u;
constexpr uint32_t VMASK_Q0_W          = 32u;
constexpr uint32_t VMASK_Q0_WORD       = 4u;
constexpr uint32_t VMASK_Q0_BIT        = 128u;
constexpr uint32_t VMASK_M             = 0x00FFFFFFu;  // w5[23:0] columns
constexpr uint32_t VMASK_M_LSB         = 0u;
constexpr uint32_t VMASK_M_W           = 24u;
constexpr uint32_t VMASK_M_WORD        = 5u;
constexpr uint32_t VMASK_M_BIT         = 160u;
constexpr uint32_t VMASK_ROWS          = 0x0000FFFFu;  // w6[15:0] rows
constexpr uint32_t VMASK_ROWS_LSB      = 0u;
constexpr uint32_t VMASK_ROWS_W        = 16u;
constexpr uint32_t VMASK_ROWS_WORD     = 6u;
constexpr uint32_t VMASK_ROWS_BIT      = 192u;
constexpr uint32_t VMASK_G             = 0xFFFF0000u;  // w6[31:16] rows per query position (>= 1)
constexpr uint32_t VMASK_G_LSB         = 16u;
constexpr uint32_t VMASK_G_W           = 16u;
constexpr uint32_t VMASK_G_WORD        = 6u;
constexpr uint32_t VMASK_G_BIT         = 208u;
constexpr uint32_t VMASK_X_STRIDE      = 0x0000FFFFu;  // w7[15:0] lines
constexpr uint32_t VMASK_X_STRIDE_LSB  = 0u;
constexpr uint32_t VMASK_X_STRIDE_W    = 16u;
constexpr uint32_t VMASK_X_STRIDE_WORD = 7u;
constexpr uint32_t VMASK_X_STRIDE_BIT  = 224u;
constexpr uint32_t VMASK_P             = 0xFFFF0000u;  // w7[31:16] row period; 0 = none
constexpr uint32_t VMASK_P_LSB         = 16u;
constexpr uint32_t VMASK_P_W           = 16u;
constexpr uint32_t VMASK_P_WORD        = 7u;
constexpr uint32_t VMASK_P_BIT         = 240u;
constexpr uint32_t VMASK_SUB_WINDOW    = 0u;           // position window; others ILLEGAL_OP
constexpr uint32_t VMASK_RSV_W0        = 0xFFFFF000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t VMASK_RSV_W1        = 0xFF000000u;
constexpr uint32_t VMASK_RSV_W2        = 0x00000000u;
constexpr uint32_t VMASK_RSV_W3        = 0x00000000u;
constexpr uint32_t VMASK_RSV_W4        = 0x00000000u;
constexpr uint32_t VMASK_RSV_W5        = 0xFF000000u;
constexpr uint32_t VMASK_RSV_W6        = 0x00000000u;
constexpr uint32_t VMASK_RSV_W7        = 0x00000000u;
constexpr uint32_t VMASK_WORDS         = 0xFFu;        // words holding a field (w0 always)
struct vmask_f {
    uint32_t sub = 0;       // w0[11:8]
    uint32_t x = 0;         // w1[23:0] operand line (in place)
    uint32_t win = 0;       // w2 window size W, u32; 0 = no window
    uint32_t k0 = 0;        // w3 i32 position of column 0
    uint32_t q0 = 0;        // w4 i32 query position of row group 0
    uint32_t m = 0;         // w5[23:0] columns
    uint32_t rows = 0;      // w6[15:0] rows
    uint32_t g = 0;         // w6[31:16] rows per query position (>= 1)
    uint32_t x_stride = 0;  // w7[15:0] lines
    uint32_t p = 0;         // w7[31:16] row period; 0 = none
};
constexpr op VMASK(const vmask_f& f) {
    op o{};
    o.w[0] = OP_VMASK;
    o.w[0] |= (f.sub & 0xFu) << 8;
    o.w[1] |= (f.x & 0xFFFFFFu) << 0;
    o.w[2] |= f.win;
    o.w[3] |= f.k0;
    o.w[4] |= f.q0;
    o.w[5] |= (f.m & 0xFFFFFFu) << 0;
    o.w[6] |= (f.rows & 0xFFFFu) << 0;
    o.w[6] |= (f.g & 0xFFFFu) << 16;
    o.w[7] |= (f.x_stride & 0xFFFFu) << 0;
    o.w[7] |= (f.p & 0xFFFFu) << 16;
    return op_finish(o);
}
// SOP 0x40 (VU §5.8, owner VU)
constexpr uint32_t SOP_SUB       = 0x00000F00u;  // w0[11:8] function (SOP_FUNCS)
constexpr uint32_t SOP_SUB_LSB   = 8u;
constexpr uint32_t SOP_SUB_W     = 4u;
constexpr uint32_t SOP_SUB_WORD  = 0u;
constexpr uint32_t SOP_SUB_BIT   = 8u;
constexpr uint32_t SOP_SD        = 0x0000000Fu;  // w1[3:0] destination s register
constexpr uint32_t SOP_SD_LSB    = 0u;
constexpr uint32_t SOP_SD_W      = 4u;
constexpr uint32_t SOP_SD_WORD   = 1u;
constexpr uint32_t SOP_SD_BIT    = 32u;
constexpr uint32_t SOP_SA        = 0x000000F0u;  // w1[7:4] source s register a
constexpr uint32_t SOP_SA_LSB    = 4u;
constexpr uint32_t SOP_SA_W      = 4u;
constexpr uint32_t SOP_SA_WORD   = 1u;
constexpr uint32_t SOP_SA_BIT    = 36u;
constexpr uint32_t SOP_SB        = 0x00000F00u;  // w1[11:8] source s register b
constexpr uint32_t SOP_SB_LSB    = 8u;
constexpr uint32_t SOP_SB_W      = 4u;
constexpr uint32_t SOP_SB_WORD   = 1u;
constexpr uint32_t SOP_SB_BIT    = 40u;
constexpr uint32_t SOP_IMM       = 0xFFFFFFFFu;  // w2 f32 bits
constexpr uint32_t SOP_IMM_LSB   = 0u;
constexpr uint32_t SOP_IMM_W     = 32u;
constexpr uint32_t SOP_IMM_WORD  = 2u;
constexpr uint32_t SOP_IMM_BIT   = 64u;
constexpr uint32_t SOP_ADDR      = 0x0FFFFFFFu;  // w3[27:0] SPM word address (LDS / STS)
constexpr uint32_t SOP_ADDR_LSB  = 0u;
constexpr uint32_t SOP_ADDR_W    = 28u;
constexpr uint32_t SOP_ADDR_WORD = 3u;
constexpr uint32_t SOP_ADDR_BIT  = 96u;
constexpr uint32_t SOP_SUB_MOVI  = 0u;           // sd <- imm (raw) (uses SD, IMM)
constexpr uint32_t SOP_SUB_ADD   = 1u;           // sd <- FA(sa, sb) (uses SD, SA, SB)
constexpr uint32_t SOP_SUB_MUL   = 2u;           // sd <- FM(sa, sb) (uses SD, SA, SB)
constexpr uint32_t SOP_SUB_ADDI  = 3u;           // sd <- FA(sa, imm) (uses SD, SA, IMM)
constexpr uint32_t SOP_SUB_MULI  = 4u;           // sd <- FM(sa, imm) (uses SD, SA, IMM)
constexpr uint32_t SOP_SUB_SQRT  = 5u;           // sd <- SQ(sa) (uses SD, SA)
constexpr uint32_t SOP_SUB_RECIP = 6u;           // sd <- RC(sa) (uses SD, SA)
constexpr uint32_t SOP_SUB_LDS   = 7u;           // sd <- SPM word (raw) (uses SD, ADDR)
constexpr uint32_t SOP_SUB_STS   = 8u;           // SPM word <- sa (raw; exactly 4 bytes) (uses SA, ADDR)
constexpr uint32_t SOP_SUB_SUB   = 9u;           // sd <- FS(sa, sb) (uses SD, SA, SB)
constexpr uint32_t SOP_SUB_DIV   = 10u;          // sd <- FD(sa, sb) (uses SD, SA, SB)
constexpr uint32_t SOP_SUB_RSQRT = 11u;          // sd <- RC(SQ(sa)), two roundings (uses SD, SA)
constexpr uint32_t SOP_SUB_MAX   = 12u;          // sd <- MAXT(sa, sb) (uses SD, SA, SB)
constexpr uint32_t SOP_SUB_MIN   = 13u;          // sd <- MINT(sa, sb) (uses SD, SA, SB)
constexpr uint32_t SOP_SUB_EXP   = 14u;          // sd <- EXPK(sa, P1) (uses SD, SA)
constexpr uint32_t SOP_RSV_W0    = 0xFFFFF000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t SOP_RSV_W1    = 0xFFFFF000u;
constexpr uint32_t SOP_RSV_W2    = 0x00000000u;
constexpr uint32_t SOP_RSV_W3    = 0xF0000000u;
constexpr uint32_t SOP_RSV_W4    = 0xFFFFFFFFu;
constexpr uint32_t SOP_RSV_W5    = 0xFFFFFFFFu;
constexpr uint32_t SOP_RSV_W6    = 0xFFFFFFFFu;
constexpr uint32_t SOP_RSV_W7    = 0xFFFFFFFFu;
constexpr uint32_t SOP_WORDS     = 15u;          // words holding a field (w0 always)
struct sop_f {
    uint32_t sub = 0;   // w0[11:8] function (SOP_FUNCS)
    uint32_t sd = 0;    // w1[3:0] destination s register
    uint32_t sa = 0;    // w1[7:4] source s register a
    uint32_t sb = 0;    // w1[11:8] source s register b
    uint32_t imm = 0;   // w2 f32 bits
    uint32_t addr = 0;  // w3[27:0] SPM word address (LDS / STS)
};
constexpr op SOP(const sop_f& f) {
    op o{};
    o.w[0] = OP_SOP;
    o.w[0] |= (f.sub & 0xFu) << 8;
    o.w[1] |= (f.sd & 0xFu) << 0;
    o.w[1] |= (f.sa & 0xFu) << 4;
    o.w[1] |= (f.sb & 0xFu) << 8;
    o.w[2] |= f.imm;
    o.w[3] |= (f.addr & 0xFFFFFFFu) << 0;
    return op_finish(o);
}
constexpr uint32_t VOP_FUNC_N = 25u;   // function codes 0..24 are defined
constexpr uint32_t VOP_READS_A = 1u, VOP_READS_B = 2u, VOP_READS_C = 4u;
// Operands a VOP function reads; an operand it does not read must be 0 (VU §5.2).
constexpr uint32_t vop_func_reads(uint32_t func) {
    switch (func) {
    case 0x00u: case 0x01u: case 0x02u: case 0x03u: case 0x07u: case 0x0Au: case 0x0Fu: case 0x13u:
    case 0x18u: return 3u;
    case 0x04u: case 0x05u: case 0x0Bu: case 0x10u: return 5u;
    case 0x06u: case 0x08u: case 0x09u: case 0x11u: case 0x12u: case 0x14u: case 0x15u: case 0x17u:
    return 1u;
    case 0x0Cu: case 0x0Du: case 0x0Eu: return 7u;
    case 0x16u: return 4u;
    default: return 0u;
    }
}
constexpr bool vop_func_half(uint32_t func) {
    switch (func) {
    case 0x06u: case 0x07u: case 0x11u: case 0x12u: case 0x13u: case 0x14u: case 0x15u: case 0x17u:
    case 0x18u: return true;
    default: return false;
    }
}
constexpr uint32_t SOP_SUB_N = 15u;
constexpr uint32_t SOP_USES_SD = 1u, SOP_USES_SA = 2u, SOP_USES_SB = 4u, SOP_USES_IMM = 8u, SOP_USES_ADDR = 16u;
// Fields an SOP function uses; the others must be 0 (VU §5.8).
constexpr uint32_t sop_uses(uint32_t sub) {
    switch (sub) {
    case 0u: return 9u;
    case 1u: case 2u: case 9u: case 10u: case 12u: case 13u: return 7u;
    case 3u: case 4u: return 11u;
    case 5u: case 6u: case 11u: case 14u: return 3u;
    case 7u: return 17u;
    case 8u: return 18u;
    default: return 0u;
    }
}

}  // namespace tsr

#endif  // TSR_VU_HPP
