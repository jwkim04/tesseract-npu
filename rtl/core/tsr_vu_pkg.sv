// tsr_vu_pkg.sv — constants, lane micro-ops and internal structs of the vector unit (area VU).
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §6 numerics, §7 microarchitecture; W4-B task "vu").
// The ISA-level fields come from the generated tsr_tables_pkg (TSR_VOP_*, TSR_RED_*, ...); this
// package holds what only the VU's blocks share: f32 constants of VU Appendix A, the fixed latencies
// of the arithmetic blocks, the lane micro-op codes and the slot / sub-slot structs of tsr_vu.

package tsr_vu_pkg;

  /* verilator lint_off UNUSEDPARAM */
  // f32 constants (VU Appendix A, §6.5, AT §5.5-§5.7)
  localparam logic [31:0] TSR_VU_ZERO   = 32'h0000_0000;
  localparam logic [31:0] TSR_VU_ONE    = 32'h3F80_0000;
  localparam logic [31:0] TSR_VU_HALF   = 32'h3F00_0000;
  localparam logic [31:0] TSR_VU_THREE  = 32'h4040_0000;
  localparam logic [31:0] TSR_VU_THIRD  = 32'h3EAA_AAAB;
  localparam logic [31:0] TSR_VU_KG     = 32'h41B2_E92F;
  localparam logic [31:0] TSR_VU_TEN    = 32'h4120_0000;
  localparam logic [31:0] TSR_VU_INF    = 32'h7F80_0000;
  localparam logic [31:0] TSR_VU_NINF   = 32'hFF80_0000;
  localparam logic [31:0] TSR_VU_NAN    = 32'h7FC0_0000;
  localparam logic [31:0] TSR_VU_MFLOOR = 32'hF180_0000;   // AT §6.3 M_FLOOR = -2^100
  localparam logic [31:0] TSR_VU_LFLOOR = 32'h0080_0000;   // AT §5.7: 2^-126

  // EXPK κ codes (VU §6.3)
  localparam logic [1:0] TSR_VU_K_P1 = 2'd0;
  localparam logic [1:0] TSR_VU_K_N1 = 2'd1;
  localparam logic [1:0] TSR_VU_K_N2 = 2'd2;
  localparam logic [1:0] TSR_VU_K_NG = 2'd3;

  // Fixed latencies (cycles from the input register's D side to the output register's Q side).
  localparam int unsigned TSR_VU_EXP_LAT  = 26;   // tsr_vu_exp
  localparam int unsigned TSR_VU_RCP_LAT  = 21;   // tsr_vu_rcp
  localparam int unsigned TSR_VU_SQRT_LAT = 28;   // tsr_vu_sqrt
  localparam int unsigned TSR_VU_SC_LAT   = 56;   // tsr_vu_sincos
  localparam int unsigned TSR_VU_FMUL_LAT = 4;    // tsr_fp_mul in the VU (its 300 MHz default)
  localparam int unsigned TSR_VU_FADD_LAT = 5;    // tsr_fp_add in the VU
  // Lane exits, counted from the lane's input register (cycle 0, tsr_vu_lane header)
  // One four-lane local batch. Local operations reserve the bank before injecting.
  typedef struct packed {
    logic [3:0] valid;
    logic       scale_only;
    logic       tanh_op;
    logic [3:0][31:0] a;
    logic [31:0] max_value;
    logic [31:0] scale;
  } tsr_vu_local_req_t;

  typedef struct packed {
    logic [3:0] valid;
    logic [3:0][31:0] y;
    logic [1:0] back_valid;
    logic [1:0][31:0] back_y;
  } tsr_vu_local_rsp_t;

  localparam int unsigned TSR_VU_LF = 42;         // front exit (full-rate functions)
  localparam int unsigned TSR_VU_LB = 78;         // back exit (back-lane functions)

  // Lane micro-ops (tsr_vu_lane): one per element. The VOP functions keep their order (func + 1).
  typedef enum logic [5:0] {
    U_NOP     = 6'd0,
    U_ADD     = 6'd1,  U_SUB     = 6'd2,  U_MUL     = 6'd3,  U_MAX     = 6'd4,  U_MULC    = 6'd5,
    U_ADDC    = 6'd6,  U_SILU    = 6'd7,  U_SWIGLU  = 6'd8,  U_EXP     = 6'd9,  U_COPY    = 6'd10,
    U_MIN     = 6'd11, U_SUBC    = 6'd12, U_MULCB   = 6'd13, U_MULCADD = 6'd14, U_ADDMULC = 6'd15,
    U_EXPSUB  = 6'd16, U_EXPSUBC = 6'd17, U_SIGMOID = 6'd18, U_GELU    = 6'd19, U_GEGLU   = 6'd20,
    U_TANH    = 6'd21, U_RECIP   = 6'd22, U_SETC    = 6'd23, U_GELU16  = 6'd24, U_GEGLU16 = 6'd25,
    // reductions (accumulate into the lane's partials; no exit)
    U_RSUM    = 6'd32, U_RSUMSQ  = 6'd33, U_RMAX    = 6'd34,
    // ROPE rotation / AMERGE combine: FA(FM(a, c1), ±FM(b, c2)) (neg: subtract)
    U_ROT     = 6'd35,
    // AT's vector ops (AT §5.4-§5.7)
    U_ASMAX   = 6'd36,     // EXPK(FM(FS(a, c1), c2), P1)
    U_ESCB    = 6'd37,     // EXPK(FM(FS(a, b), c2), P1)   (AMERGE prepass a / b)
    U_AMASK   = 6'd38,     // b == -inf ? -inf : FA(a, b); MAX-accumulate with M_FLOOR
    U_RCPMAX  = 6'd39      // RC(MAXT(a, 2^-126))          (ANORM prepass)
    // DQROW (VU §6.8) runs on U_ADDMULC (Q8_0) and U_ROT (Q4_K), see tsr_vu_dq
  } tsr_vu_uop_e;

  // F32 of a 4-bit unsigned integer (DQROW Q4_K quants; exact)
  function automatic logic [31:0] tsr_vu_u4_to_f32(input logic [3:0] q);
    logic [1:0]  p;
    logic [22:0] f;
    if (q == 4'd0) return 32'd0;
    p = q[3] ? 2'd3 : q[2] ? 2'd2 : q[1] ? 2'd1 : 2'd0;           // msb index
    f = 23'(32'(q) << (5'd23 - 5'(p)));                            // the leading one drops out at bit 23
    return {1'b0, 8'd127 + 8'(p), f};
  endfunction

  function automatic tsr_vu_uop_e tsr_vu_uop_of_func(input logic [4:0] func);   // func <= 0x18
    return tsr_vu_uop_e'(6'(func) + 6'd1);
  endfunction

  // -------------------------------------------------------------------------------------------
  // Half-width bank controls accompanying the selected operands.
  typedef struct packed {
    logic [1:0] load;       // load coefficient cache 1 or 2 from the A/B payload
    logic [1:0] elem;       // select per-element cache instead of scalar C
    logic [1:0] rope;       // zero, NEOX, NORM
    logic [5:0] angle_addr; // buffer half and entry
    logic       upper_quad; // four-wide subslot selects the upper four elements
    logic       angle_hi;   // NORM angle group within the entry
  } tsr_vu_bank_coeff_t;

  typedef struct packed {
    logic        valid;
    logic [8:0]  addr;     // half, entry, local angle index
    logic [63:0] data;     // {cosine, sine}, already rounded by the producer
  } tsr_vu_bank_angle_t;

  // Slots (tsr_vu_gen -> tsr_vu_rd -> tsr_vu_inj): one SPM line position of an op. Line fields are 24 bits
  // (VU §5.1; AT's 32-bit addresses are range-checked below the SPM's line count).
  // -------------------------------------------------------------------------------------------
  localparam int unsigned TSR_VU_LW = 24;

  typedef enum logic [3:0] {
    SK_LANE = 4'd0,     // a lane slot: nsub injection cycles
    SK_CL1  = 4'd1,     // load the c1-line register from source A
    SK_CL2  = 4'd2,     // load the c2-line register from source A
    SK_TBL  = 4'd3,     // ROPE: source A's line into table RAM line d0[4:0]
    SK_POS  = 4'd4,     // ROPE: source A's line into pos buffer half d0[0]
    SK_TOPK = 4'd5,     // TOPK: source A's line to the TOPK unit
    SK_REDE = 4'd6,     // RED: an empty row (m = 0 with TO_S): a row token without lines
    SK_SOPX = 4'd7      // SOP SQRT / RSQRT / DIV: the scalar path (operands in the op constants)
  } tsr_vu_sk_e;

  // operand source of a slot: none (0), an SPM line, a local-memory line, the op constant (k1 for a,
  // k2 for b; SOP's scalar operands)
  typedef enum logic [1:0] { SRC_NONE = 2'd0, SRC_SPM = 2'd1, SRC_LM = 2'd2, SRC_K = 2'd3 } tsr_vu_src_e;
  typedef enum logic [1:0] { DS_NONE = 2'd0, DS_SPM = 2'd1, DS_LM = 2'd2, DS_SREG = 2'd3 } tsr_vu_dst_e;
  // c operand of a lane slot
  typedef enum logic [2:0] {
    C_ZERO = 3'd0, C_K = 3'd1,       // 0; the op constant (k1 for c1, k2 for c2)
    C_CLW  = 3'd2,                   // word cword of the c-line register (a per-row scalar)
    C_CLE  = 3'd3,                   // element l of the c-line register (an element-wise vector)
    C_ANG  = 3'd4                    // ROPE: the angle buffer (c / s by mode)
  } tsr_vu_csel_e;

  typedef struct packed {
    logic               tag;         // held-op tag
    tsr_vu_sk_e         kind;
    tsr_vu_uop_e        uop;
    logic [2:0]         nsub;        // injection cycles: 1, 2 or 4 (SPLIT) / 2 (ROPE NEOX pair)
    logic               exitb;       // back exit
    tsr_vu_src_e        asrc;
    logic [TSR_VU_LW-1:0] aline;
    tsr_vu_src_e        bsrc;
    logic [TSR_VU_LW-1:0] bline;
    tsr_vu_csel_e       c1sel;
    tsr_vu_csel_e       c2sel;
    logic [3:0]         cword;       // C_CLW: word of the c-line registers (row mod 16)
    tsr_vu_dst_e        dst;
    logic [TSR_VU_LW-1:0] d0;        // destination line (SPM / LM), SREG index, TBL / POS index
    logic [TSR_VU_LW-1:0] d1;        // ROPE NEOX pair: the second output line
    logic [4:0]         tail;        // words of the output line that are results (1..16; the rest 0)
    logic [3:0]         dword;       // DS_SREG: word of the result line; STS: the word written
    logic               sts;         // SOP STS: write 4 bytes (word dword) only
    logic               keep;        // write only the words < tail (AT row vectors: the rest keep their bits)
    // reductions
    logic               red;         // accumulate into the partials
    logic               r_first;     // first line of a row
    logic               r_last;      // last line of a row (the row token goes to tsr_vu_red)
    logic [2:0]         r_li;        // line index in the row mod 8 (partial phase)
    logic               r_fresh;     // line index < 8
    logic [3:0]         r_ng;        // groups to drain - 1 (min(ceil(m/8), 16) - 1)
    logic [15:0]        r_row;       // row index in the op
    logic               r_lrow;      // last row of the op
    // ROPE
    logic [1:0]         rope;        // 0 none, 1 NEOX pair, 2 NORM
    logic [4:0]         ang_e;       // angle buffer entry (line index in the half / 2)
    logic               ang_hi;      // NORM: banks 8..15
    logic               ang_h;       // angle buffer half (token parity)
    logic               tok_last;    // last rotation slot of the token (releases the half)
    // DQROW (tsr_vu_dq): dq = {DQROW slot, fmt Q4_K, byte quarter p (bytes 16p..16p+15 of the quant
    // line), Q4_K high nibble, 0}; dqb = the constant-table entry (Q8_0 block / Q4_K 8·superblock +
    // sub-block in the chunk). The chunk's meta line is an SK_CL2 pseudo-slot with dq[5] set.
    logic [5:0]         dq;
    logic [4:0]         dqb;
  } tsr_vu_slot_t;

  // Per-op constants, latched by tsr_vu_gen at the op's start (s registers read then, VU §4.4)
  typedef struct packed {
    logic [31:0] k1, k2;      // lane constants: VOP c (cmode 1), ASMAX / AMERGE scale, SOP operands
    logic [1:0]  rkind;       // RED: 0 SUM, 1 SUMSQ, 2 MAX; 3 AMASK row maxima (MAX from M_FLOOR)
    logic [1:0]  post;        // RED post-op (VU §6.2)
    logic [31:0] sk, se;      // RMS operands
    logic        to_s;        // RED result to s[sd]
    logic [3:0]  sd;
    logic [TSR_VU_LW-1:0] rd; // row-result vector line (RED d, AMASK mdst)
    logic        keep;        // AMASK mdst: words past R keep their bits (AT tsr_ref_at.c)
    logic [15:0] rows;        // RED / AMASK rows (the last result line's word count)
    logic [4:0]  k;           // TOPK k
  } tsr_vu_opk_t;

  // Row token (tsr_vu_inj -> tsr_vu_red), at the injection of a row's last line
  typedef struct packed {
    logic               tag;
    logic [3:0]         ring;        // first partial entry of the row (ring of 16)
    logic [3:0]         np;          // partial entries of the row - 1 (min(L, 8) - 1)
    logic [3:0]         ng;          // groups - 1
    logic               empty;       // no groups (m = 0)
    logic [15:0]        row;
    logic               lrow;
  } tsr_vu_row_t;

  // Exit descriptor of one sub-slot (tsr_vu_inj -> tsr_vu_wr, delayed to its lane exit)
  typedef struct packed {
    logic               tag;
    tsr_vu_dst_e        dst;
    logic [TSR_VU_LW-1:0] line;
    logic [1:0]         part;        // sub-slot index of a split slot
    logic [2:0]         nparts;      // sub-slots that form the line (1 for a ROPE pair's two lines)
    logic [4:0]         tail;
    logic [3:0]         dword;
    logic               sts;
    logic               keep;
  } tsr_vu_desc_t;

  // Back-lane micro-ops (exit at TSR_VU_LB)
  function automatic logic tsr_vu_uop_back(input tsr_vu_uop_e u);
    case (u)
      U_SILU, U_SWIGLU, U_SIGMOID, U_GELU, U_GEGLU, U_TANH, U_RECIP, U_GELU16, U_GEGLU16, U_RCPMAX:
        return 1'b1;
      default: return 1'b0;
    endcase
  endfunction

  // Front-exit micro-ops that need the lane's EXPK core
  function automatic logic tsr_vu_uop_exp(input tsr_vu_uop_e u);
    case (u)
      U_EXP, U_EXPSUB, U_EXPSUBC, U_ASMAX, U_ESCB: return 1'b1;
      default: return 1'b0;
    endcase
  endfunction
  /* verilator lint_on UNUSEDPARAM */

endpackage
