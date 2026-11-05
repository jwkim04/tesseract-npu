// tsr_vu_chk.sv — the VU checker class on tsr_chk's external-class port: dispatch checks, ERR_INFO
// and operand extents of the unit-3 opcodes: DQROW 0x12, VOP 0x30, RED 0x31, ROPE 0x32, TOPK 0x33,
// SOP 0x40 and AT's vector ops ASMAX 0x34, AMASK 0x35, AMERGE 0x36, ANORM 0x37 (AT §5.8 rows; the VU
// executes them, VU §5.11).
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §5.1-§5.9; CC §8.2 checker classes; W4-B task "vu").
// Port contract: rtl/core/tsr_xc_pkg.sv (one class of tsr_chk's xc_* port; AT's opcodes belong to the
// AT class and are ignored here). The command payload is the eight instruction words (the VU decodes
// them itself, tsr_vu.sv).
//
// Oracle: the simulator's checks sw/sim/tsr_vu.c chk_vop / chk_red / chk_rope / chk_topk / chk_sop /
// chk_dqrow and sw/sim/tsr_at.c chk_asmax / chk_amask / chk_amerge / chk_anorm (tb/vu test_vu_chk
// compares code, ERR_INFO and the SPM extents bit for bit). Rules:
//   ILLEGAL_OP  a reserved bit of the encoding tables (tsr_tables_pkg TSR_<OP>_RSV_W*: x16 bits, flags,
//               w[31:24] of line addresses and m), an undefined enumeration (VOP func >= 25 or cmode >= 3,
//               RED kind >= 3 or post 3, ROPE mode >= 2, SOP sub 15), and every field the op or
//               function does not use (VU §5.1): VOP a / a_stride without a read of a, b / b_stride
//               likewise, c unless cmode 2, si unless cmode 1, a c-reading function with cmode 0 or a
//               function without c with cmode != 0; RED MAX with a post-op, sd without TO_S, d with
//               TO_S, sk / se without RMS; ROPE ATAB with FF, FS or sts != 0 (VU §5.9: before OPERAND),
//               sfs without FS, a table line without FF or ATAB; SOP fields its sub does not use
//               (tsr_sop_uses).
//   BAD_FMT / K_RANGE  DQROW only: fmt >= 2; K = 0, K % block != 0 or K > K_MAX.
//   OPERAND     VU §5.9 in its order, ERR_INFO = the raw field: VOP rows > 1 and d_stride < L(m)
//               (d_stride); RED TO_S and rows != 1 (rows); ROPE head_dim (head_dim), n_dims (n_dims),
//               T > 1, n_heads > 0 and d_stride' < H (d_stride); TOPK k (k), rows > 1 and d_stride' <
//               ceil(k/8) (d_stride); DQROW rows > 1 and d_stride' < L(K) (d_stride).
//   AT ops      reserved bits ILLEGAL_OP; OPERAND (AT §5.8): ASMAX n_lines not in [1, 4095] (n_lines),
//               R not in [1, 1024] (R), scale not finite, not normal or <= 0 (the scale bits); AMASK
//               n_lines, R, rpq = 0 (0); AMERGE R, scale; ANORM R. Range operands: ASMAX dst, src
//               ((R-1)·stride' + n_lines), mvec (ceil(R/16)); AMASK dst, src, mask ((ceil(R/rpq) - 1)·
//               mstride + n_lines), mdst (w7); AMERGE Orun (R·D/16), parts (C·cstride' + C·ceil(R/16),
//               C = NUM_CH), mk, st (2·ceil(R/16)); ANORM out, Orun (R·D/16), l_run. A merging AMERGE
//               lists six extents (Orun W and R, parts, mk, st W and R); the class slot (CC §8.2) holds
//               four, so the RTL keeps the first four, and the simulator's tile-breakpoint matcher takes
//               the same four (DB §5.3): a watch on st never hits. ceil(R/rpq) uses a serial divider
//               (AMASK only).
//   SPM_RANGE   operands in field order (w1, w2, w3, w4; AMASK w7), start + extent > SPM lines (48-bit),
//               ERR_INFO = the start line; SOP LDS / STS: word address + 1 > SPM_BYTES / 4, ERR_INFO =
//               the word address. Extents (VU §5.9): E(n, s, len) = 0 if n = 0 or len = 0, else
//               (n - 1)·s + len.
// Extents of a passing op: the SPM operands with a nonzero extent in field order (VOP d W, a R, b R,
// c R; RED d W, a R; ROPE d W, src R, pos R, table R; TOPK d W, a R; DQROW d W, src R; SOP LDS the
// line of the word R, STS W), compacted into slots 0..3 (tsr_ext_add skips empty ones). The s
// registers the simulator also lists have no RTL slot.
//
// Pipeline (one op at a time; every stage starts from registers):
//   K0 start: capture the words (req_valid && !busy)
//   K1 decode: fields, reserved / use / enumeration checks, L(m), ceil(rows/16), ROPE H operands,
//      DQROW K checks and ROW_BEATS operands
//   K2 strides resolved, the OPERAND conditions, (n - 1) for every extent
//   K3 the three products (n - 1)·s (DSP)
//   K4 extents and end addresses
//   K5 limit compares
//   K6 priority, ERR_INFO, extents -> rsp_valid (registered, one cycle)
// busy is set from K0 until req_valid has been seen low after the answer (tsr_xc_pkg contract).

module tsr_vu_chk
  import tsr_pkg::*;
  import tsr_xc_pkg::*;
#(
  parameter bit          LOCAL_INTERMEDIATES = 1'b0,
  parameter int unsigned SPM_BYTES = TSR_DEF_SPM_BYTES,
  parameter int unsigned K_MAX     = TSR_DEF_K_MAX,
  parameter int unsigned NUM_CH    = TSR_DEF_NUM_CH
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         req_valid,
  input  tsr_insn_t                    req_insn,
  output logic                         rsp_valid,
  output logic [3:0]                   rsp_code,
  output logic [31:0]                  rsp_info,
  output tsr_ext_t [TSR_EXT_N-1:0]     rsp_ext,
  output logic [TSR_XC_CMD_W-1:0]      rsp_cmd
);
  import tsr_tables_pkg::*;

  localparam logic [TSR_EXT_AW-1:0] SPM_LIM = TSR_EXT_AW'(SPM_BYTES / TSR_LINE_BYTES);
  localparam logic [TSR_EXT_AW-1:0] SPM_WRD = TSR_EXT_AW'(SPM_BYTES / 4);
  localparam int unsigned AW = TSR_EXT_AW;

  // op classes (one-hot)
  localparam int unsigned C_VOP = 0, C_RED = 1, C_ROPE = 2, C_TOPK = 3, C_SOP = 4, C_DQ = 5;
  localparam int unsigned C_ASM = 6, C_AMK = 7, C_AMG = 8, C_ANM = 9, NC = 10;

  function automatic logic rsv_any(input tsr_insn_t w, input logic [7:0][31:0] m);
    logic r;
    r = 1'b0;
    for (int i = 0; i < 8; i++) r |= ((w[i] & m[i]) != 32'd0);
    return r;
  endfunction

  function automatic logic [7:0][31:0] rsv_of(input logic [7:0] op);
    case (op)
      tsr_tables_pkg::TSR_OP_VOP:  return {TSR_VOP_RSV_W7, TSR_VOP_RSV_W6, TSR_VOP_RSV_W5, TSR_VOP_RSV_W4,
                           TSR_VOP_RSV_W3, TSR_VOP_RSV_W2, TSR_VOP_RSV_W1, TSR_VOP_RSV_W0};
      tsr_tables_pkg::TSR_OP_RED:  return {TSR_RED_RSV_W7, TSR_RED_RSV_W6, TSR_RED_RSV_W5, TSR_RED_RSV_W4,
                           TSR_RED_RSV_W3, TSR_RED_RSV_W2, TSR_RED_RSV_W1, TSR_RED_RSV_W0};
      tsr_tables_pkg::TSR_OP_ROPE: return {TSR_ROPE_RSV_W7, TSR_ROPE_RSV_W6, TSR_ROPE_RSV_W5, TSR_ROPE_RSV_W4,
                           TSR_ROPE_RSV_W3, TSR_ROPE_RSV_W2, TSR_ROPE_RSV_W1, TSR_ROPE_RSV_W0};
      tsr_tables_pkg::TSR_OP_TOPK: return {TSR_TOPK_RSV_W7, TSR_TOPK_RSV_W6, TSR_TOPK_RSV_W5, TSR_TOPK_RSV_W4,
                           TSR_TOPK_RSV_W3, TSR_TOPK_RSV_W2, TSR_TOPK_RSV_W1, TSR_TOPK_RSV_W0};
      tsr_tables_pkg::TSR_OP_SOP:  return {TSR_SOP_RSV_W7, TSR_SOP_RSV_W6, TSR_SOP_RSV_W5, TSR_SOP_RSV_W4,
                           TSR_SOP_RSV_W3, TSR_SOP_RSV_W2, TSR_SOP_RSV_W1, TSR_SOP_RSV_W0};
      TSR_OP_ASMAX:  return {TSR_ASMAX_RSV_W7, TSR_ASMAX_RSV_W6, TSR_ASMAX_RSV_W5, TSR_ASMAX_RSV_W4,
                             TSR_ASMAX_RSV_W3, TSR_ASMAX_RSV_W2, TSR_ASMAX_RSV_W1, TSR_ASMAX_RSV_W0};
      TSR_OP_AMASK:  return {TSR_AMASK_RSV_W7, TSR_AMASK_RSV_W6, TSR_AMASK_RSV_W5, TSR_AMASK_RSV_W4,
                             TSR_AMASK_RSV_W3, TSR_AMASK_RSV_W2, TSR_AMASK_RSV_W1, TSR_AMASK_RSV_W0};
      TSR_OP_AMERGE: return {TSR_AMERGE_RSV_W7, TSR_AMERGE_RSV_W6, TSR_AMERGE_RSV_W5, TSR_AMERGE_RSV_W4,
                             TSR_AMERGE_RSV_W3, TSR_AMERGE_RSV_W2, TSR_AMERGE_RSV_W1, TSR_AMERGE_RSV_W0};
      TSR_OP_ANORM:  return {TSR_ANORM_RSV_W7, TSR_ANORM_RSV_W6, TSR_ANORM_RSV_W5, TSR_ANORM_RSV_W4,
                             TSR_ANORM_RSV_W3, TSR_ANORM_RSV_W2, TSR_ANORM_RSV_W1, TSR_ANORM_RSV_W0};
      default:     return {TSR_DQROW_RSV_W7, TSR_DQROW_RSV_W6, TSR_DQROW_RSV_W5, TSR_DQROW_RSV_W4,
                           TSR_DQROW_RSV_W3, TSR_DQROW_RSV_W2, TSR_DQROW_RSV_W1, TSR_DQROW_RSV_W0};
    endcase
  endfunction

  // -------------------------------------------------------------------------------------------
  // Control
  // -------------------------------------------------------------------------------------------
  logic [6:0] st_q;                    // K0..K6 hold the op
  logic       stall2;                  // K1 -> K2 waits for the AMASK divider
  logic       busy_q;                  // from K0 until req_valid is seen low after the answer
  logic       ans_q;                   // answered, waiting for req_valid low
  logic       rsp_q;
  wire        start = req_valid && !busy_q && (tsr_xc_class(req_insn[0][7:0]) == TSR_XC_VU);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q   <= '0;
      busy_q <= 1'b0;
      ans_q  <= 1'b0;
      rsp_q  <= 1'b0;
    end else begin
      st_q  <= {st_q[5:2], st_q[1] && !stall2, st_q[0] || (st_q[1] && stall2), start};
      rsp_q <= st_q[6];
      if (start) busy_q <= 1'b1;
      if (st_q[6]) ans_q <= 1'b1;
      if (ans_q && !rsp_q && !req_valid) begin
        ans_q  <= 1'b0;
        busy_q <= 1'b0;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // K0: capture
  // -------------------------------------------------------------------------------------------
  tsr_insn_t w_q;
  always_ff @(posedge clk) if (start) w_q <= req_insn;

  wire [7:0]  op  = w_q[0][7:0];
  wire [3:0]  sub = w_q[0][11:8];
  wire [15:0] x16 = w_q[0][31:16];

  // -------------------------------------------------------------------------------------------
  // K1: decode
  // -------------------------------------------------------------------------------------------
  logic [NC-1:0] d_cls;
  logic        d_ill, d_fmt, d_krng;
  logic [2:0]  d_reads;                // VOP a / b / c
  logic        d_tos, d_tab, d_neox;
  logic [31:0] d_lm;                   // L(m) / L(K)
  logic [31:0] d_lr;                   // ceil(rows / 16)
  logic [31:0] d_rb;                   // DQROW ROW_BEATS
  logic [4:0]  d_sop_uses;
  logic [31:0] d_h;                    // ROPE H = n_heads·head_dim/16 (lines per token)
  always_comb begin
    d_cls = '0;
    case (op)
      tsr_tables_pkg::TSR_OP_VOP:  d_cls[C_VOP]  = 1'b1;
      tsr_tables_pkg::TSR_OP_RED:  d_cls[C_RED]  = 1'b1;
      tsr_tables_pkg::TSR_OP_ROPE: d_cls[C_ROPE] = 1'b1;
      tsr_tables_pkg::TSR_OP_TOPK: d_cls[C_TOPK] = 1'b1;
      tsr_tables_pkg::TSR_OP_SOP:  d_cls[C_SOP]  = 1'b1;
      TSR_OP_ASMAX:  d_cls[C_ASM] = 1'b1;
      TSR_OP_AMASK:  d_cls[C_AMK] = 1'b1;
      TSR_OP_AMERGE: d_cls[C_AMG] = 1'b1;
      TSR_OP_ANORM:  d_cls[C_ANM] = 1'b1;
      default:     d_cls[C_DQ]   = 1'b1;
    endcase
    d_ill      = rsv_any(w_q, rsv_of(op));
    if (op == TSR_OP_ASMAX) begin
      tsr_insn_t w_std;
      w_std = w_q;
      w_std[7] = '0;
      d_ill = rsv_any(w_std, rsv_of(op)) ||
              (w_q[7] != 0 && (!LOCAL_INTERMEDIATES || w_q[7][1:0] == 0 || w_q[7][31:18] != 0));
    end
    d_fmt      = 1'b0;
    d_krng     = 1'b0;
    d_reads    = tsr_vop_func_reads(x16[7:0]);
    d_tos      = w_q[0][12];
    d_tab      = w_q[0][12] || w_q[0][14];
    d_neox     = (sub == TSR_ROPE_MODE_NEOX);
    d_sop_uses = tsr_sop_uses(sub);
    d_lm       = (32'(w_q[5][23:0]) + 32'd15) >> 4;
    d_lr       = (32'(w_q[6][15:0]) + 32'd15) >> 4;
    d_rb       = tsr_row_beats(sub[0], w_q[5]);
    d_h        = (32'(w_q[6][15:0]) * 32'(w_q[5][15:0])) >> 4;
    if (d_cls[C_VOP]) begin
      // VU §5.2: function, cmode, and the fields a function does not use
      if (d_reads == 3'd0 || sub > TSR_VOP_CMODE_VEC) d_ill = 1'b1;
      if (d_reads[2] != (sub != TSR_VOP_CMODE_NONE)) d_ill = 1'b1;
      if (sub != TSR_VOP_CMODE_SREG && x16[11:8] != 4'd0) d_ill = 1'b1;
      if (sub != TSR_VOP_CMODE_VEC && w_q[4] != 32'd0) d_ill = 1'b1;
      if (!d_reads[0] && (w_q[2] != 32'd0 || w_q[7][15:0] != 16'd0)) d_ill = 1'b1;
      if (!d_reads[1] && (w_q[3] != 32'd0 || w_q[7][31:16] != 16'd0)) d_ill = 1'b1;
    end
    if (d_cls[C_RED]) begin
      // VU §5.3
      if (sub > TSR_RED_KIND_MAX || x16[5:4] > TSR_RED_POST_RMS) d_ill = 1'b1;
      if (sub == TSR_RED_KIND_MAX && x16[5:4] != TSR_RED_POST_NONE) d_ill = 1'b1;
      if (d_tos ? (w_q[1] != 32'd0) : (x16[3:0] != 4'd0)) d_ill = 1'b1;
      if (x16[5:4] != TSR_RED_POST_RMS && x16[15:8] != 8'd0) d_ill = 1'b1;
    end
    if (d_cls[C_ROPE]) begin
      // VU §5.4, §5.9 (ATAB with FF, FS or sts is ILLEGAL_OP, before OPERAND)
      if (sub > TSR_ROPE_MODE_NEOX) d_ill = 1'b1;
      if (w_q[0][14] && (w_q[0][13:12] != 2'd0 || x16[7:4] != 4'd0)) d_ill = 1'b1;
      if (!w_q[0][13] && x16[11:8] != 4'd0) d_ill = 1'b1;
      if (!d_tab && w_q[4] != 32'd0) d_ill = 1'b1;
    end
    if (d_cls[C_SOP]) begin
      // VU §5.8: undefined sub, and every field its sub does not use
      if (d_sop_uses == 5'd0) d_ill = 1'b1;
      if (!d_sop_uses[0] && w_q[1][3:0]  != 4'd0) d_ill = 1'b1;
      if (!d_sop_uses[1] && w_q[1][7:4]  != 4'd0) d_ill = 1'b1;
      if (!d_sop_uses[2] && w_q[1][11:8] != 4'd0) d_ill = 1'b1;
      if (!d_sop_uses[3] && w_q[2] != 32'd0) d_ill = 1'b1;
      if (!d_sop_uses[4] && w_q[3] != 32'd0) d_ill = 1'b1;
    end
    if (d_cls[C_DQ]) begin
      // VU §5.6: BAD_FMT, then K_RANGE
      d_fmt  = (sub >= 4'd2);
      d_krng = (w_q[5] == 32'd0) || ((w_q[5] & (sub[0] ? 32'd255 : 32'd31)) != 32'd0) ||
               (w_q[5] > 32'(K_MAX));
    end
  end

  logic [NC-1:0] cls_q;
  logic        ill_q, fmt_q, krng_q, tos_q, tab_q, neox_q;
  logic [2:0]  reads_q;
  logic [31:0] lm_q, lr_q, rb_q, h_q;
  logic [4:0]  uses_q;
  always_ff @(posedge clk) begin
    if (st_q[0] && !(st_q[1] && stall2)) begin
      cls_q   <= d_cls;
      ill_q   <= d_ill;
      fmt_q   <= d_fmt;
      krng_q  <= d_krng;
      tos_q   <= d_tos;
      tab_q   <= d_tab;
      neox_q  <= d_neox;
      reads_q <= d_reads;
      lm_q    <= d_lm;
      lr_q    <= d_lr;
      rb_q    <= d_rb;
      h_q     <= d_h;
      uses_q  <= d_sop_uses;
    end
  end

  // -------------------------------------------------------------------------------------------
  // AMASK: q = ceil(R / rpq) by restoring division (17 steps from K1; K2 waits)
  // -------------------------------------------------------------------------------------------
  logic [16:0] dvn_q, dvq_q, dvr_q;
  logic [4:0]  dvc_q;
  logic        dvb_q;                  // dividing
  wire  [15:0] rpq = w_q[5][31:16];
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      dvb_q <= 1'b0;
      dvc_q <= '0;
    end else if (st_q[0] && d_cls[C_AMK]) begin
      dvb_q <= 1'b1;
      dvc_q <= 5'd17;
      dvn_q <= 17'(w_q[5][15:0]) + 17'(rpq) - 17'd1;
      dvr_q <= '0;
      dvq_q <= '0;
    end else if (dvb_q) begin
      logic [17:0] t;
      t = {dvr_q, dvn_q[16]} - 18'(rpq);
      if (!t[17]) begin
        dvr_q <= t[16:0];
        dvq_q <= {dvq_q[15:0], 1'b1};
      end else begin
        dvr_q <= {dvr_q[15:0], dvn_q[16]};
        dvq_q <= {dvq_q[15:0], 1'b0};
      end
      dvn_q <= {dvn_q[15:0], 1'b0};
      dvc_q <= dvc_q - 5'd1;
      if (dvc_q == 5'd1) dvb_q <= 1'b0;
    end
  end
  assign stall2 = cls_q[C_AMK] && (dvb_q || st_q[0]);

  // -------------------------------------------------------------------------------------------
  // K2: resolved strides, OPERAND, the range operands (start, n, s, len per slot) and the extent list
  // -------------------------------------------------------------------------------------------
  // Range slots k = 0..3 (fields w1..w4; AMASK slot 3 = w7): start a, present, E(n, s, len). Extent list:
  // up to six (slot, write) entries in the simulator's tsr_ext_add order, compacted in K6.
  wire [15:0] rows = w_q[6][15:0];
  wire [15:0] hd   = w_q[5][15:0];
  wire [15:0] nd   = w_q[5][31:16];
  wire [15:0] nh   = w_q[6][15:0];
  wire [15:0] tt   = w_q[6][31:16];
  wire [31:0] H    = h_q;                               // ROPE: lines per token
  wire [4:0]  kk   = x16[4:0];
  wire [31:0] kl   = (32'(kk) + 32'd7) >> 3;            // TOPK ceil(k/8)

  function automatic logic scale_bad(input logic [8:0] se);   // AT §5.4: finite, normal, > 0 ({sign, exp})
    return se[8] || se[7:0] == 8'd0 || se[7:0] == 8'hFF;
  endfunction

  logic        c2_opd;                 // OPERAND
  logic [31:0] c2_opd_info;
  logic [3:0]  c2_pres;
  logic [3:0][31:0] c2_a;
  logic [3:0][16:0] c2_n;              // rows / T' (17 bits: (n - 1) needs n >= 1)
  logic [3:0][31:0] c2_s;
  logic [3:0][AW-1:0] c2_len;
  logic [5:0]       c2_ev;             // extent list
  logic [5:0][1:0]  c2_es;
  logic [5:0]       c2_ew;
  logic        c2_sop_oor;
  always_comb begin
    logic [31:0] dsp, ssp, rr, rl, cs, r16;
    logic [15:0] tp, nl, st;
    logic [4:0]  q;
    c2_opd      = 1'b0;
    c2_opd_info = 32'd0;
    c2_pres     = '0;
    c2_a        = '0;
    c2_n        = '0;
    c2_s        = '0;
    c2_len      = '0;
    c2_ev       = '0;
    c2_es       = '0;
    c2_ew       = '0;
    c2_sop_oor  = 1'b0;
    dsp = 32'd0;
    ssp = 32'd0;
    tp  = 16'd0;
    q   = 5'd0;
    rr  = 32'd0;
    rl  = 32'd0;
    cs  = 32'd0;
    r16 = 32'd0;
    nl  = w_q[4][15:0];
    st  = (w_q[4][31:16] != 16'd0) ? w_q[4][31:16] : w_q[4][15:0];
    for (int k = 0; k < 4; k++) c2_a[k] = 32'(w_q[k + 1][23:0]);
    // default extent list: the slots in field order, slot 0 written
    for (int k = 0; k < 4; k++) begin
      c2_ev[k] = 1'b1;
      c2_es[k] = 2'(k);
      c2_ew[k] = (k == 0);
    end
    if (cls_q[C_VOP]) begin
      if (rows > 16'd1 && 32'(w_q[6][31:16]) < lm_q) begin
        c2_opd = 1'b1; c2_opd_info = 32'(w_q[6][31:16]);
      end
      c2_pres = {sub == TSR_VOP_CMODE_VEC, reads_q[1], reads_q[0], 1'b1};
      c2_n[0] = 17'(rows); c2_s[0] = 32'(w_q[6][31:16]); c2_len[0] = AW'(lm_q);
      c2_n[1] = 17'(rows); c2_s[1] = 32'(w_q[7][15:0]);  c2_len[1] = AW'(lm_q);
      c2_n[2] = 17'(rows); c2_s[2] = 32'(w_q[7][31:16]); c2_len[2] = AW'(lm_q);
      c2_n[3] = 17'(rows != 16'd0); c2_s[3] = 32'd0;     c2_len[3] = AW'(lr_q);   // ceil(rows/16)
    end
    if (cls_q[C_RED]) begin
      if (tos_q && rows != 16'd1) begin
        c2_opd = 1'b1; c2_opd_info = 32'(rows);
      end
      c2_pres = {2'b00, 1'b1, !tos_q};
      c2_n[0] = 17'(!tos_q && rows != 16'd0); c2_s[0] = 32'd0; c2_len[0] = AW'(lr_q);
      c2_n[1] = 17'(rows); c2_s[1] = 32'(w_q[7][15:0]); c2_len[1] = AW'(lm_q);
    end
    if (cls_q[C_ROPE]) begin
      q   = neox_q ? 5'd31 : 5'd15;
      dsp = (w_q[7][31:16] != 16'd0) ? 32'(w_q[7][31:16]) : H;
      ssp = (w_q[7][15:0] != 16'd0) ? 32'(w_q[7][15:0]) : H;
      tp  = (nh != 16'd0) ? tt : 16'd0;
      if (hd == 16'd0 || hd[3:0] != 4'd0 || 32'(hd) > TSR_ROPE_MAX_HEAD_DIM) begin
        c2_opd = 1'b1; c2_opd_info = 32'(hd);
      end else if (nd == 16'd0 || nd > hd || (nd[4:0] & q) != 5'd0) begin
        c2_opd = 1'b1; c2_opd_info = 32'(nd);
      end else if (tt > 16'd1 && nh != 16'd0 && dsp < H) begin
        c2_opd = 1'b1; c2_opd_info = 32'(w_q[7][31:16]);
      end
      c2_pres = {tab_q, 3'b111};
      c2_n[0] = 17'(tp); c2_s[0] = dsp;   c2_len[0] = AW'(H);
      c2_n[1] = 17'(tp); c2_s[1] = ssp;   c2_len[1] = AW'(H);
      c2_n[2] = 17'(tp != 16'd0); c2_s[2] = 32'd0; c2_len[2] = AW'(32'((32'(tp) + 32'd15) >> 4));
      c2_n[3] = 17'(tp != 16'd0); c2_s[3] = 32'd0; c2_len[3] = AW'(32'((32'(nd) + 32'd31) >> 5));
    end
    if (cls_q[C_TOPK]) begin
      dsp = (w_q[6][31:16] != 16'd0) ? 32'(w_q[6][31:16]) : kl;
      if (kk == 5'd0 || kk > 5'd16) begin
        c2_opd = 1'b1; c2_opd_info = 32'(kk);
      end else if (rows > 16'd1 && dsp < kl) begin
        c2_opd = 1'b1; c2_opd_info = 32'(w_q[6][31:16]);
      end
      c2_pres = 4'b0011;
      c2_n[0] = 17'(rows); c2_s[0] = dsp;                c2_len[0] = AW'(kl);
      c2_n[1] = 17'(rows); c2_s[1] = 32'(w_q[7][15:0]);  c2_len[1] = AW'(lm_q);
    end
    if (cls_q[C_DQ]) begin
      dsp = (w_q[7][31:16] != 16'd0) ? 32'(w_q[7][31:16]) : lm_q;
      ssp = (w_q[7][15:0] != 16'd0) ? 32'(w_q[7][15:0]) : rb_q;
      if (rows > 16'd1 && dsp < lm_q) begin
        c2_opd = 1'b1; c2_opd_info = 32'(w_q[7][31:16]);
      end
      c2_pres = 4'b0011;
      c2_n[0] = 17'(rows); c2_s[0] = dsp; c2_len[0] = AW'(lm_q);
      c2_n[1] = 17'(rows); c2_s[1] = ssp; c2_len[1] = AW'(rb_q);
    end
    if (cls_q[C_SOP]) begin
      // LDS / STS: one word (VU §5.9), its line as the extent
      c2_sop_oor = uses_q[4] && (TSR_EXT_AW'(w_q[3]) + TSR_EXT_AW'(1) > SPM_WRD);
      c2_pres    = {3'b000, uses_q[4]};
      c2_ew[0]   = (sub == TSR_SOP_SUB_STS);
      c2_a[0]    = {4'd0, w_q[3][31:4]};
      c2_n[0]    = 17'(uses_q[4]); c2_s[0] = 32'd0; c2_len[0] = AW'(1);
    end
    // AT's vector ops: 32-bit line fields
    if (cls_q[C_ASM] || cls_q[C_AMK] || cls_q[C_AMG] || cls_q[C_ANM]) begin
      for (int k = 0; k < 4; k++) c2_a[k] = w_q[k + 1];
    end
    if (cls_q[C_ASM] || cls_q[C_AMK]) begin
      rr = cls_q[C_ASM] ? w_q[5] : 32'(w_q[5][15:0]);
      if (nl == 16'd0 || nl > 16'd4095) begin
        c2_opd = 1'b1; c2_opd_info = 32'(nl);
      end else if (rr == 32'd0 || rr > 32'd1024) begin
        c2_opd = 1'b1; c2_opd_info = rr;
      end else if (cls_q[C_ASM] && scale_bad(w_q[6][31:23])) begin
        c2_opd = 1'b1; c2_opd_info = w_q[6];
      end else if (cls_q[C_ASM] && LOCAL_INTERMEDIATES && w_q[7][1:0] != 0 &&
                   (rr > 32'(tsr_att_pkg::TSR_LOCAL_MAX_ROWS) || nl > 16'd128 ||
                    w_q[4][31:16] != ((nl + 16'd3) & 16'hfffc) || w_q[1] != w_q[2] ||
                    rr * 32'(st >> 2) > 32'(tsr_att_pkg::TSR_LOCAL_SCORE_LINES))) begin
        c2_opd = 1'b1; c2_opd_info = rr;
      end else if (cls_q[C_AMK] && rpq == 16'd0) begin
        c2_opd = 1'b1; c2_opd_info = 32'd0;
      end
      r16 = (rr + 32'd15) >> 4;
      c2_n[0] = 17'(rr); c2_s[0] = 32'(st); c2_len[0] = AW'(nl);
      c2_n[1] = 17'(rr); c2_s[1] = 32'(st); c2_len[1] = AW'(nl);
      if (cls_q[C_ASM]) begin
        c2_pres = 4'b0111;
        c2_n[2] = 17'd1; c2_len[2] = AW'(r16);
      end else begin
        c2_pres = 4'b1111;
        c2_n[2] = dvq_q; c2_s[2] = w_q[6]; c2_len[2] = AW'(nl);
        c2_a[3] = w_q[7];
        c2_n[3] = 17'd1; c2_len[3] = AW'(r16);
        c2_ew[3] = 1'b1;
      end
    end
    if (cls_q[C_AMG]) begin
      rr  = 32'(w_q[5][15:0]);
      rl  = rr << (2 + 32'(w_q[5][17:16]));            // R·D/16
      cs  = (w_q[7] != 32'd0) ? w_q[7] : rl;
      r16 = (rr + 32'd15) >> 4;
      if (rr == 32'd0 || rr > 32'd1024) begin
        c2_opd = 1'b1; c2_opd_info = rr;
      end else if (scale_bad(w_q[6][31:23])) begin
        c2_opd = 1'b1; c2_opd_info = w_q[6];
      end
      c2_pres = 4'b1111;
      c2_n[0] = 17'd1; c2_len[0] = AW'(rl);
      c2_n[1] = 17'd1; c2_len[1] = AW'(NUM_CH) * AW'(cs) + AW'(NUM_CH) * AW'(r16);
      c2_n[2] = 17'd1; c2_len[2] = AW'(r16);
      c2_n[3] = 17'd1; c2_len[3] = AW'(2 * r16);
      // Orun W, [Orun R], parts R, mk R, st W, [st R] (merging: both directions of Orun and st)
      if (w_q[0][12]) begin
        c2_ev = 6'b001111; c2_es = {2'd0, 2'd0, 2'd3, 2'd2, 2'd1, 2'd0}; c2_ew = 6'b001001;
      end else begin
        c2_ev = 6'b111111; c2_es = {2'd3, 2'd3, 2'd2, 2'd1, 2'd0, 2'd0}; c2_ew = 6'b010001;
      end
    end
    if (cls_q[C_ANM]) begin
      rr  = 32'(w_q[4][15:0]);
      rl  = rr << (2 + 32'(w_q[4][17:16]));
      if (rr == 32'd0 || rr > 32'd1024) begin
        c2_opd = 1'b1; c2_opd_info = rr;
      end
      c2_pres = 4'b0111;
      c2_n[0] = 17'd1; c2_len[0] = AW'(rl);
      c2_n[1] = 17'd1; c2_len[1] = AW'(rl);
      c2_n[2] = 17'd1; c2_len[2] = AW'(32'((rr + 32'd15) >> 4));
    end
  end

  logic        opd_q, sop_oor_q;
  logic [31:0] opd_info_q;
  logic [3:0]  pres_q;
  logic [3:0][31:0] a_q, s_q;
  logic [3:0][AW-1:0] len_q;
  logic [3:0][15:0] nm1_q;             // n - 1 (n >= 1), else 0
  logic [3:0]       nz_q;              // n != 0 and len != 0
  logic [5:0]       ev_q, ew_q;
  logic [5:0][1:0]  es_q;
  always_ff @(posedge clk) begin
    if (st_q[1]) begin
      opd_q      <= c2_opd;
      opd_info_q <= c2_opd_info;
      sop_oor_q  <= c2_sop_oor;
      pres_q     <= c2_pres;
      a_q        <= c2_a;
      s_q        <= c2_s;
      len_q      <= c2_len;
      ev_q       <= c2_ev;
      es_q       <= c2_es;
      ew_q       <= c2_ew;
      for (int k = 0; k < 4; k++) begin
        nz_q[k]  <= (c2_n[k] != 17'd0) && (c2_len[k] != '0);
        nm1_q[k] <= (c2_n[k] != 17'd0) ? 16'(c2_n[k] - 17'd1) : 16'd0;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // K3: (n - 1)·s;  K4: extents and ends;  K5: compares
  // -------------------------------------------------------------------------------------------
  logic [3:0][AW-1:0] prod_q, ext_q, end_q;
  logic [3:0]         oor_q;
  always_ff @(posedge clk) begin
    if (st_q[2]) begin
      for (int k = 0; k < 4; k++) prod_q[k] <= AW'(nm1_q[k]) * AW'(s_q[k]);
    end
    if (st_q[3]) begin
      for (int k = 0; k < 4; k++) begin
        ext_q[k] <= nz_q[k] ? prod_q[k] + len_q[k] : '0;
        end_q[k] <= AW'(a_q[k]) + (nz_q[k] ? prod_q[k] + len_q[k] : '0);
      end
    end
    if (st_q[4]) begin
      for (int k = 0; k < 4; k++) oor_q[k] <= pres_q[k] && (end_q[k] > SPM_LIM);
    end
  end

  // -------------------------------------------------------------------------------------------
  // K6: priority, ERR_INFO, extents
  // -------------------------------------------------------------------------------------------
  logic [3:0]  c6_code;
  logic [31:0] c6_info;
  tsr_ext_t [TSR_EXT_N-1:0] c6_ext;
  always_comb begin
    c6_code = tsr_tables_pkg::TSR_ERR_NONE;
    c6_info = 32'd0;
    if (ill_q) begin
      c6_code = tsr_tables_pkg::TSR_ERR_ILLEGAL_OP;
    end else if (fmt_q) begin
      c6_code = tsr_tables_pkg::TSR_ERR_BAD_FMT;
    end else if (krng_q) begin
      c6_code = tsr_tables_pkg::TSR_ERR_K_RANGE;
    end else if (opd_q) begin
      c6_code = tsr_tables_pkg::TSR_ERR_OPERAND;
      c6_info = opd_info_q;
    end else if (sop_oor_q) begin
      c6_code = tsr_tables_pkg::TSR_ERR_SPM_RANGE;
      c6_info = w_q[3];
    end else begin
      for (int k = 3; k >= 0; k--) begin
        if (oor_q[k]) begin
          c6_code = tsr_tables_pkg::TSR_ERR_SPM_RANGE;
          c6_info = a_q[k];
        end
      end
    end
    c6_ext = '0;
    begin
      int unsigned j;
      j = 0;
      for (int e = 0; e < 6; e++) begin
        if (ev_q[e] && pres_q[es_q[e]] && ext_q[es_q[e]] != '0 && j < 4) begin
          c6_ext[j[1:0]] = '{v: 1'b1, dram: 1'b0, wr: ew_q[e], ch: 8'd0, lo: AW'(a_q[es_q[e]]),
                            hi: end_q[es_q[e]]};
          j = j + 1;
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rsp_code <= '0;
      rsp_info <= '0;
      rsp_ext  <= '0;
      rsp_cmd  <= '0;
    end else if (st_q[6]) begin
      rsp_code <= c6_code;
      rsp_info <= c6_info;
      rsp_ext  <= (c6_code == tsr_tables_pkg::TSR_ERR_NONE) ? c6_ext : '0;
      rsp_cmd  <= TSR_XC_CMD_W'(w_q);
    end else begin
      rsp_code <= '0;
      rsp_info <= '0;
      rsp_ext  <= '0;
      rsp_cmd  <= '0;
    end
  end
  assign rsp_valid = rsp_q;

  wire unused_ok = &{1'b0, x16[15:12], uses_q[3:0], reads_q[2]};

endmodule
