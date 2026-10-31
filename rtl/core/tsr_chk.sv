// tsr_chk.sv — dispatch checker: one op at a time, its error code (EX §8.1), ERR_INFO, its four
// tagged operand extents and its unit command (CC §8).
//
// Owner: issue path (docs/design/04-control-core.md "CC" §8.1-§8.2, §13.2; W4 item 1). The
// dispatcher (tsr_issue, CC §8.3) starts it for every OP entry of the issue FIFO and merges the
// BAD_DEP rule, which is its own (the dispatcher's counts), into the result.
//
// Oracle: the simulator's checker classes (sw/sim/tsr_sim.c chk_sync / chk_ldst / chk_qact /
// chk_mvm, and tsr_sim_accept_op for the refusal rule); tb/chk compares code, info, extents and
// command fields bit for bit.
//
// Classes (EX §7.1-§7.3, spec §21 "Implementation clarifications"):
//   refused   an opcode the issue path does not accept: reserved / retired (tsr_op_legal = 0),
//             CAPS lacking its tsr_op_caps bits, or its unit not built (Q depth 0). Unit SYNC,
//             ILLEGAL_OP, nothing else checked (CC §7.4).
//   NOP FENCE w0[31:8] = 0 and w1..w7 = 0, else ILLEGAL_OP. No extents. (SYNC)
//   LD ST     w0[31:12] = 0, w4..w7 = 0, channel (sub) < NUM_CH, else ILLEGAL_OP; then SPM_RANGE
//             (A1 = w1, n = w3 lines), DRAM_RANGE (A2 = w2 on channel sub). n = 0 passes and is
//             flagged rsp_null: the head completes it without the DMA (docs/rtl-m2.md §3.2: the
//             unit never sees n_lines = 0). (DMA)
//   QACT      w0[31:12] = 0, w6 = w7 = 0, kind (sub) < 2, else ILLEGAL_OP; K = w3: K_RANGE if
//             K = 0, K % block or K > K_MAX; T = w4: T_RANGE if T = 0 or T > T_MAX; SPM_RANGE for
//             dst = w1 with T x QA_LINES, then src = w2 with (T-1) x stride + K/16 (stride w5,
//             0 = K/16). (QACT)
//   MVM       w0[31:13] = 0 (flag bit 12 = ACC), w5[31:20] = 0, w7 = 0, else ILLEGAL_OP; fmt
//             (sub) >= 2 BAD_FMT; K = w5[15:0] K_RANGE (0, % block, > K_MAX); T = w5[19:16]
//             T_RANGE; SPM_RANGE dst = w1 with (T-1) x stride + ceil(N/16) (stride w6, 0 =
//             ceil(N/16)), x = w2 with T x QA_LINES; DRAM_RANGE w_off = w3 with R x ROW_BEATS on
//             every channel, R = ceil(N / NUM_CH). (MVM)
//   external  every other accepted opcode (VU ops, AT ops; W4-B): the class lives outside this
//             block and answers through the xc_* port (CC §8.2: "opcodes from VU / AT plug in as
//             classes"); its code must already follow the EX §8.1 order (OPERAND included).
// Range rule (spec §21): an operand with start a (the 32-bit immediate) and extent e is in range
// iff a + e <= limit, computed in 48 bits without wrap; SPM limit SPM_BYTES / 64 lines, DRAM
// limit min(CH_BYTES / 64, 2^ADDR_BEAT_W) beats. ERR_INFO = a for SPM_RANGE / DRAM_RANGE, else 0.
// Check order (EX §8.1): ILLEGAL_OP -> (BAD_DEP, the dispatcher's) -> BAD_FMT -> K_RANGE ->
// T_RANGE -> OPERAND -> SPM_RANGE -> DRAM_RANGE, range operands in field order.
// Extents: the operands with e > 0 in field order, compacted into slots 0..3 (the simulator's
// tsr_ext_add skips empty operands); MVM ACC lists dst again as a read after the weights.
// Only meaningful for code 0 (the dispatcher clears them otherwise).
//
// Pipeline (CC §8.2), one op at a time; every stage starts from registers:
//   C0 capture the words and per-word nonzero flags
//   C1 decode: opcode class, refusal, reserved fields, sub codes -> ILLEGAL_OP / BAD_FMT
//   C2 K / T checks, QA_LINES, ROW_BEATS, resolved strides, ceil(N/16), MVM R (power-of-two
//      NUM_CH: a shift; else a 33-step serial divider that stalls the pipeline here)
//   C3 (T-1) x stride, T x QA_LINES; MVM R x ROW_BEATS stage 1 (operands registered)
//   C4 extents (48 bits); R x ROW_BEATS stage 2 (the product)
//   C5 end addresses
//   C6 limit compares
//   C7 priority, ERR_INFO, extents, command build -> rsp_valid (one-cycle strobe)
// = 8 cycles from req_valid to rsp_valid for every core class with power-of-two NUM_CH (MVM with
// odd NUM_CH: + 33); a refused op answers after the same 8 cycles. External classes: C1 raises
// xc_req_valid (held with xc_req_insn until xc_rsp_valid), rsp_valid the cycle after
// xc_rsp_valid.
// Interface: req_valid starts a check when !busy (ignored while busy); the rsp_* payload is
// valid with rsp_valid and held until the next start. rsp_kind is the class's kind assuming its
// own code (the dispatcher turns a NOP / FENCE that it poisons with BAD_DEP into a plain poisoned
// entry, as the simulator does).

module tsr_chk
  import tsr_pkg::*;
#(
  parameter int unsigned     NUM_CH      = TSR_DEF_NUM_CH,
  parameter int unsigned     T_MAX       = TSR_DEF_T_MAX,
  parameter int unsigned     K_MAX       = TSR_DEF_K_MAX,
  parameter int unsigned     SPM_BYTES   = TSR_DEF_SPM_BYTES,
  parameter longint unsigned CH_BYTES    = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter logic [31:0]     CAPS        = 32'd0,
  // unit built (queue depth > 0, CC §7.4): 0 DMA, 1 QACT, 2 MVM, 3 VU, 4 AT, 5 SYNC
  parameter logic [5:0]      UNITS       = 6'b101111,
  parameter int unsigned     XC_CMD_W    = 256      // command payload of an external class
) (
  input  logic                  clk,
  input  logic                  rst_n,

  input  logic                  req_valid,
  input  tsr_insn_t             req_insn,
  output logic                  busy,

  output logic                  rsp_valid,
  output logic [3:0]            rsp_unit,
  output logic [3:0]            rsp_code,
  output logic [31:0]           rsp_info,
  output logic [1:0]            rsp_kind,
  output logic                  rsp_null,
  output logic                  rsp_xc,
  output tsr_ext_t [TSR_EXT_N-1:0] rsp_ext,
  output tsr_dma_cmd_t          rsp_dma,
  output tsr_qact_cmd_t         rsp_qact,
  output tsr_mvm_cmd_t          rsp_mvm,
  output logic [XC_CMD_W-1:0]   rsp_xcmd,

  // external checker classes (VU / AT, CC §8.2, §16)
  output logic                  xc_req_valid,
  output tsr_insn_t             xc_req_insn,
  input  logic                  xc_rsp_valid,
  input  logic [3:0]            xc_rsp_code,
  input  logic [31:0]           xc_rsp_info,
  input  tsr_ext_t [TSR_EXT_N-1:0] xc_rsp_ext,
  input  logic [XC_CMD_W-1:0]   xc_rsp_cmd
);

  // -------------------------------------------------------------------------------------------
  // Constants
  // -------------------------------------------------------------------------------------------
  localparam logic [TSR_EXT_AW-1:0] SPM_LIM  = TSR_EXT_AW'(SPM_BYTES / TSR_LINE_BYTES);
  localparam logic [TSR_EXT_AW-1:0] DRAM_LIM = TSR_EXT_AW'(tsr_dram_limit(CH_BYTES, ADDR_BEAT_W));
  localparam bit          POW2_CH = (NUM_CH & (NUM_CH - 1)) == 0;
  localparam int unsigned LOG2_CH = (NUM_CH > 1) ? $clog2(NUM_CH) : 0;

  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX) begin : g_chk_ch
    $fatal(1, "tsr_chk: NUM_CH must be in [1, 16]");
  end
  if (!UNITS[5]) begin : g_chk_sync
    $fatal(1, "tsr_chk: the SYNC unit must be built (CC §7.4)");
  end

  localparam logic [3:0] U_SYNC = tsr_tables_pkg::TSR_UNIT_SYNC;

  // core classes (one-hot index)
  localparam int unsigned CL_SYNC = 0, CL_LDST = 1, CL_QACT = 2, CL_MVM = 3, CL_XC = 4, CL_REF = 5;

  // -------------------------------------------------------------------------------------------
  // Stage control: st_q[k] = stage Ck holds the op (one op at a time)
  // -------------------------------------------------------------------------------------------
  logic [7:0] st_q;
  logic       xc_wait_q;               // waiting for the external class
  logic       xc_done_q;               // its answer is registered: rsp next
  logic       rsp_q;

  // C0 / C1
  tsr_insn_t  w_q;
  logic [7:0] nz_q;                    // word i != 0
  // C1 results
  logic [5:0] cls_q;                   // one-hot class
  logic [3:0] unit_q;
  logic       ill_q, fmt_bad_q, is_st_q, is_fence_q, acc_q;
  logic       kind_q;                  // QACT kind / MVM fmt (bit 0 of sub, legal values only)
  // C2 results
  logic       krng_q, trng_q;
  logic [31:0] xl_q;                   // QACT K/16
  logic [31:0] ss_q;                   // QACT resolved source stride
  logic [31:0] qal_q;                  // QA_LINES (QACT kind or the MVM fmt's QA kind)
  logic [31:0] yl_q;                   // MVM ceil(N/16)
  logic [31:0] ds_q;                   // MVM resolved dst stride
  logic [31:0] rb_q;                   // ROW_BEATS
  logic [32:0] r_q;                    // MVM R = ceil(N / NUM_CH)
  logic [3:0]  t_q;                    // T (4 LSBs: only used when T is legal)
  // C3 / C4
  logic [TSR_EXT_AW-1:0] e_q [4];      // operand extents in field order (core classes)
  logic [TSR_EXT_AW-1:0] tms_q;        // (T-1) x stride
  logic [32:0]           rr_q;         // R for the product
  logic [15:0]           rbp_q;        // ROW_BEATS for the product (<= 1084)
  // C5 / C6
  logic [TSR_EXT_AW-1:0] a_q [4];      // operand start addresses (zero-extended immediates)
  logic [TSR_EXT_AW-1:0] end_q [4];
  logic [2:0]            oor_q;        // operand k out of range

  // -------------------------------------------------------------------------------------------
  // C1 decode (combinational from the C0 registers)
  // -------------------------------------------------------------------------------------------
  wire [7:0]  op  = w_q[0][7:0];
  wire [3:0]  sub = w_q[0][11:8];

  logic [3:0] d_uof;
  logic       d_legal, d_accept;
  logic [31:0] d_need;
  logic [5:0] d_cls;
  logic       d_ill, d_fmt_bad;
  always_comb begin
    d_uof    = tsr_tables_pkg::tsr_op_unit(op);
    d_legal  = tsr_tables_pkg::tsr_op_legal(op);
    d_need   = tsr_tables_pkg::tsr_op_caps(op);
    d_accept = d_legal && ((CAPS & d_need) == d_need) && UNITS[d_uof[2:0]] && (d_uof < 4'd6);
    d_cls    = '0;
    if (!d_accept) d_cls[CL_REF] = 1'b1;
    else begin
      case (op)
        TSR_OP_NOP, TSR_OP_FENCE: d_cls[CL_SYNC] = 1'b1;
        TSR_OP_LD, TSR_OP_ST:     d_cls[CL_LDST] = 1'b1;
        TSR_OP_QACT:              d_cls[CL_QACT] = 1'b1;
        TSR_OP_MVM:               d_cls[CL_MVM]  = 1'b1;
        default:                  d_cls[CL_XC]   = 1'b1;
      endcase
    end
    // reserved fields and sub codes (EX §7.1; spec §21)
    d_ill     = 1'b0;
    d_fmt_bad = 1'b0;
    if (d_cls[CL_REF]) d_ill = 1'b1;
    if (d_cls[CL_SYNC] && (w_q[0][31:8] != 24'd0 || nz_q[7:1] != 7'd0)) d_ill = 1'b1;
    if (d_cls[CL_LDST] && (w_q[0][31:12] != 20'd0 || nz_q[7:4] != 4'd0 || 32'(sub) >= 32'(NUM_CH)))
      d_ill = 1'b1;
    if (d_cls[CL_QACT] && (w_q[0][31:12] != 20'd0 || nz_q[7:6] != 2'd0 || sub >= 4'd2)) d_ill = 1'b1;
    if (d_cls[CL_MVM] && (w_q[0][31:13] != 19'd0 || w_q[5][31:20] != 12'd0 || nz_q[7])) d_ill = 1'b1;
    if (d_cls[CL_MVM] && sub >= 4'd2) d_fmt_bad = 1'b1;
  end

  // -------------------------------------------------------------------------------------------
  // C2 (combinational from the C1 registers)
  // -------------------------------------------------------------------------------------------
  wire [31:0] w3 = w_q[3], w4 = w_q[4], w5 = w_q[5], w6 = w_q[6];

  logic        c2_krng, c2_trng;
  logic [31:0] c2_xl, c2_ss, c2_qal, c2_yl, c2_ds, c2_rb;
  logic [32:0] c2_ndiv;                // N + NUM_CH - 1
  logic [3:0]  c2_t;
  always_comb begin
    c2_krng = 1'b0;
    c2_trng = 1'b0;
    c2_t    = 4'd0;
    c2_xl   = 32'd0;
    c2_ss   = 32'd0;
    c2_qal  = 32'd0;
    c2_yl   = 32'd0;
    c2_ds   = 32'd0;
    c2_rb   = 32'd0;
    c2_ndiv = 33'(w4) + 33'(NUM_CH - 1);
    if (cls_q[CL_QACT]) begin
      c2_krng = (w3 == 32'd0) || ((w3 & (tsr_qa_block_size(kind_q) - 32'd1)) != 32'd0) ||
                (w3 > 32'(K_MAX));
      c2_trng = (w4 == 32'd0) || (w4 > 32'(T_MAX));
      c2_t    = w4[3:0];
      c2_xl   = w3 >> 4;                                   // K * 4 / 64 (K % 32 = 0)
      c2_ss   = (w5 != 32'd0) ? w5 : c2_xl;                // spec §21 (Stride 0)
      c2_qal  = tsr_qa_lines(kind_q, {16'd0, w3[15:0]});
    end
    if (cls_q[CL_MVM]) begin
      c2_krng = (w5[15:0] == 16'd0) || ((w5[15:0] & (kind_q ? 16'd255 : 16'd31)) != 16'd0) ||
                (32'(w5[15:0]) > 32'(K_MAX));
      c2_trng = (w5[19:16] == 4'd0) || (32'(w5[19:16]) > 32'(T_MAX));
      c2_t    = w5[19:16];
      c2_qal  = tsr_qa_lines(kind_q, {16'd0, w5[15:0]});   // QA kind = fmt (Q8_0 -> A32, Q4_K -> A256)
      c2_yl   = 32'((33'(w4) + 33'd15) >> 4);              // ceil(4N / 64)
      c2_ds   = (w6 != 32'd0) ? w6 : c2_yl;
      c2_rb   = tsr_row_beats(kind_q, {16'd0, w5[15:0]});
    end
  end

  // Odd NUM_CH: restoring division of N + C - 1 by C, quotient MSB first (33 steps).
  logic [32:0] dv_rem_q, dv_quo_q;
  logic [5:0]  dv_cnt_q;
  logic        dv_done;
  if (POW2_CH) begin : g_div_shift
    assign dv_done  = 1'b1;
    assign dv_rem_q = '0;
    assign dv_quo_q = '0;
    assign dv_cnt_q = '0;
    wire unused_dv = &{1'b0, dv_rem_q, dv_cnt_q};
  end else begin : g_div_serial
    logic [32:0] dv_x_q;
    wire  [33:0] dv_try = {dv_rem_q, dv_x_q[32]} - 34'(NUM_CH);
    assign dv_done = (dv_cnt_q == 6'd0);
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        dv_cnt_q <= '0;
      end else if (st_q[1] && cls_q[CL_MVM]) begin
        dv_x_q   <= c2_ndiv;
        dv_rem_q <= '0;
        dv_quo_q <= '0;
        dv_cnt_q <= 6'd33;
      end else if (dv_cnt_q != 6'd0) begin
        if (!dv_try[33]) begin
          dv_rem_q <= dv_try[32:0];
          dv_quo_q <= {dv_quo_q[31:0], 1'b1};
        end else begin
          dv_rem_q <= {dv_rem_q[31:0], dv_x_q[32]};
          dv_quo_q <= {dv_quo_q[31:0], 1'b0};
        end
        dv_x_q   <= {dv_x_q[31:0], 1'b0};
        dv_cnt_q <= dv_cnt_q - 6'd1;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Pipeline registers
  // -------------------------------------------------------------------------------------------
  wire start   = req_valid && !busy;
  wire c2_stall = st_q[2] && !POW2_CH && cls_q[CL_MVM] && !dv_done;   // C2 -> C3 waits for R

  assign busy = (st_q != 8'd0) || xc_wait_q || xc_done_q || rsp_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q      <= '0;
      xc_wait_q <= 1'b0;
      xc_done_q <= 1'b0;
      rsp_q     <= 1'b0;
    end else begin
      rsp_q <= 1'b0;
      // C0
      st_q[0] <= start;
      // C1: an external class leaves the pipeline here
      st_q[1] <= st_q[0];
      if (st_q[1] && cls_q[CL_XC]) begin
        st_q[2]   <= 1'b0;
        xc_wait_q <= 1'b1;
      end else begin
        st_q[2] <= st_q[1];
      end
      if (xc_wait_q && xc_rsp_valid) begin
        xc_wait_q <= 1'b0;
        xc_done_q <= 1'b1;
      end
      if (xc_done_q) begin
        xc_done_q <= 1'b0;
        rsp_q     <= 1'b1;
      end
      // C2 -> C3 (stall for the serial divider)
      if (c2_stall) st_q[3] <= 1'b0;
      else          st_q[3] <= st_q[2];
      if (c2_stall) st_q[2] <= 1'b1;
      st_q[4] <= st_q[3];
      st_q[5] <= st_q[4];
      st_q[6] <= st_q[5];
      st_q[7] <= st_q[6];
      if (st_q[7]) rsp_q <= 1'b1;
    end
  end

  // C0: capture
  always_ff @(posedge clk) begin
    if (start) begin
      w_q <= req_insn;
      for (int i = 0; i < 8; i++) nz_q[i] <= (req_insn[i] != 32'd0);
    end
  end

  // C1: decode
  always_ff @(posedge clk) begin
    if (st_q[0]) begin
      cls_q      <= d_cls;
      unit_q     <= d_accept ? d_uof : U_SYNC;
      ill_q      <= d_ill;
      fmt_bad_q  <= d_fmt_bad;
      is_st_q    <= (op == TSR_OP_ST);
      is_fence_q <= (op == TSR_OP_FENCE);
      acc_q      <= w_q[0][12];
      kind_q     <= sub[0];
    end
  end

  // C2
  always_ff @(posedge clk) begin
    if (st_q[1]) begin
      krng_q <= c2_krng;
      trng_q <= c2_trng;
      t_q    <= c2_t;
      xl_q   <= c2_xl;
      ss_q   <= c2_ss;
      qal_q  <= c2_qal;
      yl_q   <= c2_yl;
      ds_q   <= c2_ds;
      rb_q   <= c2_rb;
      r_q    <= c2_ndiv >> LOG2_CH;        // power-of-two NUM_CH (CC §8.1)
    end
  end

  // C3: products with T, R x ROW_BEATS operands
  logic [TSR_EXT_AW-1:0] tq_x;           // T x QA_LINES
  logic [TSR_EXT_AW-1:0] tm1_x;          // (T-1) x stride
  always_comb begin
    tq_x  = TSR_EXT_AW'(t_q) * TSR_EXT_AW'(qal_q[15:0]);
    tm1_x = TSR_EXT_AW'(t_q - 4'd1) * TSR_EXT_AW'(cls_q[CL_MVM] ? ds_q : ss_q);
  end
  // e_q is written by C3 and C4 from one block: a variable with two procedural drivers is a
  // multi-driven net in synthesis (Vivado Synth 8-6858 keeps the constant driver and ignores the
  // other one). The stages never overlap: the checker holds one op at a time.
  always_ff @(posedge clk) begin
    if (st_q[2] && !c2_stall) begin
      tms_q <= tm1_x;
      rr_q  <= !cls_q[CL_MVM] ? 33'd0 : POW2_CH ? r_q : dv_quo_q;
      rbp_q <= rb_q[15:0];
      e_q[0] <= '0;
      e_q[1] <= '0;
      e_q[2] <= '0;
      e_q[3] <= '0;
      if (cls_q[CL_LDST]) begin
        e_q[0] <= TSR_EXT_AW'(w3);            // SPM lines
        e_q[1] <= TSR_EXT_AW'(w3);            // DRAM beats
      end
      if (cls_q[CL_QACT]) e_q[0] <= tq_x;     // dst: T x QA_LINES
      if (cls_q[CL_MVM])  e_q[1] <= tq_x;     // x:   T x QA_LINES
    end
    // C4: extents
    if (st_q[3]) begin
      if (cls_q[CL_QACT]) e_q[1] <= tms_q + TSR_EXT_AW'(xl_q);          // src
      if (cls_q[CL_MVM]) begin
        e_q[0] <= tms_q + TSR_EXT_AW'(yl_q);                            // dst
        e_q[2] <= TSR_EXT_AW'(rr_q) * TSR_EXT_AW'(rbp_q);               // weights, every channel
      end
    end
  end

  // C5: end addresses (operand k starts at its immediate: LD/ST w1 w2, QACT w1 w2, MVM w1 w2 w3)
  always_ff @(posedge clk) begin
    if (st_q[4]) begin
      a_q[0] <= TSR_EXT_AW'(w_q[1]);
      a_q[1] <= TSR_EXT_AW'(w_q[2]);
      a_q[2] <= TSR_EXT_AW'(w_q[3]);
      a_q[3] <= TSR_EXT_AW'(w_q[1]);
      for (int k = 0; k < 3; k++) end_q[k] <= TSR_EXT_AW'(w_q[k + 1]) + e_q[k];
      end_q[3] <= TSR_EXT_AW'(w_q[1]) + e_q[0];
    end
  end

  // C6: limit compares. Operand kinds per class: LD/ST 0 SPM, 1 DRAM; QACT 0 SPM, 1 SPM;
  // MVM 0 SPM, 1 SPM, 2 DRAM.
  always_ff @(posedge clk) begin
    if (st_q[5]) begin
      oor_q[0] <= end_q[0] > SPM_LIM;
      oor_q[1] <= end_q[1] > (cls_q[CL_LDST] ? DRAM_LIM : SPM_LIM);
      oor_q[2] <= cls_q[CL_MVM] && (end_q[2] > DRAM_LIM);
    end
  end

  // -------------------------------------------------------------------------------------------
  // C7: priority, info, extents, commands
  // -------------------------------------------------------------------------------------------
  logic [3:0]  c7_code;
  logic [31:0] c7_info;
  tsr_ext_t    cand [4];
  tsr_ext_t [TSR_EXT_N-1:0] c7_ext;
  always_comb begin
    // code in EX §8.1 order (BAD_DEP is merged by the dispatcher)
    c7_code = TSR_ERR_NONE;
    c7_info = 32'd0;
    if (ill_q) begin
      c7_code = TSR_ERR_ILLEGAL_OP;
    end else if (fmt_bad_q) begin
      c7_code = TSR_ERR_BAD_FMT;
    end else if ((cls_q[CL_QACT] || cls_q[CL_MVM]) && krng_q) begin
      c7_code = TSR_ERR_K_RANGE;
    end else if ((cls_q[CL_QACT] || cls_q[CL_MVM]) && trng_q) begin
      c7_code = TSR_ERR_T_RANGE;
    end else if (cls_q[CL_LDST] || cls_q[CL_QACT] || cls_q[CL_MVM]) begin
      if (oor_q[0]) begin
        c7_code = TSR_ERR_SPM_RANGE;
        c7_info = w_q[1];
      end else if (oor_q[1]) begin
        c7_code = cls_q[CL_LDST] ? TSR_ERR_DRAM_RANGE : TSR_ERR_SPM_RANGE;
        c7_info = w_q[2];
      end else if (oor_q[2]) begin
        c7_code = TSR_ERR_DRAM_RANGE;
        c7_info = w_q[3];
      end
    end
    // candidate extents in field order (v = e > 0)
    for (int k = 0; k < 4; k++) cand[k] = '0;
    if (cls_q[CL_LDST]) begin
      cand[0] = '{v: e_q[0] != '0, dram: 1'b0, wr: !is_st_q, ch: 8'd0,  lo: a_q[0], hi: end_q[0]};
      cand[1] = '{v: e_q[1] != '0, dram: 1'b1, wr: is_st_q,  ch: 8'(w_q[0][11:8]), lo: a_q[1], hi: end_q[1]};
    end
    if (cls_q[CL_QACT]) begin
      cand[0] = '{v: e_q[0] != '0, dram: 1'b0, wr: 1'b1, ch: 8'd0, lo: a_q[0], hi: end_q[0]};
      cand[1] = '{v: e_q[1] != '0, dram: 1'b0, wr: 1'b0, ch: 8'd0, lo: a_q[1], hi: end_q[1]};
    end
    if (cls_q[CL_MVM]) begin
      cand[0] = '{v: e_q[0] != '0, dram: 1'b0, wr: 1'b1, ch: 8'd0, lo: a_q[0], hi: end_q[0]};
      cand[1] = '{v: e_q[1] != '0, dram: 1'b0, wr: 1'b0, ch: 8'd0, lo: a_q[1], hi: end_q[1]};
      cand[2] = '{v: e_q[2] != '0, dram: 1'b1, wr: 1'b0, ch: TSR_EXT_CH_ALL, lo: a_q[2], hi: end_q[2]};
      cand[3] = '{v: acc_q && e_q[0] != '0, dram: 1'b0, wr: 1'b0, ch: 8'd0, lo: a_q[3], hi: end_q[3]};
    end
    // compact: slot j = the j-th valid candidate
    c7_ext = '0;
    begin
      int unsigned j;
      j = 0;
      for (int k = 0; k < 4; k++) begin
        if (cand[k].v) begin
          c7_ext[j[1:0]] = cand[k];
          j = j + 1;
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (st_q[7]) begin
      rsp_unit <= unit_q;
      rsp_code <= c7_code;
      rsp_info <= c7_info;
      rsp_xc   <= 1'b0;
      rsp_kind <= cls_q[CL_SYNC] ? (is_fence_q ? TSR_QK_FENCE : TSR_QK_NOP) : TSR_QK_OP;
      rsp_null <= cls_q[CL_LDST] && (w_q[3] == 32'd0) && (c7_code == TSR_ERR_NONE);
      rsp_ext  <= (c7_code == TSR_ERR_NONE) ? c7_ext : '0;
      rsp_dma  <= '{is_st: is_st_q, ch: w_q[0][11:8], spm_line: w_q[1], dram_beat: w_q[2],
                    n_lines: w_q[3]};
      rsp_qact <= '{kind: kind_q, dst_line: w_q[1], src_line: w_q[2], k: w_q[3][15:0],
                    t: w_q[4][3:0], src_stride: ss_q, qa_lines: qal_q[15:0]};
      rsp_mvm  <= '{fmt: kind_q, acc: acc_q, dst_line: w_q[1], x_line: w_q[2], w_off: w_q[3],
                    n: w_q[4], k: w_q[5][15:0], t: w_q[5][19:16], dst_stride: ds_q,
                    y_lines: yl_q, rows_per_ch: rr_q[31:0], row_beats: rb_q, qa_lines: qal_q[15:0]};
      rsp_xcmd <= '0;
    end
    if (xc_wait_q && xc_rsp_valid) begin
      rsp_unit <= unit_q;
      rsp_code <= xc_rsp_code;
      rsp_info <= (xc_rsp_code == TSR_ERR_NONE) ? 32'd0 : xc_rsp_info;
      rsp_xc   <= 1'b1;
      rsp_kind <= TSR_QK_OP;
      rsp_null <= 1'b0;
      rsp_ext  <= (xc_rsp_code == TSR_ERR_NONE) ? xc_rsp_ext : '0;
      rsp_dma  <= '0;
      rsp_qact <= '0;
      rsp_mvm  <= '0;
      rsp_xcmd <= xc_rsp_cmd;
    end
  end

  assign rsp_valid    = rsp_q;
  assign xc_req_valid = xc_wait_q;
  assign xc_req_insn  = w_q;

  wire unused_ok = &{1'b0, rb_q[31:16], qal_q[31:16], nz_q[0]};

`ifndef SYNTHESIS
  // Simulation checks: one op at a time; an external answer only when asked.
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      assert (!(xc_rsp_valid && !xc_wait_q))
        else $error("tsr_chk: xc_rsp_valid without a pending external-class request");
      assert ($onehot0(st_q))
        else $error("tsr_chk: more than one op in the pipeline (%b)", st_q);
    end
  end
`endif

endmodule
