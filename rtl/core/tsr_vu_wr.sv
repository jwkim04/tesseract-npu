// tsr_vu_wr.sv — the VU's write side: the lane-exit assembler, the output FIFO, the reduction and TOPK
// result FIFOs, the SPM write port (both banks of client 4), and the local-memory and s-register writes
// of lane results.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §7.1 output FIFO, §7.7 writes and flow control, §4.2 tail
// rule).
// Exits: the injector's descriptors are delayed to their lanes' exits (front TSR_VU_LF + 1, back TSR_VU_LB + 1
// cycles: the lanes register the injector's outputs once more; LANE_X more behind the lane pipes of
// tsr_vu_lanes). The injector's exit-order rule makes
// the exits strictly ordered, so one assembler serves both: a descriptor with nparts n and part k
// places lanes 0 .. 16/n - 1 at words 16k/n ..; the last part emits the line with the tail rule (words
// >= tail written 0, VU §4.2) to its destination: the output FIFO (SPM), the local memory, or the s
// register d0 (word dword of the line). SOP STS writes 4 bytes (VU §5.8).
// Flow control (R4, VU §7.7): the generator reserves output-FIFO and RED-FIFO places before the slot's
// reads (oc_take, rc_take); a place is free again when its line leaves for the SPM. The lanes therefore
// never wait, and a stalled SPM write only stops new reads.
// SPM write: one request register (R2) loaded from the FIFO heads in turn (RED, TOPK, output);
// wr_line[b] / wr_valid[b] for the line's bank b. A write is done when tsr_spm accepts it (VU §4.5):
// ev_w[tag] pulses then (and for s-register writes), for the top's done count.

module tsr_vu_wr
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
#(
  parameter  int unsigned SPM_BYTES = TSR_DEF_SPM_BYTES,
  parameter  int unsigned OFD       = 64,     // output FIFO lines
  parameter  int unsigned RFD       = 4,      // RED result lines
  parameter  int unsigned TFD       = 4,      // TOPK result lines
  parameter  int unsigned LMAW      = 9,
  parameter  int unsigned LANE_X    = 0,      // cycles the lane pipes add to every exit (tsr_vu_lanes: 2 x stages)
  localparam int unsigned LINES     = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW        = (LINES > 1) ? $clog2(LINES) : 1
) (
  input  logic                           clk,
  input  logic                           rst_n,
  // descriptors (tsr_vu_inj) and the lane exits
  input  logic                           d_v,
  input  logic                           d_back,
  input  tsr_vu_desc_t                   d_desc,
  input  logic [15:0][31:0]              f_y,
  input  logic [15:0][31:0]              b_y,
  // credits
  input  logic [1:0]                     oc_take,
  output logic [7:0]                     oc_free,
  input  logic                           rc_take,
  output logic [3:0]                     rc_free,
  // RED result lines (tsr_vu_red; places reserved through rc_take)
  input  logic                           r_v,
  input  logic                           r_tag,
  input  logic [TSR_VU_LW-1:0]           r_line,
  input  logic [TSR_LINE_BITS-1:0]       r_data,
  input  logic [TSR_SPM_BE_W-1:0]        r_be,
  // TOPK result lines
  input  logic                           t_v,
  output logic                           t_ready,
  input  logic                           t_tag,
  input  logic [TSR_VU_LW-1:0]           t_line,
  input  logic [TSR_LINE_BITS-1:0]       t_data,
  // local memory and s-register writes of lane results
  output logic                           lm_we,
  output logic [LMAW-1:0]                lm_wa,
  output logic [TSR_LINE_BITS-1:0]       lm_wd,
  output logic                           s_we,
  output logic [3:0]                     s_wi,
  output logic [31:0]                    s_wd,
  output logic                           s_wtag,
  // SPM client 4, write side
  output logic [1:0]                     wr_valid,
  input  logic [1:0]                     wr_ready,
  output logic [1:0][AW-1:0]             wr_line,
  output logic [1:0][TSR_LINE_BITS-1:0]  wr_data,
  output logic [1:0][TSR_SPM_BE_W-1:0]   wr_be,
  // events
  output logic [1:0]                     ev_w,        // [tag]: a write of that op completed
  output logic                           ev_line,     // an SPM line was written (VU_LINES)
  output logic                           wr_wait,     // a write request is raised and not granted
  output logic                           idle
);

  localparam int unsigned DF = TSR_VU_LF + LANE_X + 1;
  localparam int unsigned DB = TSR_VU_LB + LANE_X + 1;

  // -------------------------------------------------------------------------------------------
  // Descriptor delay lines to the exits
  // -------------------------------------------------------------------------------------------
  localparam int unsigned DW = $bits(tsr_vu_desc_t);
  logic         fv, bv;
  logic [DF-1:0] fv_sr;
  logic [DB-1:0] bv_sr;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      fv_sr <= '0;
      bv_sr <= '0;
    end else begin
      fv_sr <= {fv_sr[DF-2:0], d_v && !d_back};
      bv_sr <= {bv_sr[DB-2:0], d_v && d_back};
    end
  end
  assign fv = fv_sr[DF-1];
  assign bv = bv_sr[DB-1];
  // the descriptors arrive one cycle before their exit: the placement is decoded into registers then
  tsr_vu_desc_t fd1, bd1;
  tsr_vu_dly #(.W(DW), .D(DF - 1)) u_fd (.clk, .d(d_desc), .q(fd1));
  tsr_vu_dly #(.W(DW), .D(DB - 1)) u_bd (.clk, .d(d_desc), .q(bd1));
  wire fv1 = fv_sr[DF-2];
  wire bv1 = bv_sr[DB-2];

  // -------------------------------------------------------------------------------------------
  // Assembler
  // -------------------------------------------------------------------------------------------
  // P (the cycle before the exit, registered): the exiting part's descriptor, which words it fills,
  // which words are zero (tail rule, applied on the last part), whether the line is complete
  tsr_vu_desc_t      xd;                                 // the exiting part's descriptor
  logic              bsel_q, emit_q;
  logic [15:0]       plc_q, zr_q;
  logic [1:0]        wsel_q;                             // part width: 0 16 words, 1 8, 2 4
  always_ff @(posedge clk) begin
    tsr_vu_desc_t d;
    int unsigned  w;
    d = bv1 ? bd1 : fd1;
    w = (d.nparts == 3'd4) ? 4 : (d.nparts == 3'd2) ? 8 : 16;
    xd     <= d;
    bsel_q <= bv1;
    emit_q <= (fv1 || bv1) && (32'(d.part) == 32'(d.nparts) - 1);
    wsel_q <= (w == 4) ? 2'd2 : (w == 8) ? 2'd1 : 2'd0;
    for (int i = 0; i < 16; i++) begin
      plc_q[i] <= (32'(i) / w == 32'(d.part));
      zr_q[i]  <= (32'(i) >= 32'(d.tail));
    end
  end
  wire xv   = fv || bv;
  wire emit = emit_q;
  wire unused_xd = &{1'b0, xd.part, xd.nparts};        // decoded into plc_q / wsel_q / emit_q

  logic [15:0][31:0] xy;
  assign xy = bsel_q ? b_y : f_y;

  logic [15:0][31:0] lb_q;                               // the line being assembled
  logic [15:0][31:0] ln;                                 // the line with this part placed
  always_comb begin
    for (int i = 0; i < 16; i++) begin
      logic [31:0] v;
      case (wsel_q)
        2'd2:    v = xy[i % 4];
        2'd1:    v = xy[i % 8];
        default: v = xy[i];
      endcase
      ln[i] = zr_q[i] ? 32'd0 : plc_q[i] ? v : lb_q[i];
    end
  end
  always_ff @(posedge clk) if (xv) lb_q <= ln;

  // destinations
  logic                     o_push;
  logic [TSR_VU_LW-1:0]     o_line;
  logic [TSR_LINE_BITS-1:0] o_data;
  logic [TSR_SPM_BE_W-1:0]  o_be;
  logic                     o_tag;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      o_push <= 1'b0;
      lm_we  <= 1'b0;
      s_we   <= 1'b0;
    end else begin
      o_push <= emit && xd.dst == DS_SPM;
      lm_we  <= emit && xd.dst == DS_LM;
      s_we   <= emit && xd.dst == DS_SREG;
    end
    o_line <= xd.line;
    o_tag  <= xd.tag;
    o_data <= ln;
    o_be   <= xd.sts ? (TSR_SPM_BE_W'(4'hF) << (4 * 32'(xd.dword))) : '1;
    if (xd.keep) begin
      for (int i = 0; i < 16; i++) o_be[4*i +: 4] <= (32'(i) < 32'(xd.tail)) ? 4'hF : 4'h0;
    end
    if (xd.sts) o_data <= {16{ln[0]}};
    lm_wa  <= LMAW'(xd.line);
    lm_wd  <= ln;
    s_wi   <= xd.line[3:0];
    s_wd   <= ln[xd.dword];
    s_wtag <= xd.tag;
  end

  // -------------------------------------------------------------------------------------------
  // FIFOs (line, data, be, tag)
  // -------------------------------------------------------------------------------------------
  localparam int unsigned EW = 1 + TSR_VU_LW + TSR_LINE_BITS + TSR_SPM_BE_W;

  logic          of_v, rf_v, tf_v;
  logic [EW-1:0] of_h, rf_h, tf_h;
  logic          of_pop, rf_pop, tf_pop;
  logic [$clog2(OFD):0] of_n;
  logic [$clog2(RFD):0] rf_n;
  logic [$clog2(TFD):0] tf_n;

  tsr_vu_fifo #(.W(EW), .D(OFD)) u_of (
    .clk, .rst_n, .push(o_push), .din({o_tag, o_line, o_data, o_be}), .pop(of_pop), .v(of_v), .dout(of_h), .n(of_n)
  );
  tsr_vu_fifo #(.W(EW), .D(RFD)) u_rf (
    .clk, .rst_n, .push(r_v), .din({r_tag, r_line, r_data, r_be}), .pop(rf_pop), .v(rf_v), .dout(rf_h), .n(rf_n)
  );
  tsr_vu_fifo #(.W(EW), .D(TFD)) u_tf (
    .clk, .rst_n, .push(t_v && t_ready), .din({t_tag, t_line, t_data, {TSR_SPM_BE_W{1'b1}}}), .pop(tf_pop),
    .v(tf_v), .dout(tf_h), .n(tf_n)
  );
  assign t_ready = (32'(tf_n) < TFD);

  // credits: places reserved and not yet freed (freed when the line leaves the FIFO). A reservation is
  // registered first (oc_take_q, rc_take_q: off the generator's load path) and counts as taken from then.
  logic [$clog2(OFD)+1:0] oc_res;
  logic [$clog2(RFD)+1:0] rc_res;
  logic [1:0]             oc_take_q;
  logic                   rc_take_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      oc_res    <= '0;
      rc_res    <= '0;
      oc_take_q <= '0;
      rc_take_q <= 1'b0;
    end else begin
      oc_take_q <= oc_take;
      rc_take_q <= rc_take;
      oc_res    <= oc_res + ($clog2(OFD)+2)'(oc_take_q) - ($clog2(OFD)+2)'(of_pop);
      rc_res    <= rc_res + ($clog2(RFD)+2)'(rc_take_q) - ($clog2(RFD)+2)'(rf_pop);
    end
  end
  always_comb begin
    logic [31:0] f;
    f = OFD - 32'(oc_res) - 32'(oc_take_q);
    oc_free = (f > 32'd255) ? 8'd255 : 8'(f);
    f = RFD - 32'(rc_res) - 32'(rc_take_q);
    rc_free = (f > 32'd15) ? 4'd15 : 4'(f);
  end

  // -------------------------------------------------------------------------------------------
  // SPM write request register
  // -------------------------------------------------------------------------------------------
  logic          wq_v;
  logic [EW-1:0] wq;
  wire           wq_acc = wq_v && wr_ready[wq[TSR_LINE_BITS + TSR_SPM_BE_W]];   // the line's bit 0
  wire           wq_free = !wq_v || wq_acc;
  assign rf_pop = wq_free && rf_v;
  assign tf_pop = wq_free && !rf_v && tf_v;
  assign of_pop = wq_free && !rf_v && !tf_v && of_v;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wq_v <= 1'b0;
    end else if (wq_free) begin
      wq_v <= rf_v || tf_v || of_v;
    end
    if (wq_free) wq <= rf_v ? rf_h : tf_v ? tf_h : of_h;
  end
  wire                     wq_tag  = wq[EW-1];
  wire [TSR_VU_LW-1:0]     wq_line = wq[TSR_LINE_BITS + TSR_SPM_BE_W +: TSR_VU_LW];
  for (genvar b = 0; b < 2; b++) begin : g_wb
    assign wr_valid[b] = wq_v && (wq_line[0] == 1'(b));
    assign wr_line[b]  = AW'(wq_line);
    assign wr_data[b]  = wq[TSR_SPM_BE_W +: TSR_LINE_BITS];
    assign wr_be[b]    = wq[TSR_SPM_BE_W-1:0];
  end

  always_comb begin
    ev_w = '0;
    if (wq_acc) ev_w[wq_tag] = 1'b1;
  end
  assign ev_line = wq_acc;
  assign wr_wait = wq_v && !wq_acc;
  assign idle    = !wq_v && !of_v && !rf_v && !tf_v && (fv_sr == '0) && (bv_sr == '0) && !o_push;

  wire unused_ok = &{1'b0, wq_line, of_n, rf_n};

endmodule

// One FIFO with a first-word-fall-through output (dout = the oldest entry while v): memory + head
// register. Helper of tsr_vu_wr / tsr_vu (lives here so the VU keeps its files tsr_vu*.sv).
/* verilator lint_off DECLFILENAME */
module tsr_vu_fifo #(
  parameter  int unsigned W = 8,
  parameter  int unsigned D = 4,
  localparam int unsigned PW = (D > 1) ? $clog2(D) : 1
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         push,
  input  logic [W-1:0] din,
  input  logic         pop,
  output logic         v,
  output logic [W-1:0] dout,
  output logic [PW:0]  n
);
  logic [W-1:0]  mem [D];
  logic [PW-1:0] rp, wp;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rp <= '0;
      wp <= '0;
      n  <= '0;
    end else begin
      if (push) wp <= (32'(wp) == D - 1) ? '0 : wp + 1'b1;
      if (pop)  rp <= (32'(rp) == D - 1) ? '0 : rp + 1'b1;
      n <= n + (PW+1)'(push) - (PW+1)'(pop);
    end
  end
  always_ff @(posedge clk) if (push) mem[wp] <= din;
  assign v    = (n != '0);
  assign dout = mem[rp];
`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    if (rst_n) begin
      assert (!(push && !pop && 32'(n) >= D)) else $error("%m: FIFO overflow");
      assert (!(pop && n == '0)) else $error("%m: FIFO underflow");
    end
  end
`endif
endmodule
/* verilator lint_on DECLFILENAME */
