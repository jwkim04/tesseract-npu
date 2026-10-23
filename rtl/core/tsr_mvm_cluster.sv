// tsr_mvm_cluster.sv — one MVM cluster (spec 클러스터: the MVM engine of one channel): front end,
// activation buffer, back end, and the glue between them and the broadcast / result pipes.
//
// Owner: top agent (docs/rtl-m3.md §2 block (top); contract §3.1, §3.2, §3.5, §6.2).
//
// Contract:
//   - broadcast input (after tsr_mvm's pipe, no backpressure, docs/rtl-m3.md §3.1): per MVM with
//     N >= 1 exactly one bc_cmd (tsr_mvm_ccmd_t), then T * QA_LINES bc_line items, the final one
//     with last = 1; the next MVM's bc_cmd comes only after the previous MVM finished everywhere
//     (tsr_mvm waits for every expected last result). The cluster accepts every item in any
//     state: bc_line always goes into the activation buffer (tsr_mvm_actbuf write port).
//   - setup on bc_cmd: first = CH * R (R = rows_per_ch, 64-bit product), rows_c =
//     tsr_mvm_rows_c(n, R, first), beats_c = rows_c * row_beats (multi-cycle arithmetic allowed);
//     if rows_c >= 1: one tsr_mvm_fcmd_t {fmt, k, row_beats, w_off, rows_c, beats_c} to the front
//     end (which starts streaming weights at once, overlapping the broadcast), cfg_fmt /
//     cfg_d_lines to the back end. rows_c = 0: the cluster stays silent (no memory traffic, no
//     result) and only absorbs the broadcast.
//   - activation gating: front -> back beats are passed only after the bc_line with last = 1 of
//     this MVM has been written into the activation buffer (and until the MVM's last result).
//   - results (R1/R2, through tsr_mvm's result pipe): one item per row in row order:
//     res_row = first + r (global row index n), res_last = (r == rows_c - 1), res_y = the back
//     end's T_MAX results.
//   - memory: the front end's WS read source port, unchanged (docs/rtl-m3.md §4).
//   - idle: no MVM in progress in this cluster (setup done or not started, front and back idle).
// Placement: one cluster per channel, in the SLR of that channel's DDR controller (spec 배치);
//   only the broadcast pipe, the result pipe and the memory port leave it.
//
// Interface addition (additive, top agent): parameter WS_RD_LAT (weight FIFO RAM read latency,
//   docs/rtl-m3.md §1, default TSR_DEF_WS_RD_LAT) is passed to tsr_mvm_front.
//
// Microarchitecture:
//   - bc_line -> one register (aw_*: the activation-buffer write port; docs/rtl-m3.md §9.2 item 1:
//     one more register in front of the 3 x T_MAX activation RAMs) -> tsr_mvm_actbuf.
//     act_ready_q := 0 at bc_cmd, := 1 at the edge that writes the line with last = 1.
//   - setup pipeline (one-hot su_q; the front command is valid 8 cycles after the bc_cmd cycle):
//       su0: CH * R as two partial products (CH & 3) * R and (CH >> 2) * R (constant shifts / adds)
//       su1: first = sum (36 bits)
//       su2: N > first, N - first
//       su3: rows_c = min(R, N - first) or 0
//       su4..su6: beats = rows_c * row_beats (32 x 16 multiplier with 3 pipeline registers: maps
//                 to cascaded DSPs with M / P registers; ~2 DSP48E2 per cluster); the front
//                 command becomes valid with the last product register (rows_c >= 1) and is held
//                 until the front end takes it (R2)
//     The front end starts streaming at once; the setup hides behind the broadcast (T * QA_LINES
//     lines) and the memory round trip.
//   - beat gating (docs/rtl-m3.md §3.2): beat_valid_back = beat_valid & act_ready_q,
//     beat_ready_front = beat_ready & act_ready_q (flops, one LUT).
//   - result tagging: row_q (= first + r) and left_q (rows still to come) count result
//     handshakes; res_last_q = (left_q == 1). res_valid / res_y / res_ready pass straight between
//     the back end and the result pipe (no combinational input -> output path: the back end's
//     res_valid does not depend on res_ready, R2).
//   Simulation checks: protocol of the broadcast (one MVM at a time), results only for this
//   cluster's rows, every row's result exactly once.
//   Cost (RTL structure): ~1.1 K FF (527 activation-buffer write register, ~400 setup) and the
//   32 x 16 product (2 DSP48E2 expected) besides the front end, activation buffer and back end.

module tsr_mvm_cluster
  import tsr_pkg::*;
#(
  parameter int unsigned CH            = 0,     // channel index c of this cluster
  parameter int unsigned K_MAX         = TSR_DEF_K_MAX,
  parameter int unsigned T_MAX         = TSR_DEF_T_MAX,
  parameter int unsigned ADDR_BEAT_W   = TSR_DEF_ADDR_BEAT_W,
  parameter int unsigned BEAT_BITS     = TSR_DEF_BEAT_BITS,
  parameter int unsigned WS_FIFO_BEATS = TSR_DEF_WS_FIFO_BEATS,
  parameter int unsigned ACT_RD_LAT    = TSR_DEF_ACT_RD_LAT,
  parameter int unsigned MAC_LAT       = TSR_DEF_MAC_LAT,
  parameter int unsigned MUL_LAT       = TSR_DEF_FMUL_LAT,
  parameter int unsigned ADD_LAT       = TSR_DEF_FADD_LAT,
  parameter int unsigned WS_RD_LAT     = TSR_DEF_WS_RD_LAT
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // ---- broadcast stream (tsr_mvm -> cluster, no backpressure) ----
  input  logic                          bc_cmd_valid,
  input  tsr_mvm_ccmd_t                 bc_cmd,
  input  logic                          bc_line_valid,
  input  tsr_mvm_bcline_t               bc_line,

  // ---- read source port (tsr_mem_arb source TSR_SRC_WS of channel CH) ----
  output logic                          rd_req_valid,
  input  logic                          rd_req_ready,
  output logic [ADDR_BEAT_W-1:0]        rd_req_addr,
  output logic [TSR_LEN_W-1:0]          rd_req_len,
  output logic [TSR_SRC_ID_W-1:0]       rd_req_id,
  input  logic                          rd_rsp_valid,
  output logic                          rd_rsp_ready,
  input  logic [BEAT_BITS-1:0]          rd_rsp_data,
  input  logic                          rd_rsp_last,
  input  logic [TSR_SRC_ID_W-1:0]       rd_rsp_id,

  // ---- row results (cluster -> tsr_mvm gather) ----
  output logic                          res_valid,
  input  logic                          res_ready,
  output logic [31:0]                   res_row,
  output logic                          res_last,
  output logic [T_MAX-1:0][31:0]        res_y,

  output logic                          idle
);

  if (CH >= TSR_NUM_CH_MAX) begin : g_chk_ch
    $fatal(1, "tsr_mvm_cluster: CH must be < 16");
  end

  localparam int unsigned QS_WORDS  = tsr_actbuf_qs_words(K_MAX);
  localparam int unsigned DBS_LINES = tsr_actbuf_dbs_lines(K_MAX);
  localparam int unsigned QS_AW     = (QS_WORDS > 1) ? $clog2(QS_WORDS) : 1;
  localparam int unsigned DBS_AW    = (DBS_LINES > 1) ? $clog2(DBS_LINES) : 1;
  localparam logic [35:0] CH_LO     = 36'(CH & 3);
  localparam logic [35:0] CH_HI     = 36'(CH >> 2);

  // -------------------------------------------------------------------------------------------
  // Broadcast: activation-buffer write register and activation gating
  // -------------------------------------------------------------------------------------------
  logic            aw_v_q;
  tsr_mvm_bcline_t aw_q;
  logic            act_ready_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      aw_v_q      <= 1'b0;
      act_ready_q <= 1'b0;
    end else begin
      aw_v_q <= bc_line_valid;
      if (bc_cmd_valid)            act_ready_q <= 1'b0;
      else if (aw_v_q && aw_q.last) act_ready_q <= 1'b1;
    end
  end
  always_ff @(posedge clk) begin
    if (bc_line_valid) aw_q <= bc_line;
  end

  // -------------------------------------------------------------------------------------------
  // Setup (docs/rtl-m3.md §3.2)
  // -------------------------------------------------------------------------------------------
  logic [6:0]                 su_q;        // one-hot setup step
  logic                       busy_q;      // MVM in progress (bc_cmd .. last result / no rows)
  logic                       fmt_q;
  logic [15:0]                k_q, rb_q;
  logic [31:0]                n_q, r_q, woff_q;
  logic [7:0]                 dl_q;
  logic [35:0]                plo_q, phi_q, first_q, left_q;
  logic                       gt_q;
  logic [31:0]                rows_q;
  logic                       has_rows_q;
  logic [47:0]                mul1_q, mul2_q, mul3_q;
  logic                       fc_v_q;
  logic                       f_cmd_ready;

  // result tagging
  logic [31:0]                row_q, left_rows_q;
  logic                       res_last_q;

  wire res_fire = res_valid & res_ready;
  wire fc_fire  = fc_v_q & f_cmd_ready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      su_q        <= '0;
      busy_q      <= 1'b0;
      fmt_q       <= 1'b0;                 // back-end configuration: defined after reset
      dl_q        <= '0;
      fc_v_q      <= 1'b0;
      has_rows_q  <= 1'b0;
      row_q       <= '0;
      left_rows_q <= '0;
      res_last_q  <= 1'b0;
    end else begin
      su_q <= {su_q[5:0], bc_cmd_valid};
      if (bc_cmd_valid) begin
        busy_q <= 1'b1;
        fmt_q  <= bc_cmd.fmt;
        dl_q   <= bc_cmd.d_lines;
      end
      if (su_q[3]) begin
        has_rows_q  <= gt_q;
        row_q       <= first_q[31:0];
      end
      if (su_q[4]) begin
        left_rows_q <= rows_q;
        res_last_q  <= (rows_q == 32'd1);
        if (!has_rows_q) busy_q <= 1'b0;   // rows_c = 0: only the broadcast is absorbed
      end
      if (su_q[6] && has_rows_q) fc_v_q <= 1'b1;
      else if (fc_fire)          fc_v_q <= 1'b0;
      if (res_fire) begin
        row_q       <= row_q + 32'd1;
        left_rows_q <= left_rows_q - 32'd1;
        res_last_q  <= (left_rows_q == 32'd2);
        if (res_last_q) busy_q <= 1'b0;
      end
    end
  end

  // Setup datapath (no reset: qualified by su_q).
  always_ff @(posedge clk) begin
    if (bc_cmd_valid) begin
      k_q    <= bc_cmd.k;
      n_q    <= bc_cmd.n;
      r_q    <= bc_cmd.rows_per_ch;
      rb_q   <= bc_cmd.row_beats;
      woff_q <= bc_cmd.w_off;
    end
    // su0: CH * R = (CH & 3) * R + ((CH >> 2) * R) << 2 (constant factors <= 3)
    plo_q   <= CH_LO * 36'(r_q);
    phi_q   <= CH_HI * 36'(r_q);
    // su1
    first_q <= plo_q + (phi_q << 2);
    // su2
    gt_q    <= (36'(n_q) > first_q);
    left_q  <= 36'(n_q) - first_q;
    // su3: rows_c = max(0, min(R, N - first)) (tsr_mvm_rows_c)
    rows_q  <= !gt_q ? 32'd0 : ((left_q < 36'(r_q)) ? left_q[31:0] : r_q);
    // su4..su6: beats_c = rows_c * row_beats
    mul1_q  <= 48'(rows_q) * 48'(rb_q);
    mul2_q  <= mul1_q;
    mul3_q  <= mul2_q;
  end

  // -------------------------------------------------------------------------------------------
  // Front end, activation buffer, back end
  // -------------------------------------------------------------------------------------------
  tsr_mvm_fcmd_t fc;
  always_comb begin
    fc.fmt       = fmt_q;
    fc.k         = k_q;
    fc.row_beats = rb_q;
    fc.w_off     = woff_q;
    fc.rows      = rows_q;
    fc.beats     = mul3_q[TSR_MVM_BEATS_W-1:0];
  end

  logic          f_beat_valid, f_beat_ready, f_idle;
  tsr_mvm_beat_t f_beat;
  logic          b_beat_ready, b_idle;
  logic          qs_re, dbs_re;
  logic [QS_AW-1:0]  qs_addr;
  logic [DBS_AW-1:0] dbs_addr;
  logic [T_MAX-1:0][1023:0] qs_data;
  logic [T_MAX-1:0][511:0]  dbs_data;

  tsr_mvm_front #(
    .K_MAX(K_MAX), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .WS_FIFO_BEATS(WS_FIFO_BEATS),
    .WS_RD_LAT(WS_RD_LAT)
  ) u_front (
    .clk, .rst_n,
    .cmd_valid(fc_v_q), .cmd_ready(f_cmd_ready), .cmd(fc),
    .rd_req_valid, .rd_req_ready, .rd_req_addr, .rd_req_len, .rd_req_id,
    .rd_rsp_valid, .rd_rsp_ready, .rd_rsp_data, .rd_rsp_last, .rd_rsp_id,
    .beat_valid(f_beat_valid), .beat_ready(f_beat_ready), .beat(f_beat),
    .idle(f_idle)
  );

  assign f_beat_ready = b_beat_ready & act_ready_q;

  tsr_mvm_actbuf #(.K_MAX(K_MAX), .T_MAX(T_MAX), .RD_LAT(ACT_RD_LAT)) u_act (
    .clk, .rst_n,
    .wr_valid(aw_v_q), .wr_line(aw_q),
    .qs_re, .qs_addr, .qs_data, .dbs_re, .dbs_addr, .dbs_data
  );

  tsr_mvm_back #(
    .K_MAX(K_MAX), .T_MAX(T_MAX), .ACT_RD_LAT(ACT_RD_LAT), .MAC_LAT(MAC_LAT), .MUL_LAT(MUL_LAT),
    .ADD_LAT(ADD_LAT)
  ) u_back (
    .clk, .rst_n,
    .cfg_fmt(fmt_q), .cfg_d_lines(dl_q),
    .beat_valid(f_beat_valid & act_ready_q), .beat_ready(b_beat_ready), .beat(f_beat),
    .qs_re, .qs_addr, .qs_data, .dbs_re, .dbs_addr, .dbs_data,
    .res_valid, .res_ready, .res_y,
    .idle(b_idle)
  );

  assign res_row  = row_q;
  assign res_last = res_last_q;
  assign idle     = ~busy_q & ~fc_v_q & f_idle & b_idle;

  wire unused_ok = &{1'b0, mul3_q[47:TSR_MVM_BEATS_W], left_q[35:32]};

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      assert (!(bc_cmd_valid && bc_line_valid)) else $error("%m: bc_cmd and bc_line in one cycle");
      if (bc_cmd_valid) begin
        assert (!busy_q && su_q == '0) else $error("%m: bc_cmd while an MVM is in progress");
        assert (bc_cmd.n != 32'd0 && bc_cmd.rows_per_ch != 32'd0) else $error("%m: bc_cmd with N = 0");
        assert (32'(bc_cmd.row_beats) == tsr_row_beats(bc_cmd.fmt, 32'(bc_cmd.k)))
          else $error("%m: row_beats %0d != ROW_BEATS", bc_cmd.row_beats);
        assert (32'(bc_cmd.d_lines) == tsr_qa_d_lines(bc_cmd.fmt, 32'(bc_cmd.k)))
          else $error("%m: d_lines %0d != QA D lines", bc_cmd.d_lines);
      end
      if (su_q[4]) begin
        assert (rows_q == tsr_mvm_rows_c(n_q, r_q, 64'(CH) * 64'(r_q)))
          else $error("%m: rows_c %0d", rows_q);
      end
      if (fc_fire) begin
        assert (64'(fc.beats) == 64'(rows_q) * 64'(rb_q)) else $error("%m: beats != rows_c * row_beats");
      end
      if (res_valid) begin
        assert (busy_q && has_rows_q && left_rows_q != 32'd0 && su_q == '0)
          else $error("%m: result without a pending row");
      end
      if (f_beat_valid && f_beat_ready) begin
        assert (act_ready_q) else $error("%m: beat passed before the activations");
      end
    end
  end
`endif

endmodule
