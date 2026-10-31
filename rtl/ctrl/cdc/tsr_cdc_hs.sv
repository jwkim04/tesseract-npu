// tsr_cdc_hs.sv — tagged request / response handshake between two clock domains (CC §11.2: SPM
// bridge, core window, LQ / LCQ registers, RT_* mirrors, LOAD_*, DMI, error records;
// docs/ctrl-riscv.md §7.2 "tsr_cdc_hs ... with a 1-bit request tag, used by every request path that
// can time out").
//
// Owner: ctrl agent (W4 item 3; CC §2.2 `tsr_cdc_hs`).
//
// Protocol: two-phase toggles. The requester (domain A) holds the request word in a register and
// flips req_tgl; the responder (domain B) sees req_tgl != ack_tgl through a synchroniser, answers
// with the response word in a register and flips ack_tgl; A sees ack_tgl == req_tgl through a
// synchroniser. The toggle value is the request's tag: A changes the request register only when
// the tags agree again, so B always reads a stable request, and A reads the response register only
// after B wrote it (both registers cross with set_max_delay -datapath_only, CC §11.3).
//
// A side (clk_a):
//   a_ready      a new request may be given (the previous one is done; also while a timed-out one
//                still drains)
//   a_req        one-cycle request with a_req_data (only while a_ready)
//   a_local      sampled with a_req: answer the request locally at once (a_done with a_timeout,
//                nothing crosses), for "nothing crosses into a domain in reset" (CC §11.2)
//   a_done       one-cycle pulse: the request is finished; a_rsp_data valid in that cycle;
//                a_timeout = 1 if it was answered locally or did not complete within TIMEOUT clk_a
//                cycles (counted from a_req). A late response of a timed-out request is discarded
//                by its tag; a request given meanwhile waits until it has drained, its own TIMEOUT
//                counting from its a_req (so every request is done within TIMEOUT + 1 cycles).
//                TIMEOUT = 0: no bound.
// B side (clk_b):
//   b_valid      a request is pending (level, from a register); b_req_data (a B-side register,
//                loaded from the A-side request register when the toggle arrives) stable while
//                b_valid
//   b_rsp        one-cycle answer with b_rsp_data (only while b_valid); b_valid is 0 from the
//                next cycle on
// Resets: synchronous, active low, from the same hardware reset on both sides.

module tsr_cdc_hs #(
  parameter int unsigned REQ_W   = 32,
  parameter int unsigned RSP_W   = 32,
  parameter int unsigned TIMEOUT = 4096,
  parameter int unsigned STAGES  = 2
) (
  input  logic             clk_a,
  input  logic             rst_a_n,
  output logic             a_ready,
  input  logic             a_req,
  input  logic             a_local,
  input  logic [REQ_W-1:0] a_req_data,
  output logic             a_done,
  output logic             a_timeout,
  output logic [RSP_W-1:0] a_rsp_data,
  input  logic             clk_b,
  input  logic             rst_b_n,
  output logic             b_valid,
  output logic [REQ_W-1:0] b_req_data,
  input  logic             b_rsp,
  input  logic [RSP_W-1:0] b_rsp_data
);

  localparam int unsigned TW = (TIMEOUT > 1) ? $clog2(TIMEOUT + 1) : 1;

  // ---------------------------------------------------------------- A side
  typedef enum logic [1:0] { A_IDLE, A_BUSY, A_DRAIN, A_DRAIN_PEND } a_st_e;
  a_st_e            a_st_q;
  logic             req_tgl_q, ack_tgl_as;
  logic             ack_tgl_q;   // B side's acknowledge toggle (declared before its first use)
  logic [REQ_W-1:0] req_q, pend_q;
  logic [RSP_W-1:0] rsp_q;
  logic [TW-1:0]    tmr_q;
  logic             a_ack;   // the response of the request in flight arrived

  tsr_sync #(.W(1), .STAGES(STAGES)) u_sync_ack (
    .clk(clk_a), .rst_n(rst_a_n), .d(ack_tgl_q), .q(ack_tgl_as)
  );
  assign a_ack   = ack_tgl_as == req_tgl_q;
  assign a_ready = (a_st_q == A_IDLE) || (a_st_q == A_DRAIN);

  logic tmr_hit;
  assign tmr_hit = (TIMEOUT != 0) && (tmr_q == TW'(TIMEOUT));

  always_ff @(posedge clk_a) begin
    if (!rst_a_n) begin
      a_st_q    <= A_IDLE;
      req_tgl_q <= 1'b0;
      tmr_q     <= '0;
      a_done    <= 1'b0;
      a_timeout <= 1'b0;
    end else begin
      a_done    <= 1'b0;
      a_timeout <= 1'b0;
      if (a_st_q != A_IDLE && !tmr_hit) tmr_q <= tmr_q + 1'b1;
      unique case (a_st_q)
        A_IDLE: begin
          if (a_req && a_local) begin
            a_done    <= 1'b1;
            a_timeout <= 1'b1;
          end else if (a_req) begin
            req_q     <= a_req_data;
            req_tgl_q <= ~req_tgl_q;
            tmr_q     <= TW'(1);
            a_st_q    <= A_BUSY;
          end
        end
        A_BUSY: begin
          if (a_ack) begin
            a_done     <= 1'b1;
            a_rsp_data <= rsp_q;
            a_st_q     <= A_IDLE;
          end else if (tmr_hit) begin
            a_done    <= 1'b1;
            a_timeout <= 1'b1;
            a_st_q    <= A_DRAIN;
          end
        end
        A_DRAIN: begin
          if (a_req && a_local) begin
            a_done    <= 1'b1;
            a_timeout <= 1'b1;
          end else if (a_req) begin
            pend_q <= a_req_data;
            tmr_q  <= TW'(1);
            a_st_q <= a_ack ? A_BUSY : A_DRAIN_PEND;
            if (a_ack) begin
              req_q     <= a_req_data;
              req_tgl_q <= ~req_tgl_q;
            end
          end else if (a_ack) begin
            a_st_q <= A_IDLE;
          end
        end
        A_DRAIN_PEND: begin
          if (a_ack) begin
            req_q     <= pend_q;
            req_tgl_q <= ~req_tgl_q;
            a_st_q    <= A_BUSY;
          end else if (tmr_hit) begin
            a_done    <= 1'b1;
            a_timeout <= 1'b1;
            a_st_q    <= A_DRAIN;
          end
        end
        default: a_st_q <= A_IDLE;
      endcase
    end
  end

  // ---------------------------------------------------------------- B side
  logic req_tgl_bs;

  tsr_sync #(.W(1), .STAGES(STAGES)) u_sync_req (
    .clk(clk_b), .rst_n(rst_b_n), .d(req_tgl_q), .q(req_tgl_bs)
  );
  // The request word is captured into a B-side register when the toggle arrives, so the only
  // path out of req_q ends in an enabled register (no B-side logic or memory port sees the other
  // domain's flops: report_cdc, CC §11.3); b_valid follows one cycle later.
  logic             cap_v_q;
  logic [REQ_W-1:0] cap_q;
  assign b_valid    = cap_v_q;
  assign b_req_data = cap_q;

  always_ff @(posedge clk_b) begin
    if (!rst_b_n) begin
      ack_tgl_q <= 1'b0;
      cap_v_q   <= 1'b0;
    end else if (cap_v_q && b_rsp) begin
      rsp_q     <= b_rsp_data;
      ack_tgl_q <= ~ack_tgl_q;
      cap_v_q   <= 1'b0;
    end else if (!cap_v_q && req_tgl_bs != ack_tgl_q) begin
      cap_q   <= req_q;
      cap_v_q <= 1'b1;
    end
  end

`ifndef SYNTHESIS
  // armed once the reset has been seen (the simulation's initial state is random)
  bit arm_a, arm_b;
  initial begin arm_a = 1'b0; arm_b = 1'b0; end
  // not always_ff: the initial block writes arm_a / arm_b too (IEEE 1800 9.2.2.4)
  always @(posedge clk_a) begin
    if (!rst_a_n) arm_a <= 1'b1;
    else if (arm_a && a_req && !a_ready) $error("tsr_cdc_hs: request while not ready");
  end
  always @(posedge clk_b) begin
    if (!rst_b_n) arm_b <= 1'b1;
    else if (arm_b && b_rsp && !b_valid) $error("tsr_cdc_hs: response without a request");
  end
`endif

endmodule
