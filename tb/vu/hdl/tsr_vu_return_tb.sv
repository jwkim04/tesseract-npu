// Return-path timing miter: original separate pipes + writer mux against the
// shared lane-side mux. Arithmetic payloads are independent arbitrary bits.
module tsr_vu_return_tb
  import tsr_vu_pkg::*;
#(parameter int unsigned PIPE = 2) (
  input logic clk, rst_n,
  input logic exit_valid, exit_back,
  input logic [511:0] raw_front, raw_back,
  output logic valid, collision,
  output logic [511:0] expected, actual
);
  localparam int unsigned DF = TSR_VU_LF + 2 * PIPE + 1;
  localparam int unsigned DB = TSR_VU_LB + 2 * PIPE + 1;
  logic [DF-1:0] fv;
  logic [DB-1:0] bv;
  always_ff @(posedge clk) begin
    if (!rst_n) begin fv <= '0; bv <= '0; end
    else begin
      fv <= {fv[DF-2:0], exit_valid && !exit_back};
      bv <= {bv[DB-2:0], exit_valid && exit_back};
    end
  end
  assign valid = fv[DF-1] || bv[DB-1];
  assign collision = fv[DF-1] && bv[DB-1];

  logic local_back, local_rst_n;
  logic [511:0] old_front, old_back, shared;
  if (PIPE == 0) begin : g_direct
    assign local_back = exit_valid && exit_back;
    assign local_rst_n = rst_n;
    assign old_front = raw_front;
    assign old_back = raw_back;
    assign actual = shared;
  end else begin : g_pipe
    logic [PIPE-1:0] back_q, reset_q;
    logic [PIPE-1:0][511:0] front_q, old_back_q, shared_q;
    always_ff @(posedge clk) begin
      back_q[0] <= exit_valid && exit_back;
      reset_q[0] <= rst_n;
      front_q[0] <= raw_front;
      old_back_q[0] <= raw_back;
      shared_q[0] <= shared;
      for (int unsigned i = 1; i < PIPE; i++) begin
        back_q[i] <= back_q[i-1];
        reset_q[i] <= reset_q[i-1];
        front_q[i] <= front_q[i-1];
        old_back_q[i] <= old_back_q[i-1];
        shared_q[i] <= shared_q[i-1];
      end
    end
    assign local_back = back_q[PIPE-1];
    assign local_rst_n = reset_q[PIPE-1];
    assign old_front = front_q[PIPE-1];
    assign old_back = old_back_q[PIPE-1];
    assign actual = shared_q[PIPE-1];
  end
  tsr_vu_lane_share u_share (
    .clk, .rst_n(local_rst_n), .back_valid(local_back),
    .front_y(raw_front), .back_y(raw_back), .y(shared)
  );
  assign expected = bv[DB-1] ? old_back : old_front;
endmodule
