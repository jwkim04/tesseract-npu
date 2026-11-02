// tsr_vu_lane_share.sv — select the one scheduled lane exit before the return pipe.
// Owner: VU (docs/design/02-vector-unit.md §7.1, §7.10).
// back_valid is aligned with the lane input, before its S0 register. The
// injector orders all exits, so front and back are never consumed together.
// No arithmetic, buffering or extra latency is added; reduction drains separately.
module tsr_vu_lane_share
  import tsr_vu_pkg::*;
(
  input logic clk, rst_n,
  input logic back_valid,
  input logic [15:0][31:0] front_y, back_y,
  output logic [15:0][31:0] y
);
  // S0 registers the injector outputs once, then LB cycles reach the back exit.
  // Reset is required: stale back tokens must not mask new front results after
  // a short reset, before an unreset delay line would have drained.
  logic [TSR_VU_LB:0] back_q;
  always_ff @(posedge clk) begin
    if (!rst_n) back_q <= '0;
    else back_q <= {back_q[TSR_VU_LB-1:0], back_valid};
  end
  assign y = back_q[TSR_VU_LB] ? back_y : front_y;
endmodule
