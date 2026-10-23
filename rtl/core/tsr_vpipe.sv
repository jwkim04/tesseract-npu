// tsr_vpipe.sv — fixed-latency pipe without backpressure (valid bits + data), DEPTH stages.
//
// Owner: architecture (docs/rtl-m3.md §1, §6). Shared infrastructure: the MVM broadcast pipe
// (tsr_mvm -> clusters, crosses SLRs) and any fixed delay line.
// Contract: out_valid / out_data are in_valid / in_data delayed by exactly DEPTH cycles
//   (DEPTH = 0: wires). The valid bits are reset to 0 (control); the data registers have no reset
//   and load every cycle (qualified by the valid bits). No backpressure: the consumer must take
//   every item (rule R4 of docs/rtl-m2.md applies to whoever feeds the pipe).
// Timing: every stage is a plain register, so a placer may move stages along a long route
//   (SLR crossing: one stage on each side, docs/rtl-m3.md §9).

module tsr_vpipe #(
  parameter int unsigned VW    = 1,   // valid / control bits (reset)
  parameter int unsigned W     = 1,   // data bits (no reset)
  parameter int unsigned DEPTH = 1    // register stages, >= 0
) (
  input  logic          clk,
  input  logic          rst_n,
  input  logic [VW-1:0] in_valid,
  input  logic [W-1:0]  in_data,
  output logic [VW-1:0] out_valid,
  output logic [W-1:0]  out_data
);

  if (VW < 1 || W < 1) begin : g_chk
    $fatal(1, "tsr_vpipe: VW >= 1 and W >= 1 required");
  end

  if (DEPTH == 0) begin : g_wire
    assign out_valid = in_valid;
    assign out_data  = in_data;
    wire unused_ok = &{1'b0, clk, rst_n};
  end else begin : g_regs
    logic [DEPTH-1:0][VW-1:0] v_q;
    logic [DEPTH-1:0][W-1:0]  d_q;

    always_ff @(posedge clk) begin
      if (!rst_n) v_q <= '0;
      else begin
        v_q[0] <= in_valid;
        for (int unsigned i = 1; i < DEPTH; i++) v_q[i] <= v_q[i-1];
      end
    end

    always_ff @(posedge clk) begin
      d_q[0] <= in_data;
      for (int unsigned i = 1; i < DEPTH; i++) d_q[i] <= d_q[i-1];
    end

    assign out_valid = v_q[DEPTH-1];
    assign out_data  = d_q[DEPTH-1];
  end

endmodule
