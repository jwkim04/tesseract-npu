// tsr_vpipe.sv — fixed-latency pipe without backpressure (valid bits + data), DEPTH stages.
//
// Owner: architecture (docs/rtl-mvm.md §1, §6). Shared infrastructure: the MVM broadcast pipe
// (tsr_mvm -> clusters, crosses SLRs) and any fixed delay line.
// Contract: out_valid / out_data are in_valid / in_data delayed by exactly DEPTH cycles
//   (DEPTH = 0: wires). The valid bits are reset to 0 (control); the data registers have no reset
//   and load every cycle (qualified by the valid bits). No backpressure: the consumer must take
//   every item (rule R4 of docs/rtl-core.md applies to whoever feeds the pipe).
// Timing: every stage is a plain register, so a placer may move stages along a long route
//   (SLR crossing: one stage on each side, docs/rtl-mvm.md §9).
// KEEP_REGS = 1 (implementation, no functional effect): the stage registers carry the Xilinx `keep`
//   and `shreg_extract = "no"` attributes, so synthesis neither merges the pipe with an
//   equivalent pipe nor packs it into SRLs. Needed where several pipes start at the same source
//   and must stay separate registers next to their own consumers: the per-cluster broadcast
//   pipes of tsr_mvm, which Vivado 2024.2 otherwise merges into one pipe feeding all clusters
//   across the SLRs (docs/rtl-mvm.md §14). Verilator ignores the attributes. (Not named KEEP:
//   Vivado reads an instance parameter named KEEP as the instance's KEEP property, [Netlist 29-72].)

module tsr_vpipe #(
  parameter int unsigned VW        = 1,   // valid / control bits (reset)
  parameter int unsigned W         = 1,   // data bits (no reset)
  parameter int unsigned DEPTH     = 1,   // register stages, >= 0
  parameter bit          KEEP_REGS = 1'b0 // keep every stage register (no merging / SRL), see above
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
  end else if (KEEP_REGS) begin : g_keep
    (* keep = "true", shreg_extract = "no" *) logic [DEPTH-1:0][VW-1:0] v_q;
    (* keep = "true", shreg_extract = "no" *) logic [DEPTH-1:0][W-1:0]  d_q;

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
