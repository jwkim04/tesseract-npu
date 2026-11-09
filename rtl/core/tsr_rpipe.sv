// tsr_rpipe.sv — valid/ready pipe of DEPTH tsr_rslice stages (full throughput).
//
// Spec: docs/rtl-mvm.md §1, §6. Shared infrastructure: the MVM result pipe
// (clusters -> tsr_mvm gather, crosses SLRs) and any registered valid/ready boundary.
// Contract: a valid/ready stream (rules R1/R2 of docs/rtl-core.md on both sides); items leave in
//   order, unchanged; DEPTH = 0 is a wire. With DEPTH >= 1, s_ready, m_valid and m_data are
//   registers (no combinational path through the pipe in either direction); each stage adds one
//   cycle of latency and holds up to two items, so the pipe never limits throughput.
// tsr_rslice is defined in tsr_mem_arb.sv (rtl/filelist.f lists it before this file).

module tsr_rpipe #(
  parameter int unsigned W     = 8,
  parameter int unsigned DEPTH = 1    // tsr_rslice stages, >= 0
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         s_valid,
  output logic         s_ready,
  input  logic [W-1:0] s_data,
  output logic         m_valid,
  input  logic         m_ready,
  output logic [W-1:0] m_data
);

  if (W < 1) begin : g_chk
    $fatal(1, "tsr_rpipe: W >= 1 required");
  end

  if (DEPTH == 0) begin : g_wire
    assign m_valid = s_valid;
    assign s_ready = m_ready;
    assign m_data  = s_data;
    wire unused_ok = &{1'b0, clk, rst_n};
  end else begin : g_slices
    logic [DEPTH:0]        v, r;
    logic [DEPTH:0][W-1:0] d;
    assign v[0]    = s_valid;
    assign d[0]    = s_data;
    assign s_ready = r[0];
    for (genvar i = 0; i < DEPTH; i++) begin : g_s
      tsr_rslice #(.W(W)) u_slice (
        .clk, .rst_n,
        .s_valid(v[i]), .s_ready(r[i]), .s_data(d[i]),
        .m_valid(v[i+1]), .m_ready(r[i+1]), .m_data(d[i+1])
      );
    end
    assign m_valid    = v[DEPTH];
    assign m_data     = d[DEPTH];
    assign r[DEPTH]   = m_ready;
  end

endmodule
