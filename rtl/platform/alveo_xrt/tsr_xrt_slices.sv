// tsr_xrt_slices.sv — N >= 1 tsr_rslice stages in series: the register slices of the platform
// adapter's AXI channels (tsr_xrt_axi_bridge; full throughput, s_ready, m_valid and m_data are
// flops, so both sides are flop-to-flop and the chain can span an SLR crossing).
//
// Owner: adapter agent (docs/platform-u200.md §4). tsr_rslice is defined in rtl/core/tsr_mem_arb.sv.

module tsr_xrt_slices #(
  parameter int unsigned W = 8,
  parameter int unsigned N = 2
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

  if (N < 1) begin : g_chk_n
    $fatal(1, "tsr_xrt_slices: N must be >= 1");
  end

  logic [N:0]        v, r;
  logic [N:0][W-1:0] d;

  assign v[0]    = s_valid;
  assign d[0]    = s_data;
  assign s_ready = r[0];

  for (genvar k = 0; k < N; k++) begin : g_st
    tsr_rslice #(.W(W)) u_s (
      .clk, .rst_n,
      .s_valid(v[k]), .s_ready(r[k]), .s_data(d[k]),
      .m_valid(v[k+1]), .m_ready(r[k+1]), .m_data(d[k+1])
    );
  end

  assign m_valid = v[N];
  assign m_data  = d[N];
  assign r[N]    = m_ready;

endmodule
