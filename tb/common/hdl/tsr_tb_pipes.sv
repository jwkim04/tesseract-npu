// tsr_tb_pipes.sv — testbench-only toplevel for the tb/common self-test of tsr_vpipe / tsr_rpipe:
// three instances of each, with DEPTH 0, 1 and 3 (index i of every port vector).

module tsr_tb_pipes #(
  parameter int unsigned W = 40
) (
  input  logic                 clk,
  input  logic                 rst_n,
  // tsr_vpipe, VW = 2
  input  logic [2:0][1:0]      v_in_valid,
  input  logic [2:0][W-1:0]    v_in_data,
  output logic [2:0][1:0]      v_out_valid,
  output logic [2:0][W-1:0]    v_out_data,
  // tsr_rpipe
  input  logic [2:0]           r_s_valid,
  output logic [2:0]           r_s_ready,
  input  logic [2:0][W-1:0]    r_s_data,
  output logic [2:0]           r_m_valid,
  input  logic [2:0]           r_m_ready,
  output logic [2:0][W-1:0]    r_m_data
);

  localparam int unsigned D0 = 0, D1 = 1, D2 = 3;

  for (genvar i = 0; i < 3; i++) begin : g_p
    localparam int unsigned D = (i == 0) ? D0 : (i == 1) ? D1 : D2;
    tsr_vpipe #(.VW(2), .W(W), .DEPTH(D)) u_v (
      .clk, .rst_n, .in_valid(v_in_valid[i]), .in_data(v_in_data[i]),
      .out_valid(v_out_valid[i]), .out_data(v_out_data[i])
    );
    tsr_rpipe #(.W(W), .DEPTH(D)) u_r (
      .clk, .rst_n, .s_valid(r_s_valid[i]), .s_ready(r_s_ready[i]), .s_data(r_s_data[i]),
      .m_valid(r_m_valid[i]), .m_ready(r_m_ready[i]), .m_data(r_m_data[i])
    );
  end

endmodule
