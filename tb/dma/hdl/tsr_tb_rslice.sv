// Register slices on one stream: FORM 0, FORM 1 and FORM 1 with eight pointer / occupancy copies (RPC 8), and the
// two-flit pair receiver with one and with eight pointer copies. The copies must be cycle-identical to their single
// flops (tb/dma/test_rslice.py).
module tsr_tb_rslice #(
  parameter int unsigned W = 577
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         s_valid,
  input  logic [W-1:0] s_data,
  input  logic         m_ready,
  output logic [3:0]   s_ready,
  output logic [3:0]   m_valid,
  output logic [3:0][W-1:0] m_data,
  // the pair receiver: 2 flits of (W + 1) / 2 bits
  input  logic         f_valid,
  input  logic [(W+1)/2-1:0] f_data,
  output logic [1:0]   f_ready,
  output logic [1:0]   p_valid,
  output logic [1:0][W-1:0] p_data
);
  tsr_rslice #(.W(W), .FORM(0)) u_f0 (
    .clk, .rst_n, .s_valid, .s_ready(s_ready[0]), .s_data, .m_valid(m_valid[0]), .m_ready, .m_data(m_data[0])
  );
  tsr_rslice #(.W(W), .FORM(1)) u_f1 (
    .clk, .rst_n, .s_valid, .s_ready(s_ready[1]), .s_data, .m_valid(m_valid[1]), .m_ready, .m_data(m_data[1])
  );
  tsr_rslice #(.W(W), .FORM(1), .RPC(8)) u_f1c (
    .clk, .rst_n, .s_valid, .s_ready(s_ready[2]), .s_data, .m_valid(m_valid[2]), .m_ready, .m_data(m_data[2])
  );
  tsr_rslice #(.W(W), .FORM(1), .RPC(3)) u_f1d (
    .clk, .rst_n, .s_valid, .s_ready(s_ready[3]), .s_data, .m_valid(m_valid[3]), .m_ready, .m_data(m_data[3])
  );
  tsr_mvm_pair_rx #(.W(W), .RPC(1)) u_rx1 (
    .clk, .rst_n, .s_valid(f_valid), .s_ready(f_ready[0]), .s_data(f_data), .m_valid(p_valid[0]), .m_ready, .m_data(p_data[0])
  );
  tsr_mvm_pair_rx #(.W(W), .RPC(8)) u_rx8 (
    .clk, .rst_n, .s_valid(f_valid), .s_ready(f_ready[1]), .s_data(f_data), .m_valid(p_valid[1]), .m_ready, .m_data(p_data[1])
  );
endmodule
