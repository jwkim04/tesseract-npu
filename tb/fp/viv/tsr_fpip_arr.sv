// tsr_fpip_arr.sv — array wrapper of the Xilinx Floating-Point Operator IP (instance name fpip_op, created
// by tb/fp/viv/fp_area.tcl) for the area reference of experiment E1(d). Not part of the RTL filelist.
// The IP has no registers at its ports, so the wrapper registers the operands and the result (two more
// cycles than the IP latency). Add or multiply only (the IP is created with one fixed operation); sub is unused.
module tsr_fpip_arr #(
  parameter int unsigned N = 32
) (
  input  logic              clk,
  input  logic              rst_n,
  input  logic [N-1:0]      in_valid,
  input  logic [N-1:0][31:0] a,
  input  logic [N-1:0][31:0] b,
  input  logic [N-1:0]      sub,
  output logic [N-1:0]      out_valid,
  output logic [N-1:0][31:0] y
);
  for (genvar i = 0; i < N; i++) begin : g_i
    logic        v_q;
    logic [31:0] a_q, b_q, r_w;
    logic        rv_w;
    always_ff @(posedge clk) begin
      if (!rst_n) v_q <= 1'b0;
      else        v_q <= in_valid[i];
      a_q <= a[i];
      b_q <= b[i];
    end
    fpip_op u_ip (
      .aclk(clk),
      .s_axis_a_tvalid(v_q), .s_axis_a_tdata(a_q),
      .s_axis_b_tvalid(v_q), .s_axis_b_tdata(b_q),
      .m_axis_result_tvalid(rv_w), .m_axis_result_tdata(r_w)
    );
    always_ff @(posedge clk) begin
      out_valid[i] <= rv_w;
      y[i]         <= r_w;
    end
    wire unused_sub = sub[i];
  end
endmodule
