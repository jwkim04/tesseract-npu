// tsr_vu_lane_tb.sv — testbench-only toplevel of the lane sweep (tb/vu/cpp/vu_lane_sweep.cpp): N lanes
// with EXPK and back lane, flat port vectors, the exits' latencies as constant outputs.

module tsr_vu_lane_tb #(
  parameter int unsigned N = 16
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic [N-1:0]       in_v,
  input  logic [N-1:0][5:0]  in_uop,
  input  logic [N-1:0][31:0] in_a,
  input  logic [N-1:0][31:0] in_b,
  input  logic [N-1:0][31:0] in_c1,
  input  logic [N-1:0][31:0] in_c2,
  input  logic [N-1:0]       in_neg,
  input  logic [N-1:0]       acc_lv,
  input  logic [N-1:0]       acc_fresh,
  input  logic [N-1:0][3:0]  acc_pa,
  input  logic [3:0]         dr_addr,
  output logic [N-1:0][31:0] dr_data,
  output logic [N-1:0][31:0] f_y,
  output logic [N-1:0][31:0] b_y,
  output logic [7:0]         lat_f,
  output logic [7:0]         lat_b
);
  import tsr_vu_pkg::*;

  for (genvar i = 0; i < N; i++) begin : g_l
    logic [31:0] unused_drain_hi;
    tsr_vu_lane #(.HAS_EXP(1'b1), .HAS_BACK(1'b1)) u_lane (
      .clk, .rst_n, .in_v(in_v[i]), .in_uop(tsr_vu_uop_e'(in_uop[i])), .in_a(in_a[i]), .in_b(in_b[i]),
      .in_c1(in_c1[i]), .in_c2(in_c2[i]), .in_neg(in_neg[i]), .acc_lv(acc_lv[i]), .acc_fresh(acc_fresh[i]),
      .acc_pa(acc_pa[i]), .dr_addr, .dr_data_hi(unused_drain_hi), .dr_data(dr_data[i]), .f_y(f_y[i]), .b_y(b_y[i])
    );
  end
  assign lat_f = 8'(TSR_VU_LF);
  assign lat_b = 8'(TSR_VU_LB);
endmodule
