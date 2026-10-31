// tsr_vu_arith_tb.sv — testbench-only toplevel of the VU arithmetic sweeps (tb/vu/cpp/vu_sweep.cpp):
// N parallel instances of each fixed-latency block, inputs and outputs as flat port vectors, and the
// blocks' latencies as constant outputs (the harness aligns its expectations with them).

module tsr_vu_arith_tb #(
  parameter int unsigned N = 16
) (
  input  logic                 clk,
  // tsr_vu_exp
  input  logic [N-1:0][31:0]   e_v,
  input  logic [N-1:0][1:0]    e_k,
  output logic [N-1:0][31:0]   e_y,
  // tsr_vu_rcp
  input  logic [N-1:0][31:0]   r_x,
  output logic [N-1:0][31:0]   r_y,
  // tsr_vu_sqrt
  input  logic [N-1:0][31:0]   s_x,
  output logic [N-1:0][31:0]   s_y,
  // tsr_vu_sincos
  input  logic                 rst_n,
  input  logic [N-1:0][31:0]   c_t,
  output logic [N-1:0][31:0]   c_s,
  output logic [N-1:0][31:0]   c_c,
  // latencies
  output logic [7:0]           lat_sc,
  output logic [7:0]           lat_exp,
  output logic [7:0]           lat_rcp,
  output logic [7:0]           lat_sqrt
);
  import tsr_vu_pkg::*;

  for (genvar i = 0; i < N; i++) begin : g_i
    tsr_vu_exp  u_exp  (.clk, .v(e_v[i]), .kappa(e_k[i]), .y(e_y[i]));
    tsr_vu_rcp  u_rcp  (.clk, .x(r_x[i]), .y(r_y[i]));
    tsr_vu_sqrt u_sqrt (.clk, .x(s_x[i]), .y(s_y[i]));
    tsr_vu_sincos u_sc (.clk, .rst_n, .t(c_t[i]), .s(c_s[i]), .c(c_c[i]));
  end

  assign lat_sc   = 8'(TSR_VU_SC_LAT);
  assign lat_exp  = 8'(TSR_VU_EXP_LAT);
  assign lat_rcp  = 8'(TSR_VU_RCP_LAT);
  assign lat_sqrt = 8'(TSR_VU_SQRT_LAT);

endmodule
