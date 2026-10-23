// tsr_tb_mvm_i2f.sv — testbench-only toplevel of the exhaustive tsr_mvm_i2f sweep
// (tb/mvmback/cpp/i2f_sweep.cpp): LANES independent 26-bit converters (the width tsr_mvm_back
// instantiates), so the harness checks LANES inputs per cycle against libtsr_ref tsr_i32_to_f32.

module tsr_tb_mvm_i2f #(
  parameter int unsigned LANES = 16,
  parameter int unsigned IW    = 26
) (
  input  logic                    clk,
  input  logic [LANES-1:0][IW-1:0] v,
  output logic [LANES-1:0][31:0]   y
);
  for (genvar i = 0; i < LANES; i++) begin : g_lane
    tsr_mvm_i2f #(.IW(IW)) u_i2f (.clk, .v(v[i]), .y(y[i]));
  end
endmodule
