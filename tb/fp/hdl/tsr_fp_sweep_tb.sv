// tsr_fp_sweep_tb.sv — testbench-only toplevel of the Verilator C++ sweep harness
// (tb/fp/cpp/fp_sweep.cpp): the tsr_fp_pkg function lanes (tsr_fp_funcs_tb), MLANES tsr_fp_mul,
// DLANES tsr_fp_div and ALANES tsr_fp_add instances side by side, so the harness checks many
// vectors per eval. Multipliers share ce / in_valid; adders share ce / in_valid (each lane has its
// own sub); dividers share in_valid / out_ready (they run in lockstep).

module tsr_fp_sweep_tb #(
  parameter int unsigned LANES   = 16,
  parameter int unsigned MLANES  = 16,
  parameter int unsigned MUL_LAT = 4,
  parameter int unsigned DLANES  = 16,
  parameter int unsigned ALANES  = 32,
  parameter int unsigned ADD_LAT = 5
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // ---- package functions (combinational) ----
  input  logic [LANES-1:0][31:0]  x,
  input  logic [LANES-1:0][15:0]  h,
  output logic [LANES-1:0][31:0]  daz,
  output logic [LANES-1:0]        is_nan,
  output logic [LANES-1:0][30:0]  amag,
  output logic [LANES-1:0][31:0]  cand,
  output logic [LANES-1:0][31:0]  h2f,
  output logic [LANES-1:0][15:0]  f2h,
  output logic [LANES-1:0][7:0]   q8a,
  output logic [LANES-1:0][7:0]   q8e,
  output logic [LANES-1:0][31:0]  i2f,
  output logic [30:0]             tree_amax,
  output logic [31:0]             tree_max,
  output logic [30:0]             chain_amax,
  output logic [31:0]             chain_max,

  // ---- multipliers ----
  input  logic                    mul_ce,
  input  logic                    mul_in_valid,
  input  logic [MLANES-1:0][31:0] mul_a,
  input  logic [MLANES-1:0][31:0] mul_b,
  output logic [MLANES-1:0]       mul_out_valid,
  output logic [MLANES-1:0][31:0] mul_y,

  // ---- dividers ----
  input  logic                    div_in_valid,
  output logic [DLANES-1:0]       div_in_ready,
  input  logic [DLANES-1:0][31:0] div_a,
  input  logic [DLANES-1:0][31:0] div_b,
  output logic [DLANES-1:0]       div_out_valid,
  input  logic                    div_out_ready,
  output logic [DLANES-1:0][31:0] div_y,

  // ---- adders / subtractors ----
  input  logic                    add_ce,
  input  logic                    add_in_valid,
  input  logic [ALANES-1:0][31:0] add_a,
  input  logic [ALANES-1:0][31:0] add_b,
  input  logic [ALANES-1:0]       add_sub,
  output logic [ALANES-1:0]       add_out_valid,
  output logic [ALANES-1:0][31:0] add_y
);

  tsr_fp_funcs_tb #(.LANES(LANES)) u_funcs (
    .x, .h, .daz, .is_nan, .amag, .cand, .h2f, .f2h, .q8a, .q8e, .i2f,
    .tree_amax, .tree_max, .chain_amax, .chain_max
  );

  for (genvar i = 0; i < int'(MLANES); i++) begin : g_mul
    tsr_fp_mul #(.LAT(MUL_LAT)) u_mul (
      .clk, .rst_n, .ce(mul_ce), .in_valid(mul_in_valid), .a(mul_a[i]), .b(mul_b[i]),
      .out_valid(mul_out_valid[i]), .y(mul_y[i])
    );
  end

  for (genvar i = 0; i < int'(DLANES); i++) begin : g_div
    tsr_fp_div u_div (
      .clk, .rst_n, .in_valid(div_in_valid), .in_ready(div_in_ready[i]), .a(div_a[i]),
      .b(div_b[i]), .out_valid(div_out_valid[i]), .out_ready(div_out_ready), .y(div_y[i])
    );
  end

  for (genvar i = 0; i < int'(ALANES); i++) begin : g_add
    tsr_fp_add #(.LAT(ADD_LAT)) u_add (
      .clk, .rst_n, .ce(add_ce), .in_valid(add_in_valid), .a(add_a[i]), .b(add_b[i]),
      .sub(add_sub[i]), .out_valid(add_out_valid[i]), .y(add_y[i])
    );
  end

endmodule
