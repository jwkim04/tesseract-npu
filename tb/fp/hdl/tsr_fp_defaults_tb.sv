// tsr_fp_defaults_tb.sv — testbench-only wrapper for test_fp_default_lats (tb/fp/test_fp.py):
// tsr_fp_mul and tsr_fp_add instantiated with their default LAT (no override), next to the
// tsr_pkg constants the MVM unit passes them (TSR_DEF_FMUL_LAT / TSR_DEF_FADD_LAT, docs/rtl-mvm.md
// §1). The test measures each default instance's latency and checks that the package constants
// equal it, so the blocks that default MUL_LAT / ADD_LAT to tsr_pkg (and so the U200 bitstream)
// run the latency that tb/fp swept and timing-closed. Not part of the core.

module tsr_fp_defaults_tb (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        mul_out_valid,
  output logic [31:0] mul_y,          // a * b, tsr_fp_mul at its default LAT
  output logic        add_out_valid,
  output logic [31:0] add_y,          // a + b, tsr_fp_add at its default LAT
  output logic [31:0] pkg_fmul_lat,   // tsr_pkg::TSR_DEF_FMUL_LAT
  output logic [31:0] pkg_fadd_lat    // tsr_pkg::TSR_DEF_FADD_LAT
);
  assign pkg_fmul_lat = 32'(tsr_pkg::TSR_DEF_FMUL_LAT);
  assign pkg_fadd_lat = 32'(tsr_pkg::TSR_DEF_FADD_LAT);

  tsr_fp_mul u_mul (
    .clk, .rst_n, .ce(1'b1), .in_valid, .a, .b, .out_valid(mul_out_valid), .y(mul_y)
  );

  tsr_fp_add u_add (
    .clk, .rst_n, .ce(1'b1), .in_valid, .a, .b, .sub(1'b0), .out_valid(add_out_valid), .y(add_y)
  );

endmodule
