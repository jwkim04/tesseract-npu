// tsr_vu_gdn_ulane.sv — one lane of the GDN state update (docs/design/02-vector-unit.md §7.12.5):
//     y = FA(FM(alpha, s), FM(e, k))
// with the VU primitives tsr_fp_mul and tsr_fp_add (DAZ, RNE, FTZ, canonical NaN; bit-exact with
// tsr_ref_gdn_upd_elem). alpha and e are constant while a row streams through, s and k change every
// beat. Fully pipelined, one element per cycle, latency FM_LAT + FA_LAT.

module tsr_vu_gdn_ulane #(
  parameter int unsigned FM_LAT = 4,
  parameter int unsigned FA_LAT = 5
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  input  logic [31:0] s,
  input  logic [31:0] k,
  input  logic [31:0] alpha,
  input  logic [31:0] e,
  output logic        out_valid,
  output logic [31:0] y
);

  logic        v_m1, v_m2;
  logic [31:0] p1, p2;

  tsr_fp_mul #(.LAT(FM_LAT)) u_m1 (
    .clk, .rst_n, .ce(1'b1), .in_valid, .a(alpha), .b(s), .out_valid(v_m1), .y(p1));
  tsr_fp_mul #(.LAT(FM_LAT)) u_m2 (
    .clk, .rst_n, .ce(1'b1), .in_valid, .a(e), .b(k), .out_valid(v_m2), .y(p2));
  tsr_fp_add #(.LAT(FA_LAT)) u_a (
    .clk, .rst_n, .ce(1'b1), .in_valid(v_m1), .a(p1), .b(p2), .sub(1'b0), .out_valid, .y);

  wire unused_ok = &{1'b0, v_m2};

endmodule
