// tsr_fp_arr.sv — array wrapper of an f32 operator for the area runs of tb/fp/viv/fp_area.tcl.
// Not part of the RTL filelist. N independent instances, each with its own ports (so synthesis cannot
// merge them); the operators have registered inputs and outputs themselves (LAT >= 2), so the wrapper
// adds no register. ce is tied to 1 (as in the vector unit); rst_n is shared.
//   KIND 0  tsr_fp_add      KIND 1  tsr_fp_mul      KIND 2  tsr_fp_add_dsp      KIND 3  tsr_fp_add_dsp, NORM_DSP = 1
module tsr_fp_arr #(
  parameter int unsigned N    = 32,
  parameter int unsigned KIND = 0,
  parameter int unsigned LAT  = 5
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
    if (KIND == 0) begin : g_add
      tsr_fp_add #(.LAT(LAT)) u (.clk, .rst_n, .ce(1'b1), .in_valid(in_valid[i]), .a(a[i]), .b(b[i]), .sub(sub[i]),
                                 .out_valid(out_valid[i]), .y(y[i]));
    end else if (KIND == 1) begin : g_mul
      tsr_fp_mul #(.LAT(LAT)) u (.clk, .rst_n, .ce(1'b1), .in_valid(in_valid[i]), .a(a[i]), .b(b[i]),
                                 .out_valid(out_valid[i]), .y(y[i]));
    end else begin : g_adsp
      tsr_fp_add_dsp #(.LAT(LAT), .NORM_DSP(KIND == 3)) u (.clk, .rst_n, .ce(1'b1), .in_valid(in_valid[i]), .a(a[i]), .b(b[i]), .sub(sub[i]),
                                     .out_valid(out_valid[i]), .y(y[i]));
    end
  end
endmodule
