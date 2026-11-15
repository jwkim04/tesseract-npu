// tsr_qact_scale_tb.sv — testbench-only toplevel of the A32 scale sweeps (tb/qact/cpp/qact_scale_sweep.cpp):
// N parallel instances of tsr_qact_scale, inputs and outputs as flat port vectors, and the unit's latency as a
// constant output (the harness aligns its expectations with it).

module tsr_qact_scale_tb #(
  parameter int unsigned N = 64
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic [N-1:0]       in_valid,
  input  logic [N-1:0][31:0] amag,        // bit 31 is not used
  output logic [N-1:0][31:0] scale,
  output logic [N-1:0][31:0] aux,         // {14'd0, zero, out_valid, h16}
  output logic [7:0]         lat
);
  for (genvar i = 0; i < N; i++) begin : g_i
    logic        ov, z;
    logic [15:0] h;
    tsr_qact_scale u_s (
      .clk, .rst_n, .in_valid(in_valid[i]), .amag(amag[i][30:0]),
      .out_valid(ov), .scale(scale[i]), .h16(h), .zero(z)
    );
    assign aux[i] = {14'd0, z, ov, h};
    wire unused_ok = &{1'b0, amag[i][31]};
  end

  assign lat = 8'(g_i[0].u_s.LAT);
endmodule
