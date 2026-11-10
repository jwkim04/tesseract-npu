// tsr_tech_flat_mac.sv — wiring-only wrapper of tsr_mac_lane with flat vector ports, so that the
// funcsim netlist of tb/tech/viv/net_sim.sh keeps one port per signal (write_verilog splits packed
// 2-D ports into escaped per-element ports). Synthesized as the top of the mac netlist; no logic.
module tsr_tech_flat_mac #(
  parameter  int unsigned N     = 32,
  parameter  int unsigned T     = 1,
  parameter  int unsigned LAT   = 3,
  localparam int unsigned SUM_W = 16 + $clog2(N)
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   ce,
  input  logic                   in_valid,
  input  logic [N*8-1:0]         w,
  input  logic [T*N*8-1:0]       x,
  output logic                   out_valid,
  output logic [T*SUM_W-1:0]     y
);
  tsr_mac_lane #(.N(N), .T(T), .LAT(LAT)) u_mac (
    .clk, .rst_n, .ce, .in_valid, .w, .x, .out_valid, .y
  );
endmodule
