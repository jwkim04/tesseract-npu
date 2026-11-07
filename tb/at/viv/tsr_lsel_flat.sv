// tsr_lsel_flat.sv — wiring-only wrappers of the ATTPV drain's lane select with flat vector ports, so that the funcsim
// netlist of tb/at/viv/lsel_net.sh keeps one port per signal (write_verilog splits packed 2-D ports into escaped
// per-element ports). The select is registered at the outputs, as the drain registers it (c1_q).
//
//   tsr_lsel_flat         tsr_mvm_pv_lsel, the module of the RTL
//   tsr_lsel_legacy_flat  the form the RTL had until the card run of 2026-11-08: the lane number as the integer
//                         8 l + k, used as `a[t][6'(L)]` and `nf[t][L / 32]` inside the clocked process. Vivado 2024.2
//                         synthesises it without the top bit of l; the netlist test keeps it as the control that
//                         shows the bench can see the fault.
module tsr_lsel_flat #(
  parameter  int unsigned T_MAX = 4,
  localparam int unsigned TW    = (T_MAX > 1) ? $clog2(T_MAX) : 1
) (
  input  logic                    clk,
  input  logic [2:0]              l,
  input  logic [TW-1:0]           t,
  input  logic [T_MAX*64*38-1:0]  a_flat,
  input  logic [T_MAX*2-1:0]      nf_flat,
  output logic [8*38-1:0]         y_flat,
  output logic [7:0]              f
);
  logic [T_MAX-1:0][63:0][37:0] a;
  logic [T_MAX-1:0][1:0]        nf;
  logic [7:0][37:0]             y, y_q;
  logic [7:0]                   f_w, f_q;
  assign a  = a_flat;
  assign nf = nf_flat;
  tsr_mvm_pv_lsel #(.T_MAX(T_MAX)) u_lsel (.l, .t, .a, .nf, .y, .f(f_w));
  always_ff @(posedge clk) begin
    y_q <= y;
    f_q <= f_w;
  end
  assign y_flat = y_q;
  assign f      = f_q;
endmodule

module tsr_lsel_legacy_flat #(
  parameter  int unsigned T_MAX = 4,
  localparam int unsigned TW    = (T_MAX > 1) ? $clog2(T_MAX) : 1
) (
  input  logic                    clk,
  input  logic [2:0]              l,
  input  logic [TW-1:0]           t,
  input  logic [T_MAX*64*38-1:0]  a_flat,
  input  logic [T_MAX*2-1:0]      nf_flat,
  output logic [8*38-1:0]         y_flat,
  output logic [7:0]              f
);
  logic [T_MAX-1:0][63:0][37:0] a;
  logic [T_MAX-1:0][1:0]        nf;
  logic [7:0][37:0]             y_q;
  logic [7:0]                   f_q;
  assign a  = a_flat;
  assign nf = nf_flat;
  always_ff @(posedge clk) begin
    for (int k = 0; k < 8; k++) begin
      automatic int L = 8 * int'(l) + k;
      y_q[k] <= a[t][6'(L)];
      f_q[k] <= nf[t][L / 32];
    end
  end
  assign y_flat = y_q;
  assign f      = f_q;
endmodule
