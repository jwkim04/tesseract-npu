// tsr_tb_lsel_net.sv — xsim testbench of tb/at/viv/lsel_net.sh: the Vivado-synthesised netlist of the ATTPV drain's
// lane select (tsr_lsel_flat, or with +define LEGACY the old integer form tsr_lsel_legacy_flat, both from
// tsr_lsel_flat.sv; write_verilog -funcsim, top renamed <top>_net, unisim primitives) against the selection itself:
// output lane k of drain step (l, t) is lane 8 l + k of slot t, and its non-finite flag is flag[t][lane / 32].
// Random capture banks and steps; prints "NETSIM lsel <form> cases <n> mismatches <m>" (m counts the cases with any
// wrong output).
`timescale 1ns / 1ps

module tsr_tb_lsel_net #(
  parameter int T_MAX = 4,
  parameter int CASES = 4000,
  parameter int SEED  = 1
);
  localparam int TW = (T_MAX > 1) ? $clog2(T_MAX) : 1;
`ifdef LEGACY
  localparam string FORM = "legacy";
`else
  localparam string FORM = "rtl";
`endif
  logic                         clk = 1'b0;
  logic [2:0]                   l;
  logic [TW-1:0]                t;
  logic [T_MAX-1:0][63:0][37:0] a;
  logic [T_MAX-1:0][1:0]        nf;
  logic [8*38-1:0]              y;
  logic [7:0]                   f;
`ifdef LEGACY
  tsr_lsel_legacy_flat_net u_net (.clk, .l, .t, .a_flat(a), .nf_flat(nf), .y_flat(y), .f);
`else
  tsr_lsel_flat_net u_net (.clk, .l, .t, .a_flat(a), .nf_flat(nf), .y_flat(y), .f);
`endif
  always #1.6665 clk = ~clk;

  int cases = 0, fails = 0, lane_fails [8];
  logic [7:0][37:0] exp_y;
  logic [7:0]       exp_f;

  initial begin
    process::self().srandom(SEED);
    #200;                                     // glbl GSR (100 ns)
    for (int c = 0; c < CASES; c++) begin
      @(negedge clk);
      l = 3'($urandom);
      t = TW'($urandom % T_MAX);
      for (int i = 0; i < T_MAX; i++) begin
        for (int j = 0; j < 64; j++) a[i][j] = {$urandom, $urandom};
        nf[i] = 2'($urandom);
      end
      for (int k = 0; k < 8; k++) begin
        exp_y[k] = a[t][8 * int'(l) + k];
        exp_f[k] = nf[t][(8 * int'(l) + k) / 32];
      end
      @(negedge clk);                         // registered at the rising edge in between
      cases++;
      if (y !== exp_y || f !== exp_f) begin
        if (fails < 4) $display("NETSIM lsel mismatch: l %0d t %0d y[0] %h expected %h", l, t, y[37:0], exp_y[0]);
        fails++;
      end
    end
    $display("NETSIM lsel %s cases %0d mismatches %0d", FORM, cases, fails);
    $finish;
  end
endmodule
