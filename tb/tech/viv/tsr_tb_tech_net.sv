// tsr_tb_tech_net.sv — xsim testbenches of tb/tech/viv/net_sim.sh: the Vivado-synthesised netlist
// of an rtl/tech/xilinx module (write_verilog -mode funcsim, top renamed <top>_net, unisim
// primitives: DSP48E2, URAM288, RAMB36E2, LUTRAM, ...; the MAC lane through the flat-port wrapper
// tb/tech/viv/tsr_tech_flat_mac.sv) against the generic RTL of the same module, cycle by cycle
// (!==, after a flush that defines every register of the RTL).
//
//   tsr_tb_tech_net_mac   tsr_mac_lane: 16 flush cycles (reset, ce = 1), the corner sweeps of
//                         tb/tech/cpp/mac_equiv.cpp (216 per-slot triples of
//                         {-128, -127, -1, 0, 1, 127}^3, 216 uniform beats), then CYCLES random
//                         beats (uniform / corner / Q4_K weights / extremes / sparse; ce 85 %,
//                         in_valid 70 %, short resets).
//   tsr_tb_tech_net_sram  tsr_sram_1r1w: writes every address (all bytes), one read, then CYCLES
//                         random cycles in phases of 500 (uniform, 8-address window, forced
//                         same-cycle collisions, read of the previous write address, sparse) with
//                         random byte enables; rdata compared in every cycle with a read result.
// Each prints "NETSIM <name> cycles <n> compared <n> mismatches <n>" and the first mismatches.
`timescale 1ns / 1ps

module tsr_tb_tech_net_mac #(
  parameter int N      = 32,
  parameter int T      = 8,
  parameter int LAT    = 5,
  parameter int CYCLES = 20000,
  parameter int SEED   = 1
);
  localparam int SW = 16 + $clog2(N);
  localparam logic [7:0] CORNER [6] = '{8'h80, 8'h81, 8'hff, 8'h00, 8'h01, 8'h7f};

  logic                     clk = 1'b0;
  logic                     rst_n, ce, in_valid;
  logic [N-1:0][7:0]        w;
  logic [T-1:0][N-1:0][7:0] x;
  logic                     ov_r, ov_n;
  logic [T-1:0][SW-1:0]     y_r, y_n;

  tsr_mac_lane #(.N(N), .T(T), .LAT(LAT)) u_ref (
    .clk, .rst_n, .ce, .in_valid, .w, .x, .out_valid(ov_r), .y(y_r)
  );
  tsr_tech_flat_mac_net u_net (
    .clk, .rst_n, .ce, .in_valid, .w, .x, .out_valid(ov_n), .y(y_n)
  );

  always #1.6665 clk = ~clk;

  int cycles = 0, compared = 0, fails = 0;
  bit check = 1'b0;

  always @(posedge clk) begin
    #0.5;
    cycles++;
    if (check) begin
      compared++;
      // after the flush every output bit is defined: X / Z counts as a mismatch
      if (ov_r !== ov_n || y_r !== y_n || $isunknown({ov_r, y_r})) begin
        if (fails < 10) $display("NETSIM mismatch cycle %0d: ov %b/%b y %h / %h", cycles, ov_r, ov_n, y_r, y_n);
        fails++;
      end
    end
  end

  task automatic beat_random(int mode);
    for (int k = 0; k < N; k++) begin
      case (mode)
        0: begin w[k] = 8'($urandom); for (int t = 0; t < T; t++) x[t][k] = 8'($urandom); end
        1: begin w[k] = CORNER[$urandom % 6]; for (int t = 0; t < T; t++) x[t][k] = CORNER[$urandom % 6]; end
        2: begin w[k] = 8'($urandom % 16); for (int t = 0; t < T; t++) x[t][k] = 8'($urandom); end
        3: begin
          w[k] = ($urandom % 2) ? 8'h80 : 8'h7f;
          for (int t = 0; t < T; t++) x[t][k] = ($urandom % 3 == 0) ? 8'h7f : 8'h80;
        end
        default: begin
          w[k] = ($urandom % 4) ? 8'h00 : 8'($urandom);
          for (int t = 0; t < T; t++) x[t][k] = ($urandom % 4) ? 8'h00 : 8'($urandom);
        end
      endcase
    end
  endtask

  initial begin
    int rst_left;
    process::self().srandom(SEED);
    rst_n = 1'b0; ce = 1'b1; in_valid = 1'b0; w = '0; x = '0;
    #200;                                   // glbl GSR (100 ns)
    @(negedge clk);
    for (int i = 0; i < 16; i++) begin       // flush in reset
      beat_random(0);
      in_valid = 1'($urandom);
      @(negedge clk);
    end
    rst_n = 1'b1; in_valid = 1'b1; check = 1'b1;
    for (int b = 0; b < 216; b++) begin      // per-slot corner triples
      for (int k = 0; k < N; k++) begin
        int i;
        i = (b + 5 * k) % 216;
        w[k] = CORNER[i / 36];
        for (int t = 0; t < T; t++) x[t][k] = (t % 2 == 0) ? CORNER[((i / 6) % 6 + t / 2) % 6] : CORNER[(i % 6 + t) % 6];
      end
      @(negedge clk);
    end
    for (int b = 0; b < 216; b++) begin      // uniform: extreme sums
      for (int k = 0; k < N; k++) begin
        w[k] = CORNER[b / 36];
        for (int t = 0; t < T; t++) x[t][k] = (t % 2 == 0) ? CORNER[(b / 6) % 6] : CORNER[b % 6];
      end
      @(negedge clk);
    end
    rst_left = 0;
    for (int i = 0; i < CYCLES; i++) begin
      if (rst_left > 0) rst_left--;
      else if ($urandom % 1000 < 3) rst_left = 1 + $urandom % 3;
      rst_n = (rst_left == 0);
      ce = ($urandom % 100) < 85;
      in_valid = ($urandom % 100) < 70;
      beat_random($urandom % 5);
      @(negedge clk);
    end
    repeat (LAT + 2) @(negedge clk);
    $display("NETSIM mac N %0d T %0d LAT %0d cycles %0d compared %0d mismatches %0d", N, T, LAT, cycles, compared, fails);
    $finish;
  end
endmodule


module tsr_tb_tech_net_sram #(
  parameter int DEPTH  = 16384,
  parameter int WIDTH  = 512,
  parameter int RD_LAT = 2,
  parameter int CYCLES = 30000,
  parameter int SEED   = 1
);
  localparam int AW = (DEPTH > 1) ? $clog2(DEPTH) : 1;
  localparam int BW = WIDTH / 8;

  logic             clk = 1'b0;
  logic             we, re;
  logic [AW-1:0]    waddr, raddr, prev_wa;
  logic [WIDTH-1:0] wdata, rd_r, rd_n;
  logic [BW-1:0]    wbe;

  tsr_sram_1r1w #(.DEPTH(DEPTH), .WIDTH(WIDTH), .RD_LAT(RD_LAT)) u_ref (
    .clk, .we, .waddr, .wdata, .wbe, .re, .raddr, .rdata(rd_r)
  );
  tsr_sram_1r1w_net u_net (
    .clk, .we, .waddr, .wdata, .wbe, .re, .raddr, .rdata(rd_n)
  );

  always #1.6665 clk = ~clk;

  int cycles = 0, compared = 0, fails = 0, coll = 0, raw1 = 0;
  bit check = 1'b0;
  logic [RD_LAT-1:0] rv = '0;     // rv[i]: a read was issued i + 1 edges ago

  // rdata is compared in the cycles with a read result (the contract; the xilinx URAM output
  // stages hold in the other cycles while the generic ones shift)
  always @(posedge clk) begin
    if (we && re && waddr == raddr) coll++;
    if (re && raddr == prev_wa) raw1++;
    rv <= RD_LAT'({rv, re});
    #0.5;
    cycles++;
    if (check && rv[RD_LAT-1]) begin
      compared++;
      if (rd_r !== rd_n || $isunknown(rd_r)) begin
        if (fails < 10) $display("NETSIM mismatch cycle %0d: rd %h / %h", cycles, rd_r, rd_n);
        fails++;
      end
    end
  end

  function automatic logic [AW-1:0] ra();
    return AW'($urandom % DEPTH);
  endfunction

  task automatic rand_data();
    logic [32 * ((WIDTH + 31) / 32) - 1:0] r;
    for (int i = 0; i < (WIDTH + 31) / 32; i++) r[32 * i +: 32] = $urandom;
    wdata = r[WIDTH-1:0];
  endtask

  initial begin
    int mode, bmode;
    process::self().srandom(SEED);
    we = 1'b0; re = 1'b0; waddr = '0; raddr = '0; wdata = '0; wbe = '0; prev_wa = '1;
    #200;                                   // glbl GSR (100 ns)
    @(negedge clk);
    for (int a = 0; a < DEPTH; a++) begin    // every address, all bytes
      we = 1'b1; waddr = AW'(a); wbe = '1; rand_data();
      @(negedge clk);
    end
    we = 1'b0; re = 1'b1; raddr = '0;
    @(negedge clk);
    re = 1'b0;
    repeat (RD_LAT + 1) @(negedge clk);
    check = 1'b1;
    for (int i = 0; i < CYCLES; i++) begin
      if (i % 500 == 0) begin
        mode = $urandom % 5;
        bmode = $urandom % 4;
      end
      we = ($urandom % 100) < ((mode == 4) ? 15 : 70);
      re = ($urandom % 100) < ((mode == 4) ? 15 : 70);
      prev_wa = waddr;
      case (mode)
        1: begin waddr = AW'($urandom % ((DEPTH < 8) ? DEPTH : 8)); raddr = AW'($urandom % ((DEPTH < 8) ? DEPTH : 8)); end
        2: begin waddr = ra(); raddr = waddr; end
        3: begin raddr = waddr; waddr = ra(); end
        default: begin waddr = ra(); raddr = ra(); end
      endcase
      case (bmode)
        0: wbe = '1;
        1: for (int b = 0; b < BW; b++) wbe[b] = 1'($urandom);
        2: begin wbe = '0; wbe[$urandom % BW] = 1'b1; end
        default: wbe = '0;
      endcase
      rand_data();
      @(negedge clk);
    end
    we = 1'b0; re = 1'b0;
    repeat (RD_LAT + 2) @(negedge clk);
    $display("NETSIM sram DEPTH %0d WIDTH %0d RD_LAT %0d cycles %0d compared %0d collisions %0d raw1 %0d mismatches %0d",
             DEPTH, WIDTH, RD_LAT, cycles, compared, coll, raw1, fails);
    $finish;
  end
endmodule
