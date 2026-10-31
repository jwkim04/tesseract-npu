// tb/chk/viv/tsr_tb_chk_net.sv — post-synthesis netlist of tsr_chk (Vivado write_verilog -mode funcsim,
// module renamed tsr_chk_net by net_chk.sh) against the simulator's answers. ops.memh (gen_ops.py) holds
// one op per line: the 256-bit instruction, the simulator's error code and ERR_INFO. Each op is run to
// rsp_valid; rsp_code / rsp_info must equal the file's. Prints NETCHK lines: net_chk.sh passes iff the
// last one reports 0 mismatches. The netlist ports are the flattened names Vivado writes for the unpacked
// arrays (req_insn[k], xc_rsp_ext[k][field]); the external class stays idle (no VU / AT ops).
`timescale 1ns / 1ps
module tsr_tb_chk_net;
  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  logic         req_valid = 1'b0;
  logic [255:0] req_insn  = '0;
  wire          busy, rsp_valid;
  wire  [3:0]   rsp_code;
  wire  [31:0]  rsp_info;
  tsr_chk_net dut (
    .\req_insn[0] (req_insn[0*32 +: 32]) ,
    .\req_insn[1] (req_insn[1*32 +: 32]) ,
    .\req_insn[2] (req_insn[2*32 +: 32]) ,
    .\req_insn[3] (req_insn[3*32 +: 32]) ,
    .\req_insn[4] (req_insn[4*32 +: 32]) ,
    .\req_insn[5] (req_insn[5*32 +: 32]) ,
    .\req_insn[6] (req_insn[6*32 +: 32]) ,
    .\req_insn[7] (req_insn[7*32 +: 32]) ,
    .\xc_rsp_ext[3][v] ('0) ,
    .\xc_rsp_ext[3][dram] ('0) ,
    .\xc_rsp_ext[3][wr] ('0) ,
    .\xc_rsp_ext[3][ch] ('0) ,
    .\xc_rsp_ext[3][lo] ('0) ,
    .\xc_rsp_ext[3][hi] ('0) ,
    .\xc_rsp_ext[2][v] ('0) ,
    .\xc_rsp_ext[2][dram] ('0) ,
    .\xc_rsp_ext[2][wr] ('0) ,
    .\xc_rsp_ext[2][ch] ('0) ,
    .\xc_rsp_ext[2][lo] ('0) ,
    .\xc_rsp_ext[2][hi] ('0) ,
    .\xc_rsp_ext[1][v] ('0) ,
    .\xc_rsp_ext[1][dram] ('0) ,
    .\xc_rsp_ext[1][wr] ('0) ,
    .\xc_rsp_ext[1][ch] ('0) ,
    .\xc_rsp_ext[1][lo] ('0) ,
    .\xc_rsp_ext[1][hi] ('0) ,
    .\xc_rsp_ext[0][v] ('0) ,
    .\xc_rsp_ext[0][dram] ('0) ,
    .\xc_rsp_ext[0][wr] ('0) ,
    .\xc_rsp_ext[0][ch] ('0) ,
    .\xc_rsp_ext[0][lo] ('0) ,
    .\xc_rsp_ext[0][hi] ('0) ,
    .clk(clk), .rst_n(rst_n), .req_valid(req_valid), .busy(busy),
    .rsp_valid(rsp_valid), .rsp_code(rsp_code), .rsp_info(rsp_info),
    .xc_rsp_valid(1'b0), .xc_rsp_code(4'd0), .xc_rsp_info(32'd0), .xc_rsp_cmd('0)
  );
  logic [255:0] ins [256];
  logic [3:0]   ecode [256];
  logic [31:0]  einfo [256];
  integer n, i, fd, bad;
  initial begin
    fd = $fopen("ops.memh", "r");
    n = 0;
    while (!$feof(fd) && n < 256) begin
      if ($fscanf(fd, "%h %h %h\n", ins[n], ecode[n], einfo[n]) == 3) n = n + 1;
    end
    $fclose(fd);
    bad = 0;
    repeat (30) @(posedge clk);   // glbl GSR is active for the first 100 ns
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
    for (i = 0; i < n; i = i + 1) begin
      @(posedge clk);
      req_insn  <= ins[i];
      req_valid <= 1'b1;
      @(posedge clk);
      req_valid <= 1'b0;
      begin : wt
        integer k;
        k = 0;
        while (rsp_valid !== 1'b1 && k < 200) begin
          @(posedge clk);
          k = k + 1;
        end
        if (k >= 200) begin
          bad = bad + 1;
          $display("NETCHK TIMEOUT op %0d busy %b rsp_valid %b", i, busy, rsp_valid);
        end
      end
      #1;
      if (rsp_code !== ecode[i] || rsp_info !== einfo[i]) begin
        bad = bad + 1;
        $display("NETCHK MISMATCH op %0d w0..w5 %h: netlist code %0d info %h, simulator code %0d info %h", i,
                 ins[i][191:0], rsp_code, rsp_info, ecode[i], einfo[i]);
      end
      @(posedge clk);
    end
    $display("NETCHK %0d ops, %0d mismatches", n, bad);
    $finish;
  end
endmodule
