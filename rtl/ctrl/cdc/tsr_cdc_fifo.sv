// tsr_cdc_fifo.sv — asynchronous FIFO with gray-coded pointers (CC §11.2: the issue FIFO, 32 x 387 b
// clk_rv -> core, CC §7.2; the log FIFO, 16 x 32 b clk_rv -> core, CC §6.5; docs/ctrl-riscv.md §7.2).
//
// Spec: CC §2.2 `tsr_cdc_fifo`.
//
// Contract:
//   write side (wclk): wr_free = free entries (0..DEPTH) as the write side sees them (the read
//     pointer arrives through a synchroniser, so it may under-report, never over-report); wr_en
//     writes wr_data and is legal only while wr_free != 0 (asserted). wr_free updates in the cycle
//     after a write. wr_empty = no entry written and not yet consumed by the read side, as the write
//     side sees it (CC §5.1 QIDLE: "the issue FIFO empty").
//   read side (rclk): first-word-fall-through with a registered output: rd_valid / rd_data show the
//     oldest entry; rd_ready (with rd_valid) consumes it (valid/ready rules R1-R3, docs/rtl-core.md
//     §1: rd_valid does not depend on rd_ready). An entry counts as used until it is consumed, so
//     DEPTH entries are the bound for everything written and not yet consumed. rd_count = entries
//     written and not yet consumed as the read side sees them (the write pointer arrives through a
//     synchroniser: it may under-report, never over-report), the presented one included
//     (Q_FIFO, EX §5.4).
//   Memory: an array written in wclk (LUTRAM on the FPGA) and read into a register in rclk; the
//     array -> rd_data path and the gray pointers are the crossing paths (set_max_delay
//     -datapath_only, set_bus_skew on the pointers: rtl/ctrl/viv/tsr_ctrl_cdc.xdc, CC §11.3).
//   Latency: an entry written at a wclk edge is visible at rd_valid after 1 wclk + STAGES + 1 rclk
//   cycles; its slot is free again 1 rclk + STAGES wclk cycles after it was consumed.
// DEPTH: a power of two >= 2. Resets: synchronous, active low, both from the same hardware reset.

module tsr_cdc_fifo #(
  parameter int unsigned W      = 32,
  parameter int unsigned DEPTH  = 16,
  parameter int unsigned STAGES = 2,
  localparam int unsigned AW    = $clog2(DEPTH)
) (
  input  logic          wclk,
  input  logic          wrst_n,
  input  logic          wr_en,
  input  logic [W-1:0]  wr_data,
  output logic [AW:0]   wr_free,
  output logic          wr_empty,
  input  logic          rclk,
  input  logic          rrst_n,
  output logic          rd_valid,
  output logic [W-1:0]  rd_data,
  input  logic          rd_ready,
  output logic [AW:0]   rd_count
);

  if (DEPTH < 2 || (DEPTH & (DEPTH - 1)) != 0) begin : g_chk
    $fatal(1, "tsr_cdc_fifo: DEPTH must be a power of two >= 2");
  end

  function automatic logic [AW:0] bin2gray(input logic [AW:0] b);
    return b ^ (b >> 1);
  endfunction
  function automatic logic [AW:0] gray2bin(input logic [AW:0] g);
    logic [AW:0] b;
    b[AW] = g[AW];
    for (int i = int'(AW) - 1; i >= 0; i--) b[i] = b[i+1] ^ g[i];
    return b;
  endfunction

  logic [W-1:0] mem [DEPTH];

  // Pointers of both sides, declared before their first use (each side reads the other's gray
  // pointer through its synchroniser).
  logic [AW:0] wbin_q, wgray_q, rgray_ws, rbin_ws;
  logic [AW:0] rbin_q, rgray_q, wgray_rs, wbin_rs, rbin_nxt;

  // ---------------------------------------------------------------- write side (wclk)

  always_ff @(posedge wclk) begin
    if (wr_en) mem[wbin_q[AW-1:0]] <= wr_data;
  end

  always_ff @(posedge wclk) begin
    if (!wrst_n) begin
      wbin_q  <= '0;
      wgray_q <= '0;
    end else if (wr_en) begin
      wbin_q  <= wbin_q + 1'b1;
      wgray_q <= bin2gray(wbin_q + 1'b1);
    end
  end

  tsr_sync #(.W(AW + 1), .STAGES(STAGES)) u_sync_r2w (
    .clk(wclk), .rst_n(wrst_n), .d(rgray_q), .q(rgray_ws)
  );
  assign rbin_ws  = gray2bin(rgray_ws);
  assign wr_free  = (AW+1)'(DEPTH) - (wbin_q - rbin_ws);
  assign wr_empty = wbin_q == rbin_ws;

  // ---------------------------------------------------------------- read side (rclk)

  tsr_sync #(.W(AW + 1), .STAGES(STAGES)) u_sync_w2r (
    .clk(rclk), .rst_n(rrst_n), .d(wgray_q), .q(wgray_rs)
  );
  assign wbin_rs  = gray2bin(wgray_rs);
  assign rbin_nxt = rbin_q + (AW+1)'(rd_valid && rd_ready);
  assign rd_count = wbin_rs - rbin_q;

  always_ff @(posedge rclk) begin
    if (!rrst_n) begin
      rbin_q   <= '0;
      rgray_q  <= '0;
      rd_valid <= 1'b0;
    end else begin
      rbin_q   <= rbin_nxt;
      rgray_q  <= bin2gray(rbin_nxt);
      rd_valid <= wbin_rs != rbin_nxt;
    end
  end

  // Output register: always the entry at the (next) read pointer (FWFT).
  always_ff @(posedge rclk) begin
    rd_data <= mem[rbin_nxt[AW-1:0]];
  end

`ifndef SYNTHESIS
  bit arm;   // armed once the reset has been seen
  initial arm = 1'b0;
  always @(posedge wclk) begin   // not always_ff: the initial block writes arm too (IEEE 1800 9.2.2.4)
    if (!wrst_n) arm <= 1'b1;
    else if (arm && wr_en && wr_free == '0) $error("tsr_cdc_fifo: write while full");
  end
`endif

endmodule
