// tsr_sram_1r1w.sv — Xilinx UltraScale+ (U200) variant of the simple dual-port SRAM: one write
// port and one read port, one clock, byte write enables, registered read.
//
// Owner: xtech agent (M3 part C; docs/rtl-m3.md §9.2, docs/rtl-m2.md §5 and §12 "Consequences
// for M4" item 1). Selected by rtl/filelist_xilinx.f in place of rtl/tech/generic.
//
// Contract: identical to rtl/tech/generic/tsr_sram_1r1w.sv (same module name, parameters and
// ports; tb/tech compares the two cycle by cycle):
//   write: when we = 1, bytes b of mem[waddr] with wbe[b] = 1 take wdata[8b +: 8] at the edge.
//   read : when re = 1, rdata = mem[raddr] exactly RD_LAT cycles later (RD_LAT >= 1, fully
//          pipelined). Stage 1 is the RAM's own read register (holds while re = 0). rdata is
//          undefined in cycles without a read result.
//   collision: a same-cycle read and write of one address return the OLD data (read-first); a
//          read in any later cycle returns the new data.
//   no reset of the array or of the read pipeline; contents undefined at power-up; addresses
//          >= DEPTH are never presented.
//   Cycle behaviour against generic: identical in every cycle for "block" and "distributed"
//   (stages 2..RD_LAT free-running as there); identical in every cycle with a read result for
//   "ultra", whose stages 2..RD_LAT only load behind a read (below) and hold otherwise.
//
// Mapping (inference + attributes only, no unisim primitive, so Verilator simulates this file):
//   RAM_STYLE by size (one rule for every instance of the core, docs/rtl-m3.md §9.3):
//     "ultra"       DEPTH >= 4096 and DEPTH x WIDTH >= 4096 x 72 (at least one full URAM288):
//                   the SPM banks (U200: 4 banks x 16 K lines x 512 bits = 4 x 8 columns x
//                   4-deep cascades = 128 URAM; 72-bit columns with 9 byte enables each).
//     "distributed" DEPTH <= 32 (LUTRAM; only the reduced test configurations, e.g. the K_MAX 256
//                   activation buffers of 2 words).
//     "block"       everything else: activation buffer QS / DBS (256 / 72 x 512 at K_MAX 32768),
//                   weight FIFO (64..1024 x 512), QACT line buffer (64 x 512): RAMB36, 512 x 72 SDP
//                   (8 data bytes per RAMB36, parity bits unused; 1 K deep: 1 K x 36).
//   Read-first collision: BRAM SDP with a common clock supports READ_FIRST (Vivado sets
//   WRITE_MODE READ_FIRST); URAM288 executes port A before port B, and Vivado maps the read to
//   port A and the write to port B (RDB_WR_A = 0, RDB_WR_B = 1). tb/tech/viv/net_sim.sh checks
//   both on the synthesised netlists (collisions included).
//   Read pipeline, so that the stages land in the primitive's registers:
//     block       stage 1 = the BRAM read latch, stage 2 = DOA/DOB_REG (a free-running register
//                 right behind the read register is absorbed, REGCE = 1); stages 3..RD_LAT are
//                 fabric flops.
//     ultra       stage 1 = the URAM read, stages 2..RD_LAT = OREG, then the cascade pipeline
//                 registers (REG_CAS): Vivado 2024.2 absorbs URAM output stages only when they are
//                 enabled by the read enable delayed alongside (UG901 template; a free-running
//                 stage stays in fabric with "no pipeline registers"), so here stage k loads iff
//                 a read result reaches it (en_q). 16 K x 512: RD_LAT 2 / 3 / 4 absorb 1 / 2 / 3
//                 stages, no fabric flop left.
//     distributed the read register is a fabric flop after the asynchronous LUTRAM read.
//   Fabric stages: SRL extraction off (timing registers; a 512-bit SRL output is slow).

module tsr_sram_1r1w #(
  parameter  int unsigned DEPTH  = 1024,
  parameter  int unsigned WIDTH  = 512,
  parameter  int unsigned RD_LAT = 1,
  localparam int unsigned AW     = (DEPTH > 1) ? $clog2(DEPTH) : 1,
  localparam int unsigned BW     = WIDTH / 8
) (
  input  logic             clk,
  input  logic             we,
  input  logic [AW-1:0]    waddr,
  input  logic [WIDTH-1:0] wdata,
  input  logic [BW-1:0]    wbe,
  input  logic             re,
  input  logic [AW-1:0]    raddr,
  output logic [WIDTH-1:0] rdata
);

  if (DEPTH < 1 || WIDTH < 8 || (WIDTH % 8) != 0 || RD_LAT < 1) begin : g_chk_cfg
    $fatal(1, "tsr_sram_1r1w: DEPTH >= 1, WIDTH a positive multiple of 8, RD_LAT >= 1");
  end

  // 64-bit products: DEPTH x WIDTH may exceed 2^32 in principle (not in any configuration here).
  localparam bit IS_ULTRA = (DEPTH >= 4096) && (64'(DEPTH) * 64'(WIDTH) >= 64'(4096 * 72));
  localparam bit IS_DIST  = !IS_ULTRA && (DEPTH <= 32);
  // STYLE is read only by the ram_style attribute below, which Verilator ignores.
  /* verilator lint_off UNUSEDPARAM */
  localparam     STYLE    = IS_ULTRA ? "ultra" : (IS_DIST ? "distributed" : "block");
  /* verilator lint_on UNUSEDPARAM */

  (* ram_style = STYLE *) logic [WIDTH-1:0] mem [DEPTH];

  // Write port: byte-granular write enables.
  always_ff @(posedge clk) begin
    if (we) begin
      for (int unsigned b = 0; b < BW; b++) begin
        if (wbe[b]) mem[waddr][8*b +: 8] <= wdata[8*b +: 8];
      end
    end
  end

  // Read port: RAM read register (read-first: it samples mem before this edge's write lands).
  logic [WIDTH-1:0] rd_q;

  always_ff @(posedge clk) begin
    if (re) rd_q <= mem[raddr];
  end

  if (RD_LAT == 1) begin : g_lat1
    assign rdata = rd_q;
  end else if (IS_ULTRA) begin : g_uram
    // Stages 2..RD_LAT (OREG, REG_CAS): stage k + 2 loads iff en_q[k] (a read result arrives).
    logic [RD_LAT-2:0]  en_q;
    logic [WIDTH-1:0]   pipe_q [RD_LAT - 1];
    always_ff @(posedge clk) en_q <= (RD_LAT - 1)'({en_q, re});
    always_ff @(posedge clk) begin
      if (en_q[0]) pipe_q[0] <= rd_q;
    end
    for (genvar i = 1; i < RD_LAT - 1; i++) begin : g_pipe
      always_ff @(posedge clk) begin
        if (en_q[i]) pipe_q[i] <= pipe_q[i-1];
      end
    end
    assign rdata = pipe_q[RD_LAT-2];
  end else begin : g_latn
    // Stage 2: the BRAM output register (DOA/DOB_REG), free-running.
    logic [WIDTH-1:0] ord_q;
    always_ff @(posedge clk) ord_q <= rd_q;

    if (RD_LAT == 2) begin : g_lat2
      assign rdata = ord_q;
    end else begin : g_lat3p
      // Stages 3..RD_LAT: fabric pipeline.
      (* shreg_extract = "no" *) logic [WIDTH-1:0] pipe_q [RD_LAT - 2];
      always_ff @(posedge clk) pipe_q[0] <= ord_q;
      for (genvar i = 1; i < RD_LAT - 2; i++) begin : g_pipe
        always_ff @(posedge clk) pipe_q[i] <= pipe_q[i-1];
      end
      assign rdata = pipe_q[RD_LAT-3];
    end
  end

endmodule
