// tsr_sram_1r1w.sv — generic (behavioural, vendor-neutral) simple dual-port SRAM:
// one write port and one read port, one clock, byte write enables, registered read.
//
// Spec: docs/rtl-core.md §2, block (d).
// Target variants (same module name and ports) live in rtl/tech/<target>/ and are selected by the
// target filelist (rtl/filelist.f = generic). Vendor primitives / attributes only there. This
// generic model follows the standard inference template for a simple dual-port RAM with byte
// write enables and an output pipeline (no attributes: the tool picks BRAM/URAM/LUTRAM).
//
// Contract:
//   write: when we = 1, bytes b of mem[waddr] with wbe[b] = 1 take wdata[8b +: 8] at the edge.
//   read : when re = 1, rdata = mem[raddr] exactly RD_LAT cycles later (RD_LAT >= 1, fully
//          pipelined: one read per cycle). The first stage is the RAM's own read register (it
//          holds its value while re = 0); stages 2..RD_LAT are free-running output registers
//          (BRAM DOB_REG / URAM OREG / cascade pipeline in a target variant). rdata is undefined
//          in cycles without a read result.
//   collision: a read and a write of the same address in the same cycle return the OLD data
//          (read-first); a read issued in any later cycle returns the new data.
//   no reset of the array or of the read pipeline (datapath only); contents are undefined at
//          power-up (SPM is undefined at start).
//   addresses >= DEPTH are never presented (callers range-check; tsr_spm asserts it in
//          simulation). DEPTH need not be a power of two.

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

  logic [WIDTH-1:0] mem [DEPTH];

  // Write port: byte-granular write enables.
  always_ff @(posedge clk) begin
    if (we) begin
      for (int unsigned b = 0; b < BW; b++) begin
        if (wbe[b]) mem[waddr][8*b +: 8] <= wdata[8*b +: 8];
      end
    end
  end

  // Read port: RAM read register (read-first: it samples mem before this edge's write lands),
  // then RD_LAT - 1 output pipeline registers.
  logic [WIDTH-1:0] rd_pipe [RD_LAT];

  always_ff @(posedge clk) begin
    if (re) rd_pipe[0] <= mem[raddr];
  end

  for (genvar i = 1; i < RD_LAT; i++) begin : g_rd_pipe
    always_ff @(posedge clk) rd_pipe[i] <= rd_pipe[i-1];
  end

  assign rdata = rd_pipe[RD_LAT-1];

endmodule
