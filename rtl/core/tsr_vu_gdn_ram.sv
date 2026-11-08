// tsr_vu_gdn_ram.sv — simple dual-port RAM of the GDN unit (docs/design/02-vector-unit.md §7.12.5):
// one write port with NWE write-enable groups, one read port, one clock, registered read.
//
// Row buffers, coefficient tables and scratch memories of tsr_vu_gdn use it. STYLE = "ultra" maps
// one RAM to URAM288 slices (UltraScale+), "block" to RAMB36, "distributed" to LUTRAM; inference and
// attributes only, no unisim primitive, so Verilator simulates the file as it is (the technique of
// rtl/tech/xilinx/tsr_sram_1r1w.sv, which this module follows for the read pipeline).
//
// Contract
//   write: when we[g] = 1, bits [g*WIDTH/NWE +: WIDTH/NWE] of mem[waddr] take wdata at the edge.
//   read : when re = 1, rdata = mem[raddr] exactly RD_LAT cycles later (RD_LAT >= 1, fully
//          pipelined). Stage 1 is the RAM's own read register; stages 2..RD_LAT only load behind a
//          read (Vivado 2024.2 absorbs URAM output stages only when they are enabled by the read
//          enable delayed alongside) and hold otherwise. rdata is undefined in cycles without a
//          read result.
//   collision: a same-cycle read and write of one address return the OLD data (read-first).
//   no reset; contents undefined at power-up.

module tsr_vu_gdn_ram #(
  parameter  int unsigned DEPTH  = 512,
  parameter  int unsigned WIDTH  = 32,
  parameter  int unsigned NWE    = 1,           // write-enable groups; WIDTH % NWE == 0
  parameter  int unsigned RD_LAT = 3,
  parameter                STYLE  = "ultra",    // "ultra" | "block" | "distributed"
  localparam int unsigned AW     = (DEPTH > 1) ? $clog2(DEPTH) : 1,
  localparam int unsigned GW     = WIDTH / NWE
) (
  input  logic             clk,
  input  logic [NWE-1:0]   we,
  input  logic [AW-1:0]    waddr,
  input  logic [WIDTH-1:0] wdata,
  input  logic             re,
  input  logic [AW-1:0]    raddr,
  output logic [WIDTH-1:0] rdata
);

  if (DEPTH < 2 || WIDTH < 1 || NWE < 1 || (WIDTH % NWE) != 0 || RD_LAT < 1) begin : g_chk_cfg
    $fatal(1, "tsr_vu_gdn_ram: DEPTH >= 2, WIDTH >= 1, WIDTH %% NWE == 0, RD_LAT >= 1");
  end

  // STYLE_ATTR is read only by the ram_style attribute, which Verilator ignores.
  /* verilator lint_off UNUSEDPARAM */
  localparam STYLE_ATTR = STYLE;
  /* verilator lint_on UNUSEDPARAM */

  (* ram_style = STYLE_ATTR *) logic [WIDTH-1:0] mem [DEPTH];

  always_ff @(posedge clk) begin
    for (int unsigned g = 0; g < NWE; g++) begin
      if (we[g]) mem[waddr][g*GW +: GW] <= wdata[g*GW +: GW];
    end
  end

  logic [WIDTH-1:0] rd_q;
  always_ff @(posedge clk) begin
    if (re) rd_q <= mem[raddr];
  end

  if (RD_LAT == 1) begin : g_lat1
    assign rdata = rd_q;
  end else begin : g_latn
    logic [RD_LAT-2:0] en_q;
    logic [WIDTH-1:0]  pipe_q [RD_LAT-1];
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
  end

endmodule
