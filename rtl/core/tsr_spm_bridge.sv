// tsr_spm_bridge.sv — core-domain end of the RISC-V SPM bridge: SPM client 3 (TSR_SPM_CL_BRIDGE)
// for the RISC-V's exact SPM window (0x4000_0000), its line-buffered alias (0x5000_0000) and the
// debug module's system-bus accesses to them (docs/design/04-control-core.md "CC" §4.1, §10.5).
//
// Owner: SPM agent. The clk_rv side (address decode and faults, the 4,096-cycle bound with
// BRIDGE_TIMEOUT, request tags and discarding of late answers, the line buffer tsr_spm_lb and its
// invalidation, byte / half / word lanes) and the tagged 4-phase crossing (CC §11.2: 64 b request,
// 513 b response) belong to the control subsystem; this module sees one request at a time with
// its payload held stable until accepted, and holds its response stable until it is taken.
//
// Request (req_valid / req_ready; req_ready = idle flop): req_waddr = SPM word address (window
//   byte offset / 4: line = req_waddr[WAW-1:4], word = req_waddr[3:0]).
//   load  (req_we = 0): reads the whole line; the response carries it (an exact load takes its
//         word on the clk_rv side, a line-buffer fill keeps the line).
//   store (req_we = 1): writes the word at its offset: wr_be = req_be shifted by 4 x word, the
//         data word replicated into every word lane (only the enabled bytes are written:
//         req_be / req_wdata are Ibex's lane-aligned data_be / data_wdata). The response comes
//         only after tsr_spm accepted the write (CC §10.5), so a tile op pushed after a completed
//         store sees it (CC §7.1, memory model rule 2), and a later load of the line returns it.
//   A line >= SPM lines (an address beyond SPM_BYTES, which the clk_rv decoder already faults)
//   answers rsp_err = 1 without an SPM access.
// Response (rsp_valid / rsp_ready, all from flops): rsp_err, rsp_line (loads; unchanged by stores
//   and errors). rsp_valid stays high, the payload unchanged, until rsp_ready.
// SPM side: single-port client (rtl-core §5 interface) for tsr_spm_cport. Every output is a flop
//   (R5); a request is raised with its payload ready and held until accepted (R2); a read is
//   issued only with the response register free (R4: one access at a time). No-deadlock (CC §10.3
//   (i)): the bridge never waits on memory while it holds an SPM request.
// Throughput: one access per 4 + RD_LAT core cycles at best (idle, request, grant, response,
//   handshake); RISC-V SPM traffic is small control data (op-list records, flags, argument
//   blocks, CC §10.5).

module tsr_spm_bridge
  import tsr_pkg::*;
#(
  parameter  int unsigned SPM_BYTES = TSR_DEF_SPM_BYTES,
  localparam int unsigned LINES     = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW        = (LINES > 1) ? $clog2(LINES) : 1,
  localparam int unsigned WAW       = AW + 4                     // SPM word address bits
) (
  input  logic                         clk,
  input  logic                         rst_n,

  // ---- request (core-domain end of the RISC-V <-> core bridge crossing) ----
  input  logic                         req_valid,
  output logic                         req_ready,
  input  logic                         req_we,
  input  logic [WAW-1:0]               req_waddr,
  input  logic [3:0]                   req_be,
  input  logic [31:0]                  req_wdata,

  // ---- response ----
  output logic                         rsp_valid,
  input  logic                         rsp_ready,
  output logic                         rsp_err,
  output logic [TSR_LINE_BITS-1:0]     rsp_line,

  // ---- SPM client (single port, through tsr_spm_cport) ----
  output logic                         spm_rd_req_valid,
  input  logic                         spm_rd_req_ready,
  output logic [AW-1:0]                spm_rd_req_line,
  input  logic                         spm_rd_rsp_valid,
  input  logic [TSR_LINE_BITS-1:0]     spm_rd_rsp_data,
  output logic                         spm_wr_valid,
  input  logic                         spm_wr_ready,
  output logic [AW-1:0]                spm_wr_line,
  output logic [TSR_LINE_BITS-1:0]     spm_wr_data,
  output logic [TSR_SPM_BE_W-1:0]      spm_wr_be
);

  if (SPM_BYTES < TSR_LINE_BYTES || (SPM_BYTES % TSR_LINE_BYTES) != 0) begin : g_chk_size
    $fatal(1, "tsr_spm_bridge: SPM_BYTES must be a positive multiple of 64");
  end

  typedef enum logic [2:0] {S_IDLE, S_RD, S_RWAIT, S_WR, S_RSP} st_e;

  st_e                      st_q;
  logic                     rd_v_q, wr_v_q, rsp_v_q, err_q;
  logic [AW-1:0]            line_q;
  logic [31:0]              wdata_q;
  logic [TSR_SPM_BE_W-1:0]  be_q;
  logic [TSR_LINE_BITS-1:0] rline_q;

  wire          req_fire = req_valid && (st_q == S_IDLE);
  wire [AW-1:0] req_lw = req_waddr[WAW-1:4];           // may exceed LINES (not a power of two)
  wire          req_bad = 32'(req_lw) >= LINES;

  // Byte enables of the line: req_be at word req_waddr[3:0].
  logic [TSR_SPM_BE_W-1:0] req_be_line;
  always_comb begin
    for (int unsigned w = 0; w < TSR_SPM_BE_W / 4; w++) begin
      req_be_line[4*w +: 4] = (req_waddr[3:0] == 4'(w)) ? req_be : 4'b0;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q    <= S_IDLE;
      rd_v_q  <= 1'b0;
      wr_v_q  <= 1'b0;
      rsp_v_q <= 1'b0;
      err_q   <= 1'b0;
    end else begin
      unique case (st_q)
        S_IDLE: if (req_fire) begin
          err_q <= req_bad;
          if (req_bad) begin
            rsp_v_q <= 1'b1;
            st_q    <= S_RSP;
          end else if (req_we) begin
            wr_v_q <= 1'b1;
            st_q   <= S_WR;
          end else begin
            rd_v_q <= 1'b1;
            st_q   <= S_RD;
          end
        end
        S_RD: if (spm_rd_req_ready) begin
          rd_v_q <= 1'b0;
          st_q   <= S_RWAIT;
        end
        S_RWAIT: if (spm_rd_rsp_valid) begin
          rsp_v_q <= 1'b1;
          st_q    <= S_RSP;
        end
        S_WR: if (spm_wr_ready) begin
          wr_v_q  <= 1'b0;
          rsp_v_q <= 1'b1;
          st_q    <= S_RSP;
        end
        S_RSP: if (rsp_ready) begin
          rsp_v_q <= 1'b0;
          st_q    <= S_IDLE;
        end
        default: st_q <= S_IDLE;
      endcase
    end
  end

  // Datapath registers (no reset; qualified by the valid flags above).
  always_ff @(posedge clk) begin
    if (req_fire) begin
      line_q  <= req_lw;
      wdata_q <= req_wdata;
      be_q    <= req_be_line;
    end
    if (st_q == S_RWAIT && spm_rd_rsp_valid) rline_q <= spm_rd_rsp_data;
  end

  assign req_ready        = (st_q == S_IDLE);
  assign rsp_valid        = rsp_v_q;
  assign rsp_err          = err_q;
  assign rsp_line         = rline_q;
  assign spm_rd_req_valid = rd_v_q;
  assign spm_rd_req_line  = line_q;
  assign spm_wr_valid     = wr_v_q;
  assign spm_wr_line      = line_q;
  assign spm_wr_data      = {(TSR_LINE_BITS / 32){wdata_q}};
  assign spm_wr_be        = be_q;

  // -------------------------------------------------------------------------------------------
  // Simulation checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      assert (!spm_rd_rsp_valid || st_q == S_RWAIT)
        else $error("%m: SPM read response without an outstanding bridge read");
      assert (!(rd_v_q && wr_v_q)) else $error("%m: read and write request at once");
    end
  end
`endif

endmodule
