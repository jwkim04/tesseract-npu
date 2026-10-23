// tsr_spm.sv — line-addressed scratchpad (SPM_BYTES / 64 lines x 512 bits) with NCL client ports.
//
// Owner: SPM agent (docs/rtl-m2.md §2, block (d); port plan §5).
//
// Built from tsr_sram_1r1w instances (rtl/tech/<target>/); no vendor primitives here.
// Clients (tsr_pkg TSR_SPM_CL_*): 0 = DMA, 1 = QACT, 2 = MVM (M3, tied off in M2).
// Contract, per client c:
//   read  : rd_req_valid[c]/rd_req_ready[c] accept rd_req_line[c]. Exactly RD_LAT cycles after the
//           accepting edge, rd_rsp_valid[c] = 1 for one cycle with the line on rd_rsp_data
//           (shared by all clients, qualified by rd_rsp_valid[c]). Responses are in request order
//           and have NO backpressure: a client issues a read only when it can take the response.
//           One read per cycle in total (single read port). "Exactly RD_LAT cycles": a request
//           accepted at edge E is sampled by the client at edge E + RD_LAT (RD_LAT = 1: the data
//           is there in the cycle right after the accepting edge).
//   write : wr_valid[c]/wr_ready[c] accept (wr_line[c], wr_data[c], wr_be[c]); byte b of the line is
//           written iff wr_be[c][b]. One write per cycle in total (single write port).
//   order : a read accepted in a cycle after a write was accepted returns the written data
//           (internal pipelining must not break this); same-cycle read and write of one line
//           returns the old data.
//   arbitration: fixed priority, lowest client index first, separately for read and write, with a
//           registered grant (rules R3/R5: no combinational path from any input to any output):
//             rd_req_ready = rd_gnt_q (one-hot or zero);
//             rd_gnt_q    <= (|rd_req_valid) ? lowest set bit of rd_req_valid : rd_gnt_q;
//             reset: no client granted (ready = 0 during reset).
//           (write side identical with wr_valid / wr_ready.) A client that is not the current
//           owner waits one cycle before its first transfer; the owner keeps the grant while the
//           port is idle, so a streaming client gets one transfer per cycle. In M2 only the
//           dispatched unit drives requests, so there is never contention; ready is still low in
//           the first request cycle after another client used the port (and during reset), and
//           clients must honour it (hold valid and payload, rule R2).
//   lines >= SPM lines are never presented (tsr_cmdproc range-checks every operand).
//   Simulation only (`ifndef SYNTHESIS): asserts that presented lines are < SPM lines and that
//   clients obey R2 (a request not yet accepted stays valid with an unchanged payload). The
//   checks are armed by the first clock edge with rst_n low; before it the inputs and the state
//   are arbitrary (Verilator's X randomisation can even produce a time-0 edge with rst_n high).
// Line data convention: byte j of a line is bits [8j+7:8j] (little-endian), f32 element i of a line
// is bits [32i+31:32i].
//
// Microarchitecture:
//   - Depth banking: NBANK = ceil(LINES / BANK_LINES) banks of BANK_LINES lines (the last bank
//     holds the remainder, so any multiple of 64 bytes works), selected by the high line-address
//     bits; only the addressed bank sees re/we. U200 default: 4 MiB = 4 banks x 16 K lines; with
//     URAM288 (4 K x 72) a bank is 8 columns x 4-deep cascade = 32 URAMs, 128 in total.
//   - Latency split (RD_LAT = IN_REG + RAM_LAT + OUT_REG, all requests take the same path):
//       IN_REG  = 1 if RD_LAT >= 2: arbitration result (enables, bank, offset, write data/be) is
//                 registered before the RAMs, so the RAM address/data fan-out starts at a flop.
//                 Reads and writes share this stage, which keeps the write->read order above.
//       OUT_REG = 1 if RD_LAT >= 4 and NBANK > 1: register after the bank read mux.
//       RAM_LAT = the rest (>= 1): tsr_sram_1r1w RD_LAT (RAM read register + output registers).
//     RD_LAT 2 (default): flop -> RAM read register -> bank mux (one LUT level for <= 4 banks)
//     -> rd_rsp_data. U200 timing (M4) is expected to want RD_LAT 4: IN_REG + RAM 2 (URAM OREG)
//     + OUT_REG; every client takes SPM_RD_LAT as a parameter.
//   - rd_rsp_valid comes from an RD_LAT-deep shift register of one-hot client bits (reset).
//   - Port plan (docs/rtl-m2.md §5): M2 clients 0/1 time-share the 1R1W port. M3 MVM uses client 2
//     unchanged: activation broadcast = a read stream of the QA lines (one line per cycle; the MVM
//     unit fans rd_rsp_data out to the per-channel clusters through its own pipelined broadcast
//     tree), result gather = y line writes on client 2's write port (ACC: old y lines read first).
//     M6 (VU, 2 reads + 1 write per cycle) adds a second read port per client and either a second
//     copy of the array or even/odd line interleaving instead of depth banking; the client
//     interface stays.

module tsr_spm
  import tsr_pkg::*;
#(
  parameter  int unsigned SPM_BYTES  = TSR_DEF_SPM_BYTES,
  parameter  int unsigned NCL        = TSR_SPM_NCL,
  parameter  int unsigned RD_LAT     = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned BANK_LINES = 16384,   // lines per depth bank (power of two, >= 2)
  localparam int unsigned LINES      = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW         = (LINES > 1) ? $clog2(LINES) : 1
) (
  input  logic                                   clk,
  input  logic                                   rst_n,
  // read port per client
  input  logic [NCL-1:0]                         rd_req_valid,
  output logic [NCL-1:0]                         rd_req_ready,
  input  logic [NCL-1:0][AW-1:0]                 rd_req_line,
  output logic [NCL-1:0]                         rd_rsp_valid,
  output logic [TSR_LINE_BITS-1:0]               rd_rsp_data,
  // write port per client
  input  logic [NCL-1:0]                         wr_valid,
  output logic [NCL-1:0]                         wr_ready,
  input  logic [NCL-1:0][AW-1:0]                 wr_line,
  input  logic [NCL-1:0][TSR_LINE_BITS-1:0]      wr_data,
  input  logic [NCL-1:0][TSR_SPM_BE_W-1:0]       wr_be
);

  // -------------------------------------------------------------------------------------------
  // Configuration
  // -------------------------------------------------------------------------------------------
  localparam int unsigned NBANK      = (LINES + BANK_LINES - 1) / BANK_LINES;
  localparam int unsigned BL_LOG     = $clog2(BANK_LINES);
  localparam int unsigned OFF_W      = (NBANK > 1) ? BL_LOG : AW;           // RAM address width
  localparam int unsigned BSEL_W     = (NBANK > 1) ? $clog2(NBANK) : 1;
  localparam int unsigned LAST_DEPTH = LINES - (NBANK - 1) * BANK_LINES;
  localparam int unsigned IN_REG     = (RD_LAT >= 2) ? 1 : 0;
  localparam int unsigned OUT_REG    = (RD_LAT >= 4 && NBANK > 1) ? 1 : 0;
  localparam int unsigned RAM_LAT    = RD_LAT - IN_REG - OUT_REG;

  if (SPM_BYTES < TSR_LINE_BYTES || (SPM_BYTES % TSR_LINE_BYTES) != 0) begin : g_chk_size
    $fatal(1, "tsr_spm: SPM_BYTES must be a positive multiple of 64");
  end
  if (NCL < 1 || RD_LAT < 1) begin : g_chk_ports
    $fatal(1, "tsr_spm: NCL >= 1 and RD_LAT >= 1 required");
  end
  if (BANK_LINES < 2 || (BANK_LINES & (BANK_LINES - 1)) != 0) begin : g_chk_bank
    $fatal(1, "tsr_spm: BANK_LINES must be a power of two >= 2");
  end

  function automatic logic [BSEL_W-1:0] bank_of(logic [AW-1:0] line);
    return (NBANK > 1) ? BSEL_W'(line >> BL_LOG) : '0;
  endfunction

  function automatic logic [NCL-1:0] lowest_set(logic [NCL-1:0] v);
    return v & (~v + NCL'(1));
  endfunction

  // -------------------------------------------------------------------------------------------
  // Arbitration: registered fixed-priority grant per port (see header)
  // -------------------------------------------------------------------------------------------
  logic [NCL-1:0] rd_gnt_q, wr_gnt_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_gnt_q <= '0;
      wr_gnt_q <= '0;
    end else begin
      if (|rd_req_valid) rd_gnt_q <= lowest_set(rd_req_valid);
      if (|wr_valid)     wr_gnt_q <= lowest_set(wr_valid);
    end
  end

  assign rd_req_ready = rd_gnt_q;
  assign wr_ready     = wr_gnt_q;

  logic [NCL-1:0] rd_take, wr_take;
  logic           rd_fire, wr_fire;
  assign rd_take = rd_req_valid & rd_gnt_q;
  assign wr_take = wr_valid & wr_gnt_q;
  assign rd_fire = |rd_take;
  assign wr_fire = |wr_take;

  // Request of the granted client (one-hot AND-OR mux, selected by the grant flop).
  logic [AW-1:0]             rd_line_sel, wr_line_sel;
  logic [TSR_LINE_BITS-1:0]  wr_data_sel;
  logic [TSR_SPM_BE_W-1:0]   wr_be_sel;

  always_comb begin
    rd_line_sel = '0;
    wr_line_sel = '0;
    wr_data_sel = '0;
    wr_be_sel   = '0;
    for (int unsigned c = 0; c < NCL; c++) begin
      if (rd_gnt_q[c]) rd_line_sel = rd_line_sel | rd_req_line[c];
      if (wr_gnt_q[c]) begin
        wr_line_sel = wr_line_sel | wr_line[c];
        wr_data_sel = wr_data_sel | wr_data[c];
        wr_be_sel   = wr_be_sel   | wr_be[c];
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // RAM input stage (IN_REG): what the banks see this cycle
  // -------------------------------------------------------------------------------------------
  logic                      rd_en_s, wr_en_s;
  logic [BSEL_W-1:0]         rd_bank_s, wr_bank_s;
  logic [OFF_W-1:0]          rd_off_s, wr_off_s;
  logic [TSR_LINE_BITS-1:0]  wr_data_s;
  logic [TSR_SPM_BE_W-1:0]   wr_be_s;

  if (IN_REG != 0) begin : g_in_reg
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        rd_en_s <= 1'b0;
        wr_en_s <= 1'b0;
      end else begin
        rd_en_s <= rd_fire;
        wr_en_s <= wr_fire;
      end
    end
    // Datapath registers load every cycle (no clock enable: rd_en_s / wr_en_s gate their use, and
    // a CE would be a ~600-flop net driven from the clients' valid flops).
    always_ff @(posedge clk) begin
      rd_bank_s <= bank_of(rd_line_sel);
      rd_off_s  <= rd_line_sel[OFF_W-1:0];
      wr_bank_s <= bank_of(wr_line_sel);
      wr_off_s  <= wr_line_sel[OFF_W-1:0];
      wr_data_s <= wr_data_sel;
      wr_be_s   <= wr_be_sel;
    end
  end else begin : g_no_in_reg
    assign rd_en_s   = rd_fire;
    assign rd_bank_s = bank_of(rd_line_sel);
    assign rd_off_s  = rd_line_sel[OFF_W-1:0];
    assign wr_en_s   = wr_fire;
    assign wr_bank_s = bank_of(wr_line_sel);
    assign wr_off_s  = wr_line_sel[OFF_W-1:0];
    assign wr_data_s = wr_data_sel;
    assign wr_be_s   = wr_be_sel;
  end

  // -------------------------------------------------------------------------------------------
  // Banks
  // -------------------------------------------------------------------------------------------
  logic [NBANK-1:0][TSR_LINE_BITS-1:0] bank_rdata;

  for (genvar b = 0; b < NBANK; b++) begin : g_bank
    localparam int unsigned DEPTH_B = (b == NBANK - 1) ? LAST_DEPTH : BANK_LINES;
    localparam int unsigned AW_B    = (DEPTH_B > 1) ? $clog2(DEPTH_B) : 1;
    logic re_b, we_b;
    assign re_b = rd_en_s && (rd_bank_s == BSEL_W'(b));
    assign we_b = wr_en_s && (wr_bank_s == BSEL_W'(b));

    tsr_sram_1r1w #(
      .DEPTH(DEPTH_B), .WIDTH(TSR_LINE_BITS), .RD_LAT(RAM_LAT)
    ) u_ram (
      .clk   (clk),
      .we    (we_b),
      .waddr (wr_off_s[AW_B-1:0]),
      .wdata (wr_data_s),
      .wbe   (wr_be_s),
      .re    (re_b),
      .raddr (rd_off_s[AW_B-1:0]),
      .rdata (bank_rdata[b])
    );
  end

  // -------------------------------------------------------------------------------------------
  // Read data: bank mux (bank select delayed by RAM_LAT), optional output register
  // -------------------------------------------------------------------------------------------
  if (NBANK > 1) begin : g_bank_mux
    logic [BSEL_W-1:0]        bsel_pipe [RAM_LAT];
    logic [TSR_LINE_BITS-1:0] rd_mux;

    always_ff @(posedge clk) bsel_pipe[0] <= rd_bank_s;
    for (genvar i = 1; i < RAM_LAT; i++) begin : g_bsel
      always_ff @(posedge clk) bsel_pipe[i] <= bsel_pipe[i-1];
    end

    always_comb begin
      rd_mux = '0;
      for (int unsigned b = 0; b < NBANK; b++) begin
        if (bsel_pipe[RAM_LAT-1] == BSEL_W'(b)) rd_mux = rd_mux | bank_rdata[b];
      end
    end

    if (OUT_REG != 0) begin : g_out_reg
      logic [TSR_LINE_BITS-1:0] rd_data_q;
      always_ff @(posedge clk) rd_data_q <= rd_mux;
      assign rd_rsp_data = rd_data_q;
    end else begin : g_no_out_reg
      assign rd_rsp_data = rd_mux;
    end
  end else begin : g_one_bank
    assign rd_rsp_data = bank_rdata[0];
  end

  // -------------------------------------------------------------------------------------------
  // Read response valid: RD_LAT-deep pipeline of the one-hot accepted client
  // -------------------------------------------------------------------------------------------
  logic [NCL-1:0] rsp_pipe [RD_LAT];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int unsigned i = 0; i < RD_LAT; i++) rsp_pipe[i] <= '0;
    end else begin
      rsp_pipe[0] <= rd_take;
      for (int unsigned i = 1; i < RD_LAT; i++) rsp_pipe[i] <= rsp_pipe[i-1];
    end
  end

  assign rd_rsp_valid = rsp_pipe[RD_LAT-1];

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks of the client contract
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic [NCL-1:0]                     chk_rd_hold, chk_wr_hold;
  logic [NCL-1:0][AW-1:0]             chk_rd_line, chk_wr_line;
  logic [NCL-1:0][TSR_LINE_BITS-1:0]  chk_wr_data;
  logic [NCL-1:0][TSR_SPM_BE_W-1:0]   chk_wr_be;

  // Armed once a reset has been seen (same scheme as tsr_mem_arb / tsr_dma): before the first
  // reset edge, chk_*_hold and the grant flops hold power-up garbage and the inputs are not driven
  // yet. The initial block runs before Verilator's first evaluation (which may already see a
  // rising clock edge, with X-randomised rst_n and inputs).
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      chk_rd_hold <= '0;
      chk_wr_hold <= '0;
    end else begin
      if (chk_on_q) begin
        for (int unsigned c = 0; c < NCL; c++) begin
          if (rd_req_valid[c]) begin
            assert (32'(rd_req_line[c]) < LINES)
              else $error("%m: client %0d read line %0d >= %0d SPM lines", c, rd_req_line[c], LINES);
          end
          if (wr_valid[c]) begin
            assert (32'(wr_line[c]) < LINES)
              else $error("%m: client %0d write line %0d >= %0d SPM lines", c, wr_line[c], LINES);
          end
          if (chk_rd_hold[c]) begin
            assert (rd_req_valid[c] && rd_req_line[c] == chk_rd_line[c])
              else $error("%m: client %0d dropped or changed a read request before it was accepted (R2)", c);
          end
          if (chk_wr_hold[c]) begin
            assert (wr_valid[c] && wr_line[c] == chk_wr_line[c] && wr_data[c] == chk_wr_data[c]
                    && wr_be[c] == chk_wr_be[c])
              else $error("%m: client %0d dropped or changed a write request before it was accepted (R2)", c);
          end
        end
        assert ($onehot0(rd_gnt_q) && $onehot0(wr_gnt_q)) else $error("%m: grant not one-hot");
      end
      chk_rd_hold <= rd_req_valid & ~rd_req_ready;
      chk_wr_hold <= wr_valid & ~wr_ready;
    end
    chk_rd_line <= rd_req_line;
    chk_wr_line <= wr_line;
    chk_wr_data <= wr_data;
    chk_wr_be   <= wr_be;
  end
`endif

endmodule
