// tsr_spm.sv — line-addressed scratchpad (SPM_BYTES / 64 lines x 512 bits) in two line-interleaved
// banks (bank E: even lines, bank O: odd lines; each 1R1W, so the SPM has 2 read and 2 write
// ports), with per-bank client ports and round-robin arbitration per bank port.
//
// Owner: SPM agent (docs/design/04-control-core.md "CC" §10, CC-12; docs/rtl-core.md §5).
//
// Built from tsr_sram_1r1w instances (rtl/tech/<target>/); no vendor primitives here.
// Clients (tsr_pkg TSR_SPM_CL_*, CC §10.2): 0 DMA, 1 QACT, 2 MVM, 3 RISC-V bridge, 4 VU, 5 AT;
// absent clients are tied off by the instantiating module (valid 0). Bank b: 0 = E, 1 = O.
// The clients with a single-port interface (DMA, QACT, MVM, bridge) reach their two bank ports
// through tsr_spm_cport; the VU drives both bank ports itself.
// Contract, per client c and bank b (CC §10.2):
//   read  : rd_req_valid[c][b] / rd_req_ready[c][b] accept rd_req_line[c][b], whose bit 0 is b
//           (asserted). Exactly RD_LAT cycles after the accepting edge rd_rsp_valid[c][b] = 1 for
//           one cycle with the line on rd_rsp_data[b] (shared by all clients of bank b, qualified
//           by rd_rsp_valid[c][b]). No backpressure (R4): a client issues a read only when it can
//           take the response. Responses in request order per (c, b). "Exactly RD_LAT cycles": a
//           request accepted at edge E is sampled by the client at edge E + RD_LAT.
//   write : wr_valid[c][b] / wr_ready[c][b] accept (wr_line, wr_data, wr_be)[c][b]; byte j of the
//           line is written iff wr_be[c][b][j].
//   A client may raise requests on both banks in the same cycle (two lines per cycle). Per bank
//   one read and one write transfer per cycle in total. R2 per (c, b): a raised request stays
//   with an unchanged payload until accepted.
//   order : a read accepted after a write of the same line was accepted returns the new data (one
//           bank owns the line; its input stage is shared by reads and writes, so internal
//           pipelining cannot reorder them); a read and a write of one line accepted at the same
//           edge: the read returns the old data (read-first).
//   arbitration, separately per bank and per port (read, write): a registered one-hot grant,
//           round robin with the current owner last:
//             gnt_q <= |req ? first requesting client after the owner (index order, wrapping;
//                             no owner: the lowest requesting index) : gnt_q;
//             reset: no client granted (ready = 0 during reset).
//           ready = gnt_q: no combinational path from any input to any output (R5). The owner
//           keeps the grant while the port idles, so a client streaming alone transfers every
//           cycle (a single-port client alternating banks keeps both grants). A request that is
//           not the owner's waits at least one cycle; a waiting client (R2) is granted after at
//           most NCL - 1 other grants, every grant move transfers in the next cycle (the new owner
//           requested and holds its request), so once the port has an owner a request waits at
//           most NCL - 1 cycles (asserted in simulation; CC §10.3 (ii), (iii)).
//   lines >= SPM lines are never presented (the checker range-checks every operand).
//   Simulation only (`ifndef SYNTHESIS): line range, line[0] == b, R2 per (c, b), one-hot grants
//   and the round-robin wait bound. The checks are armed by the first clock edge with rst_n low;
//   before it the inputs and the state are arbitrary (Verilator's X randomisation can even produce
//   a time-0 edge with rst_n high).
// Line data convention: byte j of a line is bits [8j+7:8j] (little-endian), f32 element i of a line
// is bits [32i+31:32i].
//
// No-deadlock argument (CC §10.3; it belongs to every client's contract): (i) every client raises
// a request only with its payload ready and never waits on memory while it holds one (the DMA's LD
// writes beats already in its queue, an ST reserves queue space before it reads; QACT, the MVM
// gather, the VU and AT write results they hold; the bridge holds its data); (ii) a granted
// request transfers in that cycle; (iii) round robin grants a waiting client within NCL - 1 other
// grants. So every SPM request completes within a bounded number of cycles that does not depend
// on memory progress, and with the DMA LD burst reservation (tsr_dma, CC §10.4) no memory-response
// path ever waits on the SPM.
//
// Microarchitecture:
//   - Line L lives in bank b = L[0] at bank-local line a = L >> 1. A bank of LINES_b lines
//     (LINES_E = ceil(LINES / 2), LINES_O = floor(LINES / 2)) is split into depth banks of
//     BANK_LINES lines (the last holds the remainder, so any multiple of 64 bytes works) selected
//     by the high bits of a. RAM group g_bank[i] = depth bank i / 2 of bank i % 2 (the groups of
//     bank O are one fewer than bank E's at most, so the indices stay dense). U200 default: 4 MiB =
//     2 banks x 32 K lines = 2 x 2 depth banks of 16 K lines (CC §10.1).
//   - Column slices: a RAM group is NCOL = 8 tsr_sram_1r1w instances of 64 bits (bytes 8k..8k+7
//     of the line in slice k). With URAM288 (4 K x 72) a slice is one 4-deep cascade, so a group
//     is 8 cascades = 32 URAMs, 128 in total at 4 MiB (unchanged by the interleave).
//   - Latency split (RD_LAT = IN_REG + RAM_LAT + OUT_REG, every request of both banks takes the
//     same path):
//       IN_REG  = 1 if RD_LAT >= 2: the arbitration result of each bank (enables, depth bank,
//                 offset, write data / byte enables) is registered before the RAMs, so the RAM
//                 address / data fan-out starts at a flop. Reads and writes of a bank share this
//                 stage, which keeps the write -> read order above. Every (bank, depth bank, slice)
//                 RAM has its own kept copy of the stage's enable, offset and byte-enable registers
//                 (docs/rtl-mvm.md §14.3 / §14.8 / §14.9: one shared offset register driving every
//                 cascade head was the worst path of the core), the enables decoded from the depth
//                 bank bits before the register; only the write data register is shared per bank.
//       OUT_REG = 1 if RD_LAT >= 4 and a bank has more than one depth bank: a register after the
//                 depth-bank read mux of each bank.
//       RAM_LAT = the rest (>= 1): tsr_sram_1r1w RD_LAT (RAM read register + output registers).
//     RD_LAT 2 (default): flop -> RAM read register -> depth-bank mux (2:1 at 4 MiB) ->
//     rd_rsp_data[b]. The U200 bitstreams use RD_LAT 4: IN_REG + RAM 2 (URAM OREG) + OUT_REG.
//   - Arbitration result per bank: one-hot AND-OR mux of the clients' requests, selected by the
//     grant flops (no priority logic between the grant flop and the RAM input stage).
//   - rd_rsp_valid comes from an RD_LAT-deep shift register of one-hot client bits per bank (reset).

module tsr_spm
  import tsr_pkg::*;
#(
  parameter  int unsigned SPM_BYTES  = TSR_DEF_SPM_BYTES,
  parameter  int unsigned NCL        = TSR_SPM_NCL,
  parameter  int unsigned RD_LAT     = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned BANK_LINES = 16384,   // lines per depth bank of one bank (power of two, >= 2)
  localparam int unsigned LINES      = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW         = (LINES > 1) ? $clog2(LINES) : 1
) (
  input  logic                                        clk,
  input  logic                                        rst_n,
  // read port per client and bank [c][b]
  input  logic [NCL-1:0][1:0]                         rd_req_valid,
  output logic [NCL-1:0][1:0]                         rd_req_ready,
  input  logic [NCL-1:0][1:0][AW-1:0]                 rd_req_line,
  output logic [NCL-1:0][1:0]                         rd_rsp_valid,
  output logic [1:0][TSR_LINE_BITS-1:0]               rd_rsp_data,
  // write port per client and bank [c][b]
  input  logic [NCL-1:0][1:0]                         wr_valid,
  output logic [NCL-1:0][1:0]                         wr_ready,
  input  logic [NCL-1:0][1:0][AW-1:0]                 wr_line,
  input  logic [NCL-1:0][1:0][TSR_LINE_BITS-1:0]      wr_data,
  input  logic [NCL-1:0][1:0][TSR_SPM_BE_W-1:0]       wr_be
);

  // -------------------------------------------------------------------------------------------
  // Configuration
  // -------------------------------------------------------------------------------------------
  localparam int unsigned LINES_E = (LINES + 1) / 2;                 // bank E (even lines)
  localparam int unsigned LINES_O = LINES / 2;                       // bank O (odd lines)
  localparam int unsigned NDB_E   = (LINES_E + BANK_LINES - 1) / BANK_LINES;
  localparam int unsigned NDB_O   = (LINES_O + BANK_LINES - 1) / BANK_LINES;
  localparam int unsigned NGRP    = NDB_E + NDB_O;                   // RAM groups g_bank[]
  localparam int unsigned LAW     = (AW > 1) ? AW - 1 : 1;           // bank-local line bits
  localparam int unsigned BL_LOG  = $clog2(BANK_LINES);
  localparam int unsigned DSEL_W  = (NDB_E > 1) ? $clog2(NDB_E) : 1; // depth-bank select bits
  localparam int unsigned IN_REG  = (RD_LAT >= 2) ? 1 : 0;
  localparam int unsigned OUT_REG = (RD_LAT >= 4 && NDB_E > 1) ? 1 : 0;
  localparam int unsigned RAM_LAT = RD_LAT - IN_REG - OUT_REG;
  localparam int unsigned NCOL    = 8;                               // RAM column slices per group
  localparam int unsigned COL_W   = TSR_LINE_BITS / NCOL;            // bits per slice (64)
  localparam int unsigned COL_BE  = COL_W / 8;                       // byte enables per slice

  if (SPM_BYTES < TSR_LINE_BYTES || (SPM_BYTES % TSR_LINE_BYTES) != 0) begin : g_chk_size
    $fatal(1, "tsr_spm: SPM_BYTES must be a positive multiple of 64");
  end
  if (NCL < 1 || RD_LAT < 1) begin : g_chk_ports
    $fatal(1, "tsr_spm: NCL >= 1 and RD_LAT >= 1 required");
  end
  if (BANK_LINES < 2 || (BANK_LINES & (BANK_LINES - 1)) != 0) begin : g_chk_bank
    $fatal(1, "tsr_spm: BANK_LINES must be a power of two >= 2");
  end

  // Bank-local line of a line address (line >> 1) and its depth bank.
  function automatic logic [LAW-1:0] loc_of(logic [AW-1:0] line);
    return (AW > 1) ? LAW'(line >> 1) : '0;
  endfunction

  function automatic logic [DSEL_W-1:0] dsel_of(logic [LAW-1:0] loc);
    return (NDB_E > 1) ? DSEL_W'(loc >> BL_LOG) : '0;
  endfunction

  function automatic logic [NCL-1:0] lowest_set(logic [NCL-1:0] v);
    return v & (~v + NCL'(1));
  endfunction

  // Round robin with the owner last: the lowest requester above the owner, else the lowest
  // requester (which may be the owner itself). gnt is one-hot or zero (no owner yet).
  function automatic logic [NCL-1:0] rr_next(logic [NCL-1:0] req, logic [NCL-1:0] gnt);
    logic [NCL-1:0] above;
    above = req & ~(gnt | (gnt - NCL'(1)));     // requesters with an index above the owner's
    return (above != '0) ? lowest_set(above) : lowest_set(req);
  endfunction

  // -------------------------------------------------------------------------------------------
  // Arbitration per bank: registered round-robin grants (see header)
  // -------------------------------------------------------------------------------------------
  logic [1:0][NCL-1:0] rd_req_b, wr_req_b;      // requests regrouped [bank][client]
  logic [1:0][NCL-1:0] rd_gnt_q, wr_gnt_q;

  always_comb begin
    for (int unsigned b = 0; b < 2; b++) begin
      for (int unsigned c = 0; c < NCL; c++) begin
        rd_req_b[b][c] = rd_req_valid[c][b];
        wr_req_b[b][c] = wr_valid[c][b];
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_gnt_q <= '0;
      wr_gnt_q <= '0;
    end else begin
      for (int unsigned b = 0; b < 2; b++) begin
        if (|rd_req_b[b]) rd_gnt_q[b] <= rr_next(rd_req_b[b], rd_gnt_q[b]);
        if (|wr_req_b[b]) wr_gnt_q[b] <= rr_next(wr_req_b[b], wr_gnt_q[b]);
      end
    end
  end

  always_comb begin
    for (int unsigned c = 0; c < NCL; c++) begin
      for (int unsigned b = 0; b < 2; b++) begin
        rd_req_ready[c][b] = rd_gnt_q[b][c];
        wr_ready[c][b]     = wr_gnt_q[b][c];
      end
    end
  end

  logic [1:0][NCL-1:0] rd_take;
  logic [1:0]          rd_fire, wr_fire;
  // Request of the granted client per bank (one-hot AND-OR mux, selected by the grant flops).
  logic [1:0][AW-1:0]             rd_line_sel, wr_line_sel;
  logic [1:0][TSR_LINE_BITS-1:0]  wr_data_sel;
  logic [1:0][TSR_SPM_BE_W-1:0]   wr_be_sel;

  always_comb begin
    for (int unsigned b = 0; b < 2; b++) begin
      rd_take[b]     = rd_req_b[b] & rd_gnt_q[b];
      rd_fire[b]     = |rd_take[b];
      wr_fire[b]     = |(wr_req_b[b] & wr_gnt_q[b]);
      rd_line_sel[b] = '0;
      wr_line_sel[b] = '0;
      wr_data_sel[b] = '0;
      wr_be_sel[b]   = '0;
      for (int unsigned c = 0; c < NCL; c++) begin
        if (rd_gnt_q[b][c]) rd_line_sel[b] = rd_line_sel[b] | rd_req_line[c][b];
        if (wr_gnt_q[b][c]) begin
          wr_line_sel[b] = wr_line_sel[b] | wr_line[c][b];
          wr_data_sel[b] = wr_data_sel[b] | wr_data[c][b];
          wr_be_sel[b]   = wr_be_sel[b]   | wr_be[c][b];
        end
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // RAM input stage (IN_REG), per bank
  // -------------------------------------------------------------------------------------------
  logic [1:0][LAW-1:0]    rd_loc, wr_loc;       // bank-local lines of the granted requests
  logic [1:0][DSEL_W-1:0] rd_dsel, wr_dsel;     // their depth banks
  always_comb begin
    for (int unsigned b = 0; b < 2; b++) begin
      rd_loc[b]  = loc_of(rd_line_sel[b]);
      wr_loc[b]  = loc_of(wr_line_sel[b]);
      rd_dsel[b] = dsel_of(rd_loc[b]);
      wr_dsel[b] = dsel_of(wr_loc[b]);
    end
  end

  // Shared part of the stage: write data (one load per bank) and the read depth bank (data mux
  // select).
  logic [1:0][TSR_LINE_BITS-1:0] wr_data_s;
  logic [1:0][DSEL_W-1:0]        rd_dsel_s;

  if (IN_REG != 0) begin : g_in_reg
    // Datapath registers load every cycle (no clock enable: the RAM enables gate their use, and a
    // CE would be a ~600-flop net driven from the clients' valid flops).
    always_ff @(posedge clk) begin
      rd_dsel_s <= rd_dsel;
      wr_data_s <= wr_data_sel;
    end
  end else begin : g_no_in_reg
    assign rd_dsel_s = rd_dsel;
    assign wr_data_s = wr_data_sel;
  end

  // -------------------------------------------------------------------------------------------
  // RAM groups: g_bank[i] = depth bank i / 2 of bank i % 2, NCOL column slices each, with the
  // slice's copy of the input stage
  // -------------------------------------------------------------------------------------------
  logic [NGRP-1:0][TSR_LINE_BITS-1:0] grp_rdata;

  for (genvar i = 0; i < NGRP; i++) begin : g_bank
    localparam int unsigned B       = i % 2;
    localparam int unsigned D       = i / 2;
    localparam int unsigned LINES_B = (B == 0) ? LINES_E : LINES_O;
    localparam int unsigned NDB_B   = (B == 0) ? NDB_E : NDB_O;
    localparam int unsigned DEPTH_G = (D == NDB_B - 1) ? LINES_B - D * BANK_LINES : BANK_LINES;
    localparam int unsigned AW_G    = (DEPTH_G > 1) ? $clog2(DEPTH_G) : 1;

    wire rd_hit = rd_fire[B] && (rd_dsel[B] == DSEL_W'(D));
    wire wr_hit = wr_fire[B] && (wr_dsel[B] == DSEL_W'(D));

    for (genvar k = 0; k < NCOL; k++) begin : g_col
      logic              re_k, we_k;
      logic [AW_G-1:0]   ra_k, wa_k;
      logic [COL_BE-1:0] be_k;

      if (IN_REG != 0) begin : g_in
        // keep: synthesis must not merge the copies (identical inputs) back into one register.
        (* keep = "true" *) logic              re_q, we_q;
        (* keep = "true" *) logic [AW_G-1:0]   ra_q, wa_q;
        (* keep = "true" *) logic [COL_BE-1:0] be_q;

        always_ff @(posedge clk) begin
          if (!rst_n) begin
            re_q <= 1'b0;
            we_q <= 1'b0;
          end else begin
            re_q <= rd_hit;
            we_q <= wr_hit;
          end
        end
        always_ff @(posedge clk) begin
          ra_q <= rd_loc[B][AW_G-1:0];
          wa_q <= wr_loc[B][AW_G-1:0];
          be_q <= wr_be_sel[B][k*COL_BE +: COL_BE];
        end
        assign re_k = re_q;
        assign we_k = we_q;
        assign ra_k = ra_q;
        assign wa_k = wa_q;
        assign be_k = be_q;
      end else begin : g_comb
        assign re_k = rd_hit;
        assign we_k = wr_hit;
        assign ra_k = rd_loc[B][AW_G-1:0];
        assign wa_k = wr_loc[B][AW_G-1:0];
        assign be_k = wr_be_sel[B][k*COL_BE +: COL_BE];
      end

      tsr_sram_1r1w #(
        .DEPTH(DEPTH_G), .WIDTH(COL_W), .RD_LAT(RAM_LAT)
      ) u_ram (
        .clk   (clk),
        .we    (we_k),
        .waddr (wa_k),
        .wdata (wr_data_s[B][k*COL_W +: COL_W]),
        .wbe   (be_k),
        .re    (re_k),
        .raddr (ra_k),
        .rdata (grp_rdata[i][k*COL_W +: COL_W])
      );
    end
  end

  // -------------------------------------------------------------------------------------------
  // Read data per bank: depth-bank mux (select delayed by RAM_LAT), optional output register
  // -------------------------------------------------------------------------------------------
  for (genvar b = 0; b < 2; b++) begin : g_rd
    localparam int unsigned NDB_B = (b == 0) ? NDB_E : NDB_O;

    if (NDB_B == 0) begin : g_absent
      // SPM of one line: bank O holds no line (no request can address it).
      assign rd_rsp_data[b] = '0;
      wire unused_absent = &{1'b0, rd_dsel_s[b]};
    end else if (NDB_B == 1) begin : g_one
      if (OUT_REG != 0) begin : g_out_reg
        logic [TSR_LINE_BITS-1:0] rd_data_q;
        always_ff @(posedge clk) rd_data_q <= grp_rdata[b];
        assign rd_rsp_data[b] = rd_data_q;
      end else begin : g_no_out_reg
        assign rd_rsp_data[b] = grp_rdata[b];
      end
      wire unused_one = &{1'b0, rd_dsel_s[b]};
    end else begin : g_mux
      // The select's last stage has one kept copy per column slice (fan-out 64 mux LUTs per bit
      // instead of 512); the stages before it are shared.
      logic [DSEL_W-1:0]        bsel_d;      // rd_dsel_s delayed by RAM_LAT - 1
      logic [TSR_LINE_BITS-1:0] rd_mux;

      if (RAM_LAT == 1) begin : g_bsel1
        assign bsel_d = rd_dsel_s[b];
      end else begin : g_bseln
        logic [DSEL_W-1:0] bsel_pipe [RAM_LAT - 1];
        always_ff @(posedge clk) bsel_pipe[0] <= rd_dsel_s[b];
        for (genvar j = 1; j < RAM_LAT - 1; j++) begin : g_bsel
          always_ff @(posedge clk) bsel_pipe[j] <= bsel_pipe[j-1];
        end
        assign bsel_d = bsel_pipe[RAM_LAT-2];
      end

      (* keep = "true" *) logic [NCOL-1:0][DSEL_W-1:0] bsel_q;
      always_ff @(posedge clk) bsel_q <= {NCOL{bsel_d}};

      always_comb begin
        rd_mux = '0;
        for (int unsigned k = 0; k < NCOL; k++) begin
          for (int unsigned d = 0; d < NDB_B; d++) begin
            if (bsel_q[k] == DSEL_W'(d)) begin
              rd_mux[k*COL_W +: COL_W] = rd_mux[k*COL_W +: COL_W] | grp_rdata[2*d + b][k*COL_W +: COL_W];
            end
          end
        end
      end

      if (OUT_REG != 0) begin : g_out_reg
        logic [TSR_LINE_BITS-1:0] rd_data_q;
        always_ff @(posedge clk) rd_data_q <= rd_mux;
        assign rd_rsp_data[b] = rd_data_q;
      end else begin : g_no_out_reg
        assign rd_rsp_data[b] = rd_mux;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Read response valid: RD_LAT-deep pipeline of the one-hot accepted client, per bank
  // -------------------------------------------------------------------------------------------
  logic [1:0][NCL-1:0] rsp_pipe [RD_LAT];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int unsigned i = 0; i < RD_LAT; i++) rsp_pipe[i] <= '0;
    end else begin
      rsp_pipe[0] <= rd_take;
      for (int unsigned i = 1; i < RD_LAT; i++) rsp_pipe[i] <= rsp_pipe[i-1];
    end
  end

  always_comb begin
    for (int unsigned c = 0; c < NCL; c++) begin
      for (int unsigned b = 0; b < 2; b++) rd_rsp_valid[c][b] = rsp_pipe[RD_LAT-1][b][c];
    end
  end

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks of the client contract
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  localparam int unsigned WCW = $clog2(NCL + 2);                     // wait counter bits
  logic [NCL-1:0][1:0]                    chk_rd_hold, chk_wr_hold;
  logic [NCL-1:0][1:0][AW-1:0]            chk_rd_line, chk_wr_line;
  logic [NCL-1:0][1:0][TSR_LINE_BITS-1:0] chk_wr_data;
  logic [NCL-1:0][1:0][TSR_SPM_BE_W-1:0]  chk_wr_be;
  logic [NCL-1:0][1:0][WCW-1:0]           chk_rd_wait, chk_wr_wait;   // cycles waited with an owner

  // Armed once a reset has been seen (same scheme as tsr_mem_arb / tsr_dma): before the first
  // reset edge, the check state and the grant flops hold power-up garbage and the inputs are not
  // driven yet. The initial block runs before Verilator's first evaluation (which may already see
  // a rising clock edge, with X-randomised rst_n and inputs).
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      chk_rd_hold <= '0;
      chk_wr_hold <= '0;
      chk_rd_wait <= '0;
      chk_wr_wait <= '0;
    end else begin
      if (chk_on_q) begin
        for (int unsigned c = 0; c < NCL; c++) begin
          for (int unsigned b = 0; b < 2; b++) begin
            if (rd_req_valid[c][b]) begin
              assert (32'(rd_req_line[c][b]) < LINES)
                else $error("%m: client %0d bank %0d read line %0d >= %0d SPM lines", c, b,
                            rd_req_line[c][b], LINES);
              assert (rd_req_line[c][b][0] == 1'(b))
                else $error("%m: client %0d read line %0d on bank %0d (line[0] must be the bank)", c,
                            rd_req_line[c][b], b);
            end
            if (wr_valid[c][b]) begin
              assert (32'(wr_line[c][b]) < LINES)
                else $error("%m: client %0d bank %0d write line %0d >= %0d SPM lines", c, b,
                            wr_line[c][b], LINES);
              assert (wr_line[c][b][0] == 1'(b))
                else $error("%m: client %0d write line %0d on bank %0d (line[0] must be the bank)", c,
                            wr_line[c][b], b);
            end
            if (chk_rd_hold[c][b]) begin
              assert (rd_req_valid[c][b] && rd_req_line[c][b] == chk_rd_line[c][b])
                else $error("%m: client %0d bank %0d dropped or changed a read request before it was accepted (R2)",
                            c, b);
            end
            if (chk_wr_hold[c][b]) begin
              assert (wr_valid[c][b] && wr_line[c][b] == chk_wr_line[c][b]
                      && wr_data[c][b] == chk_wr_data[c][b] && wr_be[c][b] == chk_wr_be[c][b])
                else $error("%m: client %0d bank %0d dropped or changed a write request before it was accepted (R2)",
                            c, b);
            end
            // CC §10.2 / §10.3 (iii): a waiting request is accepted within NCL - 1 cycles once
            // the port has an owner.
            assert (32'(chk_rd_wait[c][b]) < NCL && 32'(chk_wr_wait[c][b]) < NCL)
              else $error("%m: client %0d bank %0d waited %0d (read) / %0d (write) cycles: round-robin bound NCL - 1 = %0d exceeded",
                          c, b, chk_rd_wait[c][b], chk_wr_wait[c][b], NCL - 1);
          end
        end
        for (int unsigned b = 0; b < 2; b++) begin
          assert ($onehot0(rd_gnt_q[b]) && $onehot0(wr_gnt_q[b]))
            else $error("%m: bank %0d grant not one-hot", b);
        end
      end
      for (int unsigned c = 0; c < NCL; c++) begin
        for (int unsigned b = 0; b < 2; b++) begin
          chk_rd_hold[c][b] <= rd_req_valid[c][b] & ~rd_req_ready[c][b];
          chk_wr_hold[c][b] <= wr_valid[c][b] & ~wr_ready[c][b];
          if (!rd_req_valid[c][b] || rd_req_ready[c][b]) chk_rd_wait[c][b] <= '0;
          else if (rd_gnt_q[b] != '0 && chk_rd_wait[c][b] != '1) chk_rd_wait[c][b] <= chk_rd_wait[c][b] + 1'b1;
          if (!wr_valid[c][b] || wr_ready[c][b]) chk_wr_wait[c][b] <= '0;
          else if (wr_gnt_q[b] != '0 && chk_wr_wait[c][b] != '1) chk_wr_wait[c][b] <= chk_wr_wait[c][b] + 1'b1;
        end
      end
    end
    chk_rd_line <= rd_req_line;
    chk_wr_line <= wr_line;
    chk_wr_data <= wr_data;
    chk_wr_be   <= wr_be;
  end
`endif

endmodule
