// tsr_spm_cport.sv — single-port SPM client adapter: one client with one read and one write
// request per cycle (the client interface of docs/rtl-core.md §5) onto its two bank ports of the
// even / odd tsr_spm (docs/design/04-control-core.md "CC" §10.2, CC-12).
//
// Used by the DMA, QACT, MVM and the RISC-V bridge (clients 0-3); the VU drives
// both bank ports itself.
//
// Steering: a request goes to bank b = line[0]; the other bank's valid is 0. Client side:
//   rd_req_ready = s_rd_req_ready[rd_req_line[0]], wr_ready = s_wr_ready[wr_line[0]] (the steered
//                  bank's grant flop);
//   rd_rsp_valid = s_rd_rsp_valid[0] | s_rd_rsp_valid[1];
//   rd_rsp_data  = the data of the bank that answered (2:1 select by s_rd_rsp_valid[1]).
// A single-port client has at most one read accepted per cycle and both banks answer exactly
// RD_LAT cycles after the accepting edge, so at most one bank answers in a cycle and the
// responses keep the client's request order (CC §10.2: in order per (c, b), and here per c).
// Write -> read order of one line holds because both go to the line's bank. A client streaming
// alone alternates banks and keeps both grants: one line per cycle.
// R2 per bank port follows from the client's R2 (the held line keeps its bank). Write data and
// byte enables go to both bank ports unchanged (only the steered one is valid).
// Combinational only. ready depends on the client's own request line (one LUT inside the
// client's logic): accepted because rule R5 is about the tsr_spm module boundary (CC §10.2), whose
// ready outputs are flops; the client's line and valid are flops in every client.

module tsr_spm_cport
  import tsr_pkg::*;
#(
  parameter  int unsigned SPM_BYTES = TSR_DEF_SPM_BYTES,
  localparam int unsigned LINES     = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW        = (LINES > 1) ? $clog2(LINES) : 1
) (
  // ---- client side (single port) ----
  input  logic                           rd_req_valid,
  output logic                           rd_req_ready,
  input  logic [AW-1:0]                  rd_req_line,
  output logic                           rd_rsp_valid,
  output logic [TSR_LINE_BITS-1:0]       rd_rsp_data,
  input  logic                           wr_valid,
  output logic                           wr_ready,
  input  logic [AW-1:0]                  wr_line,
  input  logic [TSR_LINE_BITS-1:0]       wr_data,
  input  logic [TSR_SPM_BE_W-1:0]        wr_be,

  // ---- SPM side: this client's bank ports of tsr_spm, [b] ----
  output logic [1:0]                     s_rd_req_valid,
  input  logic [1:0]                     s_rd_req_ready,
  output logic [1:0][AW-1:0]             s_rd_req_line,
  input  logic [1:0]                     s_rd_rsp_valid,
  input  logic [1:0][TSR_LINE_BITS-1:0]  s_rd_rsp_data,
  output logic [1:0]                     s_wr_valid,
  input  logic [1:0]                     s_wr_ready,
  output logic [1:0][AW-1:0]             s_wr_line,
  output logic [1:0][TSR_LINE_BITS-1:0]  s_wr_data,
  output logic [1:0][TSR_SPM_BE_W-1:0]   s_wr_be
);

  wire rd_b = rd_req_line[0];
  wire wr_b = wr_line[0];

  assign s_rd_req_valid = {rd_req_valid & rd_b, rd_req_valid & ~rd_b};
  assign s_rd_req_line  = {rd_req_line, rd_req_line};
  assign rd_req_ready   = s_rd_req_ready[rd_b];
  assign rd_rsp_valid   = |s_rd_rsp_valid;
  assign rd_rsp_data    = s_rd_rsp_valid[1] ? s_rd_rsp_data[1] : s_rd_rsp_data[0];

  assign s_wr_valid     = {wr_valid & wr_b, wr_valid & ~wr_b};
  assign s_wr_line      = {wr_line, wr_line};
  assign s_wr_data      = {wr_data, wr_data};
  assign s_wr_be        = {wr_be, wr_be};
  assign wr_ready       = s_wr_ready[wr_b];

endmodule
