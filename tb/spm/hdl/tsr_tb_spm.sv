// tsr_tb_spm.sv — testbench-only toplevel for tb/spm (not part of the core): the even / odd
// tsr_spm with all six clients of CC §10.2, as tsr_core will wire them:
//   clients 0..2 (DMA, QACT, MVM slots): single-port clients through tsr_spm_cport, driven by
//            Python on the p_* ports (cport client side);
//   client 3: tsr_spm_bridge through tsr_spm_cport, its request / response interface on br_*;
//   clients 4, 5 (VU, AT slots): dual-bank clients on the v_* ports ([client - 4][bank]),
//            straight to tsr_spm's bank ports.
// Python observes tsr_spm's bank ports through the hierarchy (u_spm) for its scoreboard.

module tsr_tb_spm
  import tsr_pkg::*;
#(
  parameter  int unsigned SPM_BYTES  = 65536,
  parameter  int unsigned RD_LAT     = 2,
  parameter  int unsigned BANK_LINES = 256,
  localparam int unsigned NCL        = TSR_SPM_NCL,
  localparam int unsigned NP         = 3,
  localparam int unsigned LINES      = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW         = (LINES > 1) ? $clog2(LINES) : 1,
  localparam int unsigned WAW        = AW + 4
) (
  input  logic                                clk,
  input  logic                                rst_n,

  // ---- single-port clients 0..2 ----
  input  logic [NP-1:0]                       p_rd_req_valid,
  output logic [NP-1:0]                       p_rd_req_ready,
  input  logic [NP-1:0][AW-1:0]               p_rd_req_line,
  output logic [NP-1:0]                       p_rd_rsp_valid,
  output logic [NP-1:0][TSR_LINE_BITS-1:0]    p_rd_rsp_data,
  input  logic [NP-1:0]                       p_wr_valid,
  output logic [NP-1:0]                       p_wr_ready,
  input  logic [NP-1:0][AW-1:0]               p_wr_line,
  input  logic [NP-1:0][TSR_LINE_BITS-1:0]    p_wr_data,
  input  logic [NP-1:0][TSR_SPM_BE_W-1:0]     p_wr_be,

  // ---- client 3: RISC-V bridge ----
  input  logic                                br_req_valid,
  output logic                                br_req_ready,
  input  logic                                br_req_we,
  input  logic [WAW-1:0]                      br_req_waddr,
  input  logic [3:0]                          br_req_be,
  input  logic [31:0]                         br_req_wdata,
  output logic                                br_rsp_valid,
  input  logic                                br_rsp_ready,
  output logic                                br_rsp_err,
  output logic [TSR_LINE_BITS-1:0]            br_rsp_line,

  // ---- dual-bank clients 4, 5: [client - 4][bank] ----
  input  logic [1:0][1:0]                     v_rd_req_valid,
  output logic [1:0][1:0]                     v_rd_req_ready,
  input  logic [1:0][1:0][AW-1:0]             v_rd_req_line,
  output logic [1:0][1:0]                     v_rd_rsp_valid,
  output logic [1:0][TSR_LINE_BITS-1:0]       v_rd_rsp_data,
  input  logic [1:0][1:0]                     v_wr_valid,
  output logic [1:0][1:0]                     v_wr_ready,
  input  logic [1:0][1:0][AW-1:0]             v_wr_line,
  input  logic [1:0][1:0][TSR_LINE_BITS-1:0]  v_wr_data,
  input  logic [1:0][1:0][TSR_SPM_BE_W-1:0]   v_wr_be
);

  logic [NCL-1:0][1:0]                    b_rd_req_valid, b_rd_req_ready, b_rd_rsp_valid;
  logic [NCL-1:0][1:0][AW-1:0]            b_rd_req_line;
  logic [1:0][TSR_LINE_BITS-1:0]          b_rd_rsp_data;
  logic [NCL-1:0][1:0]                    b_wr_valid, b_wr_ready;
  logic [NCL-1:0][1:0][AW-1:0]            b_wr_line;
  logic [NCL-1:0][1:0][TSR_LINE_BITS-1:0] b_wr_data;
  logic [NCL-1:0][1:0][TSR_SPM_BE_W-1:0]  b_wr_be;

  tsr_spm #(
    .SPM_BYTES(SPM_BYTES), .NCL(NCL), .RD_LAT(RD_LAT), .BANK_LINES(BANK_LINES)
  ) u_spm (
    .clk, .rst_n,
    .rd_req_valid(b_rd_req_valid), .rd_req_ready(b_rd_req_ready), .rd_req_line(b_rd_req_line),
    .rd_rsp_valid(b_rd_rsp_valid), .rd_rsp_data(b_rd_rsp_data),
    .wr_valid(b_wr_valid), .wr_ready(b_wr_ready), .wr_line(b_wr_line), .wr_data(b_wr_data),
    .wr_be(b_wr_be)
  );

  // ---- clients 0..2: Python through tsr_spm_cport ----
  for (genvar c = 0; c < NP; c++) begin : g_p
    tsr_spm_cport #(.SPM_BYTES(SPM_BYTES)) u_cport (
      .rd_req_valid(p_rd_req_valid[c]), .rd_req_ready(p_rd_req_ready[c]), .rd_req_line(p_rd_req_line[c]),
      .rd_rsp_valid(p_rd_rsp_valid[c]), .rd_rsp_data(p_rd_rsp_data[c]),
      .wr_valid(p_wr_valid[c]), .wr_ready(p_wr_ready[c]), .wr_line(p_wr_line[c]), .wr_data(p_wr_data[c]),
      .wr_be(p_wr_be[c]),
      .s_rd_req_valid(b_rd_req_valid[c]), .s_rd_req_ready(b_rd_req_ready[c]),
      .s_rd_req_line(b_rd_req_line[c]), .s_rd_rsp_valid(b_rd_rsp_valid[c]), .s_rd_rsp_data(b_rd_rsp_data),
      .s_wr_valid(b_wr_valid[c]), .s_wr_ready(b_wr_ready[c]), .s_wr_line(b_wr_line[c]),
      .s_wr_data(b_wr_data[c]), .s_wr_be(b_wr_be[c])
    );
  end

  // ---- client 3: the bridge through tsr_spm_cport ----
  logic                     br_rd_req_valid, br_rd_req_ready, br_rd_rsp_valid, br_wr_valid, br_wr_ready;
  logic [AW-1:0]            br_rd_req_line, br_wr_line;
  logic [TSR_LINE_BITS-1:0] br_rd_rsp_data, br_wr_data;
  logic [TSR_SPM_BE_W-1:0]  br_wr_be;

  tsr_spm_bridge #(.SPM_BYTES(SPM_BYTES)) u_br (
    .clk, .rst_n,
    .req_valid(br_req_valid), .req_ready(br_req_ready), .req_we(br_req_we), .req_waddr(br_req_waddr),
    .req_be(br_req_be), .req_wdata(br_req_wdata),
    .rsp_valid(br_rsp_valid), .rsp_ready(br_rsp_ready), .rsp_err(br_rsp_err), .rsp_line(br_rsp_line),
    .spm_rd_req_valid(br_rd_req_valid), .spm_rd_req_ready(br_rd_req_ready), .spm_rd_req_line(br_rd_req_line),
    .spm_rd_rsp_valid(br_rd_rsp_valid), .spm_rd_rsp_data(br_rd_rsp_data),
    .spm_wr_valid(br_wr_valid), .spm_wr_ready(br_wr_ready), .spm_wr_line(br_wr_line),
    .spm_wr_data(br_wr_data), .spm_wr_be(br_wr_be)
  );

  tsr_spm_cport #(.SPM_BYTES(SPM_BYTES)) u_br_cport (
    .rd_req_valid(br_rd_req_valid), .rd_req_ready(br_rd_req_ready), .rd_req_line(br_rd_req_line),
    .rd_rsp_valid(br_rd_rsp_valid), .rd_rsp_data(br_rd_rsp_data),
    .wr_valid(br_wr_valid), .wr_ready(br_wr_ready), .wr_line(br_wr_line), .wr_data(br_wr_data),
    .wr_be(br_wr_be),
    .s_rd_req_valid(b_rd_req_valid[TSR_SPM_CL_BRIDGE]), .s_rd_req_ready(b_rd_req_ready[TSR_SPM_CL_BRIDGE]),
    .s_rd_req_line(b_rd_req_line[TSR_SPM_CL_BRIDGE]), .s_rd_rsp_valid(b_rd_rsp_valid[TSR_SPM_CL_BRIDGE]),
    .s_rd_rsp_data(b_rd_rsp_data),
    .s_wr_valid(b_wr_valid[TSR_SPM_CL_BRIDGE]), .s_wr_ready(b_wr_ready[TSR_SPM_CL_BRIDGE]),
    .s_wr_line(b_wr_line[TSR_SPM_CL_BRIDGE]), .s_wr_data(b_wr_data[TSR_SPM_CL_BRIDGE]),
    .s_wr_be(b_wr_be[TSR_SPM_CL_BRIDGE])
  );

  // ---- clients 4, 5: dual-bank, straight to the bank ports ----
  for (genvar v = 0; v < 2; v++) begin : g_v
    assign b_rd_req_valid[TSR_SPM_CL_VU + v] = v_rd_req_valid[v];
    assign b_rd_req_line[TSR_SPM_CL_VU + v]  = v_rd_req_line[v];
    assign b_wr_valid[TSR_SPM_CL_VU + v]     = v_wr_valid[v];
    assign b_wr_line[TSR_SPM_CL_VU + v]      = v_wr_line[v];
    assign b_wr_data[TSR_SPM_CL_VU + v]      = v_wr_data[v];
    assign b_wr_be[TSR_SPM_CL_VU + v]        = v_wr_be[v];
    assign v_rd_req_ready[v]                 = b_rd_req_ready[TSR_SPM_CL_VU + v];
    assign v_rd_rsp_valid[v]                 = b_rd_rsp_valid[TSR_SPM_CL_VU + v];
    assign v_wr_ready[v]                     = b_wr_ready[TSR_SPM_CL_VU + v];
  end
  assign v_rd_rsp_data = b_rd_rsp_data;

endmodule
