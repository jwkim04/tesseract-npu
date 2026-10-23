// tsr_tb_qact.sv — test wrapper (tb/qact/test_qact.py test_qact_rtl_spm): tsr_qact on client
// TSR_SPM_CL_QACT of the real tsr_spm, the testbench on client TSR_SPM_CL_DMA (preload, contention
// traffic, readback), client TSR_SPM_CL_MVM tied off — the same SPM port plan as tsr_core.
// Client 0 has priority in tsr_spm, so testbench traffic takes the read / write grants away from
// QACT in the middle of an instruction (ready low, grant handovers).

module tsr_tb_qact
  import tsr_pkg::*;
#(
  parameter  int unsigned K_MAX      = 2048,
  parameter  int unsigned T_MAX      = 4,
  parameter  int unsigned SPM_BYTES  = 65536,
  parameter  int unsigned SPM_RD_LAT = 2,
  parameter  int unsigned BANK_LINES = 256,
  localparam int unsigned SPM_LINES  = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned SPM_AW     = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1
) (
  input  logic                     clk,
  input  logic                     rst_n,
  // QACT dispatch
  input  logic                     cmd_valid,
  output logic                     cmd_ready,
  input  tsr_qact_cmd_t            cmd,
  output logic                     done,
  // testbench SPM client (client 0)
  input  logic                     x_rd_req_valid,
  output logic                     x_rd_req_ready,
  input  logic [SPM_AW-1:0]        x_rd_req_line,
  output logic                     x_rd_rsp_valid,
  output logic [TSR_LINE_BITS-1:0] x_rd_rsp_data,
  input  logic                     x_wr_valid,
  output logic                     x_wr_ready,
  input  logic [SPM_AW-1:0]        x_wr_line,
  input  logic [TSR_LINE_BITS-1:0] x_wr_data
);

  logic [TSR_SPM_NCL-1:0]                    rd_req_valid, rd_req_ready, rd_rsp_valid;
  logic [TSR_SPM_NCL-1:0][SPM_AW-1:0]        rd_req_line;
  logic [TSR_LINE_BITS-1:0]                  rd_rsp_data;
  logic [TSR_SPM_NCL-1:0]                    wr_valid, wr_ready;
  logic [TSR_SPM_NCL-1:0][SPM_AW-1:0]        wr_line;
  logic [TSR_SPM_NCL-1:0][TSR_LINE_BITS-1:0] wr_data;
  logic [TSR_SPM_NCL-1:0][TSR_SPM_BE_W-1:0]  wr_be;

  tsr_spm #(
    .SPM_BYTES(SPM_BYTES), .NCL(TSR_SPM_NCL), .RD_LAT(SPM_RD_LAT), .BANK_LINES(BANK_LINES)
  ) u_spm (
    .clk, .rst_n,
    .rd_req_valid, .rd_req_ready, .rd_req_line, .rd_rsp_valid, .rd_rsp_data,
    .wr_valid, .wr_ready, .wr_line, .wr_data, .wr_be
  );

  // Client 0: testbench.
  assign rd_req_valid[TSR_SPM_CL_DMA] = x_rd_req_valid;
  assign rd_req_line[TSR_SPM_CL_DMA]  = x_rd_req_line;
  assign x_rd_req_ready               = rd_req_ready[TSR_SPM_CL_DMA];
  assign x_rd_rsp_valid               = rd_rsp_valid[TSR_SPM_CL_DMA];
  assign x_rd_rsp_data                = rd_rsp_data;
  assign wr_valid[TSR_SPM_CL_DMA]     = x_wr_valid;
  assign wr_line[TSR_SPM_CL_DMA]      = x_wr_line;
  assign wr_data[TSR_SPM_CL_DMA]      = x_wr_data;
  assign wr_be[TSR_SPM_CL_DMA]        = {TSR_SPM_BE_W{1'b1}};
  assign x_wr_ready                   = wr_ready[TSR_SPM_CL_DMA];

  // Client 1: QACT.
  tsr_qact #(
    .K_MAX(K_MAX), .T_MAX(T_MAX), .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT)
  ) u_qact (
    .clk, .rst_n,
    .cmd_valid, .cmd_ready, .cmd, .done,
    .spm_rd_req_valid(rd_req_valid[TSR_SPM_CL_QACT]), .spm_rd_req_ready(rd_req_ready[TSR_SPM_CL_QACT]),
    .spm_rd_req_line(rd_req_line[TSR_SPM_CL_QACT]), .spm_rd_rsp_valid(rd_rsp_valid[TSR_SPM_CL_QACT]),
    .spm_rd_rsp_data(rd_rsp_data),
    .spm_wr_valid(wr_valid[TSR_SPM_CL_QACT]), .spm_wr_ready(wr_ready[TSR_SPM_CL_QACT]),
    .spm_wr_line(wr_line[TSR_SPM_CL_QACT]), .spm_wr_data(wr_data[TSR_SPM_CL_QACT]),
    .spm_wr_be(wr_be[TSR_SPM_CL_QACT])
  );

  // Client 2 (M3 MVM): tied off as in tsr_core.
  assign rd_req_valid[TSR_SPM_CL_MVM] = 1'b0;
  assign rd_req_line[TSR_SPM_CL_MVM]  = '0;
  assign wr_valid[TSR_SPM_CL_MVM]     = 1'b0;
  assign wr_line[TSR_SPM_CL_MVM]      = '0;
  assign wr_data[TSR_SPM_CL_MVM]      = '0;
  assign wr_be[TSR_SPM_CL_MVM]        = '0;

  wire unused_ok = &{1'b0, rd_req_ready[TSR_SPM_CL_MVM], rd_rsp_valid[TSR_SPM_CL_MVM],
                     wr_ready[TSR_SPM_CL_MVM]};

endmodule
