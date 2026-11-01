// tsr_vu_tb.sv — testbench-only toplevel of the VU unit bench (tb/vu/test_vu_unit.py): tsr_vu on a real
// tsr_spm (CC §10: 2R2W even / odd banks, round-robin per bank port) with two more clients:
//   client 0  the bench's port (single-port through tsr_spm_cport): loads and reads back SPM lines
//   client 1  a contender (single-port through tsr_spm_cport): random reads of random lines while
//             noise_en, so the VU's bank requests wait a random number of cycles (VU §10 item 4:
//             "random ready on both bank ports")
//   client 2  the VU, both bank ports

module tsr_vu_tb
  import tsr_pkg::*;
#(
  parameter  int unsigned SPM_BYTES  = 262144,
  parameter  int unsigned SPM_RD_LAT = 4,
  parameter  int unsigned NUM_CH     = 4,
  parameter  int unsigned EXP_LANES  = 16,
  parameter  int unsigned BACK_LANES = 8,
  parameter  int unsigned LANE_PIPE  = 0,
  parameter  bit          HAS_TOPK   = 1'b1,
  parameter  bit          HAS_DQROW  = 1'b1,
  localparam int unsigned LINES      = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW         = (LINES > 1) ? $clog2(LINES) : 1
) (
  input  logic                      clk,
  input  logic                      rst_n,
  // VU command
  input  logic                      cmd_valid,
  output logic                      cmd_ready,
  input  logic [255:0]              cmd,
  output logic                      done,
  output logic [15:0][31:0]         sreg,
  output logic                      perf_spm_wait,
  output logic                      perf_hazard,
  output logic                      perf_line,
  // bench port (client 0)
  input  logic                      t_rd_valid,
  output logic                      t_rd_ready,
  input  logic [AW-1:0]             t_rd_line,
  output logic                      t_rsp_valid,
  output logic [511:0]              t_rsp_data,
  input  logic                      t_wr_valid,
  output logic                      t_wr_ready,
  input  logic [AW-1:0]             t_wr_line,
  input  logic [511:0]              t_wr_data,
  // contender (client 1)
  input  logic                      noise_en,
  input  logic [31:0]               noise_seed
);
  localparam int unsigned NCL = 3;

  logic [NCL-1:0][1:0]               rd_req_valid, rd_req_ready, rd_rsp_valid;
  logic [NCL-1:0][1:0][AW-1:0]       rd_req_line;
  logic [1:0][TSR_LINE_BITS-1:0]     rd_rsp_data;
  logic [NCL-1:0][1:0]               wr_valid, wr_ready;
  logic [NCL-1:0][1:0][AW-1:0]       wr_line;
  logic [NCL-1:0][1:0][TSR_LINE_BITS-1:0] wr_data;
  logic [NCL-1:0][1:0][TSR_SPM_BE_W-1:0]  wr_be;

  tsr_spm #(.SPM_BYTES(SPM_BYTES), .NCL(NCL), .RD_LAT(SPM_RD_LAT), .BANK_LINES(1024)) u_spm (
    .clk, .rst_n, .rd_req_valid, .rd_req_ready, .rd_req_line, .rd_rsp_valid, .rd_rsp_data,
    .wr_valid, .wr_ready, .wr_line, .wr_data, .wr_be
  );

  // client 0: the bench
  tsr_spm_cport #(.SPM_BYTES(SPM_BYTES)) u_c0 (
    .rd_req_valid(t_rd_valid), .rd_req_ready(t_rd_ready), .rd_req_line(t_rd_line),
    .rd_rsp_valid(t_rsp_valid), .rd_rsp_data(t_rsp_data),
    .wr_valid(t_wr_valid), .wr_ready(t_wr_ready), .wr_line(t_wr_line), .wr_data(t_wr_data), .wr_be('1),
    .s_rd_req_valid(rd_req_valid[0]), .s_rd_req_ready(rd_req_ready[0]), .s_rd_req_line(rd_req_line[0]),
    .s_rd_rsp_valid(rd_rsp_valid[0]), .s_rd_rsp_data(rd_rsp_data),
    .s_wr_valid(wr_valid[0]), .s_wr_ready(wr_ready[0]), .s_wr_line(wr_line[0]), .s_wr_data(wr_data[0]),
    .s_wr_be(wr_be[0])
  );

  // client 1: random reads (an LFSR picks the line and whether to request)
  logic [31:0] lfsr_q;
  logic        n_v_q;
  logic [AW-1:0] n_line_q;
  logic        n_rdy, n_rsp_v, n_wrdy;
  logic [511:0] n_rsp_d;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      lfsr_q <= noise_seed | 32'd1;
      n_v_q  <= 1'b0;
    end else begin
      lfsr_q <= {lfsr_q[30:0], lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
      if (!n_v_q || n_rdy) begin
        n_v_q    <= noise_en && (lfsr_q[3:0] < 4'd7);
        n_line_q <= AW'(lfsr_q[31:8]);
      end
    end
  end
  tsr_spm_cport #(.SPM_BYTES(SPM_BYTES)) u_c1 (
    .rd_req_valid(n_v_q), .rd_req_ready(n_rdy), .rd_req_line(n_line_q), .rd_rsp_valid(n_rsp_v), .rd_rsp_data(n_rsp_d),
    .wr_valid(1'b0), .wr_ready(n_wrdy), .wr_line('0), .wr_data('0), .wr_be('0),
    .s_rd_req_valid(rd_req_valid[1]), .s_rd_req_ready(rd_req_ready[1]), .s_rd_req_line(rd_req_line[1]),
    .s_rd_rsp_valid(rd_rsp_valid[1]), .s_rd_rsp_data(rd_rsp_data),
    .s_wr_valid(wr_valid[1]), .s_wr_ready(wr_ready[1]), .s_wr_line(wr_line[1]), .s_wr_data(wr_data[1]),
    .s_wr_be(wr_be[1])
  );

  // client 2: the VU
  logic       dbg_x;
  logic [1:0] dbg_held;
  logic [31:0] dbg_w0;
  logic [2:0] dbg_flags;
  tsr_vu #(.SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT), .NUM_CH(NUM_CH), .EXP_LANES(EXP_LANES),
           .BACK_LANES(BACK_LANES), .LANE_PIPE(LANE_PIPE), .HAS_TOPK(HAS_TOPK), .HAS_DQROW(HAS_DQROW)) u_vu (
    .clk, .rst_n, .cmd_valid, .cmd_ready, .cmd, .done,
    .rd_req_valid(rd_req_valid[2]), .rd_req_ready(rd_req_ready[2]), .rd_req_line(rd_req_line[2]),
    .rd_rsp_valid(rd_rsp_valid[2]), .rd_rsp_data,
    .wr_valid(wr_valid[2]), .wr_ready(wr_ready[2]), .wr_line(wr_line[2]), .wr_data(wr_data[2]), .wr_be(wr_be[2]),
    .sreg, .perf_spm_wait, .perf_hazard, .perf_line, .dbg_held, .dbg_w0, .dbg_flags, .progress(dbg_x)
  );
  wire unused_ok = &{1'b0, n_rsp_v, n_rsp_d, n_wrdy, dbg_x, dbg_held, dbg_w0, dbg_flags};
endmodule
