// tsr_tb_dma.sv — testbench-only toplevel for tb/dma (not part of the core).
//
// tsr_dma + one tsr_mem_arb per channel, wired like tsr_core (DMA = source TSR_SRC_DMA of every
// channel), with the other read sources exposed so Python can generate background traffic:
//   f_*  : source TSR_SRC_AUX (0, the former fetch source) of channel 0 as a background reader
//          (channels > 0 tie it off; tsr_core ties it off on every channel, V2-CC-26)
//   w_*  : source TSR_SRC_WS of every channel (packed [NUM_CH]; the M3 weight-stream port)
//   m_*  : the channel memory ports (tsr_core naming, for common.mem_model.MemModel)
//   spm_*: the DMA's SPM client port. RTL_SPM = 0: driven by a Python SPM model. RTL_SPM = 1:
//          the DMA talks to a real tsr_spm (client TSR_SPM_CL_DMA; MVM client tied off);
//          spm_* outputs then only mirror the DMA's requests and the spm_* inputs are unused.
//   x_wr_*: RTL_SPM = 1 only: write port of tsr_spm client TSR_SPM_CL_QACT (lower fixed priority
//          than the DMA), driven by Python to contend with the DMA for the SPM write port (grant
//          handovers in both directions). RTL_SPM = 0: inputs unused, x_wr_ready = 0.
//   perf_rd_beat / perf_rd_wait: the arbiters' perf events per channel.

module tsr_tb_dma
  import tsr_pkg::*;
#(
  parameter  int unsigned NUM_CH         = 4,
  parameter  int unsigned ADDR_BEAT_W    = 20,
  parameter  int unsigned SPM_BYTES      = 65536,
  parameter  int unsigned SPM_RD_LAT     = 2,
  parameter  int unsigned RD_OUTSTANDING = 16,
  parameter  bit          RTL_SPM        = 1'b0,
  localparam int unsigned BB             = 512,
  localparam int unsigned SPM_LINES      = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned SPM_AW         = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1
) (
  input  logic                                   clk,
  input  logic                                   rst_n,

  // ---- DMA dispatch ----
  input  logic                                   cmd_valid,
  output logic                                   cmd_ready,
  input  tsr_dma_cmd_t                           cmd,
  output logic                                   done,

  // ---- DMA SPM client ----
  output logic                                   spm_rd_req_valid,
  input  logic                                   spm_rd_req_ready,
  output logic [SPM_AW-1:0]                      spm_rd_req_line,
  input  logic                                   spm_rd_rsp_valid,
  input  logic [TSR_LINE_BITS-1:0]               spm_rd_rsp_data,
  output logic                                   spm_wr_valid,
  input  logic                                   spm_wr_ready,
  output logic [SPM_AW-1:0]                      spm_wr_line,
  output logic [TSR_LINE_BITS-1:0]               spm_wr_data,
  output logic [TSR_SPM_BE_W-1:0]                spm_wr_be,

  // ---- RTL_SPM = 1: contending SPM writer (tsr_spm client TSR_SPM_CL_QACT, wr_be all ones) ----
  input  logic                                   x_wr_valid,
  output logic                                   x_wr_ready,
  input  logic [SPM_AW-1:0]                      x_wr_line,
  input  logic [TSR_LINE_BITS-1:0]               x_wr_data,

  // ---- background source AUX (channel 0) ----
  input  logic                                   f_rd_req_valid,
  output logic                                   f_rd_req_ready,
  input  logic [ADDR_BEAT_W-1:0]                 f_rd_req_addr,
  input  logic [TSR_LEN_W-1:0]                   f_rd_req_len,
  input  logic [TSR_SRC_ID_W-1:0]                f_rd_req_id,
  output logic                                   f_rd_rsp_valid,
  input  logic                                   f_rd_rsp_ready,
  output logic [BB-1:0]                          f_rd_rsp_data,
  output logic                                   f_rd_rsp_last,
  output logic [TSR_SRC_ID_W-1:0]                f_rd_rsp_id,

  // ---- background sources WS (one per channel) ----
  input  logic [NUM_CH-1:0]                      w_rd_req_valid,
  output logic [NUM_CH-1:0]                      w_rd_req_ready,
  input  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     w_rd_req_addr,
  input  logic [NUM_CH-1:0][TSR_LEN_W-1:0]       w_rd_req_len,
  input  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]    w_rd_req_id,
  output logic [NUM_CH-1:0]                      w_rd_rsp_valid,
  input  logic [NUM_CH-1:0]                      w_rd_rsp_ready,
  output logic [NUM_CH-1:0][BB-1:0]              w_rd_rsp_data,
  output logic [NUM_CH-1:0]                      w_rd_rsp_last,
  output logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]    w_rd_rsp_id,

  // ---- memory ports ----
  output logic [NUM_CH-1:0]                      m_rd_req_valid,
  input  logic [NUM_CH-1:0]                      m_rd_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_rd_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_rd_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_rd_req_id,
  input  logic [NUM_CH-1:0]                      m_rd_rsp_valid,
  output logic [NUM_CH-1:0]                      m_rd_rsp_ready,
  input  logic [NUM_CH-1:0][BB-1:0]              m_rd_rsp_data,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_rd_rsp_id,
  input  logic [NUM_CH-1:0]                      m_rd_rsp_last,
  output logic [NUM_CH-1:0]                      m_wr_req_valid,
  input  logic [NUM_CH-1:0]                      m_wr_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_wr_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_wr_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_req_id,
  output logic [NUM_CH-1:0]                      m_wr_dat_valid,
  input  logic [NUM_CH-1:0]                      m_wr_dat_ready,
  output logic [NUM_CH-1:0][BB-1:0]              m_wr_dat_data,
  output logic [NUM_CH-1:0]                      m_wr_dat_last,
  input  logic [NUM_CH-1:0]                      m_wr_rsp_valid,
  output logic [NUM_CH-1:0]                      m_wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_rsp_id,

  // ---- perf events ----
  output logic [NUM_CH-1:0]                      perf_rd_beat,
  output logic [NUM_CH-1:0]                      perf_rd_wait
);

  // DMA <-> arbiters
  logic [NUM_CH-1:0]                   d_rd_req_valid, d_rd_req_ready, d_rd_rsp_valid, d_rd_rsp_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  d_rd_req_addr, d_wr_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    d_rd_req_len, d_wr_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_rd_req_id, d_wr_req_id, d_wr_rsp_id;
  logic [NUM_CH-1:0]                   d_wr_req_valid, d_wr_req_ready;
  logic [NUM_CH-1:0]                   d_wr_dat_valid, d_wr_dat_ready, d_wr_dat_last;
  logic [NUM_CH-1:0][BB-1:0]           d_wr_dat_data;
  logic [NUM_CH-1:0]                   d_wr_rsp_valid, d_wr_rsp_ready;

  logic [NUM_CH-1:0][TSR_NSRC-1:0]                   s_rd_req_valid, s_rd_req_ready;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][ADDR_BEAT_W-1:0]  s_rd_req_addr;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][TSR_LEN_W-1:0]    s_rd_req_len;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][TSR_SRC_ID_W-1:0] s_rd_req_id;
  logic [NUM_CH-1:0][TSR_NSRC-1:0]                   s_rd_rsp_valid, s_rd_rsp_ready;
  logic [NUM_CH-1:0][BB-1:0]                         s_rd_rsp_data;
  logic [NUM_CH-1:0]                                 s_rd_rsp_last;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]               s_rd_rsp_id;

  // DMA SPM client (internal)
  logic                     i_rd_req_valid, i_rd_req_ready, i_rd_rsp_valid, i_wr_valid, i_wr_ready;
  logic [SPM_AW-1:0]        i_rd_req_line, i_wr_line;
  logic [TSR_LINE_BITS-1:0] i_rd_rsp_data, i_wr_data;
  logic [TSR_SPM_BE_W-1:0]  i_wr_be;

  tsr_dma #(
    .NUM_CH(NUM_CH), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .SPM_BYTES(SPM_BYTES),
    .SPM_RD_LAT(SPM_RD_LAT)
  ) u_dma (
    .clk, .rst_n,
    .cmd_valid, .cmd_ready, .cmd, .done,
    .rd_req_valid(d_rd_req_valid), .rd_req_ready(d_rd_req_ready), .rd_req_addr(d_rd_req_addr),
    .rd_req_len(d_rd_req_len), .rd_req_id(d_rd_req_id),
    .rd_rsp_valid(d_rd_rsp_valid), .rd_rsp_ready(d_rd_rsp_ready), .rd_rsp_data(s_rd_rsp_data),
    .rd_rsp_last(s_rd_rsp_last), .rd_rsp_id(s_rd_rsp_id),
    .wr_req_valid(d_wr_req_valid), .wr_req_ready(d_wr_req_ready), .wr_req_addr(d_wr_req_addr),
    .wr_req_len(d_wr_req_len), .wr_req_id(d_wr_req_id),
    .wr_dat_valid(d_wr_dat_valid), .wr_dat_ready(d_wr_dat_ready), .wr_dat_data(d_wr_dat_data),
    .wr_dat_last(d_wr_dat_last),
    .wr_rsp_valid(d_wr_rsp_valid), .wr_rsp_ready(d_wr_rsp_ready), .wr_rsp_id(d_wr_rsp_id),
    .spm_rd_req_valid(i_rd_req_valid), .spm_rd_req_ready(i_rd_req_ready),
    .spm_rd_req_line(i_rd_req_line), .spm_rd_rsp_valid(i_rd_rsp_valid),
    .spm_rd_rsp_data(i_rd_rsp_data),
    .spm_wr_valid(i_wr_valid), .spm_wr_ready(i_wr_ready), .spm_wr_line(i_wr_line),
    .spm_wr_data(i_wr_data), .spm_wr_be(i_wr_be)
  );

  assign spm_rd_req_valid = i_rd_req_valid;
  assign spm_rd_req_line  = i_rd_req_line;
  assign spm_wr_valid     = i_wr_valid;
  assign spm_wr_line      = i_wr_line;
  assign spm_wr_data      = i_wr_data;
  assign spm_wr_be        = i_wr_be;

  if (RTL_SPM) begin : g_rtl_spm
    logic [TSR_SPM_NCL-1:0]                    rq_v, rq_r, rs_v, wr_v, wr_r;
    logic [TSR_SPM_NCL-1:0][SPM_AW-1:0]        rq_l, wr_l;
    logic [TSR_SPM_NCL-1:0][TSR_LINE_BITS-1:0] wr_d;
    logic [TSR_SPM_NCL-1:0][TSR_SPM_BE_W-1:0]  wr_b;
    always_comb begin
      rq_v = '0; rq_l = '0; wr_v = '0; wr_l = '0; wr_d = '0; wr_b = '0;
      rq_v[TSR_SPM_CL_DMA] = i_rd_req_valid;
      rq_l[TSR_SPM_CL_DMA] = i_rd_req_line;
      wr_v[TSR_SPM_CL_DMA] = i_wr_valid;
      wr_l[TSR_SPM_CL_DMA] = i_wr_line;
      wr_d[TSR_SPM_CL_DMA] = i_wr_data;
      wr_b[TSR_SPM_CL_DMA] = i_wr_be;
      wr_v[TSR_SPM_CL_QACT] = x_wr_valid;
      wr_l[TSR_SPM_CL_QACT] = x_wr_line;
      wr_d[TSR_SPM_CL_QACT] = x_wr_data;
      wr_b[TSR_SPM_CL_QACT] = '1;
    end
    tsr_spm #(.SPM_BYTES(SPM_BYTES), .NCL(TSR_SPM_NCL), .RD_LAT(SPM_RD_LAT)) u_spm (
      .clk, .rst_n,
      .rd_req_valid(rq_v), .rd_req_ready(rq_r), .rd_req_line(rq_l),
      .rd_rsp_valid(rs_v), .rd_rsp_data(i_rd_rsp_data),
      .wr_valid(wr_v), .wr_ready(wr_r), .wr_line(wr_l), .wr_data(wr_d), .wr_be(wr_b)
    );
    assign i_rd_req_ready = rq_r[TSR_SPM_CL_DMA];
    assign i_rd_rsp_valid = rs_v[TSR_SPM_CL_DMA];
    assign i_wr_ready     = wr_r[TSR_SPM_CL_DMA];
    assign x_wr_ready     = wr_r[TSR_SPM_CL_QACT];
    wire unused_spm = &{1'b0, rq_r, rs_v, wr_r, spm_rd_req_ready, spm_rd_rsp_valid, spm_rd_rsp_data,
                        spm_wr_ready};
  end else begin : g_py_spm
    assign i_rd_req_ready = spm_rd_req_ready;
    assign i_rd_rsp_valid = spm_rd_rsp_valid;
    assign i_rd_rsp_data  = spm_rd_rsp_data;
    assign i_wr_ready     = spm_wr_ready;
    assign x_wr_ready     = 1'b0;
    wire unused_x = &{1'b0, x_wr_valid, x_wr_line, x_wr_data};
  end

  for (genvar c = 0; c < NUM_CH; c++) begin : g_ch
    if (c == 0) begin : g_aux
      assign s_rd_req_valid[c][TSR_SRC_AUX] = f_rd_req_valid;
      assign s_rd_req_addr[c][TSR_SRC_AUX]  = f_rd_req_addr;
      assign s_rd_req_len[c][TSR_SRC_AUX]   = f_rd_req_len;
      assign s_rd_req_id[c][TSR_SRC_AUX]    = f_rd_req_id;
      assign s_rd_rsp_ready[c][TSR_SRC_AUX] = f_rd_rsp_ready;
    end else begin : g_noaux
      assign s_rd_req_valid[c][TSR_SRC_AUX] = 1'b0;
      assign s_rd_req_addr[c][TSR_SRC_AUX]  = '0;
      assign s_rd_req_len[c][TSR_SRC_AUX]   = '0;
      assign s_rd_req_id[c][TSR_SRC_AUX]    = '0;
      assign s_rd_rsp_ready[c][TSR_SRC_AUX] = 1'b1;
    end
    assign s_rd_req_valid[c][TSR_SRC_DMA] = d_rd_req_valid[c];
    assign s_rd_req_addr[c][TSR_SRC_DMA]  = d_rd_req_addr[c];
    assign s_rd_req_len[c][TSR_SRC_DMA]   = d_rd_req_len[c];
    assign s_rd_req_id[c][TSR_SRC_DMA]    = d_rd_req_id[c];
    assign s_rd_rsp_ready[c][TSR_SRC_DMA] = d_rd_rsp_ready[c];
    assign d_rd_req_ready[c]              = s_rd_req_ready[c][TSR_SRC_DMA];
    assign d_rd_rsp_valid[c]              = s_rd_rsp_valid[c][TSR_SRC_DMA];
    assign s_rd_req_valid[c][TSR_SRC_WS]  = w_rd_req_valid[c];
    assign s_rd_req_addr[c][TSR_SRC_WS]   = w_rd_req_addr[c];
    assign s_rd_req_len[c][TSR_SRC_WS]    = w_rd_req_len[c];
    assign s_rd_req_id[c][TSR_SRC_WS]     = w_rd_req_id[c];
    assign s_rd_rsp_ready[c][TSR_SRC_WS]  = w_rd_rsp_ready[c];
    assign w_rd_req_ready[c]              = s_rd_req_ready[c][TSR_SRC_WS];
    assign w_rd_rsp_valid[c]              = s_rd_rsp_valid[c][TSR_SRC_WS];
    assign w_rd_rsp_data[c]               = s_rd_rsp_data[c];
    assign w_rd_rsp_last[c]               = s_rd_rsp_last[c];
    assign w_rd_rsp_id[c]                 = s_rd_rsp_id[c];

    tsr_mem_arb #(
      .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .RD_OUTSTANDING(RD_OUTSTANDING)
    ) u_arb (
      .clk, .rst_n,
      .s_rd_req_valid(s_rd_req_valid[c]), .s_rd_req_ready(s_rd_req_ready[c]),
      .s_rd_req_addr(s_rd_req_addr[c]), .s_rd_req_len(s_rd_req_len[c]), .s_rd_req_id(s_rd_req_id[c]),
      .s_rd_rsp_valid(s_rd_rsp_valid[c]), .s_rd_rsp_ready(s_rd_rsp_ready[c]),
      .s_rd_rsp_data(s_rd_rsp_data[c]), .s_rd_rsp_last(s_rd_rsp_last[c]), .s_rd_rsp_id(s_rd_rsp_id[c]),
      .s_wr_req_valid(d_wr_req_valid[c]), .s_wr_req_ready(d_wr_req_ready[c]),
      .s_wr_req_addr(d_wr_req_addr[c]), .s_wr_req_len(d_wr_req_len[c]), .s_wr_req_id(d_wr_req_id[c]),
      .s_wr_dat_valid(d_wr_dat_valid[c]), .s_wr_dat_ready(d_wr_dat_ready[c]),
      .s_wr_dat_data(d_wr_dat_data[c]), .s_wr_dat_last(d_wr_dat_last[c]),
      .s_wr_rsp_valid(d_wr_rsp_valid[c]), .s_wr_rsp_ready(d_wr_rsp_ready[c]), .s_wr_rsp_id(d_wr_rsp_id[c]),
      .m_rd_req_valid(m_rd_req_valid[c]), .m_rd_req_ready(m_rd_req_ready[c]),
      .m_rd_req_addr(m_rd_req_addr[c]), .m_rd_req_len(m_rd_req_len[c]), .m_rd_req_id(m_rd_req_id[c]),
      .m_rd_rsp_valid(m_rd_rsp_valid[c]), .m_rd_rsp_ready(m_rd_rsp_ready[c]),
      .m_rd_rsp_data(m_rd_rsp_data[c]), .m_rd_rsp_id(m_rd_rsp_id[c]), .m_rd_rsp_last(m_rd_rsp_last[c]),
      .m_wr_req_valid(m_wr_req_valid[c]), .m_wr_req_ready(m_wr_req_ready[c]),
      .m_wr_req_addr(m_wr_req_addr[c]), .m_wr_req_len(m_wr_req_len[c]), .m_wr_req_id(m_wr_req_id[c]),
      .m_wr_dat_valid(m_wr_dat_valid[c]), .m_wr_dat_ready(m_wr_dat_ready[c]),
      .m_wr_dat_data(m_wr_dat_data[c]), .m_wr_dat_last(m_wr_dat_last[c]),
      .m_wr_rsp_valid(m_wr_rsp_valid[c]), .m_wr_rsp_ready(m_wr_rsp_ready[c]), .m_wr_rsp_id(m_wr_rsp_id[c]),
      .perf_rd_beat(perf_rd_beat[c]), .perf_rd_wait(perf_rd_wait[c])
    );
  end

  assign f_rd_req_ready = s_rd_req_ready[0][TSR_SRC_AUX];
  assign f_rd_rsp_valid = s_rd_rsp_valid[0][TSR_SRC_AUX];
  assign f_rd_rsp_data  = s_rd_rsp_data[0];
  assign f_rd_rsp_last  = s_rd_rsp_last[0];
  assign f_rd_rsp_id    = s_rd_rsp_id[0];

  logic unused_tb;
  always_comb begin
    unused_tb = 1'b0;
    for (int c = 1; c < NUM_CH; c++) begin
      unused_tb |= s_rd_req_ready[c][TSR_SRC_AUX] | s_rd_rsp_valid[c][TSR_SRC_AUX];
    end
  end

endmodule
