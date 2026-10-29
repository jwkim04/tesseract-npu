// tsr_tb_mvm.sv — testbench-only toplevel for tb/mvm (not part of the core).
//
// The MVM unit as in the M3 core (docs/rtl-m3.md §2): tsr_mvm with its NUM_CH clusters, on SPM
// client TSR_SPM_CL_MVM of a real tsr_spm and on read source TSR_SRC_WS of one real tsr_mem_arb
// per channel, whose memory ports are exposed with the tsr_core naming (m_*, packed per channel,
// for common.mem_model.MemModel; memory-side ids {source, local}).
//   - SPM client TSR_SPM_CL_DMA (h_*): the host side of the test (Python fills x / old y lines
//     and reads the y lines back); client TSR_SPM_CL_QACT is tied off.
//   - arbiter sources AUX / DMA and the write source: tied off (the MVM unit runs alone).
//   - o_*: observation copies of the MVM unit's SPM client port and WS source ports.
// tb/mvm/test_mvm.py builds it with the real rtl/core files or with the test doubles of
// tsr_mvm_back / tsr_mvm_actbuf (tb/mvm/hdl/dbl, same module names).

module tsr_tb_mvm
  import tsr_pkg::*;
#(
  parameter  int unsigned NUM_CH         = 4,
  parameter  int unsigned T_MAX          = 1,
  parameter  int unsigned K_MAX          = 32768,
  parameter  int unsigned SPM_BYTES      = 262144,
  parameter  int unsigned SPM_RD_LAT     = 2,
  parameter  int unsigned ADDR_BEAT_W    = 28,
  parameter  int unsigned WS_FIFO_BEATS  = 512,
  parameter  int unsigned BC_PIPE        = 1,
  parameter  int unsigned RES_PIPE       = 1,
  parameter  int unsigned ACT_RD_LAT     = 2,
  parameter  int unsigned MAC_LAT        = 3,
  parameter  int unsigned MUL_LAT        = 4,
  parameter  int unsigned ADD_LAT        = 5,
  parameter  int unsigned WS_RD_LAT      = 2,
  parameter  int unsigned RD_OUTSTANDING = 16,
  localparam int unsigned BB             = 512,
  localparam int unsigned AW             = tsr_spm_aw(SPM_BYTES / TSR_LINE_BYTES)
) (
  input  logic                                clk,
  input  logic                                rst_n,

  // ---- dispatch ----
  input  logic                                cmd_valid,
  output logic                                cmd_ready,
  input  tsr_mvm_cmd_t                        cmd,
  output logic                                done,
  output logic                                perf_mvm_busy,

  // ---- host SPM port (client 0) ----
  input  logic                                h_rd_req_valid,
  output logic                                h_rd_req_ready,
  input  logic [AW-1:0]                       h_rd_req_line,
  output logic                                h_rd_rsp_valid,
  output logic [TSR_LINE_BITS-1:0]            h_rd_rsp_data,
  input  logic                                h_wr_valid,
  output logic                                h_wr_ready,
  input  logic [AW-1:0]                       h_wr_line,
  input  logic [TSR_LINE_BITS-1:0]            h_wr_data,
  input  logic [TSR_SPM_BE_W-1:0]             h_wr_be,

  // ---- observation: MVM SPM client, WS source ports ----
  output logic                                o_rd_req_valid,
  output logic                                o_rd_req_ready,
  output logic [AW-1:0]                       o_rd_req_line,
  output logic                                o_rd_rsp_valid,
  output logic                                o_wr_valid,
  output logic                                o_wr_ready,
  output logic [AW-1:0]                       o_wr_line,
  output logic [TSR_SPM_BE_W-1:0]             o_wr_be,
  output logic [NUM_CH-1:0]                   o_ws_req_valid,
  output logic [NUM_CH-1:0]                   o_ws_req_ready,
  output logic [NUM_CH-1:0]                   o_ws_rsp_valid,
  output logic [NUM_CH-1:0]                   o_ws_rsp_ready,

  // ---- memory ports ----
  output logic [NUM_CH-1:0]                   m_rd_req_valid,
  input  logic [NUM_CH-1:0]                   m_rd_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  m_rd_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]    m_rd_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0] m_rd_req_id,
  input  logic [NUM_CH-1:0]                   m_rd_rsp_valid,
  output logic [NUM_CH-1:0]                   m_rd_rsp_ready,
  input  logic [NUM_CH-1:0][BB-1:0]           m_rd_rsp_data,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0] m_rd_rsp_id,
  input  logic [NUM_CH-1:0]                   m_rd_rsp_last,
  output logic [NUM_CH-1:0]                   m_wr_req_valid,
  input  logic [NUM_CH-1:0]                   m_wr_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  m_wr_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]    m_wr_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0] m_wr_req_id,
  output logic [NUM_CH-1:0]                   m_wr_dat_valid,
  input  logic [NUM_CH-1:0]                   m_wr_dat_ready,
  output logic [NUM_CH-1:0][BB-1:0]           m_wr_dat_data,
  output logic [NUM_CH-1:0]                   m_wr_dat_last,
  input  logic [NUM_CH-1:0]                   m_wr_rsp_valid,
  output logic [NUM_CH-1:0]                   m_wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0] m_wr_rsp_id
);

  // ---- MVM unit ----
  logic                            mv_rd_req_valid, mv_rd_req_ready, mv_rd_rsp_valid;
  logic [AW-1:0]                   mv_rd_req_line;
  logic                            mv_wr_valid, mv_wr_ready;
  logic [AW-1:0]                   mv_wr_line;
  logic [TSR_LINE_BITS-1:0]        mv_wr_data, spm_rd_data;
  logic [TSR_SPM_BE_W-1:0]         mv_wr_be;
  logic [NUM_CH-1:0]               ws_req_valid, ws_req_ready, ws_rsp_valid, ws_rsp_ready, ws_rsp_last;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  ws_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    ws_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] ws_req_id, ws_rsp_id;
  logic [NUM_CH-1:0][BB-1:0]           ws_rsp_data;

  tsr_mvm #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT),
    .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .WS_FIFO_BEATS(WS_FIFO_BEATS), .BC_PIPE(BC_PIPE),
    .RES_PIPE(RES_PIPE), .ACT_RD_LAT(ACT_RD_LAT), .MAC_LAT(MAC_LAT), .MUL_LAT(MUL_LAT),
    .ADD_LAT(ADD_LAT), .WS_RD_LAT(WS_RD_LAT)
  ) u_mvm (
    .clk, .rst_n,
    .cmd_valid, .cmd_ready, .cmd, .done,
    .spm_rd_req_valid(mv_rd_req_valid), .spm_rd_req_ready(mv_rd_req_ready),
    .spm_rd_req_line(mv_rd_req_line), .spm_rd_rsp_valid(mv_rd_rsp_valid),
    .spm_rd_rsp_data(spm_rd_data),
    .spm_wr_valid(mv_wr_valid), .spm_wr_ready(mv_wr_ready), .spm_wr_line(mv_wr_line),
    .spm_wr_data(mv_wr_data), .spm_wr_be(mv_wr_be),
    .ws_rd_req_valid(ws_req_valid), .ws_rd_req_ready(ws_req_ready), .ws_rd_req_addr(ws_req_addr),
    .ws_rd_req_len(ws_req_len), .ws_rd_req_id(ws_req_id),
    .ws_rd_rsp_valid(ws_rsp_valid), .ws_rd_rsp_ready(ws_rsp_ready), .ws_rd_rsp_data(ws_rsp_data),
    .ws_rd_rsp_last(ws_rsp_last), .ws_rd_rsp_id(ws_rsp_id),
    .perf_mvm_busy
  );

  // ---- SPM: client 0 = host, 1 = tied off, 2 = MVM ----
  logic [TSR_SPM_NCL-1:0]                     s_rd_req_valid, s_rd_req_ready, s_rd_rsp_valid;
  logic [TSR_SPM_NCL-1:0][AW-1:0]             s_rd_req_line;
  logic [TSR_SPM_NCL-1:0]                     s_wr_valid, s_wr_ready;
  logic [TSR_SPM_NCL-1:0][AW-1:0]             s_wr_line;
  logic [TSR_SPM_NCL-1:0][TSR_LINE_BITS-1:0]  s_wr_data;
  logic [TSR_SPM_NCL-1:0][TSR_SPM_BE_W-1:0]   s_wr_be;

  always_comb begin
    s_rd_req_valid = '0;
    s_rd_req_line  = '0;
    s_wr_valid     = '0;
    s_wr_line      = '0;
    s_wr_data      = '0;
    s_wr_be        = '0;
    s_rd_req_valid[TSR_SPM_CL_DMA] = h_rd_req_valid;
    s_rd_req_line[TSR_SPM_CL_DMA]  = h_rd_req_line;
    s_wr_valid[TSR_SPM_CL_DMA]     = h_wr_valid;
    s_wr_line[TSR_SPM_CL_DMA]      = h_wr_line;
    s_wr_data[TSR_SPM_CL_DMA]      = h_wr_data;
    s_wr_be[TSR_SPM_CL_DMA]        = h_wr_be;
    s_rd_req_valid[TSR_SPM_CL_MVM] = mv_rd_req_valid;
    s_rd_req_line[TSR_SPM_CL_MVM]  = mv_rd_req_line;
    s_wr_valid[TSR_SPM_CL_MVM]     = mv_wr_valid;
    s_wr_line[TSR_SPM_CL_MVM]      = mv_wr_line;
    s_wr_data[TSR_SPM_CL_MVM]      = mv_wr_data;
    s_wr_be[TSR_SPM_CL_MVM]        = mv_wr_be;
  end

  tsr_spm #(.SPM_BYTES(SPM_BYTES), .NCL(TSR_SPM_NCL), .RD_LAT(SPM_RD_LAT)) u_spm (
    .clk, .rst_n,
    .rd_req_valid(s_rd_req_valid), .rd_req_ready(s_rd_req_ready), .rd_req_line(s_rd_req_line),
    .rd_rsp_valid(s_rd_rsp_valid), .rd_rsp_data(spm_rd_data),
    .wr_valid(s_wr_valid), .wr_ready(s_wr_ready), .wr_line(s_wr_line), .wr_data(s_wr_data),
    .wr_be(s_wr_be)
  );

  assign h_rd_req_ready  = s_rd_req_ready[TSR_SPM_CL_DMA];
  assign h_rd_rsp_valid  = s_rd_rsp_valid[TSR_SPM_CL_DMA];
  assign h_rd_rsp_data   = spm_rd_data;
  assign h_wr_ready      = s_wr_ready[TSR_SPM_CL_DMA];
  assign mv_rd_req_ready = s_rd_req_ready[TSR_SPM_CL_MVM];
  assign mv_rd_rsp_valid = s_rd_rsp_valid[TSR_SPM_CL_MVM];
  assign mv_wr_ready     = s_wr_ready[TSR_SPM_CL_MVM];

  // ---- one arbiter per channel: WS source only ----
  for (genvar c = 0; c < NUM_CH; c++) begin : g_ch
    logic [TSR_NSRC-1:0]                   a_req_valid, a_req_ready, a_rsp_valid, a_rsp_ready;
    logic [TSR_NSRC-1:0][ADDR_BEAT_W-1:0]  a_req_addr;
    logic [TSR_NSRC-1:0][TSR_LEN_W-1:0]    a_req_len;
    logic [TSR_NSRC-1:0][TSR_SRC_ID_W-1:0] a_req_id;
    logic                                  a_wr_req_ready, a_wr_dat_ready, a_wr_rsp_valid;
    logic                                  a_perf_beat, a_perf_wait;
    logic [TSR_SRC_ID_W-1:0]               a_wr_rsp_id;

    always_comb begin
      a_req_valid = '0;
      a_req_addr  = '0;
      a_req_len   = '0;
      a_req_id    = '0;
      a_rsp_ready = '1;
      a_req_valid[TSR_SRC_WS] = ws_req_valid[c];
      a_req_addr[TSR_SRC_WS]  = ws_req_addr[c];
      a_req_len[TSR_SRC_WS]   = ws_req_len[c];
      a_req_id[TSR_SRC_WS]    = ws_req_id[c];
      a_rsp_ready[TSR_SRC_WS] = ws_rsp_ready[c];
    end

    tsr_mem_arb #(
      .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .RD_OUTSTANDING(RD_OUTSTANDING)
    ) u_arb (
      .clk, .rst_n,
      .s_rd_req_valid(a_req_valid), .s_rd_req_ready(a_req_ready), .s_rd_req_addr(a_req_addr),
      .s_rd_req_len(a_req_len), .s_rd_req_id(a_req_id),
      .s_rd_rsp_valid(a_rsp_valid), .s_rd_rsp_ready(a_rsp_ready), .s_rd_rsp_data(ws_rsp_data[c]),
      .s_rd_rsp_last(ws_rsp_last[c]), .s_rd_rsp_id(ws_rsp_id[c]),
      .s_wr_req_valid(1'b0), .s_wr_req_ready(a_wr_req_ready), .s_wr_req_addr('0),
      .s_wr_req_len('0), .s_wr_req_id('0),
      .s_wr_dat_valid(1'b0), .s_wr_dat_ready(a_wr_dat_ready), .s_wr_dat_data('0),
      .s_wr_dat_last(1'b0),
      .s_wr_rsp_valid(a_wr_rsp_valid), .s_wr_rsp_ready(1'b1), .s_wr_rsp_id(a_wr_rsp_id),
      .m_rd_req_valid(m_rd_req_valid[c]), .m_rd_req_ready(m_rd_req_ready[c]),
      .m_rd_req_addr(m_rd_req_addr[c]), .m_rd_req_len(m_rd_req_len[c]),
      .m_rd_req_id(m_rd_req_id[c]),
      .m_rd_rsp_valid(m_rd_rsp_valid[c]), .m_rd_rsp_ready(m_rd_rsp_ready[c]),
      .m_rd_rsp_data(m_rd_rsp_data[c]), .m_rd_rsp_id(m_rd_rsp_id[c]),
      .m_rd_rsp_last(m_rd_rsp_last[c]),
      .m_wr_req_valid(m_wr_req_valid[c]), .m_wr_req_ready(m_wr_req_ready[c]),
      .m_wr_req_addr(m_wr_req_addr[c]), .m_wr_req_len(m_wr_req_len[c]),
      .m_wr_req_id(m_wr_req_id[c]),
      .m_wr_dat_valid(m_wr_dat_valid[c]), .m_wr_dat_ready(m_wr_dat_ready[c]),
      .m_wr_dat_data(m_wr_dat_data[c]), .m_wr_dat_last(m_wr_dat_last[c]),
      .m_wr_rsp_valid(m_wr_rsp_valid[c]), .m_wr_rsp_ready(m_wr_rsp_ready[c]),
      .m_wr_rsp_id(m_wr_rsp_id[c]),
      .perf_rd_beat(a_perf_beat), .perf_rd_wait(a_perf_wait)
    );

    assign ws_req_ready[c] = a_req_ready[TSR_SRC_WS];
    assign ws_rsp_valid[c] = a_rsp_valid[TSR_SRC_WS];

    wire unused_ok = &{1'b0, a_req_ready[TSR_SRC_DMA], a_req_ready[TSR_SRC_AUX],
                       a_rsp_valid[TSR_SRC_DMA], a_rsp_valid[TSR_SRC_AUX], a_wr_req_ready,
                       a_wr_dat_ready, a_wr_rsp_valid, a_wr_rsp_id, a_perf_beat, a_perf_wait};
  end

  // ---- observation ----
  assign o_rd_req_valid = mv_rd_req_valid;
  assign o_rd_req_ready = mv_rd_req_ready;
  assign o_rd_req_line  = mv_rd_req_line;
  assign o_rd_rsp_valid = mv_rd_rsp_valid;
  assign o_wr_valid     = mv_wr_valid;
  assign o_wr_ready     = mv_wr_ready;
  assign o_wr_line      = mv_wr_line;
  assign o_wr_be        = mv_wr_be;
  assign o_ws_req_valid = ws_req_valid;
  assign o_ws_req_ready = ws_req_ready;
  assign o_ws_rsp_valid = ws_rsp_valid;
  assign o_ws_rsp_ready = ws_rsp_ready;

  wire unused_ok = &{1'b0, s_rd_rsp_valid[TSR_SPM_CL_QACT], s_rd_req_ready[TSR_SPM_CL_QACT],
                     s_wr_ready[TSR_SPM_CL_QACT]};

endmodule
