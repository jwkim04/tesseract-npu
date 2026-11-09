// tsr_tb_mvm_front.sv — testbench-only toplevel for tb/mvmfront (not part of the core).
//
// One channel as in the M3 core (docs/rtl-mvm.md §2, §4): tsr_mvm_front on read source
// TSR_SRC_WS of a real tsr_mem_arb; the arbiter's memory port is exposed with the tsr_core
// naming (m_*, for common.mem_model.MemModel, memory-side ids {source, local}).
//   - source TSR_SRC_AUX (f_*; 0, the former fetch source): exposed; Python leaves it idle or
//     drives background bursts
//     (shared RD_OUTSTANDING budget, head-of-line response routing).
//   - source TSR_SRC_DMA and the write source: tied off (idle), as the MVM unit runs alone.
//   - ws_*: observation copies of the WS source port between the front end and the arbiter
//     (request handshake, response valid / ready) for the burst-list and rd_rsp_ready checks, and
//     of tsr_mvm_ws's own idle output (its contract: no command, nothing outstanding, FIFO empty).

module tsr_tb_mvm_front
  import tsr_pkg::*;
#(
  parameter  int unsigned K_MAX          = 32768,
  parameter  int unsigned ADDR_BEAT_W    = 28,
  parameter  int unsigned WS_FIFO_BEATS  = 512,
  parameter  int unsigned WS_RD_LAT      = 2,
  parameter  int unsigned RD_OUTSTANDING = 16,
  localparam int unsigned BB             = 512
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // ---- front end command / beats ----
  input  logic                          cmd_valid,
  output logic                          cmd_ready,
  input  tsr_mvm_fcmd_t                 cmd,
  output logic                          beat_valid,
  input  logic                          beat_ready,
  output tsr_mvm_beat_t                 beat,
  output logic                          idle,

  // ---- observation of the WS source port ----
  output logic                          ws_req_valid,
  output logic                          ws_req_ready,
  output logic [ADDR_BEAT_W-1:0]        ws_req_addr,
  output logic [TSR_LEN_W-1:0]          ws_req_len,
  output logic [TSR_SRC_ID_W-1:0]       ws_req_id,
  output logic                          ws_rsp_valid,
  output logic                          ws_rsp_ready,
  output logic                          ws_idle,

  // ---- background source AUX ----
  input  logic                          f_rd_req_valid,
  output logic                          f_rd_req_ready,
  input  logic [ADDR_BEAT_W-1:0]        f_rd_req_addr,
  input  logic [TSR_LEN_W-1:0]          f_rd_req_len,
  input  logic [TSR_SRC_ID_W-1:0]       f_rd_req_id,
  output logic                          f_rd_rsp_valid,
  input  logic                          f_rd_rsp_ready,
  output logic [BB-1:0]                 f_rd_rsp_data,
  output logic                          f_rd_rsp_last,
  output logic [TSR_SRC_ID_W-1:0]       f_rd_rsp_id,

  // ---- memory port (channel 0) ----
  output logic                          m_rd_req_valid,
  input  logic                          m_rd_req_ready,
  output logic [ADDR_BEAT_W-1:0]        m_rd_req_addr,
  output logic [TSR_LEN_W-1:0]          m_rd_req_len,
  output logic [TSR_MEM_ID_W-1:0]       m_rd_req_id,
  input  logic                          m_rd_rsp_valid,
  output logic                          m_rd_rsp_ready,
  input  logic [BB-1:0]                 m_rd_rsp_data,
  input  logic [TSR_MEM_ID_W-1:0]       m_rd_rsp_id,
  input  logic                          m_rd_rsp_last,
  output logic                          m_wr_req_valid,
  input  logic                          m_wr_req_ready,
  output logic [ADDR_BEAT_W-1:0]        m_wr_req_addr,
  output logic [TSR_LEN_W-1:0]          m_wr_req_len,
  output logic [TSR_MEM_ID_W-1:0]       m_wr_req_id,
  output logic                          m_wr_dat_valid,
  input  logic                          m_wr_dat_ready,
  output logic [BB-1:0]                 m_wr_dat_data,
  output logic [BB/32-1:0]                 m_wr_dat_mask,
  output logic                          m_wr_dat_last,
  input  logic                          m_wr_rsp_valid,
  output logic                          m_wr_rsp_ready,
  input  logic [TSR_MEM_ID_W-1:0]       m_wr_rsp_id,

  output logic                          perf_rd_beat,
  output logic                          perf_rd_wait
);

  // WS source <-> arbiter
  logic                    w_req_valid, w_req_ready, w_rsp_ready;
  logic [ADDR_BEAT_W-1:0]  w_req_addr;
  logic [TSR_LEN_W-1:0]    w_req_len;
  logic [TSR_SRC_ID_W-1:0] w_req_id;

  logic [TSR_NSRC-1:0]                   s_rd_req_valid, s_rd_req_ready, s_rd_rsp_valid, s_rd_rsp_ready;
  logic [TSR_NSRC-1:0][ADDR_BEAT_W-1:0]  s_rd_req_addr;
  logic [TSR_NSRC-1:0][TSR_LEN_W-1:0]    s_rd_req_len;
  logic [TSR_NSRC-1:0][TSR_SRC_ID_W-1:0] s_rd_req_id;
  logic [BB-1:0]                         s_rd_rsp_data;
  logic                                  s_rd_rsp_last;
  logic [TSR_SRC_ID_W-1:0]               s_rd_rsp_id;
  logic                                  s_wr_req_ready, s_wr_dat_ready, s_wr_rsp_valid;
  logic [TSR_SRC_ID_W-1:0]               s_wr_rsp_id;

  tsr_mvm_front #(
    .K_MAX(K_MAX), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .WS_FIFO_BEATS(WS_FIFO_BEATS),
    .WS_RD_LAT(WS_RD_LAT)
  ) u_front (
    .clk, .rst_n,
    .cmd_valid, .cmd_ready, .cmd,
    .cmd_att(1'b0), .cmd_rbeats(cmd.beats), .cmd_stride(32'd0),   // W4-B at: one range (MVM)
    .rd_req_valid(w_req_valid), .rd_req_ready(w_req_ready), .rd_req_addr(w_req_addr),
    .rd_req_len(w_req_len), .rd_req_id(w_req_id),
    .rd_rsp_valid(s_rd_rsp_valid[TSR_SRC_WS]), .rd_rsp_ready(w_rsp_ready),
    .rd_rsp_data(s_rd_rsp_data), .rd_rsp_last(s_rd_rsp_last), .rd_rsp_id(s_rd_rsp_id),
    .beat_valid, .beat_ready, .beat,
    .idle
  );

  always_comb begin
    s_rd_req_valid = '0;
    s_rd_req_addr  = '0;
    s_rd_req_len   = '0;
    s_rd_req_id    = '0;
    s_rd_rsp_ready = '1;
    // AUX: background traffic from Python
    s_rd_req_valid[TSR_SRC_AUX] = f_rd_req_valid;
    s_rd_req_addr[TSR_SRC_AUX]  = f_rd_req_addr;
    s_rd_req_len[TSR_SRC_AUX]   = f_rd_req_len;
    s_rd_req_id[TSR_SRC_AUX]    = f_rd_req_id;
    s_rd_rsp_ready[TSR_SRC_AUX] = f_rd_rsp_ready;
    // DMA: idle (valid 0, ready 1). WS: the front end.
    s_rd_req_valid[TSR_SRC_WS]    = w_req_valid;
    s_rd_req_addr[TSR_SRC_WS]     = w_req_addr;
    s_rd_req_len[TSR_SRC_WS]      = w_req_len;
    s_rd_req_id[TSR_SRC_WS]       = w_req_id;
    s_rd_rsp_ready[TSR_SRC_WS]    = w_rsp_ready;
  end

  tsr_mem_arb #(
    .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .RD_OUTSTANDING(RD_OUTSTANDING)
  ) u_arb (
    .clk, .rst_n,
    .s_rd_req_valid, .s_rd_req_ready, .s_rd_req_addr, .s_rd_req_len, .s_rd_req_id,
    .s_rd_rsp_valid, .s_rd_rsp_ready, .s_rd_rsp_data, .s_rd_rsp_last, .s_rd_rsp_id,
    .s_wr_req_valid(1'b0), .s_wr_req_ready, .s_wr_req_addr('0), .s_wr_req_len('0), .s_wr_req_id('0),
    .s_wr_dat_valid(1'b0), .s_wr_dat_ready, .s_wr_dat_data('0), .s_wr_dat_mask('1), .s_wr_dat_last(1'b0),
    .s_wr_rsp_valid, .s_wr_rsp_ready(1'b1), .s_wr_rsp_id,
    .m_rd_req_valid, .m_rd_req_ready, .m_rd_req_addr, .m_rd_req_len, .m_rd_req_id,
    .m_rd_rsp_valid, .m_rd_rsp_ready, .m_rd_rsp_data, .m_rd_rsp_id, .m_rd_rsp_last,
    .m_wr_req_valid, .m_wr_req_ready, .m_wr_req_addr, .m_wr_req_len, .m_wr_req_id,
    .m_wr_dat_valid, .m_wr_dat_ready, .m_wr_dat_data, .m_wr_dat_mask, .m_wr_dat_last,
    .m_wr_rsp_valid, .m_wr_rsp_ready, .m_wr_rsp_id,
    .perf_rd_beat, .perf_rd_wait
  );

  assign f_rd_req_ready = s_rd_req_ready[TSR_SRC_AUX];
  assign f_rd_rsp_valid = s_rd_rsp_valid[TSR_SRC_AUX];
  assign f_rd_rsp_data  = s_rd_rsp_data;
  assign f_rd_rsp_last  = s_rd_rsp_last;
  assign f_rd_rsp_id    = s_rd_rsp_id;

  assign ws_req_valid = w_req_valid;
  assign ws_req_ready = s_rd_req_ready[TSR_SRC_WS];
  assign w_req_ready  = s_rd_req_ready[TSR_SRC_WS];
  assign ws_req_addr  = w_req_addr;
  assign ws_req_len   = w_req_len;
  assign ws_req_id    = w_req_id;
  assign ws_rsp_valid = s_rd_rsp_valid[TSR_SRC_WS];
  assign ws_rsp_ready = w_rsp_ready;
  assign ws_idle      = u_front.ws_idle;

  wire unused_ok = &{1'b0, s_rd_req_ready[TSR_SRC_DMA], s_rd_rsp_valid[TSR_SRC_DMA], s_wr_req_ready,
                     s_wr_dat_ready, s_wr_rsp_valid, s_wr_rsp_id};

endmodule
