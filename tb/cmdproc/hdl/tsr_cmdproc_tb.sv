// tsr_cmdproc_tb.sv — testbench-only toplevel for block (a): tsr_csr + tsr_cmdproc + tsr_fetch.
//
//   - CSR port and done: the tsr_core boundary signals (common.csr_driver.CsrDriver attaches).
//   - Memory ports m_* in tsr_core naming for NUM_CH channels (common.mem_model.MemModel
//     attaches). Only the channel-0 read port is used: it carries tsr_fetch's requests with the
//     memory-side id {TSR_SRC_FETCH, local id}, as tsr_mem_arb would. Every other channel and the
//     write channels stay idle (their idleness is part of the test: the model flags any traffic).
//   - The unit dispatch ports of tsr_cmdproc (DMA, QACT, MVM) are toplevel ports driven by Python
//     unit models (tb/cmdproc/test_cmdproc.py UnitStub).
//   - Perf events: tb_perf_* inputs, plus channel-0 fetch read beats.

module tsr_cmdproc_tb
  import tsr_pkg::*;
#(
  parameter int unsigned     NUM_CH      = TSR_DEF_NUM_CH,
  parameter int unsigned     LANES       = TSR_DEF_LANES,
  parameter int unsigned     T_MAX       = TSR_DEF_T_MAX,
  parameter int unsigned     K_MAX       = TSR_DEF_K_MAX,
  parameter int unsigned     SPM_BYTES   = TSR_DEF_SPM_BYTES,
  parameter int unsigned     VL          = TSR_DEF_VL,
  parameter int unsigned     NSREG       = TSR_DEF_NSREG,
  parameter longint unsigned CH_BYTES    = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter bit              HAS_MVM     = 1'b0,
  parameter int unsigned     FETCH_BEATS = 16
) (
  input  logic                                   clk,
  input  logic                                   rst_n,

  // ---- CSR port, done ----
  input  logic                                   csr_valid,
  output logic                                   csr_ready,
  input  logic                                   csr_we,
  input  logic [TSR_CSR_AW-1:0]                  csr_addr,
  input  logic [31:0]                            csr_wdata,
  output logic [31:0]                            csr_rdata,
  output logic                                   done,

  // ---- memory ports (tsr_core naming) ----
  output logic [NUM_CH-1:0]                      m_rd_req_valid,
  input  logic [NUM_CH-1:0]                      m_rd_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_rd_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_rd_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_rd_req_id,
  input  logic [NUM_CH-1:0]                      m_rd_rsp_valid,
  output logic [NUM_CH-1:0]                      m_rd_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_LINE_BITS-1:0]   m_rd_rsp_data,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_rd_rsp_id,
  input  logic [NUM_CH-1:0]                      m_rd_rsp_last,
  output logic [NUM_CH-1:0]                      m_wr_req_valid,
  input  logic [NUM_CH-1:0]                      m_wr_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_wr_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_wr_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_req_id,
  output logic [NUM_CH-1:0]                      m_wr_dat_valid,
  input  logic [NUM_CH-1:0]                      m_wr_dat_ready,
  output logic [NUM_CH-1:0][TSR_LINE_BITS-1:0]   m_wr_dat_data,
  output logic [NUM_CH-1:0]                      m_wr_dat_last,
  input  logic [NUM_CH-1:0]                      m_wr_rsp_valid,
  output logic [NUM_CH-1:0]                      m_wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_rsp_id,

  // ---- unit dispatch (Python unit models) ----
  output logic                                   dma_cmd_valid,
  input  logic                                   dma_cmd_ready,
  output tsr_dma_cmd_t                           dma_cmd,
  input  logic                                   dma_done,
  output logic                                   qact_cmd_valid,
  input  logic                                   qact_cmd_ready,
  output tsr_qact_cmd_t                          qact_cmd,
  input  logic                                   qact_done,
  output logic                                   mvm_cmd_valid,
  input  logic                                   mvm_cmd_ready,
  output tsr_mvm_cmd_t                           mvm_cmd,
  input  logic                                   mvm_done,

  // ---- perf events ----
  input  logic [NUM_CH-1:0]                      tb_perf_rd_beat,
  input  logic [NUM_CH-1:0]                      tb_perf_wait,
  input  logic                                   tb_perf_mvm_busy
);

  logic                 run_start, run_abort, run_soft_reset, run_finish;
  logic [63:0]          run_cmd_addr;
  logic [31:0]          run_cmd_count;
  logic [7:0][31:0]     run_args;
  logic [TSR_ERR_W-1:0] run_err_code;
  logic [31:0]          run_err_pc, run_err_info, cur_pc;
  logic                 perf_retire;
  logic [NUM_CH-1:0]    perf_ch_rd_beat;

  logic                 fetch_start, fetch_stop, fetch_idle;
  logic [63:0]          fetch_cmd_addr;
  logic [31:0]          fetch_cmd_count;
  logic                 insn_valid, insn_ready;
  tsr_fetch_insn_t      insn;

  logic                    f_rd_req_valid, f_rd_rsp_ready;
  logic [ADDR_BEAT_W-1:0]  f_rd_req_addr;
  logic [TSR_LEN_W-1:0]    f_rd_req_len;
  logic [TSR_SRC_ID_W-1:0] f_rd_req_id;

  tsr_csr #(
    .NUM_CH(NUM_CH), .LANES(LANES), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES),
    .VL(VL), .NSREG(NSREG), .CAPS(32'd0)
  ) u_csr (
    .clk, .rst_n,
    .csr_valid, .csr_ready, .csr_we, .csr_addr, .csr_wdata, .csr_rdata, .done,
    .run_start, .run_cmd_addr, .run_cmd_count, .run_args, .run_abort, .run_soft_reset,
    .run_finish, .run_err_code, .run_err_pc, .run_err_info, .cur_pc,
    .perf_retire, .perf_mvm_busy(tb_perf_mvm_busy), .perf_ch_rd_beat, .perf_ch_wait(tb_perf_wait)
  );

  tsr_cmdproc #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CH_BYTES(CH_BYTES),
    .ADDR_BEAT_W(ADDR_BEAT_W), .HAS_MVM(HAS_MVM)
  ) u_cmdproc (
    .clk, .rst_n,
    .run_start, .run_cmd_addr, .run_cmd_count, .run_args, .run_abort, .run_soft_reset,
    .run_finish, .run_err_code, .run_err_pc, .run_err_info, .cur_pc, .perf_retire,
    .fetch_start, .fetch_cmd_addr, .fetch_cmd_count, .fetch_stop, .fetch_idle,
    .insn_valid, .insn_ready, .insn,
    .dma_cmd_valid, .dma_cmd_ready, .dma_cmd, .dma_done,
    .qact_cmd_valid, .qact_cmd_ready, .qact_cmd, .qact_done,
    .mvm_cmd_valid, .mvm_cmd_ready, .mvm_cmd, .mvm_done
  );

  tsr_fetch #(
    .ADDR_BEAT_W(ADDR_BEAT_W), .CH_BYTES(CH_BYTES), .BEAT_BITS(TSR_LINE_BITS), .BUF_BEATS(FETCH_BEATS)
  ) u_fetch (
    .clk, .rst_n,
    .start(fetch_start), .cmd_addr(fetch_cmd_addr), .cmd_count(fetch_cmd_count), .stop(fetch_stop),
    .idle(fetch_idle),
    .insn_valid, .insn_ready, .insn,
    .rd_req_valid(f_rd_req_valid), .rd_req_ready(m_rd_req_ready[0]), .rd_req_addr(f_rd_req_addr),
    .rd_req_len(f_rd_req_len), .rd_req_id(f_rd_req_id),
    .rd_rsp_valid(m_rd_rsp_valid[0]), .rd_rsp_ready(f_rd_rsp_ready), .rd_rsp_data(m_rd_rsp_data[0]),
    .rd_rsp_last(m_rd_rsp_last[0]), .rd_rsp_id(m_rd_rsp_id[0][TSR_SRC_ID_W-1:0])
  );

  // Channel 0 read port = fetch (id {FETCH, local}); everything else idle.
  always_comb begin
    m_rd_req_valid = '0;
    m_rd_req_addr  = '0;
    m_rd_req_len   = '0;
    m_rd_req_id    = '0;
    m_rd_rsp_ready = '1;
    m_rd_req_valid[0] = f_rd_req_valid;
    m_rd_req_addr[0]  = f_rd_req_addr;
    m_rd_req_len[0]   = f_rd_req_len;
    m_rd_req_id[0]    = {TSR_SRC_ID_W'(TSR_SRC_FETCH), f_rd_req_id};
    m_rd_rsp_ready[0] = f_rd_rsp_ready;
  end
  assign m_wr_req_valid = '0;
  assign m_wr_req_addr  = '0;
  assign m_wr_req_len   = '0;
  assign m_wr_req_id    = '0;
  assign m_wr_dat_valid = '0;
  assign m_wr_dat_data  = '0;
  assign m_wr_dat_last  = '0;
  assign m_wr_rsp_ready = '1;

  always_comb begin
    perf_ch_rd_beat    = tb_perf_rd_beat;
    perf_ch_rd_beat[0] = tb_perf_rd_beat[0] | (m_rd_rsp_valid[0] & f_rd_rsp_ready);
  end

  wire unused_ok = &{1'b0, m_rd_req_ready, m_rd_rsp_valid, m_rd_rsp_data, m_rd_rsp_id, m_rd_rsp_last,
                     m_wr_req_ready, m_wr_dat_ready, m_wr_rsp_valid, m_wr_rsp_id};

endmodule
