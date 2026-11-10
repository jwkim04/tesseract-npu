// stub_core_portb.sv — a stand-in for tsr_core in the kernel-level bench of the read-only port B
// (tb/platform/test_adapter.py, adapter_portb). tsr_xrt_kernel instantiates it in place of the core, so the
// kernel's own gate, reset copies, bridges and control bridge are the ones under test, and the bench drives the
// core side of the memory ports itself.
//
// The read request channel and the response ready of the main read port (m_rd_*) and of the read-only port
// (mb_rd_*) come from the variables t_a_* / t_b_*, which the bench writes through the simulator interface. The write
// channels are idle (valid 0, response ready 1). The CSR port answers every access at once with 0. The parameters are
// the ones tsr_xrt_kernel passes; none of them changes the stub.

/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSEDPARAM */
module tsr_core
  import tsr_pkg::*;
#(
  parameter int unsigned     NUM_CH         = TSR_DEF_NUM_CH,
  parameter int unsigned     BEAT_BITS      = TSR_DEF_BEAT_BITS,
  parameter int unsigned     LANES          = TSR_DEF_LANES,
  parameter int unsigned     T_MAX          = TSR_DEF_T_MAX,
  parameter int unsigned     K_MAX          = TSR_DEF_K_MAX,
  parameter int unsigned     SPM_BYTES      = TSR_DEF_SPM_BYTES,
  parameter int unsigned     VL             = TSR_DEF_VL,
  parameter int unsigned     NSREG          = TSR_DEF_NSREG,
  parameter int unsigned     RD_OUTSTANDING = TSR_DEF_RD_OUTSTANDING,
  parameter longint unsigned CH_BYTES       = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W    = TSR_DEF_ADDR_BEAT_W,
  parameter bit              HAS_MVM        = 1'b1,
  parameter bit              MVM_DUAL       = 1'b0,
  parameter int unsigned     SPM_RD_LAT     = TSR_DEF_SPM_RD_LAT,
  parameter int unsigned     MVM_WS_FIFO    = TSR_DEF_WS_FIFO_BEATS,
  parameter int unsigned     MVM_BC_PIPE    = TSR_DEF_MVM_BC_PIPE,
  parameter bit              DDR_LOCAL_PIPE = 1'b0,
  parameter bit              LOCAL_SOFTCAP  = 1'b0,
  parameter bit              LOCAL_SCALE    = 1'b0,
  parameter bit              LOCAL_INTERMEDIATES = 1'b0,
  parameter bit              MVM_GATHER_PAIR = 1'b0,
  parameter int unsigned     MVM_GATHER_FLITS = 1,
  parameter int unsigned     MVM_RES_PIPE   = TSR_DEF_MVM_RES_PIPE,
  parameter int unsigned     MVM_ACT_RD_LAT = TSR_DEF_ACT_RD_LAT,
  parameter int unsigned     MVM_MAC_LAT    = TSR_DEF_MAC_LAT,
  parameter int unsigned     MVM_MUL_LAT    = TSR_DEF_FMUL_LAT,
  parameter int unsigned     MVM_ADD_LAT    = TSR_DEF_FADD_LAT,
  parameter int unsigned     MVM_WS_RD_LAT  = TSR_DEF_WS_RD_LAT,
  parameter int unsigned     RST_PIPE       = 0,
  parameter bit              RV_LOCAL_TRANSPORT = 1'b0,
  parameter bit              XSLR_REG       = 1'b0,
  parameter bit              FAST_CLK       = 1'b0,
  parameter bit              HAS_VU         = 1'b1,
  parameter bit              HAS_ATT        = HAS_MVM,
  parameter bit              HAS_GDN        = 1'b0,
  parameter int unsigned     VU_FRONT_LANES = tsr_tables_pkg::TSR_VU_FRONT_LANES,
  parameter int unsigned     VU_EXP_LANES   = tsr_tables_pkg::TSR_VU_FRONT_LANES,
  parameter int unsigned     VU_BACK_LANES  = tsr_tables_pkg::TSR_VU_BACK_LANES,
  parameter int unsigned     VU_LANE_PIPE   = 0,
  parameter int unsigned     HIST_DEPTH     = tsr_tables_pkg::TSR_HIST_DEPTH,
  parameter bit              HAS_TRW        = 1'b1
) (
  input  logic                                   clk,
  input  logic                                   rst_n,
  input  logic                                   clk_rv,
  input  logic                                   rst_rv_n,

  input  logic                                   csr_valid,
  output logic                                   csr_ready,
  input  logic                                   csr_we,
  input  logic [TSR_CSR_AW-1:0]                  csr_addr,
  input  logic [31:0]                            csr_wdata,
  output logic [31:0]                            csr_rdata,

  output logic [NUM_CH-1:0]                      m_rd_req_valid,
  input  logic [NUM_CH-1:0]                      m_rd_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_rd_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_rd_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_rd_req_id,
  input  logic [NUM_CH-1:0]                      m_rd_rsp_valid,
  output logic [NUM_CH-1:0]                      m_rd_rsp_ready,
  input  logic [NUM_CH-1:0][BEAT_BITS-1:0]       m_rd_rsp_data,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_rd_rsp_id,
  input  logic [NUM_CH-1:0]                      m_rd_rsp_last,
  output logic [NUM_CH-1:0]                      mb_rd_req_valid,
  input  logic [NUM_CH-1:0]                      mb_rd_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     mb_rd_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       mb_rd_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    mb_rd_req_id,
  input  logic [NUM_CH-1:0]                      mb_rd_rsp_valid,
  output logic [NUM_CH-1:0]                      mb_rd_rsp_ready,
  input  logic [NUM_CH-1:0][BEAT_BITS-1:0]       mb_rd_rsp_data,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    mb_rd_rsp_id,
  input  logic [NUM_CH-1:0]                      mb_rd_rsp_last,
  input  logic                                   cfg_dual_en,
  output logic [NUM_CH-1:0]                      m_wr_req_valid,
  input  logic [NUM_CH-1:0]                      m_wr_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_wr_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_wr_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_req_id,
  output logic [NUM_CH-1:0]                      m_wr_dat_valid,
  input  logic [NUM_CH-1:0]                      m_wr_dat_ready,
  output logic [NUM_CH-1:0][BEAT_BITS-1:0]       m_wr_dat_data,
  output logic [NUM_CH-1:0][BEAT_BITS/32-1:0]    m_wr_dat_mask,
  output logic [NUM_CH-1:0]                      m_wr_dat_last,
  input  logic [NUM_CH-1:0]                      m_wr_rsp_valid,
  output logic [NUM_CH-1:0]                      m_wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_rsp_id,

  input  logic                                   plat_err_i,
  output logic                                   host_irq_o
);

  // written by the bench
  logic [NUM_CH-1:0]                      t_a_req_valid, t_b_req_valid, t_a_rsp_ready, t_b_rsp_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     t_a_req_addr, t_b_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]       t_a_req_len, t_b_req_len;
  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    t_a_req_id, t_b_req_id;

  initial begin
    t_a_req_valid = '0;
    t_b_req_valid = '0;
    t_a_rsp_ready = '0;
    t_b_rsp_ready = '0;
    t_a_req_addr  = '0;
    t_b_req_addr  = '0;
    t_a_req_len   = '0;
    t_b_req_len   = '0;
    t_a_req_id    = '0;
    t_b_req_id    = '0;
  end

  assign m_rd_req_valid  = t_a_req_valid;
  assign m_rd_req_addr   = t_a_req_addr;
  assign m_rd_req_len    = t_a_req_len;
  assign m_rd_req_id     = t_a_req_id;
  assign m_rd_rsp_ready  = t_a_rsp_ready;
  assign mb_rd_req_valid = t_b_req_valid;
  assign mb_rd_req_addr  = t_b_req_addr;
  assign mb_rd_req_len   = t_b_req_len;
  assign mb_rd_req_id    = t_b_req_id;
  assign mb_rd_rsp_ready = t_b_rsp_ready;

  assign m_wr_req_valid = '0;
  assign m_wr_req_addr  = '0;
  assign m_wr_req_len   = '0;
  assign m_wr_req_id    = '0;
  assign m_wr_dat_valid = '0;
  assign m_wr_dat_data  = '0;
  assign m_wr_dat_mask  = '0;
  assign m_wr_dat_last  = '0;
  assign m_wr_rsp_ready = '1;

  assign csr_ready  = 1'b1;
  assign csr_rdata  = 32'd0;
  assign host_irq_o = 1'b0;

  wire unused_stub = &{1'b0, clk, rst_n, clk_rv, rst_rv_n, csr_valid, csr_we, csr_addr, csr_wdata, m_rd_req_ready,
                       m_rd_rsp_valid, m_rd_rsp_data, m_rd_rsp_id, m_rd_rsp_last, mb_rd_req_ready, mb_rd_rsp_valid,
                       mb_rd_rsp_data, mb_rd_rsp_id, mb_rd_rsp_last, cfg_dual_en, m_wr_req_ready, m_wr_dat_ready,
                       m_wr_rsp_valid, m_wr_rsp_id, plat_err_i};

endmodule
