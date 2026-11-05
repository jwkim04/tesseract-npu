// tsr_tb_at_dma.sv — testbench-only toplevel for tb/at (KVST, not part of the core).
//
// tsr_chk_at (the AT checker class: instruction words -> code / info / extents / payload) and
// tsr_dma with the KVST engine (HAS_KVST = 1), the DMA on read / write source TSR_SRC_DMA of one
// real tsr_mem_arb per channel, wired like tsr_core_dp; the other read sources are tied off.
//   - chk_*: the checker's class port (the test holds chk_req_valid until chk_rsp_valid).
//   - cmd_*: the DMA's dispatch port (the test drives a KVST with cmd_xc = 1 and xcmd = the
//     checker's payload, or a plain LD / ST with cmd_xc = 0).
//   - spm_*: the DMA's SPM client port (a Python SPM model, tb/at/at_tb.py SpmPort).
//   - m_*  : the channel memory ports (tsr_core naming, common.mem_model.MemModel).

module tsr_tb_at_dma
  import tsr_pkg::*;
#(
  parameter  int unsigned     NUM_CH         = 4,
  parameter  int unsigned     T_MAX          = 8,
  parameter  int unsigned     K_MAX          = 32768,
  parameter  int unsigned     SPM_BYTES      = 262144,
  parameter  longint unsigned CH_BYTES       = 64'd17179869184,
  parameter  int unsigned     SPM_RD_LAT     = 2,
  parameter  int unsigned     ADDR_BEAT_W    = 28,
  parameter  int unsigned     RD_OUTSTANDING = 16,
  localparam int unsigned     BB             = 512,
  localparam int unsigned     AW             = tsr_spm_aw(SPM_BYTES / TSR_LINE_BYTES)
) (
  input  logic                                clk,
  input  logic                                rst_n,

  // ---- checker class ----
  input  logic                                chk_req_valid,
  input  tsr_insn_t                           chk_req_insn,
  output logic                                chk_rsp_valid,
  output logic [3:0]                          chk_rsp_code,
  output logic [31:0]                         chk_rsp_info,
  output tsr_ext_t [TSR_EXT_N-1:0]            chk_rsp_ext,
  output logic [255:0]                        chk_rsp_cmd,

  // ---- DMA dispatch ----
  input  logic                                cmd_valid,
  output logic                                cmd_ready,
  input  tsr_dma_cmd_t                        cmd,
  input  logic                                cmd_xc,
  input  logic [255:0]                        xcmd,
  output logic                                done,
  output logic                                perf_kvst_rmw,

  // ---- SPM client (Python model) ----
  output logic                                spm_rd_req_valid,
  input  logic                                spm_rd_req_ready,
  output logic [AW-1:0]                       spm_rd_req_line,
  input  logic                                spm_rd_rsp_valid,
  input  logic [TSR_LINE_BITS-1:0]            spm_rd_rsp_data,
  output logic                                spm_wr_valid,
  input  logic                                spm_wr_ready,
  output logic [AW-1:0]                       spm_wr_line,
  output logic [TSR_LINE_BITS-1:0]            spm_wr_data,
  output logic [TSR_SPM_BE_W-1:0]             spm_wr_be,

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
  output logic [NUM_CH-1:0][BB/32-1:0]          m_wr_dat_mask,
  output logic [NUM_CH-1:0]                   m_wr_dat_last,
  input  logic [NUM_CH-1:0]                   m_wr_rsp_valid,
  output logic [NUM_CH-1:0]                   m_wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0] m_wr_rsp_id
);

  tsr_chk_at #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CH_BYTES(CH_BYTES),
    .ADDR_BEAT_W(ADDR_BEAT_W), .XC_CMD_W(256)
  ) u_chk (
    .clk, .rst_n,
    .req_valid(chk_req_valid), .req_insn(chk_req_insn),
    .rsp_valid(chk_rsp_valid), .rsp_code(chk_rsp_code), .rsp_info(chk_rsp_info), .rsp_ext(chk_rsp_ext),
    .rsp_cmd(chk_rsp_cmd)
  );

  // ---- DMA ----
  logic [NUM_CH-1:0][BB/32-1:0]         d_wr_dat_mask;
  logic [NUM_CH-1:0]                   d_rd_req_valid, d_rd_req_ready, d_rd_rsp_valid, d_rd_rsp_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  d_rd_req_addr, d_wr_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    d_rd_req_len, d_wr_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_rd_req_id, d_wr_req_id, s_rd_rsp_id, d_wr_rsp_id;
  logic [NUM_CH-1:0][BB-1:0]           s_rd_rsp_data, d_wr_dat_data;
  logic [NUM_CH-1:0]                   s_rd_rsp_last;
  logic [NUM_CH-1:0]                   d_wr_req_valid, d_wr_req_ready, d_wr_dat_valid, d_wr_dat_ready;
  logic [NUM_CH-1:0]                   d_wr_dat_last, d_wr_rsp_valid, d_wr_rsp_ready;

  tsr_dma #(
    .NUM_CH(NUM_CH), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .SPM_BYTES(SPM_BYTES),
    .SPM_RD_LAT(SPM_RD_LAT), .HAS_KVST(1'b1), .XC_CMD_W(256)
  ) u_dma (
    .clk, .rst_n,
    .cmd_valid, .cmd_ready, .cmd, .cmd_xc, .xcmd, .done,
    .rd_req_valid(d_rd_req_valid), .rd_req_ready(d_rd_req_ready), .rd_req_addr(d_rd_req_addr),
    .rd_req_len(d_rd_req_len), .rd_req_id(d_rd_req_id),
    .rd_rsp_valid(d_rd_rsp_valid), .rd_rsp_ready(d_rd_rsp_ready), .rd_rsp_data(s_rd_rsp_data),
    .rd_rsp_last(s_rd_rsp_last), .rd_rsp_id(s_rd_rsp_id),
    .wr_req_valid(d_wr_req_valid), .wr_req_ready(d_wr_req_ready), .wr_req_addr(d_wr_req_addr),
    .wr_req_len(d_wr_req_len), .wr_req_id(d_wr_req_id),
    .wr_dat_valid(d_wr_dat_valid), .wr_dat_ready(d_wr_dat_ready), .wr_dat_data(d_wr_dat_data),
    .wr_dat_mask(d_wr_dat_mask), .wr_dat_last(d_wr_dat_last),
    .wr_rsp_valid(d_wr_rsp_valid), .wr_rsp_ready(d_wr_rsp_ready), .wr_rsp_id(d_wr_rsp_id),
    .spm_rd_req_valid, .spm_rd_req_ready, .spm_rd_req_line, .spm_rd_rsp_valid, .spm_rd_rsp_data,
    .spm_wr_valid, .spm_wr_ready, .spm_wr_line, .spm_wr_data, .spm_wr_be,
    .perf_kvst_rmw
  );

  // ---- one arbiter per channel: source DMA live, AUX and WS tied off ----
  for (genvar c = 0; c < NUM_CH; c++) begin : g_ch
    logic [TSR_NSRC-1:0]                   s_rq_v, s_rq_r, s_rs_v, s_rs_r;
    logic [TSR_NSRC-1:0][ADDR_BEAT_W-1:0]  s_rq_a;
    logic [TSR_NSRC-1:0][TSR_LEN_W-1:0]    s_rq_l;
    logic [TSR_NSRC-1:0][TSR_SRC_ID_W-1:0] s_rq_i;
    logic                                  p_rd_beat, p_rd_wait;
    always_comb begin
      s_rq_v = '0;
      s_rq_a = '0;
      s_rq_l = '0;
      s_rq_i = '0;
      s_rs_r = '1;
      s_rq_v[TSR_SRC_DMA] = d_rd_req_valid[c];
      s_rq_a[TSR_SRC_DMA] = d_rd_req_addr[c];
      s_rq_l[TSR_SRC_DMA] = d_rd_req_len[c];
      s_rq_i[TSR_SRC_DMA] = d_rd_req_id[c];
      s_rs_r[TSR_SRC_DMA] = d_rd_rsp_ready[c];
    end
    assign d_rd_req_ready[c] = s_rq_r[TSR_SRC_DMA];
    assign d_rd_rsp_valid[c] = s_rs_v[TSR_SRC_DMA];
    tsr_mem_arb #(.ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .RD_OUTSTANDING(RD_OUTSTANDING)) u_arb (
      .clk, .rst_n,
      .s_rd_req_valid(s_rq_v), .s_rd_req_ready(s_rq_r), .s_rd_req_addr(s_rq_a), .s_rd_req_len(s_rq_l),
      .s_rd_req_id(s_rq_i), .s_rd_rsp_valid(s_rs_v), .s_rd_rsp_ready(s_rs_r), .s_rd_rsp_data(s_rd_rsp_data[c]),
      .s_rd_rsp_last(s_rd_rsp_last[c]), .s_rd_rsp_id(s_rd_rsp_id[c]),
      .s_wr_req_valid(d_wr_req_valid[c]), .s_wr_req_ready(d_wr_req_ready[c]), .s_wr_req_addr(d_wr_req_addr[c]),
      .s_wr_req_len(d_wr_req_len[c]), .s_wr_req_id(d_wr_req_id[c]),
      .s_wr_dat_valid(d_wr_dat_valid[c]), .s_wr_dat_ready(d_wr_dat_ready[c]), .s_wr_dat_data(d_wr_dat_data[c]),
      .s_wr_dat_mask(d_wr_dat_mask[c]), .s_wr_dat_last(d_wr_dat_last[c]),
      .s_wr_rsp_valid(d_wr_rsp_valid[c]), .s_wr_rsp_ready(d_wr_rsp_ready[c]), .s_wr_rsp_id(d_wr_rsp_id[c]),
      .m_rd_req_valid(m_rd_req_valid[c]), .m_rd_req_ready(m_rd_req_ready[c]), .m_rd_req_addr(m_rd_req_addr[c]),
      .m_rd_req_len(m_rd_req_len[c]), .m_rd_req_id(m_rd_req_id[c]),
      .m_rd_rsp_valid(m_rd_rsp_valid[c]), .m_rd_rsp_ready(m_rd_rsp_ready[c]), .m_rd_rsp_data(m_rd_rsp_data[c]),
      .m_rd_rsp_id(m_rd_rsp_id[c]), .m_rd_rsp_last(m_rd_rsp_last[c]),
      .m_wr_req_valid(m_wr_req_valid[c]), .m_wr_req_ready(m_wr_req_ready[c]), .m_wr_req_addr(m_wr_req_addr[c]),
      .m_wr_req_len(m_wr_req_len[c]), .m_wr_req_id(m_wr_req_id[c]),
      .m_wr_dat_valid(m_wr_dat_valid[c]), .m_wr_dat_ready(m_wr_dat_ready[c]), .m_wr_dat_data(m_wr_dat_data[c]),
      .m_wr_dat_mask(m_wr_dat_mask[c]), .m_wr_dat_last(m_wr_dat_last[c]),
      .m_wr_rsp_valid(m_wr_rsp_valid[c]), .m_wr_rsp_ready(m_wr_rsp_ready[c]), .m_wr_rsp_id(m_wr_rsp_id[c]),
      .perf_rd_beat(p_rd_beat), .perf_rd_wait(p_rd_wait)
    );
    wire unused_ch = &{1'b0, s_rs_v[TSR_SRC_AUX], s_rs_v[TSR_SRC_WS], s_rq_r[TSR_SRC_AUX], s_rq_r[TSR_SRC_WS],
                       p_rd_beat, p_rd_wait};
  end

endmodule
