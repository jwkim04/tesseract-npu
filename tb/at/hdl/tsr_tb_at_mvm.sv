// tsr_tb_at_mvm.sv — testbench-only toplevel for tb/at (ATTQK / ATTPV, not part of the core).
//
// tsr_chk_at (the AT checker class) and tsr_mvm with the attention modes (HAS_ATT = 1), the weight
// streams on read source TSR_SRC_WS of one real tsr_mem_arb per channel (AUX / DMA and the write
// source tied off), wired like tsr_core_dp.
//   - chk_*: the checker's class port (the test holds chk_req_valid until chk_rsp_valid).
//   - cmd_*: the MVM unit's dispatch port (ATTQK / ATTPV: cmd_xc = 1, xcmd = the checker's
//     payload; a plain MVM: cmd_xc = 0 and cmd).
//   - spm_*: the MVM unit's SPM client port (a Python SPM model, tb/at/at_tb.py SpmPort).
//   - m_*  : the channel memory ports (tsr_core naming, common.mem_model.MemModel).

module tsr_tb_at_mvm
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
  parameter  int unsigned     BC_PIPE        = 1,
  parameter  int unsigned     RES_PIPE       = 1,
  parameter  int unsigned     ACT_RD_LAT     = 2,
  parameter  int unsigned     MAC_LAT        = 3,
  parameter  int unsigned     MUL_LAT        = 4,
  parameter  int unsigned     ADD_LAT        = 5,
  parameter  int unsigned     WS_FIFO_BEATS  = 512,
  parameter  int unsigned     RST_LOCAL      = 0,
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

  // ---- MVM dispatch ----
  input  logic                                cmd_valid,
  output logic                                cmd_ready,
  input  tsr_mvm_cmd_t                        cmd,
  input  logic                                cmd_xc,
  input  logic [255:0]                        xcmd,
  output logic                                done,
  output logic                                perf_mvm_busy,
  output logic                                perf_att_qk,
  output logic                                perf_att_pv,
  output logic                                perf_att_stall,
  output logic [2:0]                          o_bc_seen,

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

  // ---- MVM unit ----
  logic [NUM_CH-1:0]                   ws_req_valid, ws_req_ready, ws_rsp_valid, ws_rsp_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  ws_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    ws_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] ws_req_id, s_rd_rsp_id;
  logic [NUM_CH-1:0][BB-1:0]           s_rd_rsp_data;
  logic [NUM_CH-1:0]                   s_rd_rsp_last;

  tsr_mvm #(
    .HAS_ATT(1'b1), .XC_CMD_W(256), .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES),
    .SPM_RD_LAT(SPM_RD_LAT), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .WS_FIFO_BEATS(WS_FIFO_BEATS),
    .BC_PIPE(BC_PIPE), .RES_PIPE(RES_PIPE), .ACT_RD_LAT(ACT_RD_LAT), .MAC_LAT(MAC_LAT), .MUL_LAT(MUL_LAT),
    .ADD_LAT(ADD_LAT), .RST_LOCAL(RST_LOCAL)
  ) u_mvm (
    .clk, .rst_n,
    .cmd_valid, .cmd_ready, .cmd, .cmd_xc, .xcmd, .done,
    .spm_rd_req_valid, .spm_rd_req_ready, .spm_rd_req_line, .spm_rd_rsp_valid, .spm_rd_rsp_data,
    .spm_wr_valid, .spm_wr_ready, .spm_wr_line, .spm_wr_data, .spm_wr_be,
    .ws_rd_req_valid(ws_req_valid), .ws_rd_req_ready(ws_req_ready), .ws_rd_req_addr(ws_req_addr),
    .ws_rd_req_len(ws_req_len), .ws_rd_req_id(ws_req_id),
    .ws_rd_rsp_valid(ws_rsp_valid), .ws_rd_rsp_ready(ws_rsp_ready), .ws_rd_rsp_data(s_rd_rsp_data),
    .ws_rd_rsp_last(s_rd_rsp_last), .ws_rd_rsp_id(s_rd_rsp_id),
    .perf_mvm_busy, .perf_att_qk, .perf_att_pv, .perf_att_stall
  );

  // The same independent original-pipe model checks actual ATT command / line mappings.
  localparam int unsigned BCW = $bits(tsr_mvm_bcline_t);
  logic [NUM_CH-1:0][2:0] bc_valid;
  logic [NUM_CH-1:0][BCW-1:0] bc_data;
  wire [NUM_CH-1:0][2:0] unused_bc_reset_discarded;
  initial o_bc_seen = '0;
  always @(posedge clk) if (u_mvm.rst_t_n) o_bc_seen <= o_bc_seen | u_mvm.bc_v_q;
  for (genvar c = 0; c < NUM_CH; c++) begin : g_bc_observe
    assign bc_valid[c] = {u_mvm.g_cl[c].u_cl.bc_acmd_valid,
                          u_mvm.g_cl[c].u_cl.bc_cmd_valid,
                          u_mvm.g_cl[c].u_cl.bc_line_valid};
    assign bc_data[c] = u_mvm.g_cl[c].u_cl.bc_line;
    always @(posedge clk) if (u_mvm.rst_t_n) begin
      if (bc_valid[c][1])
        assert (u_mvm.g_cl[c].u_cl.bc_cmd == bc_data[c][$bits(tsr_mvm_ccmd_t)-1:0])
          else $fatal(1, "%m: MVM command overlay differs");
      if (bc_valid[c][2])
        assert (u_mvm.g_cl[c].u_cl.bc_acmd == bc_data[c][$bits(tsr_att_pkg::tsr_att_ccmd_t)-1:0])
          else $fatal(1, "%m: ATT command overlay differs");
    end
  end
  tsr_mvm_bc_observer #(.NUM_CH(NUM_CH), .DEPTH(BC_PIPE), .RST_LOCAL(RST_LOCAL), .W(BCW)) u_bc_observer (
    .clk, .rst_n, .top_rst_n(u_mvm.rst_t_n), .cluster_rst_n(u_mvm.rst_c_n),
    .source_valid(u_mvm.bc_v_q), .source_data(u_mvm.bc_d_q),
    .cluster_valid(bc_valid), .cluster_data(bc_data), .reset_discarded(unused_bc_reset_discarded)
  );

  // ---- one arbiter per channel: source WS live, AUX / DMA and the writes tied off ----
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
      s_rq_v[TSR_SRC_WS] = ws_req_valid[c];
      s_rq_a[TSR_SRC_WS] = ws_req_addr[c];
      s_rq_l[TSR_SRC_WS] = ws_req_len[c];
      s_rq_i[TSR_SRC_WS] = ws_req_id[c];
      s_rs_r[TSR_SRC_WS] = ws_rsp_ready[c];
    end
    assign ws_req_ready[c] = s_rq_r[TSR_SRC_WS];
    assign ws_rsp_valid[c] = s_rs_v[TSR_SRC_WS];
    logic                    wr_req_r, wr_dat_r, wr_rsp_v;
    logic [TSR_SRC_ID_W-1:0] wr_rsp_i;
    tsr_mem_arb #(.ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BB), .RD_OUTSTANDING(RD_OUTSTANDING)) u_arb (
      .clk, .rst_n,
      .s_rd_req_valid(s_rq_v), .s_rd_req_ready(s_rq_r), .s_rd_req_addr(s_rq_a), .s_rd_req_len(s_rq_l),
      .s_rd_req_id(s_rq_i), .s_rd_rsp_valid(s_rs_v), .s_rd_rsp_ready(s_rs_r), .s_rd_rsp_data(s_rd_rsp_data[c]),
      .s_rd_rsp_last(s_rd_rsp_last[c]), .s_rd_rsp_id(s_rd_rsp_id[c]),
      .s_wr_req_valid(1'b0), .s_wr_req_ready(wr_req_r), .s_wr_req_addr('0),
      .s_wr_req_len('0), .s_wr_req_id('0),
      .s_wr_dat_valid(1'b0), .s_wr_dat_ready(wr_dat_r), .s_wr_dat_data('0),
      .s_wr_dat_last(1'b0),
      .s_wr_rsp_valid(wr_rsp_v), .s_wr_rsp_ready(1'b1), .s_wr_rsp_id(wr_rsp_i),
      .m_rd_req_valid(m_rd_req_valid[c]), .m_rd_req_ready(m_rd_req_ready[c]), .m_rd_req_addr(m_rd_req_addr[c]),
      .m_rd_req_len(m_rd_req_len[c]), .m_rd_req_id(m_rd_req_id[c]),
      .m_rd_rsp_valid(m_rd_rsp_valid[c]), .m_rd_rsp_ready(m_rd_rsp_ready[c]), .m_rd_rsp_data(m_rd_rsp_data[c]),
      .m_rd_rsp_id(m_rd_rsp_id[c]), .m_rd_rsp_last(m_rd_rsp_last[c]),
      .m_wr_req_valid(m_wr_req_valid[c]), .m_wr_req_ready(m_wr_req_ready[c]), .m_wr_req_addr(m_wr_req_addr[c]),
      .m_wr_req_len(m_wr_req_len[c]), .m_wr_req_id(m_wr_req_id[c]),
      .m_wr_dat_valid(m_wr_dat_valid[c]), .m_wr_dat_ready(m_wr_dat_ready[c]), .m_wr_dat_data(m_wr_dat_data[c]),
      .m_wr_dat_last(m_wr_dat_last[c]),
      .m_wr_rsp_valid(m_wr_rsp_valid[c]), .m_wr_rsp_ready(m_wr_rsp_ready[c]), .m_wr_rsp_id(m_wr_rsp_id[c]),
      .perf_rd_beat(p_rd_beat), .perf_rd_wait(p_rd_wait)
    );
    wire unused_ch = &{1'b0, s_rs_v[TSR_SRC_AUX], s_rs_v[TSR_SRC_DMA], s_rq_r[TSR_SRC_AUX], s_rq_r[TSR_SRC_DMA],
                       p_rd_beat, p_rd_wait, wr_req_r, wr_dat_r, wr_rsp_v, wr_rsp_i};
  end

endmodule
