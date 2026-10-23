// tsr_tb_mvm_back.sv — testbench-only toplevel for tb/mvmback (not part of the core).
//
// The cluster's back half as in tsr_mvm_cluster (docs/rtl-m3.md §2, §3.4): tsr_mvm_actbuf written
// through its broadcast-side port (wr_valid / wr_line = tsr_mvm_bcline_t items, as the broadcast
// pipe delivers them) and read by tsr_mvm_back. The tagged beats, cfg_fmt / cfg_d_lines and the
// result stream are exposed for the cocotb test.
//   - tb_rd = 1 hands the activation buffer's read ports to the testbench (tb_qs_* / tb_dbs_*):
//     used only while the back end is idle, to compare the buffer contents with
//     mvm_model.actbuf_image and to check the read latency.
//   - qs_data / dbs_data and the back end's read requests are observation copies.

module tsr_tb_mvm_back
  import tsr_pkg::*;
#(
  parameter  int unsigned K_MAX      = 4096,
  parameter  int unsigned T_MAX      = 2,
  parameter  int unsigned ACT_RD_LAT = TSR_DEF_ACT_RD_LAT,
  parameter  int unsigned MAC_LAT    = TSR_DEF_MAC_LAT,
  parameter  int unsigned MUL_LAT    = TSR_DEF_FMUL_LAT,
  parameter  int unsigned ADD_LAT    = TSR_DEF_FADD_LAT,
  localparam int unsigned QS_WORDS   = tsr_actbuf_qs_words(K_MAX),
  localparam int unsigned DBS_LINES  = tsr_actbuf_dbs_lines(K_MAX),
  localparam int unsigned QS_AW      = (QS_WORDS > 1) ? $clog2(QS_WORDS) : 1,
  localparam int unsigned DBS_AW     = (DBS_LINES > 1) ? $clog2(DBS_LINES) : 1
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // ---- broadcast side of the activation buffer ----
  input  logic                          wr_valid,
  input  tsr_mvm_bcline_t               wr_line,

  // ---- back end ----
  input  logic                          cfg_fmt,
  input  logic [7:0]                    cfg_d_lines,
  input  logic                          beat_valid,
  output logic                          beat_ready,
  input  tsr_mvm_beat_t                 beat,
  output logic                          res_valid,
  input  logic                          res_ready,
  output logic [T_MAX-1:0][31:0]        res_y,
  output logic                          idle,

  // ---- test read port / observation ----
  input  logic                          tb_rd,
  input  logic                          tb_qs_re,
  input  logic [QS_AW-1:0]              tb_qs_addr,
  input  logic                          tb_dbs_re,
  input  logic [DBS_AW-1:0]             tb_dbs_addr,
  output logic [T_MAX-1:0][1023:0]      qs_data,
  output logic [T_MAX-1:0][511:0]       dbs_data,
  output logic                          be_qs_re,
  output logic [QS_AW-1:0]              be_qs_addr,
  output logic                          be_dbs_re,
  output logic [DBS_AW-1:0]             be_dbs_addr
);

  logic              qs_re, dbs_re;
  logic [QS_AW-1:0]  qs_addr;
  logic [DBS_AW-1:0] dbs_addr;

  assign qs_re    = tb_rd ? tb_qs_re    : be_qs_re;
  assign qs_addr  = tb_rd ? tb_qs_addr  : be_qs_addr;
  assign dbs_re   = tb_rd ? tb_dbs_re   : be_dbs_re;
  assign dbs_addr = tb_rd ? tb_dbs_addr : be_dbs_addr;

  tsr_mvm_actbuf #(.K_MAX(K_MAX), .T_MAX(T_MAX), .RD_LAT(ACT_RD_LAT)) u_act (
    .clk, .rst_n,
    .wr_valid, .wr_line,
    .qs_re, .qs_addr, .qs_data,
    .dbs_re, .dbs_addr, .dbs_data
  );

  tsr_mvm_back #(
    .K_MAX(K_MAX), .T_MAX(T_MAX), .ACT_RD_LAT(ACT_RD_LAT), .MAC_LAT(MAC_LAT), .MUL_LAT(MUL_LAT),
    .ADD_LAT(ADD_LAT)
  ) u_back (
    .clk, .rst_n,
    .cfg_fmt, .cfg_d_lines,
    .beat_valid, .beat_ready, .beat,
    .qs_re(be_qs_re), .qs_addr(be_qs_addr), .qs_data,
    .dbs_re(be_dbs_re), .dbs_addr(be_dbs_addr), .dbs_data,
    .res_valid, .res_ready, .res_y,
    .idle
  );

endmodule
