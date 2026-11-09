// tsr_tb_loop.sv — testbench-only toplevel for the tb/common self-test (not part of the core).
//
//   - memory ports: the src_* side (driven by a Python master) is wired straight through to the
//     m_* side, which follows the tsr_core boundary naming, so mem_model.MemModel can be exercised
//     with arbitrary (also deliberately illegal) traffic;
//   - struct probes: tsr_pkg structs as inputs, their fields as outputs, plus $bits of each
//     struct, so tb/common/isa.py's layouts are checked against tsr_pkg.sv.

module tsr_tb_loop
  import tsr_pkg::*;
#(
  parameter int unsigned NUM_CH      = 2,
  parameter int unsigned ADDR_BEAT_W = 20,
  parameter int unsigned BEAT_BITS   = 512
) (
  input  logic                                   clk,
  input  logic                                   rst_n,

  // ---- master side (Python) ----
  input  logic [NUM_CH-1:0]                      src_rd_req_valid,
  output logic [NUM_CH-1:0]                      src_rd_req_ready,
  input  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     src_rd_req_addr,
  input  logic [NUM_CH-1:0][TSR_LEN_W-1:0]       src_rd_req_len,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    src_rd_req_id,
  output logic [NUM_CH-1:0]                      src_rd_rsp_valid,
  input  logic [NUM_CH-1:0]                      src_rd_rsp_ready,
  output logic [NUM_CH-1:0][BEAT_BITS-1:0]       src_rd_rsp_data,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    src_rd_rsp_id,
  output logic [NUM_CH-1:0]                      src_rd_rsp_last,
  input  logic [NUM_CH-1:0]                      src_wr_req_valid,
  output logic [NUM_CH-1:0]                      src_wr_req_ready,
  input  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     src_wr_req_addr,
  input  logic [NUM_CH-1:0][TSR_LEN_W-1:0]       src_wr_req_len,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    src_wr_req_id,
  input  logic [NUM_CH-1:0]                      src_wr_dat_valid,
  output logic [NUM_CH-1:0]                      src_wr_dat_ready,
  input  logic [NUM_CH-1:0][BEAT_BITS-1:0]       src_wr_dat_data,
  input  logic [NUM_CH-1:0]                      src_wr_dat_last,
  output logic [NUM_CH-1:0]                      src_wr_rsp_valid,
  input  logic [NUM_CH-1:0]                      src_wr_rsp_ready,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    src_wr_rsp_id,

  // ---- memory side (MemModel), tsr_core naming ----
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
  output logic [NUM_CH-1:0]                      m_wr_req_valid,
  input  logic [NUM_CH-1:0]                      m_wr_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_wr_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_wr_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_req_id,
  output logic [NUM_CH-1:0]                      m_wr_dat_valid,
  input  logic [NUM_CH-1:0]                      m_wr_dat_ready,
  output logic [NUM_CH-1:0][BEAT_BITS-1:0]       m_wr_dat_data,
  output logic [NUM_CH-1:0]                      m_wr_dat_last,
  input  logic [NUM_CH-1:0]                      m_wr_rsp_valid,
  output logic [NUM_CH-1:0]                      m_wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_rsp_id,

  // ---- struct probes ----
  input  tsr_dma_cmd_t                           dma_in,
  output logic                                   dma_is_st,
  output logic [3:0]                             dma_ch,
  output logic [31:0]                            dma_spm_line,
  output logic [31:0]                            dma_dram_beat,
  output logic [31:0]                            dma_n_lines,
  input  tsr_qact_cmd_t                          qact_in,
  output logic                                   qact_kind,
  output logic [31:0]                            qact_dst_line,
  output logic [31:0]                            qact_src_line,
  output logic [15:0]                            qact_k,
  output logic [3:0]                             qact_t,
  output logic [31:0]                            qact_src_stride,
  output logic [15:0]                            qact_qa_lines,
  input  tsr_mvm_cmd_t                           mvm_in,
  output logic [31:0]                            mvm_n,
  output logic [31:0]                            mvm_row_beats,
  output logic [31:0]                            bits_dma_cmd,
  output logic [31:0]                            bits_qact_cmd,
  output logic [31:0]                            bits_mvm_cmd,
  output logic [31:0]                            qa_lines_a256_27648,
  output logic [31:0]                            row_beats_q4k_27648,
  // ---- MVM struct probes (docs/rtl-mvm.md §3) ----
  input  tsr_mvm_ccmd_t                          ccmd_in,
  output logic [15:0]                            ccmd_k,
  output logic [31:0]                            ccmd_rows_per_ch,
  output logic [15:0]                            ccmd_row_beats,
  output logic [7:0]                             ccmd_d_lines,
  input  tsr_mvm_bcline_t                        bcline_in,
  output logic                                   bcline_last,
  output logic [2:0]                             bcline_tok,
  output logic                                   bcline_dbs,
  output logic [9:0]                             bcline_addr,
  output logic [31:0]                            bcline_data0,
  input  tsr_mvm_fcmd_t                          fcmd_in,
  output logic [15:0]                            fcmd_row_beats,
  output logic [31:0]                            fcmd_rows,
  output logic [39:0]                            fcmd_beats,
  input  tsr_mvm_beat_t                          beat_in,
  output logic                                   beat_last,
  output logic                                   beat_row_last,
  output logic [10:0]                            beat_blk,
  output logic                                   beat_two,
  output logic                                   beat_half,
  output logic [31:0]                            beat_d_a,
  output logic [31:0]                            beat_d_b,
  output logic [5:0]                             beat_sc3,
  output logic [5:0]                             beat_m5,
  output logic [7:0]                             beat_w0,
  output logic [7:0]                             beat_w127,
  output logic [31:0]                            bits_ccmd,
  output logic [31:0]                            bits_bcline,
  output logic [31:0]                            bits_fcmd,
  output logic [31:0]                            bits_beat,
  output logic [31:0]                            actbuf_qs_words_32768,
  output logic [31:0]                            actbuf_dbs_lines_32768,
  output logic [31:0]                            actbuf_dbs_lines_256,
  output logic [31:0]                            rows_c_probe
);

  // Memory passthrough.
  assign m_rd_req_valid   = src_rd_req_valid;
  assign src_rd_req_ready = m_rd_req_ready;
  assign m_rd_req_addr    = src_rd_req_addr;
  assign m_rd_req_len     = src_rd_req_len;
  assign m_rd_req_id      = src_rd_req_id;
  assign src_rd_rsp_valid = m_rd_rsp_valid;
  assign m_rd_rsp_ready   = src_rd_rsp_ready;
  assign src_rd_rsp_data  = m_rd_rsp_data;
  assign src_rd_rsp_id    = m_rd_rsp_id;
  assign src_rd_rsp_last  = m_rd_rsp_last;
  assign m_wr_req_valid   = src_wr_req_valid;
  assign src_wr_req_ready = m_wr_req_ready;
  assign m_wr_req_addr    = src_wr_req_addr;
  assign m_wr_req_len     = src_wr_req_len;
  assign m_wr_req_id      = src_wr_req_id;
  assign m_wr_dat_valid   = src_wr_dat_valid;
  assign src_wr_dat_ready = m_wr_dat_ready;
  assign m_wr_dat_data    = src_wr_dat_data;
  assign m_wr_dat_last    = src_wr_dat_last;
  assign src_wr_rsp_valid = m_wr_rsp_valid;
  assign m_wr_rsp_ready   = src_wr_rsp_ready;
  assign src_wr_rsp_id    = m_wr_rsp_id;

  // Struct probes.
  assign dma_is_st       = dma_in.is_st;
  assign dma_ch          = dma_in.ch;
  assign dma_spm_line    = dma_in.spm_line;
  assign dma_dram_beat   = dma_in.dram_beat;
  assign dma_n_lines     = dma_in.n_lines;
  assign qact_kind       = qact_in.kind;
  assign qact_dst_line   = qact_in.dst_line;
  assign qact_src_line   = qact_in.src_line;
  assign qact_k          = qact_in.k;
  assign qact_t          = qact_in.t;
  assign qact_src_stride = qact_in.src_stride;
  assign qact_qa_lines   = qact_in.qa_lines;
  assign mvm_n           = mvm_in.n;
  assign mvm_row_beats   = mvm_in.row_beats;
  assign bits_dma_cmd    = 32'($bits(tsr_dma_cmd_t));
  assign bits_qact_cmd   = 32'($bits(tsr_qact_cmd_t));
  assign bits_mvm_cmd    = 32'($bits(tsr_mvm_cmd_t));
  assign qa_lines_a256_27648 = tsr_qa_lines(1'b1, 32'd27648);
  assign row_beats_q4k_27648 = tsr_row_beats(1'b1, 32'd27648);

  assign ccmd_k           = ccmd_in.k;
  assign ccmd_rows_per_ch = ccmd_in.rows_per_ch;
  assign ccmd_row_beats   = ccmd_in.row_beats;
  assign ccmd_d_lines     = ccmd_in.d_lines;
  assign bcline_last      = bcline_in.last;
  assign bcline_tok       = bcline_in.tok;
  assign bcline_dbs       = bcline_in.dbs;
  assign bcline_addr      = bcline_in.addr;
  assign bcline_data0     = bcline_in.data[31:0];
  assign fcmd_row_beats   = fcmd_in.row_beats;
  assign fcmd_rows        = fcmd_in.rows;
  assign fcmd_beats       = fcmd_in.beats;
  assign beat_last        = beat_in.last;
  assign beat_row_last    = beat_in.row_last;
  assign beat_blk         = beat_in.blk;
  assign beat_two         = beat_in.two;
  assign beat_half        = beat_in.half;
  assign beat_d_a         = beat_in.d_a;
  assign beat_d_b         = beat_in.d_b;
  assign beat_sc3         = beat_in.sc[3];
  assign beat_m5          = beat_in.m[5];
  assign beat_w0          = beat_in.w[0];
  assign beat_w127        = beat_in.w[127];
  assign bits_ccmd        = 32'($bits(tsr_mvm_ccmd_t));
  assign bits_bcline      = 32'($bits(tsr_mvm_bcline_t));
  assign bits_fcmd        = 32'($bits(tsr_mvm_fcmd_t));
  assign bits_beat        = 32'($bits(tsr_mvm_beat_t));
  assign actbuf_qs_words_32768  = 32'(tsr_actbuf_qs_words(32768));
  assign actbuf_dbs_lines_32768 = 32'(tsr_actbuf_dbs_lines(32768));
  assign actbuf_dbs_lines_256   = 32'(tsr_actbuf_dbs_lines(256));
  // rows of channel 3 for N = 40, R = 10 (first = 30): 10; driven through the package function
  assign rows_c_probe     = tsr_mvm_rows_c(ccmd_in.n, ccmd_in.rows_per_ch, 64'(ccmd_in.d_lines) * 64'(ccmd_in.rows_per_ch));

  wire unused_ok = &{1'b0, clk, rst_n, mvm_in.fmt, mvm_in.acc, mvm_in.dst_line, mvm_in.x_line,
                     ccmd_in.fmt, ccmd_in.w_off, bcline_in.data[511:32], fcmd_in.fmt, fcmd_in.k,
                     fcmd_in.w_off, beat_in.sc[2:0], beat_in.m[7:6], beat_in.m[4:0], beat_in.w[126:1],
                     mvm_in.w_off, mvm_in.k, mvm_in.t, mvm_in.dst_stride, mvm_in.y_lines,
                     mvm_in.rows_per_ch, mvm_in.qa_lines, mvm_in.local_scale, mvm_in.local_softcap,
                     mvm_in.scale, mvm_in.post_scale};

endmodule
