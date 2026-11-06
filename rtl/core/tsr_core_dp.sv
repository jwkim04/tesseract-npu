// tsr_core_dp.sv — the core-clock datapath of the Tesseract core (docs/design/04-control-core.md "CC"
// §2.2, §13.2): the issue path (tsr_issue with tsr_chk), the host CSR block tsr_csr, the units
// (DMA, QACT, MVM), the SPM (2R2W even / odd banks) with its client adapters and the RISC-V SPM
// bridge, the per-channel memory arbiters, the reset distribution and the integration checks.
//
// Owner: integration (W4-A, docs/design/README.md §10 items 1-3; W4-B wiring of VU, AT and DB). tsr_core instantiates it next to
// the control subsystem tsr_ctrl (rtl/ctrl); the op-list replay bench (CC §15.3) instantiates it
// alone and drives the issue-entry stream (a simulation port only, never in a bitstream: D15).
//
// Boundary:
//   - host CSR port and one memory port per channel (spec Core boundary interface; tsr_core passes
//     them through unchanged; every memory-port output is a tsr_mem_arb register).
//   - the core-domain side of tsr_ctrl (tsr_ctrl.sv header, CC §11.2), with tsr_ctrl's port names
//     minus the _i / _o suffix: issue FIFO (ifq_valid / ifq_ready / ifq_entry, ifq_count),
//     ISSUED_SEQ, S[u] / D[u], the per-unit error-record handshake (err_valid / err_rec / err_ack,
//     tsr_ctrl_pkg::err_rec_t layout; rec_resync: send every record once more), the SPM bridge (spm_req_* and spm_rsp_*), the
//     core window, the LQ / LCQ port, log words, RT mirrors, loader, DMI, MTIME, the clk_rv
//     counters and pulses, and the levels both ways (q_hold = {debug-halt, RISC-V, host, any}).
//   - the DB block's crossings to the clk_rv side (tsr_core's tsr_dbg_rv): rec_*, smp_*, wpa_*, wph_*,
//     dfault, flags; dbg_stop / dbg_halt_tgl / dbg_cancel_tgl / dbg_halt_on_boot into tsr_ctrl, rv_halt_pend
//     from it.
//   - rst_ctrl_n: the reset copy of the control subsystem (its core-domain flops and the input of
//     its clk_rv reset synchroniser), RST_PIPE cycles behind rst_n like every block here.
//   - plat_err_i, host_irq_o (EX §9.3).
// Units and queues (CC §7.4): DMA (LD / ST, with KVST when HAS_ATT), QACT, MVM (HAS_MVM = 1; ATTQK /
//   ATTPV when HAS_ATT) and the vector unit VU (HAS_VU, unit 3: VOP / RED / ROPE / SOP and AT's vector
//   ops). The attention unit has no queue of its own (unit 4 is reserved): AT's ops run on the DMA, MVM and
//   VU queues, so QD_ATT is 0. CAPS follows the parameters (VU / ATT / TOPK / DQROW bits of the tables;
//   TOPK and DQROW are M8 RTL, VU-14 / VU-15, and stay off in the U200 M7 build). The external checker
//   classes of tsr_chk (xc_*): tsr_vu_xc (the VU class tsr_vu_chk, routing AT's classes) with tsr_chk_at
//   (KVST, ATTQK, ATTPV). The DB block (HAS_DB; tsr_dbg here, its clk_rv part tsr_dbg_rv in tsr_core,
//   the trace writer's write mux on channel TRW_CH): DBG registers, stop logic, history, trace writer,
//   DMI gate, dbg_hold / dbg_step into tsr_issue (DB §13). HAS_DB = 0: the DBG block reads 0 (DBG_ID = 0),
//   DMI_EN = 0, dbg_hold / dbg_step 0, no extension PERF slots of the DB.
// SPM clients (CC §10.2): 0 DMA, 1 QACT, 2 MVM, 3 RISC-V bridge through tsr_spm_cport each; 4 VU on
//   both bank ports directly; 5 AT tied off (AT's ops use the DMA and MVM clients).
// Parameters: tsr_core's (passed through) and the queue depths QD_DMA / QD_QACT / QD_MVM / QD_SYNC
//   (tables TSR_QDEPTH_U200_*; the MVM queue is 0 when HAS_MVM = 0), ITCM_BYTES / DTCM_BYTES
//   (RV_CFG), CTL_ID.
// Reset distribution (RST_PIPE, as before in tsr_core; docs/rtl-m3.md §14): every flop of the core
//   domain, the control subsystem's included, sees rst_n exactly RST_PIPE cycles late.
// Integration checks (simulation only, end of this file): CC §9.4 I4 per unit, and the unit-local
//   rules of docs/rtl-m2.md §3.0 / rtl-m3.md §3.0, §4, §6.4; tsr_issue asserts I1-I3 itself.

module tsr_core_dp
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
  parameter int unsigned     SPM_RD_LAT     = TSR_DEF_SPM_RD_LAT,
  parameter int unsigned     MVM_WS_FIFO    = TSR_DEF_WS_FIFO_BEATS,
  parameter int unsigned     MVM_BC_PIPE    = TSR_DEF_MVM_BC_PIPE,
  parameter bit              DDR_LOCAL_PIPE = 1'b0,
  parameter bit              RV_LOCAL_TRANSPORT = 1'b0,
  parameter bit              LOCAL_SOFTCAP = 1'b0,
  parameter bit              LOCAL_SCALE = 1'b0,
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
  // issue path and control (CC §7.4, §13.1)
  parameter int unsigned     QD_DMA         = tsr_tables_pkg::TSR_QDEPTH_U200_DMA,
  parameter int unsigned     QD_QACT        = tsr_tables_pkg::TSR_QDEPTH_U200_QACT,
  parameter int unsigned     QD_MVM         = tsr_tables_pkg::TSR_QDEPTH_U200_MVM,
  parameter int unsigned     QD_SYNC        = tsr_tables_pkg::TSR_QDEPTH_U200_SYNC,
  parameter int unsigned     ITCM_BYTES     = tsr_tables_pkg::TSR_ITCM_BYTES,
  parameter int unsigned     DTCM_BYTES     = tsr_tables_pkg::TSR_DTCM_BYTES,
  parameter logic [31:0]     CTL_ID         = 32'h0000_5D01,
  // W4-B: the vector unit (unit 3, queue depth QD_VU), attention (KVST on the DMA, ATTQK / ATTPV on the MVM
  // unit, AT's vector ops on the VU; needs HAS_VU and HAS_MVM), TOPK / DQROW on the VU (M8 RTL: off in the
  // U200 M7 build), the DB block; the VU's lane knobs (VU §7.10: 16 / 8 on k4, 8 / 4 on k1)
  parameter bit              HAS_VU         = 1'b1,
  parameter bit              HAS_ATT        = HAS_MVM,           // needs HAS_VU, HAS_MVM and MVM_MAC_LAT >= 3
  parameter bit              HAS_TOPK       = 1'b0,
  parameter bit              HAS_DQROW      = 1'b0,
  parameter bit              HAS_DB         = 1'b1,
  parameter int unsigned     QD_VU          = tsr_tables_pkg::TSR_QDEPTH_U200_VU,
  parameter int unsigned     VU_FRONT_LANES = tsr_tables_pkg::TSR_VU_FRONT_LANES,
  parameter int unsigned     VU_EXP_LANES   = tsr_tables_pkg::TSR_VU_FRONT_LANES,
  parameter int unsigned     VU_BACK_LANES  = tsr_tables_pkg::TSR_VU_BACK_LANES,
  parameter int unsigned     VU_LANE_PIPE   = 0,
  // DB block sizes (DB §10.3): history depth, trace writer, tile breakpoints, RV watchpoints, REC
  parameter int unsigned     HIST_DEPTH     = tsr_tables_pkg::TSR_HIST_DEPTH,
  parameter bit              HAS_TRW        = 1'b1,
  parameter int unsigned     NTBP           = tsr_tables_pkg::TSR_DBG_NTBP,
  parameter int unsigned     NRVWP          = tsr_tables_pkg::TSR_DBG_NRVWP,
  parameter bit              HAS_REC        = 1'b1
) (
  input  logic                                   clk,
  input  logic                                   rst_n,

  // ---- host CSR port ----
  input  logic                                   csr_valid,
  output logic                                   csr_ready,
  input  logic                                   csr_we,
  input  logic [TSR_CSR_AW-1:0]                  csr_addr,
  input  logic [31:0]                            csr_wdata,
  output logic [31:0]                            csr_rdata,

  // ---- memory ports, one per channel ----
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
  output logic [NUM_CH-1:0][BEAT_BITS/32-1:0]       m_wr_dat_mask,
  output logic [NUM_CH-1:0]                      m_wr_dat_last,
  input  logic [NUM_CH-1:0]                      m_wr_rsp_valid,
  output logic [NUM_CH-1:0]                      m_wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_rsp_id,

  // ---- control subsystem (tsr_ctrl, core-domain side) ----
  output logic                                   rst_ctrl_n,
  // issue FIFO, sequences, error records
  input  logic                                   ifq_valid,
  output logic                                   ifq_ready,
  input  logic [tsr_tables_pkg::TSR_IFIFO_BITS-1:0] ifq_entry,
  input  logic [5:0]                             ifq_count,
  input  logic [5:0][TSR_SEQ_W-1:0]              issued_seq,
  output logic [5:0][TSR_SEQ_W-1:0]              s_seq,
  output logic [5:0][TSR_SEQ_W-1:0]              d_seq,
  output logic [5:0]                             err_valid,
  output logic [5:0][127:0]                      err_rec,
  input  logic [5:0]                             err_ack,
  input  logic                                   rec_resync,
  output logic [3:0]                             q_hold,
  output logic [5:0]                             q_err,
  output logic [5:0]                             q_nonempty,
  // SPM bridge
  input  logic                                   spm_req_valid,
  input  logic                                   spm_req_we,
  input  logic [25:0]                            spm_req_addr,
  input  logic [3:0]                             spm_req_be,
  input  logic [31:0]                            spm_req_wdata,
  output logic                                   spm_rsp_valid,
  input  logic                                   spm_rsp_ready,
  output logic [TSR_LINE_BITS-1:0]               spm_rsp_line,
  output logic                                   spm_rsp_err,
  // core window, LQ / LCQ, log, RT mirrors
  input  logic                                   cwin_req_valid,
  input  logic [11:0]                            cwin_req_off,
  output logic                                   cwin_rsp_valid,
  output logic [31:0]                            cwin_rsp_rdata,
  input  logic                                   lq_req_valid,
  input  logic [11:0]                            lq_req_off,
  input  logic                                   lq_req_we,
  input  logic [31:0]                            lq_req_wdata,
  input  logic [7:0][31:0]                       lq_req_lcq_w,
  output logic                                   lq_rsp_valid,
  output logic [31:0]                            lq_rsp_rdata,
  output logic                                   lq_rsp_fault,
  input  logic                                   log_valid,
  input  logic [31:0]                            log_data,
  output logic                                   log_ready,
  input  logic [31:0]                            rt_state,
  input  logic [31:0]                            rt_info,
  // loader, DMI
  input  logic                                   ld_ready,
  output logic                                   ld_req,
  output logic                                   ld_we,
  output logic [31:0]                            ld_addr,
  output logic [31:0]                            ld_wdata,
  input  logic                                   ld_rsp_valid,
  input  logic [31:0]                            ld_rsp_rdata,
  input  logic                                   ld_rsp_err,
  input  logic                                   ld_rsp_timeout,
  input  logic                                   dmi_ready,
  output logic                                   dmi_req,
  output logic [6:0]                             dmi_addr,
  output logic [1:0]                             dmi_op,
  output logic [31:0]                            dmi_wdata,
  input  logic                                   dmi_rsp_valid,
  input  logic [31:0]                            dmi_rsp_rdata,
  input  logic [1:0]                             dmi_rsp_resp,
  input  logic                                   dmi_rsp_timeout,
  // time, counters, levels
  output logic [63:0]                            mtime,
  input  logic [15:0]                            rv_cyc,
  input  logic [31:0]                            perf_push_stall,
  input  logic [31:0]                            perf_bridge_acc,
  input  logic [31:0]                            perf_bridge_wait,
  input  logic                                   push_timeout,
  input  logic                                   bridge_timeout,
  output logic                                   rv_run,
  output logic [31:0]                            rv_boot,
  output logic                                   irq_lq,
  output logic                                   irq_lcq,
  output logic                                   perf_time_en,
  output logic                                   perf_evt_en,
  input  logic                                   rv_halted,
  input  logic                                   rv_sleep,
  input  logic                                   rv_running,
  input  logic                                   rv_reset_done,
  input  logic                                   rv_clk_ok,
  input  logic                                   rv_ndmreset,
  input  logic                                   rv_dmactive,

  // ---- platform (EX §9.3) ----
  input  logic                                   plat_err_i,
  output logic                                   host_irq_o,

  // ---- DB block (W4-B): tsr_ctrl's DB inputs and the core side of tsr_dbg_rv (tsr_core wires them) ----
  input  logic                                   rv_halt_pend,
  output logic                                   dbg_stop,
  output logic                                   dbg_halt_tgl,
  output logic                                   dbg_cancel_tgl,
  output logic                                   dbg_halt_on_boot,
  output logic                                   rec_wr,
  output logic [127:0]                           rec_wdata,
  input  logic                                   rec_room,
  input  logic                                   rec_all,
  input  logic                                   smp_ready,
  output logic                                   smp_req,
  output logic                                   smp_local,
  input  logic                                   smp_done,
  input  logic                                   smp_timeout,
  input  logic [163:0]                           smp_data,
  input  logic                                   wpa_ready,
  output logic                                   wpa_req,
  output logic                                   wpa_local,
  output logic [135:0]                           wpa_data,
  input  logic                                   wpa_done,
  input  logic                                   wpa_timeout,
  input  logic                                   wph_v,
  input  logic [1:0]                             wph_mask,
  input  logic [31:0]                            wph_addr,
  input  logic                                   wph_halt,
  input  logic                                   dfault,
  output logic [4:0]                             flags
);

  // -------------------------------------------------------------------------------------------
  // Elaboration checks (same limits as tsr_sim_create's cfg_problem(), plus M2 restrictions)
  // -------------------------------------------------------------------------------------------
  if (BEAT_BITS != 512) begin : g_chk_beat
    $fatal(1, "tsr_core_dp: BEAT_BITS must be 512 (a beat is one 64-byte SPM line)");
  end
  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX) begin : g_chk_ch
    $fatal(1, "tsr_core_dp: NUM_CH must be in [1, 16] (4-bit LD/ST channel field)");
  end
  if (T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH) begin : g_chk_t
    $fatal(1, "tsr_core_dp: T_MAX must be in [1, 8]");
  end
  if (K_MAX < 256 || K_MAX > 65280 || (K_MAX % 256) != 0) begin : g_chk_k
    $fatal(1, "tsr_core_dp: K_MAX must be a multiple of 256 in [256, 65280]");
  end
  if (SPM_BYTES < TSR_LINE_BYTES || (SPM_BYTES % TSR_LINE_BYTES) != 0) begin : g_chk_spm
    $fatal(1, "tsr_core_dp: SPM_BYTES must be a positive multiple of 64");
  end
  if (NSREG < 1 || NSREG > 16 || VL < 1 || VL > 255 || LANES < 1 || LANES > 4095) begin : g_chk_cfg
    $fatal(1, "tsr_core_dp: NSREG in [1, 16], VL in [1, 255], LANES in [1, 4095] (CFG field widths)");
  end
  if (CH_BYTES < 64 || (CH_BYTES % 64) != 0 || ADDR_BEAT_W < 1 || ADDR_BEAT_W > 40) begin : g_chk_dram
    $fatal(1, "tsr_core_dp: CH_BYTES must be a positive multiple of 64, ADDR_BEAT_W in [1, 40]");
  end
  if (RD_OUTSTANDING < 1) begin : g_chk_rdo
    $fatal(1, "tsr_core_dp: RD_OUTSTANDING must be >= 1");
  end
  if (HAS_MVM && LANES != TSR_MVM_LANES) begin : g_chk_lanes
    $fatal(1, "tsr_core_dp: HAS_MVM = 1 needs LANES == 128 (= BEAT_BITS / 4, one Q4_K beat; docs/rtl-m3.md §1)");
  end
  if (HAS_MVM && (MVM_WS_FIFO < 64 || (MVM_WS_FIFO & (MVM_WS_FIFO - 1)) != 0)) begin : g_chk_wsfifo
    $fatal(1, "tsr_core_dp: MVM_WS_FIFO must be a power of two >= 64 (docs/rtl-m3.md §4)");
  end
  if (RST_PIPE > 8) begin : g_chk_rst
    $fatal(1, "tsr_core_dp: RST_PIPE must be in [0, 8]");
  end
  if (HAS_MVM && (MVM_ADD_LAT < 1 || MVM_ADD_LAT > 8 || MVM_WS_RD_LAT < 1 || MVM_WS_RD_LAT > 4 ||
                  MVM_ACT_RD_LAT < 1 || MVM_MAC_LAT < 1 || MVM_MUL_LAT < 1)) begin : g_chk_mvmlat
    $fatal(1, "tsr_core_dp: MVM latencies: ADD 1..8, WS_RD 1..4, ACT_RD / MAC / MUL >= 1 (docs/rtl-m3.md §1)");
  end
  if (QD_DMA > TSR_QD_MAX || QD_QACT > TSR_QD_MAX || QD_MVM > TSR_QD_MAX || QD_SYNC < 1 ||
      QD_SYNC > TSR_QD_MAX) begin : g_chk_qd
    $fatal(1, "tsr_core_dp: queue depths in [0, 31], SYNC >= 1 (CC §7.4)");
  end

  if (HAS_ATT && (!HAS_VU || !HAS_MVM)) begin : g_chk_att
    $fatal(1, "tsr_core_dp: HAS_ATT needs HAS_VU (AT's vector ops 0x34-0x37) and HAS_MVM (ATTQK / ATTPV)");
  end
  if (LOCAL_INTERMEDIATES && !HAS_ATT) begin : g_chk_local_att
    $fatal(1, "tsr_core_dp: local attention needs HAS_ATT");
  end
  if (LOCAL_SOFTCAP && !LOCAL_SCALE) begin : g_chk_local_softcap
    $fatal(1, "tsr_core_dp: local softcap needs the local SCALE execution path");
  end
  if (LOCAL_SCALE && (!HAS_VU || !HAS_MVM || !MVM_GATHER_PAIR)) begin : g_chk_local_scale
    $fatal(1, "tsr_core_dp: local SCALE needs both compute units and paired gather");
  end
  if (HAS_ATT && MVM_MAC_LAT < 3) begin : g_chk_attlat
    $fatal(1, "tsr_core_dp: HAS_ATT needs MVM_MAC_LAT >= 3 (the exact PV accumulation of tsr_mvm_back)");
  end
  if ((HAS_TOPK || HAS_DQROW) && !HAS_VU) begin : g_chk_topk
    $fatal(1, "tsr_core_dp: HAS_TOPK / HAS_DQROW need HAS_VU");
  end
  if (HAS_VU && (QD_VU < 1 || QD_VU > TSR_QD_MAX)) begin : g_chk_qdvu
    $fatal(1, "tsr_core_dp: with HAS_VU the VU queue depth must be in [1, 31]");
  end

  // The capability bits and the depths follow the parameters (tsr_core builds the same values for
  // tsr_ctrl, CC §7.5). The attention unit has no queue (unit 4 reserved): QD_ATT = 0.
  localparam logic [31:0] CAPS      = (HAS_VU    ? tsr_tables_pkg::TSR_CAPS_VU    : 32'd0) |
                                      (HAS_ATT   ? tsr_tables_pkg::TSR_CAPS_ATT   : 32'd0) |
                                      (HAS_TOPK  ? tsr_tables_pkg::TSR_CAPS_TOPK  : 32'd0) |
                                      (HAS_DQROW ? tsr_tables_pkg::TSR_CAPS_DQROW : 32'd0) |
                                      (LOCAL_INTERMEDIATES ? tsr_tables_pkg::TSR_CAPS_ATT_LOCAL : 32'd0) |
                                      (LOCAL_SCALE ? tsr_tables_pkg::TSR_CAPS_MVM_LOCAL_SCALE : 32'd0) |
                                     (LOCAL_SOFTCAP ? tsr_tables_pkg::TSR_CAPS_MVM_LOCAL_SOFTCAP : 32'd0);
  // CFG3 (EX §7.5, VU §5.10, AT §5.10): the VU's lane counts (information), PCAP / 16 with attention
  localparam logic [31:0] CFG3      = (HAS_VU  ? ((32'(VU_EXP_LANES) & 32'h1F) | ((32'(VU_BACK_LANES) & 32'h1F) << 5)) : 32'd0) |
                                      (HAS_ATT ? (tsr_qa_lines(1'b1, 32'(K_MAX)) << 16) : 32'd0);
  localparam int unsigned QD_MVM_E  = HAS_MVM ? QD_MVM : 0;
  localparam int unsigned QD_VU_E   = HAS_VU ? QD_VU : 0;
  localparam int unsigned QD_ATT    = 0;
  localparam int unsigned XC_CMD_W  = 256;
  localparam int unsigned PERF_X    = 5 + 2 * NUM_CH;                       // PERF extension index X (EX §6)
  localparam int unsigned SPM_LINES = SPM_BYTES / TSR_LINE_BYTES;
  localparam int unsigned SPM_AW    = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1;
  localparam int unsigned SPM_WAW   = SPM_AW + 4;                    // SPM word address bits

  // -------------------------------------------------------------------------------------------
  // Reset distribution (RST_PIPE, header): rst_pre_n after the RST_SH shared stages, then
  // RST_BLK (<= 2) stages per block: rst_b_n[i]; tsr_mvm makes its own RST_BLK stages per part
  // (RST_LOCAL). Two per-block stages let the first copy sit at the SLR boundary and the second
  // (replicated by phys_opt) next to the loads. The stage registers start at 0 (initial block:
  // INIT 0) and load in a plain always: IEEE 1800 9.2.2.4 forbids a second writer of an always_ff
  // variable (XSim rejects the initial block then; Verilator and Vivado synthesis accept it).
  // -------------------------------------------------------------------------------------------
  localparam int unsigned RB_CSR = 0, RB_DMA = 1, RB_QACT = 2, RB_SPM = 3, RB_CHK = 4,
                          RB_ISSUE = 5, RB_BRIDGE = 6, RB_CTRL = 7, RB_VU = 8, RB_DBG = 9,
                          RB_ARB = 10;                                   // RB_ARB + c: arbiter c
  localparam int unsigned NRB     = RB_ARB + NUM_CH;
  localparam int unsigned RST_BLK = (RST_PIPE > 2) ? 2 : RST_PIPE;
  localparam int unsigned RST_SH  = RST_PIPE - RST_BLK;
  logic [NRB-1:0] rst_b_n;
  logic           rst_pre_n;

  if (RST_SH == 0) begin : g_sh0
    assign rst_pre_n = rst_n;
  end else begin : g_sh
    (* shreg_extract = "no" *) logic [RST_SH-1:0] sh_q;      // shared stages 1 .. RST_SH
    initial sh_q = '0;
    always @(posedge clk) begin
      sh_q[0] <= rst_n;
      for (int unsigned k = 1; k < RST_SH; k++) sh_q[k] <= sh_q[k-1];
    end
    assign rst_pre_n = sh_q[RST_SH-1];
  end
  if (RST_BLK == 0) begin : g_rst0
    assign rst_b_n = {NRB{rst_pre_n}};
  end else begin : g_rst
    (* keep = "true" *) logic [RST_BLK-1:0][NRB-1:0] rb_q;   // per block: stages RST_SH+1 ..
    initial rb_q = '0;
    always @(posedge clk) begin
      rb_q[0] <= {NRB{rst_pre_n}};
      for (int unsigned k = 1; k < RST_BLK; k++) rb_q[k] <= rb_q[k-1];
    end
    assign rst_b_n = rb_q[RST_BLK-1];
  end
  assign rst_ctrl_n = rst_b_n[RB_CTRL];

  // -------------------------------------------------------------------------------------------
  // Unit command ports (driven by tsr_issue)
  // -------------------------------------------------------------------------------------------
  logic                 dma_cmd_valid, dma_cmd_ready, dma_done;
  tsr_dma_cmd_t         dma_cmd;
  logic                 qact_cmd_valid, qact_cmd_ready, qact_done;
  tsr_qact_cmd_t        qact_cmd;
  logic                 mvm_cmd_valid, mvm_cmd_ready, mvm_done;
  tsr_mvm_cmd_t         mvm_cmd;
  logic                 perf_mvm_busy;
  logic [NUM_CH-1:0]    perf_ch_rd_beat, perf_ch_wait;
  logic                 vu_cmd_valid, vu_cmd_ready, vu_done;       // unit 3 (HAS_VU = 0: no command ever)
  logic [XC_CMD_W-1:0]  vu_cmd;

  // -------------------------------------------------------------------------------------------
  // Issue path: tsr_issue (dispatcher, tsr_chk, queues, heads, SYNC, records; CC §8-§9) and
  // tsr_csr (EX §5 map, rings, SR sequencer, loader; CC §6, §12)
  // -------------------------------------------------------------------------------------------
  logic                       host_hold, halt_hold, qabort, rv_reset, host_clear, sr_busy;
  tsr_err_rec_t [5:0]         q_err_rec;
  logic [31:0]                q_pidx;
  logic [5:0][8:0]            q_state;
  logic                       err_any, abort_active, qidle, qquiescent;
  // A FIFO head remains pending while its issue segments are in flight.
  // The retained FIFO count also covers the gap between queued entries.
  wire ifq_drained = !RV_LOCAL_TRANSPORT || ifq_count == 6'd0;

  logic [2:0]                 hold_src;
  logic [2:0]                 perf_retired;
  logic                       perf_push, perf_mvm_wait, perf_mvm_empty, perf_hold;
  logic [5:0]                 perf_busy;
  logic [5:0]                 unit_progress;
  // external payloads of tsr_issue (KVST on the DMA, ATTQK / ATTPV on the MVM unit), the unit-4 command
  // port (reserved: depth 0, nothing is ever offered), the external checker port (xc_*) and the DB taps
  logic                       dma_cmd_xc, mvm_cmd_xc, att_cmd_valid, xc_req_valid;
  logic [XC_CMD_W-1:0]        dma_xcmd, mvm_xcmd, att_cmd;
  tsr_insn_t                  xc_req_insn;
  logic                       xc_rsp_valid;
  logic [3:0]                 xc_rsp_code;
  logic [31:0]                xc_rsp_info;
  tsr_ext_t [TSR_EXT_N-1:0]   xc_rsp_ext;
  logic [XC_CMD_W-1:0]        xc_rsp_cmd;
  logic [5:0]                 dbg_hold, dbg_step;
  logic                       tap_enq, tap_enq_rec, tap_mark;
  logic [3:0]                 tap_enq_unit, tap_enq_code;
  logic [TSR_SEQ_W-1:0]       tap_enq_seq;
  logic [31:0]                tap_enq_pidx, tap_enq_info, tap_enq_dep0, tap_enq_dep1, tap_enq_pc;
  logic [1:0]                 tap_enq_kind;
  tsr_insn_t                  tap_enq_w;
  tsr_ext_t [TSR_EXT_N-1:0]   tap_enq_ext;
  logic [5:0]                 tap_start, tap_done, tap_done_ran, head_valid;
  logic [5:0][31:0]           tap_start_pidx, tap_done_pidx, head_pidx, head_w0;
  logic [5:0][3:0]            tap_done_code;
  logic [31:0]                tap_mark_value, tap_mark_pidx, tap_mark_pc;

  tsr_issue #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CH_BYTES(CH_BYTES),
    .ADDR_BEAT_W(ADDR_BEAT_W), .CAPS(CAPS), .QD_DMA(QD_DMA), .QD_QACT(QD_QACT), .QD_MVM(QD_MVM_E),
    .QD_VU(QD_VU_E), .QD_ATT(QD_ATT), .QD_SYNC(QD_SYNC), .XC_CMD_W(XC_CMD_W),
    .DONE_REG(RV_LOCAL_TRANSPORT)
  ) u_issue (
    .clk, .rst_n(rst_b_n[RB_ISSUE]),
    .ifq_valid, .ifq_ready, .ifq_entry(tsr_ifent_t'(ifq_entry)),
    .xc_req_valid, .xc_req_insn, .xc_rsp_valid, .xc_rsp_code, .xc_rsp_info, .xc_rsp_ext, .xc_rsp_cmd,
    .dma_cmd_valid, .dma_cmd_ready, .dma_cmd, .dma_cmd_xc, .dma_xcmd, .dma_done,
    .qact_cmd_valid, .qact_cmd_ready, .qact_cmd, .qact_done,
    .mvm_cmd_valid, .mvm_cmd_ready, .mvm_cmd, .mvm_cmd_xc, .mvm_xcmd, .mvm_done,
    .vu_cmd_valid, .vu_cmd_ready, .vu_cmd, .vu_done,
    .att_cmd_valid, .att_cmd_ready(1'b0), .att_cmd, .att_done(1'b0),
    .host_hold, .halt_hold, .qabort, .rv_reset, .host_clear, .sr_busy, .dbg_hold, .dbg_step,
    .err_rec(q_err_rec), .err_valid, .err_hs(err_rec), .err_ack, .rec_resync,
    .seq_s(s_seq), .seq_d(d_seq), .q_pidx, .q_state, .q_nonempty, .err_any, .abort_active, .hold_src,
    .qidle, .qquiescent,
    .perf_retired, .perf_push, .perf_busy, .perf_mvm_wait, .perf_mvm_empty, .perf_hold,
    .tap_enq, .tap_enq_unit, .tap_enq_seq, .tap_enq_pidx, .tap_enq_code, .tap_enq_info, .tap_enq_kind,
    .tap_enq_dep0, .tap_enq_dep1, .tap_enq_w, .tap_enq_rec, .tap_enq_pc, .tap_enq_ext,
    .tap_start, .tap_start_pidx, .tap_done, .tap_done_code, .tap_done_ran, .tap_done_pidx,
    .tap_mark, .tap_mark_value, .tap_mark_pidx, .tap_mark_pc,
    .head_valid, .head_pidx, .head_w0
  );

  // Levels to the clk_rv side (tsr_ctrl: QSTATUS[4:1] = {debug-halt, RISC-V, host, any}).
  assign q_hold = {hold_src, |hold_src};
  for (genvar u = 0; u < 6; u++) begin : g_qerr
    assign q_err[u] = q_err_rec[u].err;
  end

  // DB block <-> tsr_csr (EX §5.1 DBG window, SREG window, PERF extension slots)
  logic              dbg_req, dbg_we, dbg_host, sr_db_clear, db_stop_active, dmi_en, unit_hung;
  logic [11:0]       dbg_addr;
  logic [31:0]       dbg_wdata, dbg_rdata;
  logic [63:0][3:0]  perf_ext, perf_ext_db, perf_ext_un;
  logic [15:0][31:0] vu_sreg;

  tsr_csr #(
    .NUM_CH(NUM_CH), .LANES(LANES), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES),
    .VL(VL), .NSREG(NSREG), .CAPS(CAPS), .CTL_ID(CTL_ID), .CFG3(CFG3),
    .ITCM_BYTES(ITCM_BYTES), .DTCM_BYTES(DTCM_BYTES),
    .QD_DMA(QD_DMA), .QD_QACT(QD_QACT), .QD_MVM(QD_MVM_E), .QD_VU(QD_VU_E), .QD_ATT(QD_ATT),
    .QD_SYNC(QD_SYNC), .HAS_VU(HAS_VU)
  ) u_csr (
    .clk, .rst_n(rst_b_n[RB_CSR]),
    .csr_valid, .csr_ready, .csr_we, .csr_addr, .csr_wdata, .csr_rdata,
    .cwin_req_valid, .cwin_req_off, .cwin_rsp_valid, .cwin_rsp_rdata,
    .lq_req_valid, .lq_req_off, .lq_req_we, .lq_req_wdata, .lq_req_lcq_w,
    .lq_rsp_valid, .lq_rsp_rdata, .lq_rsp_fault,
    .log_valid, .log_data, .log_ready, .rt_state, .rt_info,
    .ld_ready, .ld_req, .ld_we, .ld_addr, .ld_wdata,
    .ld_rsp_valid, .ld_rsp_rdata, .ld_rsp_err, .ld_rsp_timeout,
    .dmi_en, .dmi_ready, .dmi_req, .dmi_addr, .dmi_op, .dmi_wdata,
    .dmi_rsp_valid, .dmi_rsp_rdata, .dmi_rsp_resp, .dmi_rsp_timeout,
    .dbg_req, .dbg_we, .dbg_addr, .dbg_wdata, .dbg_host, .dbg_rdata, .sr_db_clear,
    .db_stop_active,
    .sreg(vu_sreg),
    .mtime, .rv_cyc, .perf_push_stall, .perf_bridge_acc, .perf_bridge_wait,
    .push_timeout, .bridge_timeout, .rv_run, .rv_boot,
    .irq_lq, .irq_lcq, .perf_time_en, .perf_evt_en,
    .rv_halted, .rv_sleep, .rv_running, .rv_reset_done, .rv_clk_ok,
    .rv_ndmreset, .issued_seq, .ifq_count,
    .host_hold, .halt_hold, .qabort, .rv_reset, .host_clear, .sr_busy,
    .qidle(qidle && ifq_drained), .qquiescent(qquiescent && ifq_drained),
    .err_any, .hold_src, .q_state, .seq_d(d_seq), .err_rec(q_err_rec), .q_pidx,
    .perf_retired, .perf_push, .perf_busy, .perf_mvm_wait, .perf_mvm_empty, .perf_hold,
    .perf_ch_rd_beat, .perf_ch_wait, .perf_ext, .unit_progress,
    .plat_err_i, .host_irq_o, .unit_hung
  );
  // PERF increments of the DB block (slots 61, 62, X + 10) and of the units (VU X + 13 .. 15, AT X + 16
  // .. 19): disjoint slots, so the arrays are OR-ed
  assign perf_ext = perf_ext_db | perf_ext_un;

  // -------------------------------------------------------------------------------------------
  // External checker classes (tsr_chk's xc_* port, CC §8.2): the VU class with the routing of AT's
  // classes (tsr_vu_xc, VU §5.9) and the AT class (tsr_chk_at: KVST, ATTQK, ATTPV; AT §5.8)
  // -------------------------------------------------------------------------------------------
  if (HAS_VU) begin : g_xc
    logic                     at_req_valid, at_rsp_valid;
    tsr_insn_t                at_req_insn;
    logic [3:0]               at_rsp_code;
    logic [31:0]              at_rsp_info;
    tsr_ext_t [TSR_EXT_N-1:0] at_rsp_ext;
    logic [XC_CMD_W-1:0]      at_rsp_cmd;

    tsr_vu_xc #(.LOCAL_INTERMEDIATES(LOCAL_INTERMEDIATES), .SPM_BYTES(SPM_BYTES), .K_MAX(K_MAX), .NUM_CH(NUM_CH)) u_xc (
      .clk, .rst_n(rst_b_n[RB_ISSUE]),
      .xc_req_valid, .xc_req_insn, .xc_rsp_valid, .xc_rsp_code, .xc_rsp_info, .xc_rsp_ext, .xc_rsp_cmd,
      .at_req_valid, .at_req_insn, .at_rsp_valid, .at_rsp_code, .at_rsp_info, .at_rsp_ext, .at_rsp_cmd
    );
    if (HAS_ATT) begin : g_at
      tsr_chk_at #(
        .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CH_BYTES(CH_BYTES),
        .ADDR_BEAT_W(ADDR_BEAT_W), .XC_CMD_W(XC_CMD_W), .LOCAL_INTERMEDIATES(LOCAL_INTERMEDIATES)
      ) u_chk_at (
        .clk, .rst_n(rst_b_n[RB_ISSUE]),
        .req_valid(at_req_valid), .req_insn(at_req_insn),
        .rsp_valid(at_rsp_valid), .rsp_code(at_rsp_code), .rsp_info(at_rsp_info), .rsp_ext(at_rsp_ext),
        .rsp_cmd(at_rsp_cmd)
      );
    end else begin : g_noat
      // no AT opcode reaches the port: tsr_chk refuses them while CAPS.ATT = 0
      assign at_rsp_valid = 1'b0;
      assign at_rsp_code  = '0;
      assign at_rsp_info  = '0;
      assign at_rsp_ext   = '0;
      assign at_rsp_cmd   = '0;
      wire unused_at = &{1'b0, at_req_valid, at_req_insn};
    end
  end else begin : g_noxc
    // HAS_VU = 0: no external opcode reaches the port (their CAPS bits are 0)
    assign xc_rsp_valid = 1'b0;
    assign xc_rsp_code  = '0;
    assign xc_rsp_info  = '0;
    assign xc_rsp_ext   = '0;
    assign xc_rsp_cmd   = '0;
    wire unused_xc = &{1'b0, xc_req_valid, xc_req_insn};
  end

  // tsr_issue outputs the DB block does not take (it derives the dependencies from the issue-FIFO snoop)
  wire unused_issue = &{1'b0, abort_active, att_cmd_valid, att_cmd, tap_enq_info, tap_enq_dep0, tap_enq_dep1,
                        tap_mark_pc, head_w0};

  // -------------------------------------------------------------------------------------------
  // Read-source ports of the per-channel arbiters: [channel][source].
  // -------------------------------------------------------------------------------------------
  logic [NUM_CH-1:0][TSR_NSRC-1:0]                   s_rd_req_valid, s_rd_req_ready;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][ADDR_BEAT_W-1:0]  s_rd_req_addr;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][TSR_LEN_W-1:0]    s_rd_req_len;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][TSR_SRC_ID_W-1:0] s_rd_req_id;
  logic [NUM_CH-1:0][TSR_NSRC-1:0]                   s_rd_rsp_valid, s_rd_rsp_ready;
  logic [NUM_CH-1:0][BEAT_BITS-1:0]                  s_rd_rsp_data;
  logic [NUM_CH-1:0]                                 s_rd_rsp_last;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]               s_rd_rsp_id;

  // DMA source ports, one per channel.
  logic [NUM_CH-1:0]                   d_rd_req_valid, d_rd_rsp_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  d_rd_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    d_rd_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_rd_req_id;
  logic [NUM_CH-1:0]                   d_rd_req_ready, d_rd_rsp_valid;
  logic [NUM_CH-1:0][BEAT_BITS-1:0]    d_rd_rsp_data;
  logic [NUM_CH-1:0]                   d_rd_rsp_last;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_rd_rsp_id;
  logic [NUM_CH-1:0]                   d_wr_req_valid, d_wr_req_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  d_wr_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    d_wr_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_wr_req_id;
  logic [NUM_CH-1:0]                   d_wr_dat_valid, d_wr_dat_ready, d_wr_dat_last;
  logic [NUM_CH-1:0][BEAT_BITS-1:0]    d_wr_dat_data;
  logic [NUM_CH-1:0][BEAT_BITS/32-1:0] d_wr_dat_mask;
  logic [NUM_CH-1:0]                   d_wr_rsp_valid, d_wr_rsp_ready;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_wr_rsp_id;

  // Write source ports of the arbiters [channel]: the DMA's, except channel TSR_TRW_CH, which has the
  // trace writer's per-burst mux in front of it (tsr_dbg_wrmux, DB §8.3; a wire-through while idle).
  logic [NUM_CH-1:0]                   a_wr_req_valid, a_wr_req_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  a_wr_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    a_wr_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] a_wr_req_id;
  logic [NUM_CH-1:0]                   a_wr_dat_valid, a_wr_dat_ready, a_wr_dat_last;
  logic [NUM_CH-1:0][BEAT_BITS-1:0]    a_wr_dat_data;
  logic [NUM_CH-1:0][BEAT_BITS/32-1:0] a_wr_dat_mask;
  logic [NUM_CH-1:0]                   a_wr_rsp_valid, a_wr_rsp_ready;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] a_wr_rsp_id;

  // Arbiter-facing writes after the optional central DDR boundary slices.
  logic [NUM_CH-1:0]                   b_wr_req_valid, b_wr_req_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  b_wr_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    b_wr_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] b_wr_req_id;
  logic [NUM_CH-1:0]                   b_wr_dat_valid, b_wr_dat_ready, b_wr_dat_last;
  logic [NUM_CH-1:0][BEAT_BITS-1:0]    b_wr_dat_data;
  logic [NUM_CH-1:0][BEAT_BITS/32-1:0] b_wr_dat_mask;
  logic [NUM_CH-1:0]                   b_wr_rsp_valid, b_wr_rsp_ready;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] b_wr_rsp_id;

  // -------------------------------------------------------------------------------------------
  // DB block (DB §13, W4-B item 6): tsr_dbg (DBG registers, stop logic, shadow queues, history, trace
  // writer, DMI gate; the clk_rv parts and the crossings are tsr_dbg_rv in tsr_core) and the trace
  // writer's write mux on channel TSR_TRW_CH. Memory taps: the handshakes at the core's memory ports.
  // -------------------------------------------------------------------------------------------
  logic                    t_wr_req_valid, t_wr_req_ready, t_wr_dat_valid, t_wr_dat_ready, t_wr_dat_last;
  logic                    t_wr_rsp_valid;
  logic [ADDR_BEAT_W-1:0]  t_wr_req_addr;
  logic [TSR_LEN_W-1:0]    t_wr_req_len;
  logic [BEAT_BITS-1:0]    t_wr_dat_data;
  logic [NUM_CH-1:0]       mt_rd_req, mt_rd_last, mt_wr_req, mt_wr_rsp;
  logic [NUM_CH-1:0]       mt_rd_req_i, mt_rd_last_i, mt_wr_req_i, mt_wr_rsp_i;
  assign mt_rd_req_i  = m_rd_req_valid & m_rd_req_ready;
  assign mt_rd_last_i = m_rd_rsp_valid & m_rd_rsp_ready & m_rd_rsp_last;
  assign mt_wr_req_i  = m_wr_req_valid & m_wr_req_ready;
  assign mt_wr_rsp_i  = m_wr_rsp_valid & m_wr_rsp_ready;
  // The taps count handshakes at the memory ports, which sit in other SLRs than the DB block when the
  // control plane is relocated (RV_LOCAL_TRANSPORT): two register stages in the glue, free to sit between
  // the ports and the counters. They only lag DBG_RDOUT / DBG_WROUT by MT_PIPE cycles (the probes are not
  // compared with the model, DB §5.5); with the taps wired straight, one handshake term (a LUT merged
  // into a bridge) sat in SLR2 between two flops of the counter in SLR0 (k4min link, 2026-11-06: -2.6 ns).
  localparam int unsigned MT_PIPE = RV_LOCAL_TRANSPORT ? 2 : 0;
  if (MT_PIPE == 0) begin : g_mt0
    assign mt_rd_req  = mt_rd_req_i;
    assign mt_rd_last = mt_rd_last_i;
    assign mt_wr_req  = mt_wr_req_i;
    assign mt_wr_rsp  = mt_wr_rsp_i;
  end else begin : g_mt
    logic [MT_PIPE-1:0][4*NUM_CH-1:0] mt_q;
    always_ff @(posedge clk) begin
      if (!rst_b_n[RB_DBG]) begin
        mt_q <= '0;
      end else begin
        mt_q[0] <= {mt_rd_req_i, mt_rd_last_i, mt_wr_req_i, mt_wr_rsp_i};
        for (int unsigned k = 1; k < MT_PIPE; k++) mt_q[k] <= mt_q[k-1];
      end
    end
    assign {mt_rd_req, mt_rd_last, mt_wr_req, mt_wr_rsp} = mt_q[MT_PIPE-1];
  end

  if (HAS_DB) begin : g_db
    tsr_dbg #(
      .NUM_CH(NUM_CH), .CH_BYTES(CH_BYTES), .ADDR_BEAT_W(ADDR_BEAT_W), .QD_DMA(QD_DMA), .QD_QACT(QD_QACT),
      .QD_MVM(QD_MVM_E), .QD_VU(QD_VU_E), .QD_ATT(QD_ATT), .QD_SYNC(QD_SYNC), .HIST_DEPTH(HIST_DEPTH),
      .HAS_TRW(HAS_TRW), .NTBP(NTBP), .NRVWP(NRVWP), .HAS_REC(HAS_REC)
    ) u_dbg (
      .clk, .rst_n(rst_b_n[RB_DBG]),
      .dbg_req, .dbg_we, .dbg_addr, .dbg_wdata, .dbg_host, .dbg_rdata, .sr_db_clear, .db_stop_active, .dmi_en,
      .perf_ext(perf_ext_db),
      .csr_valid, .csr_ready, .csr_we, .csr_addr, .csr_wdata, .dmi_rsp_valid, .dmi_rsp_resp, .dmi_rsp_timeout,
      .ifq_valid, .ifq_ready, .ifq_entry(tsr_ifent_t'(ifq_entry)),
      .tap_enq, .tap_enq_unit, .tap_enq_seq, .tap_enq_pidx, .tap_enq_code, .tap_enq_kind, .tap_enq_w,
      .tap_enq_rec, .tap_enq_pc, .tap_enq_ext, .tap_start, .tap_start_pidx, .tap_done, .tap_done_code,
      .tap_done_ran, .tap_done_pidx, .tap_mark, .tap_mark_value, .tap_mark_pidx,
      .head_valid, .head_pidx, .q_pidx, .q_state, .hold_src, .seq_d(d_seq), .err_rec(q_err_rec),
      .dbg_hold, .dbg_step,
      .sr_busy, .rv_run, .rv_running, .rv_halted, .rv_sleep, .rv_ndmreset, .rv_dmactive, .rv_halt_pend, .rv_clk_ok,
      .unit_hung, .mtime,
      .dbg_stop, .dbg_halt_tgl, .dbg_cancel_tgl, .dbg_halt_on_boot,
      .rec_wr, .rec_wdata, .rec_room, .rec_all, .smp_ready, .smp_req, .smp_local, .smp_done, .smp_timeout,
      .smp_data, .wpa_ready, .wpa_req, .wpa_local, .wpa_data, .wpa_done, .wpa_timeout,
      .wph_v, .wph_mask, .wph_addr, .wph_halt, .dfault, .flags,
      .mt_rd_req, .mt_rd_last, .mt_wr_req, .mt_wr_rsp,
      .trw_wr_req_valid(t_wr_req_valid), .trw_wr_req_ready(t_wr_req_ready), .trw_wr_req_addr(t_wr_req_addr),
      .trw_wr_req_len(t_wr_req_len), .trw_wr_dat_valid(t_wr_dat_valid), .trw_wr_dat_ready(t_wr_dat_ready),
      .trw_wr_dat_data(t_wr_dat_data), .trw_wr_dat_last(t_wr_dat_last), .trw_wr_rsp_valid(t_wr_rsp_valid)
    );
  end else begin : g_nodb
    // HAS_DB = 0: the DBG block reads 0 (DBG_ID = 0), DMI_EN = 0, nothing holds or steps a queue head
    assign dbg_rdata        = '0;
    assign db_stop_active   = 1'b0;
    assign dmi_en           = 1'b0;
    assign perf_ext_db      = '0;
    assign dbg_hold         = '0;
    assign dbg_step         = '0;
    assign dbg_stop         = 1'b0;
    assign dbg_halt_tgl     = 1'b0;
    assign dbg_cancel_tgl   = 1'b0;
    assign dbg_halt_on_boot = 1'b0;
    assign rec_wr           = 1'b0;
    assign rec_wdata        = '0;
    assign smp_req          = 1'b0;
    assign smp_local        = 1'b0;
    assign wpa_req          = 1'b0;
    assign wpa_local        = 1'b0;
    assign wpa_data         = '0;
    assign flags            = '0;
    assign t_wr_req_valid   = 1'b0;
    assign t_wr_req_addr    = '0;
    assign t_wr_req_len     = '0;
    assign t_wr_dat_valid   = 1'b0;
    assign t_wr_dat_data    = '0;
    assign t_wr_dat_last    = 1'b0;
    wire unused_nodb = &{1'b0, dbg_req, dbg_we, dbg_addr, dbg_wdata, dbg_host, sr_db_clear, unit_hung,
                         rv_halt_pend, rv_dmactive, rec_room, rec_all, smp_ready, smp_done, smp_timeout, smp_data,
                         wpa_ready, wpa_done, wpa_timeout, wph_v, wph_mask, wph_addr, wph_halt, dfault,
                         mt_rd_req, mt_rd_last, mt_wr_req,
                         mt_wr_rsp, tap_enq, tap_enq_rec, tap_mark, tap_enq_unit, tap_enq_code, tap_enq_seq,
                         tap_enq_pidx, tap_enq_info, tap_enq_dep0, tap_enq_dep1, tap_enq_pc, tap_enq_kind,
                         tap_enq_w, tap_enq_ext, tap_start, tap_done, tap_done_ran, head_valid,
                         tap_start_pidx, tap_done_pidx, head_pidx, head_w0, tap_done_code, tap_mark_value,
                         tap_mark_pidx, tap_mark_pc};
  end
  if (!(HAS_DB && HAS_TRW)) begin : g_notrw
    // no trace writer: no write mux either, nothing answers its (absent) bursts
    assign t_wr_req_ready = 1'b0;
    assign t_wr_dat_ready = 1'b0;
    assign t_wr_rsp_valid = 1'b0;
    wire unused_notrw = &{1'b0, t_wr_req_valid, t_wr_req_addr, t_wr_req_len, t_wr_dat_valid, t_wr_dat_data,
                          t_wr_dat_last, t_wr_req_ready, t_wr_dat_ready, t_wr_rsp_valid};
  end

  // Single-port SPM clients [TSR_SPM_NSP] (docs/rtl-m2.md §5 interface): 0 DMA, 1 QACT, 2 MVM
  // (tied off when HAS_MVM = 0), 3 RISC-V bridge; each reaches its two bank ports of tsr_spm
  // through a tsr_spm_cport (CC §10.2).
  logic [TSR_SPM_NSP-1:0]                         spm_rd_req_valid, spm_rd_req_ready, spm_rd_rsp_valid;
  logic [TSR_SPM_NSP-1:0][SPM_AW-1:0]             spm_rd_req_line;
  logic [TSR_SPM_NSP-1:0][TSR_LINE_BITS-1:0]      spm_rd_rsp_data;
  logic [TSR_SPM_NSP-1:0]                         spm_wr_valid, spm_wr_ready;
  logic [TSR_SPM_NSP-1:0][SPM_AW-1:0]             spm_wr_line;
  logic [TSR_SPM_NSP-1:0][TSR_LINE_BITS-1:0]      spm_wr_data;
  logic [TSR_SPM_NSP-1:0][TSR_SPM_BE_W-1:0]       spm_wr_be;

  // KVST runs on the DMA when CAPS has ATT (the external payload of tsr_issue)
  logic perf_kvst_rmw, perf_att_qk, perf_att_pv, perf_att_stall;
  tsr_dma #(
    .NUM_CH(NUM_CH), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .SPM_BYTES(SPM_BYTES),
    .SPM_RD_LAT(SPM_RD_LAT), .HAS_KVST(HAS_ATT), .XC_CMD_W(XC_CMD_W)
  ) u_dma (
    .clk, .rst_n(rst_b_n[RB_DMA]),
    .cmd_valid(dma_cmd_valid), .cmd_ready(dma_cmd_ready), .cmd(dma_cmd), .cmd_xc(dma_cmd_xc),
    .xcmd(dma_xcmd), .done(dma_done),
    .rd_req_valid(d_rd_req_valid), .rd_req_ready(d_rd_req_ready), .rd_req_addr(d_rd_req_addr),
    .rd_req_len(d_rd_req_len), .rd_req_id(d_rd_req_id),
    .rd_rsp_valid(d_rd_rsp_valid), .rd_rsp_ready(d_rd_rsp_ready), .rd_rsp_data(d_rd_rsp_data),
    .rd_rsp_last(d_rd_rsp_last), .rd_rsp_id(d_rd_rsp_id),
    .wr_req_valid(d_wr_req_valid), .wr_req_ready(d_wr_req_ready), .wr_req_addr(d_wr_req_addr),
    .wr_req_len(d_wr_req_len), .wr_req_id(d_wr_req_id),
    .wr_dat_valid(d_wr_dat_valid), .wr_dat_ready(d_wr_dat_ready), .wr_dat_data(d_wr_dat_data), .wr_dat_mask(d_wr_dat_mask),
    .wr_dat_last(d_wr_dat_last),
    .wr_rsp_valid(d_wr_rsp_valid), .wr_rsp_ready(d_wr_rsp_ready), .wr_rsp_id(d_wr_rsp_id),
    .spm_rd_req_valid(spm_rd_req_valid[TSR_SPM_CL_DMA]), .spm_rd_req_ready(spm_rd_req_ready[TSR_SPM_CL_DMA]),
    .spm_rd_req_line(spm_rd_req_line[TSR_SPM_CL_DMA]), .spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_DMA]),
    .spm_rd_rsp_data(spm_rd_rsp_data[TSR_SPM_CL_DMA]),
    .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_DMA]), .spm_wr_ready(spm_wr_ready[TSR_SPM_CL_DMA]),
    .spm_wr_line(spm_wr_line[TSR_SPM_CL_DMA]), .spm_wr_data(spm_wr_data[TSR_SPM_CL_DMA]),
    .spm_wr_be(spm_wr_be[TSR_SPM_CL_DMA]),
    .perf_kvst_rmw                                   // PERF KVST_RMW (AT §10)
  );

  tsr_qact #(
    .K_MAX(K_MAX), .T_MAX(T_MAX), .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT)
  ) u_qact (
    .clk, .rst_n(rst_b_n[RB_QACT]),
    .cmd_valid(qact_cmd_valid), .cmd_ready(qact_cmd_ready), .cmd(qact_cmd), .done(qact_done),
    .spm_rd_req_valid(spm_rd_req_valid[TSR_SPM_CL_QACT]), .spm_rd_req_ready(spm_rd_req_ready[TSR_SPM_CL_QACT]),
    .spm_rd_req_line(spm_rd_req_line[TSR_SPM_CL_QACT]), .spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_QACT]),
    .spm_rd_rsp_data(spm_rd_rsp_data[TSR_SPM_CL_QACT]),
    .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_QACT]), .spm_wr_ready(spm_wr_ready[TSR_SPM_CL_QACT]),
    .spm_wr_line(spm_wr_line[TSR_SPM_CL_QACT]), .spm_wr_data(spm_wr_data[TSR_SPM_CL_QACT]),
    .spm_wr_be(spm_wr_be[TSR_SPM_CL_QACT])
  );

  // -------------------------------------------------------------------------------------------
  // MVM unit (M3): SPM client TSR_SPM_CL_MVM, weight-stream source TSR_SRC_WS of every channel
  // -------------------------------------------------------------------------------------------
  // Weight-stream source ports (cluster c <-> arbiter c), [channel].
  logic [NUM_CH-1:0]                   w_rd_req_valid, w_rd_req_ready, w_rd_rsp_valid, w_rd_rsp_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  w_rd_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    w_rd_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] w_rd_req_id;

  logic scale_request, scale_grant;
  tsr_vu_pkg::tsr_vu_local_req_t [1:0] scale_req;
  tsr_vu_pkg::tsr_vu_local_rsp_t [1:0] scale_rsp;
  tsr_att_pkg::tsr_local_score_req_t [NUM_CH-1:0] local_score_req;
  logic [NUM_CH-1:0][TSR_LINE_BITS-1:0] local_score_data;

  // ATTQK / ATTPV on the MVM unit when CAPS has ATT (the external payload of tsr_issue)
  if (HAS_MVM) begin : g_mvm
    logic inner_ready;
    wire needs_scale = LOCAL_SCALE && !mvm_cmd_xc && (mvm_cmd.local_scale || mvm_cmd.local_softcap);
    assign scale_request = needs_scale && mvm_cmd_valid && inner_ready;
    assign mvm_cmd_ready = inner_ready && (!mvm_cmd_valid || !needs_scale || scale_grant);
    tsr_mvm #(
      .HAS_ATT(HAS_ATT), .LOCAL_INTERMEDIATES(LOCAL_INTERMEDIATES), .LOCAL_SCALE(LOCAL_SCALE), .XC_CMD_W(XC_CMD_W),
      .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT),
      .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .WS_FIFO_BEATS(MVM_WS_FIFO),
      .BC_PIPE(MVM_BC_PIPE), .RES_PIPE(MVM_RES_PIPE), .GATHER_PAIR(MVM_GATHER_PAIR), .GATHER_FLITS(MVM_GATHER_FLITS), .ACT_RD_LAT(MVM_ACT_RD_LAT),
      .MAC_LAT(MVM_MAC_LAT), .MUL_LAT(MVM_MUL_LAT), .ADD_LAT(MVM_ADD_LAT), .WS_RD_LAT(MVM_WS_RD_LAT),
      .RST_LOCAL(RST_BLK)
    ) u_mvm (
      .clk, .rst_n(rst_pre_n),
      .cmd_valid(mvm_cmd_valid && (!needs_scale || scale_grant)), .cmd_ready(inner_ready), .cmd(mvm_cmd), .cmd_xc(mvm_cmd_xc),
      .xcmd(mvm_xcmd), .done(mvm_done),
      .local_score_req, .local_score_data, .scale_req, .scale_rsp,
      .spm_rd_req_valid(spm_rd_req_valid[TSR_SPM_CL_MVM]), .spm_rd_req_ready(spm_rd_req_ready[TSR_SPM_CL_MVM]),
      .spm_rd_req_line(spm_rd_req_line[TSR_SPM_CL_MVM]), .spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_MVM]),
      .spm_rd_rsp_data(spm_rd_rsp_data[TSR_SPM_CL_MVM]),
      .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_MVM]), .spm_wr_ready(spm_wr_ready[TSR_SPM_CL_MVM]),
      .spm_wr_line(spm_wr_line[TSR_SPM_CL_MVM]), .spm_wr_data(spm_wr_data[TSR_SPM_CL_MVM]),
      .spm_wr_be(spm_wr_be[TSR_SPM_CL_MVM]),
      .ws_rd_req_valid(w_rd_req_valid), .ws_rd_req_ready(w_rd_req_ready), .ws_rd_req_addr(w_rd_req_addr),
      .ws_rd_req_len(w_rd_req_len), .ws_rd_req_id(w_rd_req_id),
      .ws_rd_rsp_valid(w_rd_rsp_valid), .ws_rd_rsp_ready(w_rd_rsp_ready), .ws_rd_rsp_data(s_rd_rsp_data),
      .ws_rd_rsp_last(s_rd_rsp_last), .ws_rd_rsp_id(s_rd_rsp_id),
      .perf_mvm_busy, .perf_att_qk, .perf_att_pv, .perf_att_stall   // PERF ATT_QK / PV / PATH_STALL
    );
  end else begin : g_nomvm
    assign scale_request = 1'b0;
    assign scale_req = '0;
    wire unused_scale = &{1'b0, scale_grant, scale_rsp};
    assign local_score_data = '0;
    wire unused_local_score = &{1'b0, local_score_req};
    assign perf_att_qk    = 1'b0;
    assign perf_att_pv    = 1'b0;
    assign perf_att_stall = 1'b0;
    // HAS_MVM = 0: no MVM unit; its queue has depth 0, so the dispatcher refuses MVM (ILLEGAL_OP).
    assign mvm_cmd_ready  = 1'b0;
    assign mvm_done       = 1'b0;
    assign perf_mvm_busy  = 1'b0;
    assign w_rd_req_valid = '0;
    assign w_rd_req_addr  = '0;
    assign w_rd_req_len   = '0;
    assign w_rd_req_id    = '0;
    assign w_rd_rsp_ready = '1;
    assign spm_rd_req_valid[TSR_SPM_CL_MVM] = 1'b0;
    assign spm_rd_req_line[TSR_SPM_CL_MVM]  = '0;
    assign spm_wr_valid[TSR_SPM_CL_MVM]     = 1'b0;
    assign spm_wr_line[TSR_SPM_CL_MVM]      = '0;
    assign spm_wr_data[TSR_SPM_CL_MVM]      = '0;
    assign spm_wr_be[TSR_SPM_CL_MVM]        = '0;
    wire unused_mvm = &{1'b0, rst_pre_n, w_rd_req_ready, w_rd_rsp_valid, spm_rd_req_ready[TSR_SPM_CL_MVM],
                        spm_rd_rsp_valid[TSR_SPM_CL_MVM], spm_rd_rsp_data[TSR_SPM_CL_MVM],
                        spm_wr_ready[TSR_SPM_CL_MVM], mvm_cmd_valid, mvm_cmd,
                        mvm_cmd_xc, mvm_xcmd};
  end

  // -------------------------------------------------------------------------------------------
  // RISC-V SPM bridge (client 3, CC §10.5). The direct path holds its
  // request through the response. The segmented path offers it once and
  // holds the response at the bridge until the last segment is sent.
  // The RISC-V bus has already range-checked spm_req_addr.
  // -------------------------------------------------------------------------------------------
  logic bridge_req_ready;

  tsr_spm_bridge #(.SPM_BYTES(SPM_BYTES)) u_bridge (
    .clk, .rst_n(rst_b_n[RB_BRIDGE]),
    .req_valid(spm_req_valid), .req_ready(bridge_req_ready), .req_we(spm_req_we),
    .req_waddr(spm_req_addr[SPM_WAW+1:2]), .req_be(spm_req_be), .req_wdata(spm_req_wdata),
    .rsp_valid(spm_rsp_valid), .rsp_ready(spm_rsp_ready), .rsp_err(spm_rsp_err), .rsp_line(spm_rsp_line),
    .spm_rd_req_valid(spm_rd_req_valid[TSR_SPM_CL_BRIDGE]), .spm_rd_req_ready(spm_rd_req_ready[TSR_SPM_CL_BRIDGE]),
    .spm_rd_req_line(spm_rd_req_line[TSR_SPM_CL_BRIDGE]), .spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_BRIDGE]),
    .spm_rd_rsp_data(spm_rd_rsp_data[TSR_SPM_CL_BRIDGE]),
    .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_BRIDGE]), .spm_wr_ready(spm_wr_ready[TSR_SPM_CL_BRIDGE]),
    .spm_wr_line(spm_wr_line[TSR_SPM_CL_BRIDGE]), .spm_wr_data(spm_wr_data[TSR_SPM_CL_BRIDGE]),
    .spm_wr_be(spm_wr_be[TSR_SPM_CL_BRIDGE])
  );
  if (SPM_WAW + 2 < 26) begin : g_bridge_hi
    wire unused_bridge_hi = &{1'b0, spm_req_addr[25:SPM_WAW+2]};
  end
  wire unused_bridge = &{1'b0, bridge_req_ready, spm_req_addr[1:0]};

  // Vector unit (unit 3, VU §7): SPM client 4 on both bank ports, the s bank (SREG window), PERF X + 13 ..
  // X + 15 and its debug state. cmd / done are tsr_issue's vu_* port (S - D <= 2 accepted at once).
  logic [1:0]                             vu_rd_req_valid, vu_rd_req_ready, vu_rd_rsp_valid;
  logic [1:0][SPM_AW-1:0]                 vu_rd_req_line;
  logic [1:0][TSR_LINE_BITS-1:0]          vu_rd_rsp_data;
  logic [1:0]                             vu_wr_valid, vu_wr_ready;
  logic [1:0][SPM_AW-1:0]                 vu_wr_line;
  logic [1:0][TSR_LINE_BITS-1:0]          vu_wr_data;
  logic [1:0][TSR_SPM_BE_W-1:0]           vu_wr_be;
  logic                                   vu_perf_spm_wait, vu_perf_hazard, vu_perf_line, vu_progress;
  if (HAS_VU) begin : g_vu
    logic [1:0]  vu_dbg_held;
    logic [31:0] vu_dbg_w0;
    logic [2:0]  vu_dbg_flags;
    tsr_vu #(
      .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT), .NUM_CH(NUM_CH), .FRONT_LANES(VU_FRONT_LANES), .EXP_LANES(VU_EXP_LANES),
      .BACK_LANES(VU_BACK_LANES), .LANE_PIPE(VU_LANE_PIPE), .LOCAL_INTERMEDIATES(LOCAL_INTERMEDIATES), .LOCAL_SCALE(LOCAL_SCALE), .HAS_TOPK(HAS_TOPK), .HAS_DQROW(HAS_DQROW)
    ) u_vu (
      .clk, .rst_n(rst_b_n[RB_VU]),
      .cmd_valid(vu_cmd_valid), .cmd_ready(vu_cmd_ready), .cmd(vu_cmd), .done(vu_done),
      .local_score_req, .local_score_data, .scale_request, .scale_grant, .scale_done(mvm_done), .scale_req, .scale_rsp,
      .rd_req_valid(vu_rd_req_valid), .rd_req_ready(vu_rd_req_ready), .rd_req_line(vu_rd_req_line),
      .rd_rsp_valid(vu_rd_rsp_valid), .rd_rsp_data(vu_rd_rsp_data),
      .wr_valid(vu_wr_valid), .wr_ready(vu_wr_ready), .wr_line(vu_wr_line), .wr_data(vu_wr_data),
      .wr_be(vu_wr_be),
      .sreg(vu_sreg),
      .perf_spm_wait(vu_perf_spm_wait), .perf_hazard(vu_perf_hazard), .perf_line(vu_perf_line),
      .dbg_held(vu_dbg_held), .dbg_w0(vu_dbg_w0), .dbg_flags(vu_dbg_flags),
      .progress(vu_progress)
    );
    // The VU's debug state (VU §5.10) has no port on the DB block: DB derives the unit probes from the
    // dispatcher's strobes (DB §9.1, model values).
    wire unused_vu_dbg = &{1'b0, vu_dbg_held, vu_dbg_w0, vu_dbg_flags};
  end else begin : g_novu
    assign scale_grant = 1'b0;
    assign scale_rsp = '0;
    wire unused_scale = &{1'b0, scale_request, scale_req};
    assign local_score_req = '0;
    wire unused_local_score = &{1'b0, local_score_data};
    assign vu_cmd_ready      = 1'b0;
    assign vu_done           = 1'b0;
    assign vu_rd_req_valid   = '0;
    assign vu_rd_req_line    = '0;
    assign vu_wr_valid       = '0;
    assign vu_wr_line        = '0;
    assign vu_wr_data        = '0;
    assign vu_wr_be          = '0;
    assign vu_sreg           = '0;
    assign vu_perf_spm_wait  = 1'b0;
    assign vu_perf_hazard    = 1'b0;
    assign vu_perf_line      = 1'b0;
    assign vu_progress       = 1'b0;
    wire unused_novu = &{1'b0, rst_b_n[RB_VU], vu_cmd_valid, vu_cmd, vu_rd_req_ready, vu_rd_rsp_valid,
                         vu_rd_rsp_data, vu_wr_ready};
  end

  // PERF increments of the units (EX §6): VU X + 13 .. 15, AT X + 16 .. 19 (levels are time slots, the
  // strobes events; the units' outputs are registered)
  always_comb begin
    perf_ext_un = '0;
    perf_ext_un[PERF_X + 13][0] = vu_perf_spm_wait;
    perf_ext_un[PERF_X + 14][0] = vu_perf_hazard;
    perf_ext_un[PERF_X + 15][0] = vu_perf_line;
    perf_ext_un[PERF_X + 16][0] = perf_att_qk;
    perf_ext_un[PERF_X + 17][0] = perf_att_pv;
    perf_ext_un[PERF_X + 18][0] = perf_att_stall;
    perf_ext_un[PERF_X + 19][0] = perf_kvst_rmw;
  end

  // SPM bank ports [client][bank] (CC §10.2): clients 0..3 through tsr_spm_cport, client 4 (VU) on its two
  // bank ports directly, client 5 (AT) tied off (AT's ops reach the SPM through the DMA and MVM clients).
  logic [TSR_SPM_NCL-1:0][1:0]                    sb_rd_req_valid, sb_rd_req_ready, sb_rd_rsp_valid;
  logic [TSR_SPM_NCL-1:0][1:0][SPM_AW-1:0]        sb_rd_req_line;
  logic [1:0][TSR_LINE_BITS-1:0]                  sb_rd_rsp_data;
  logic [TSR_SPM_NCL-1:0][1:0]                    sb_wr_valid, sb_wr_ready;
  logic [TSR_SPM_NCL-1:0][1:0][SPM_AW-1:0]        sb_wr_line;
  logic [TSR_SPM_NCL-1:0][1:0][TSR_LINE_BITS-1:0] sb_wr_data;
  logic [TSR_SPM_NCL-1:0][1:0][TSR_SPM_BE_W-1:0]  sb_wr_be;

  for (genvar c = 0; c < TSR_SPM_NCL; c++) begin : g_spm_cl
    if (c < TSR_SPM_NSP) begin : g_cport
      tsr_spm_cport #(.SPM_BYTES(SPM_BYTES)) u_cport (
        .rd_req_valid(spm_rd_req_valid[c]), .rd_req_ready(spm_rd_req_ready[c]),
        .rd_req_line(spm_rd_req_line[c]), .rd_rsp_valid(spm_rd_rsp_valid[c]),
        .rd_rsp_data(spm_rd_rsp_data[c]),
        .wr_valid(spm_wr_valid[c]), .wr_ready(spm_wr_ready[c]), .wr_line(spm_wr_line[c]),
        .wr_data(spm_wr_data[c]), .wr_be(spm_wr_be[c]),
        .s_rd_req_valid(sb_rd_req_valid[c]), .s_rd_req_ready(sb_rd_req_ready[c]),
        .s_rd_req_line(sb_rd_req_line[c]), .s_rd_rsp_valid(sb_rd_rsp_valid[c]),
        .s_rd_rsp_data(sb_rd_rsp_data),
        .s_wr_valid(sb_wr_valid[c]), .s_wr_ready(sb_wr_ready[c]), .s_wr_line(sb_wr_line[c]),
        .s_wr_data(sb_wr_data[c]), .s_wr_be(sb_wr_be[c])
      );
    end else if (c == TSR_SPM_CL_VU) begin : g_vucl
      assign sb_rd_req_valid[c] = vu_rd_req_valid;
      assign vu_rd_req_ready    = sb_rd_req_ready[c];
      assign sb_rd_req_line[c]  = vu_rd_req_line;
      assign vu_rd_rsp_valid    = sb_rd_rsp_valid[c];
      assign vu_rd_rsp_data     = sb_rd_rsp_data;
      assign sb_wr_valid[c]     = vu_wr_valid;
      assign vu_wr_ready        = sb_wr_ready[c];
      assign sb_wr_line[c]      = vu_wr_line;
      assign sb_wr_data[c]      = vu_wr_data;
      assign sb_wr_be[c]        = vu_wr_be;
    end else begin : g_tie
      assign sb_rd_req_valid[c] = '0;
      assign sb_rd_req_line[c]  = '0;
      assign sb_wr_valid[c]     = '0;
      assign sb_wr_line[c]      = '0;
      assign sb_wr_data[c]      = '0;
      assign sb_wr_be[c]        = '0;
      wire unused_cl = &{1'b0, sb_rd_req_ready[c], sb_rd_rsp_valid[c], sb_wr_ready[c]};
    end
  end

  tsr_spm #(
    .SPM_BYTES(SPM_BYTES), .NCL(TSR_SPM_NCL), .RD_LAT(SPM_RD_LAT)
  ) u_spm (
    .clk, .rst_n(rst_b_n[RB_SPM]),
    .rd_req_valid(sb_rd_req_valid), .rd_req_ready(sb_rd_req_ready), .rd_req_line(sb_rd_req_line),
    .rd_rsp_valid(sb_rd_rsp_valid), .rd_rsp_data(sb_rd_rsp_data),
    .wr_valid(sb_wr_valid), .wr_ready(sb_wr_ready), .wr_line(sb_wr_line), .wr_data(sb_wr_data),
    .wr_be(sb_wr_be)
  );

  // -------------------------------------------------------------------------------------------
  // Per-channel memory arbiters
  // -------------------------------------------------------------------------------------------
  for (genvar c = 0; c < NUM_CH; c++) begin : g_ch
    // Source AUX: tied off on every channel (CC-26).
    assign s_rd_req_valid[c][TSR_SRC_AUX] = 1'b0;
    assign s_rd_req_addr[c][TSR_SRC_AUX]  = '0;
    assign s_rd_req_len[c][TSR_SRC_AUX]   = '0;
    assign s_rd_req_id[c][TSR_SRC_AUX]    = '0;
    assign s_rd_rsp_ready[c][TSR_SRC_AUX] = 1'b1;
    // DDR0/3 arbiters move beside their controllers. These kept slices stay in SLR1,
    // so DMA, KVST and trace cross to the local arbiter's registered endpoints. The
    // 512-bit weight response remains entirely local; only DMA responses return here.
    if (DDR_LOCAL_PIPE && (c == 0 || c == 3)) begin : g_dma_pipe
      localparam int unsigned RQ_W = ADDR_BEAT_W + TSR_LEN_W + TSR_SRC_ID_W;
      (* keep_hierarchy = "yes" *) tsr_rslice #(.W(RQ_W), .FORM(1)) u_rq (
        .clk, .rst_n(rst_b_n[RB_ARB + c]),
        .s_valid(d_rd_req_valid[c]), .s_ready(d_rd_req_ready[c]),
        .s_data({d_rd_req_addr[c], d_rd_req_len[c], d_rd_req_id[c]}),
        .m_valid(s_rd_req_valid[c][TSR_SRC_DMA]), .m_ready(s_rd_req_ready[c][TSR_SRC_DMA]),
        .m_data({s_rd_req_addr[c][TSR_SRC_DMA], s_rd_req_len[c][TSR_SRC_DMA], s_rd_req_id[c][TSR_SRC_DMA]})
      );
      (* keep_hierarchy = "yes" *) tsr_rslice #(.W(BEAT_BITS + 1 + TSR_SRC_ID_W)) u_rsp (
        .clk, .rst_n(rst_b_n[RB_ARB + c]),
        .s_valid(s_rd_rsp_valid[c][TSR_SRC_DMA]), .s_ready(s_rd_rsp_ready[c][TSR_SRC_DMA]),
        .s_data({s_rd_rsp_data[c], s_rd_rsp_last[c], s_rd_rsp_id[c]}),
        .m_valid(d_rd_rsp_valid[c]), .m_ready(d_rd_rsp_ready[c]),
        .m_data({d_rd_rsp_data[c], d_rd_rsp_last[c], d_rd_rsp_id[c]})
      );
      (* keep_hierarchy = "yes" *) tsr_rslice #(.W(RQ_W), .FORM(1)) u_wq (
        .clk, .rst_n(rst_b_n[RB_ARB + c]),
        .s_valid(a_wr_req_valid[c]), .s_ready(a_wr_req_ready[c]),
        .s_data({a_wr_req_addr[c], a_wr_req_len[c], a_wr_req_id[c]}),
        .m_valid(b_wr_req_valid[c]), .m_ready(b_wr_req_ready[c]),
        .m_data({b_wr_req_addr[c], b_wr_req_len[c], b_wr_req_id[c]})
      );
      (* keep_hierarchy = "yes" *) tsr_rslice #(.W(BEAT_BITS + BEAT_BITS/32 + 1), .FORM(1)) u_wd (
        .clk, .rst_n(rst_b_n[RB_ARB + c]),
        .s_valid(a_wr_dat_valid[c]), .s_ready(a_wr_dat_ready[c]),
        .s_data({a_wr_dat_data[c], a_wr_dat_mask[c], a_wr_dat_last[c]}),
        .m_valid(b_wr_dat_valid[c]), .m_ready(b_wr_dat_ready[c]),
        .m_data({b_wr_dat_data[c], b_wr_dat_mask[c], b_wr_dat_last[c]})
      );
      (* keep_hierarchy = "yes" *) tsr_rslice #(.W(TSR_SRC_ID_W)) u_ws (
        .clk, .rst_n(rst_b_n[RB_ARB + c]),
        .s_valid(b_wr_rsp_valid[c]), .s_ready(b_wr_rsp_ready[c]), .s_data(b_wr_rsp_id[c]),
        .m_valid(a_wr_rsp_valid[c]), .m_ready(a_wr_rsp_ready[c]), .m_data(a_wr_rsp_id[c])
      );
    end else begin : g_dma_direct
      assign s_rd_req_valid[c][TSR_SRC_DMA] = d_rd_req_valid[c];
      assign s_rd_req_addr[c][TSR_SRC_DMA]  = d_rd_req_addr[c];
      assign s_rd_req_len[c][TSR_SRC_DMA]   = d_rd_req_len[c];
      assign s_rd_req_id[c][TSR_SRC_DMA]    = d_rd_req_id[c];
      assign s_rd_rsp_ready[c][TSR_SRC_DMA] = d_rd_rsp_ready[c];
      assign d_rd_req_ready[c]              = s_rd_req_ready[c][TSR_SRC_DMA];
      assign d_rd_rsp_valid[c]              = s_rd_rsp_valid[c][TSR_SRC_DMA];
      assign d_rd_rsp_data[c]               = s_rd_rsp_data[c];
      assign d_rd_rsp_last[c]               = s_rd_rsp_last[c];
      assign d_rd_rsp_id[c]                 = s_rd_rsp_id[c];
      assign b_wr_req_valid[c] = a_wr_req_valid[c];
      assign a_wr_req_ready[c] = b_wr_req_ready[c];
      assign b_wr_req_addr[c]  = a_wr_req_addr[c];
      assign b_wr_req_len[c]   = a_wr_req_len[c];
      assign b_wr_req_id[c]    = a_wr_req_id[c];
      assign b_wr_dat_valid[c] = a_wr_dat_valid[c];
      assign a_wr_dat_ready[c] = b_wr_dat_ready[c];
      assign b_wr_dat_data[c]  = a_wr_dat_data[c];
      assign b_wr_dat_mask[c]  = a_wr_dat_mask[c];
      assign b_wr_dat_last[c]  = a_wr_dat_last[c];
      assign a_wr_rsp_valid[c] = b_wr_rsp_valid[c];
      assign b_wr_rsp_ready[c] = a_wr_rsp_ready[c];
      assign a_wr_rsp_id[c]    = b_wr_rsp_id[c];
    end
    // Source WS (M3 weight stream of cluster c; tied off when HAS_MVM = 0).
    assign s_rd_req_valid[c][TSR_SRC_WS]  = w_rd_req_valid[c];
    assign s_rd_req_addr[c][TSR_SRC_WS]   = w_rd_req_addr[c];
    assign s_rd_req_len[c][TSR_SRC_WS]    = w_rd_req_len[c];
    assign s_rd_req_id[c][TSR_SRC_WS]     = w_rd_req_id[c];
    assign s_rd_rsp_ready[c][TSR_SRC_WS]  = w_rd_rsp_ready[c];
    assign w_rd_req_ready[c]              = s_rd_req_ready[c][TSR_SRC_WS];
    assign w_rd_rsp_valid[c]              = s_rd_rsp_valid[c][TSR_SRC_WS];

    // Write source: the DMA's port, on channel TSR_TRW_CH through the trace writer's mux (DB §8.3).
    if (HAS_DB && HAS_TRW && c == tsr_tables_pkg::TSR_TRW_CH) begin : g_trwmux
      tsr_dbg_wrmux #(.ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS)) u_wrmux (
        .clk, .rst_n(rst_b_n[RB_DBG]),
        .d_wr_req_valid(d_wr_req_valid[c]), .d_wr_req_ready(d_wr_req_ready[c]),
        .d_wr_req_addr(d_wr_req_addr[c]), .d_wr_req_len(d_wr_req_len[c]), .d_wr_req_id(d_wr_req_id[c]),
        .d_wr_dat_valid(d_wr_dat_valid[c]), .d_wr_dat_ready(d_wr_dat_ready[c]),
        .d_wr_dat_data(d_wr_dat_data[c]), .d_wr_dat_mask(d_wr_dat_mask[c]), .d_wr_dat_last(d_wr_dat_last[c]),
        .d_wr_rsp_valid(d_wr_rsp_valid[c]), .d_wr_rsp_ready(d_wr_rsp_ready[c]), .d_wr_rsp_id(d_wr_rsp_id[c]),
        .t_wr_req_valid, .t_wr_req_ready, .t_wr_req_addr, .t_wr_req_len,
        .t_wr_dat_valid, .t_wr_dat_ready, .t_wr_dat_data, .t_wr_dat_last, .t_wr_rsp_valid,
        .s_wr_req_valid(a_wr_req_valid[c]), .s_wr_req_ready(a_wr_req_ready[c]),
        .s_wr_req_addr(a_wr_req_addr[c]), .s_wr_req_len(a_wr_req_len[c]), .s_wr_req_id(a_wr_req_id[c]),
        .s_wr_dat_valid(a_wr_dat_valid[c]), .s_wr_dat_ready(a_wr_dat_ready[c]),
        .s_wr_dat_data(a_wr_dat_data[c]), .s_wr_dat_mask(a_wr_dat_mask[c]), .s_wr_dat_last(a_wr_dat_last[c]),
        .s_wr_rsp_valid(a_wr_rsp_valid[c]), .s_wr_rsp_ready(a_wr_rsp_ready[c]), .s_wr_rsp_id(a_wr_rsp_id[c])
      );
    end else begin : g_wrdirect
      assign a_wr_req_valid[c] = d_wr_req_valid[c];
      assign d_wr_req_ready[c] = a_wr_req_ready[c];
      assign a_wr_req_addr[c]  = d_wr_req_addr[c];
      assign a_wr_req_len[c]   = d_wr_req_len[c];
      assign a_wr_req_id[c]    = d_wr_req_id[c];
      assign a_wr_dat_valid[c] = d_wr_dat_valid[c];
      assign d_wr_dat_ready[c] = a_wr_dat_ready[c];
      assign a_wr_dat_data[c]  = d_wr_dat_data[c];
      assign a_wr_dat_mask[c]  = d_wr_dat_mask[c];
      assign a_wr_dat_last[c]  = d_wr_dat_last[c];
      assign d_wr_rsp_valid[c] = a_wr_rsp_valid[c];
      assign a_wr_rsp_ready[c] = d_wr_rsp_ready[c];
      assign d_wr_rsp_id[c]    = a_wr_rsp_id[c];
    end

    tsr_mem_arb #(
      .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .RD_OUTSTANDING(RD_OUTSTANDING)
    ) u_arb (
      .clk, .rst_n(rst_b_n[RB_ARB + c]),
      .s_rd_req_valid(s_rd_req_valid[c]), .s_rd_req_ready(s_rd_req_ready[c]),
      .s_rd_req_addr(s_rd_req_addr[c]), .s_rd_req_len(s_rd_req_len[c]), .s_rd_req_id(s_rd_req_id[c]),
      .s_rd_rsp_valid(s_rd_rsp_valid[c]), .s_rd_rsp_ready(s_rd_rsp_ready[c]),
      .s_rd_rsp_data(s_rd_rsp_data[c]), .s_rd_rsp_last(s_rd_rsp_last[c]), .s_rd_rsp_id(s_rd_rsp_id[c]),
      .s_wr_req_valid(b_wr_req_valid[c]), .s_wr_req_ready(b_wr_req_ready[c]),
      .s_wr_req_addr(b_wr_req_addr[c]), .s_wr_req_len(b_wr_req_len[c]), .s_wr_req_id(b_wr_req_id[c]),
      .s_wr_dat_valid(b_wr_dat_valid[c]), .s_wr_dat_ready(b_wr_dat_ready[c]),
      .s_wr_dat_data(b_wr_dat_data[c]), .s_wr_dat_mask(b_wr_dat_mask[c]), .s_wr_dat_last(b_wr_dat_last[c]),
      .s_wr_rsp_valid(b_wr_rsp_valid[c]), .s_wr_rsp_ready(b_wr_rsp_ready[c]), .s_wr_rsp_id(b_wr_rsp_id[c]),
      .m_rd_req_valid(m_rd_req_valid[c]), .m_rd_req_ready(m_rd_req_ready[c]),
      .m_rd_req_addr(m_rd_req_addr[c]), .m_rd_req_len(m_rd_req_len[c]), .m_rd_req_id(m_rd_req_id[c]),
      .m_rd_rsp_valid(m_rd_rsp_valid[c]), .m_rd_rsp_ready(m_rd_rsp_ready[c]),
      .m_rd_rsp_data(m_rd_rsp_data[c]), .m_rd_rsp_id(m_rd_rsp_id[c]), .m_rd_rsp_last(m_rd_rsp_last[c]),
      .m_wr_req_valid(m_wr_req_valid[c]), .m_wr_req_ready(m_wr_req_ready[c]),
      .m_wr_req_addr(m_wr_req_addr[c]), .m_wr_req_len(m_wr_req_len[c]), .m_wr_req_id(m_wr_req_id[c]),
      .m_wr_dat_valid(m_wr_dat_valid[c]), .m_wr_dat_ready(m_wr_dat_ready[c]),
      .m_wr_dat_data(m_wr_dat_data[c]), .m_wr_dat_mask(m_wr_dat_mask[c]), .m_wr_dat_last(m_wr_dat_last[c]),
      .m_wr_rsp_valid(m_wr_rsp_valid[c]), .m_wr_rsp_ready(m_wr_rsp_ready[c]), .m_wr_rsp_id(m_wr_rsp_id[c]),
      .perf_rd_beat(perf_ch_rd_beat[c]), .perf_rd_wait(perf_ch_wait[c])
    );
  end

  // Arbiter outputs without a consumer: AUX ready/valid on every channel.
  logic unused_aux;
  always_comb begin
    unused_aux = 1'b0;
    for (int c = 0; c < NUM_CH; c++) begin
      unused_aux |= s_rd_req_ready[c][TSR_SRC_AUX] | s_rd_rsp_valid[c][TSR_SRC_AUX];
    end
  end

  // -------------------------------------------------------------------------------------------
  // Watchdog progress (EX §8.6, tsr_csr): per datapath unit, any SPM handshake, memory beat or
  // done of the unit (the VU's own progress strobe). The AT's ops count for the units they run on; SYNC
  // entries never hold a unit.
  // -------------------------------------------------------------------------------------------
  localparam int unsigned U_DMA = 0, U_QACT = 1, U_MVM = 2;    // unit ids (CC §7.4, TSR_UNIT_*)
  logic [TSR_SPM_NSP-1:0] spm_hs;                               // per single-port SPM client
  logic                   dma_mem_beat, ws_mem_beat;
  always_comb begin
    dma_mem_beat = 1'b0;
    ws_mem_beat  = 1'b0;
    for (int c = 0; c < NUM_CH; c++) begin
      dma_mem_beat |= (d_rd_rsp_valid[c] && d_rd_rsp_ready[c]) || (d_wr_dat_valid[c] && d_wr_dat_ready[c]) ||
                      (d_wr_rsp_valid[c] && d_wr_rsp_ready[c]);
      ws_mem_beat  |= w_rd_rsp_valid[c] && w_rd_rsp_ready[c];
    end
    for (int c = 0; c < TSR_SPM_NSP; c++) begin
      spm_hs[c] = (spm_rd_req_valid[c] && spm_rd_req_ready[c]) || (spm_wr_valid[c] && spm_wr_ready[c]) ||
                  spm_rd_rsp_valid[c];
    end
  end
  assign unit_progress[U_DMA]  = dma_done || dma_mem_beat || spm_hs[TSR_SPM_CL_DMA];
  assign unit_progress[U_QACT] = qact_done || spm_hs[TSR_SPM_CL_QACT];
  assign unit_progress[U_MVM]  = mvm_done || ws_mem_beat || spm_hs[TSR_SPM_CL_MVM];
  wire   unused_spm_hs         = &{1'b0, spm_hs[TSR_SPM_CL_BRIDGE]};
  assign unit_progress[3]             = vu_progress;
  assign unit_progress[5:4]           = '0;

  // -------------------------------------------------------------------------------------------
  // Integration checks (simulation only; docs/rtl-m2.md §3.0, §5; docs/rtl-m3.md §3.0, §6.4,
  // §10.4; CC §9.4 I4). Armed by the first clock edge with rst_n low (before it, Verilator's X
  // randomisation makes state arbitrary); with RST_PIPE they see the blocks' delayed reset. Only
  // the unit-local rules are checked here: the units run concurrently, and tsr_issue asserts the
  // dispatcher's invariants (CC §9.4 I1-I3).
  //   U1 a unit pulses done only while it holds an op (never in its accepting cycle); no MVM
  //      command with HAS_MVM = 0. MVM: cmd_ready is low while it holds the op, and perf_mvm_busy
  //      equals "holds an op" in every cycle (from the cycle after the accepting edge up to and
  //      including the cycle of done = PERF_MVM_CYCLES), and so does tsr_issue's perf_busy[MVM].
  //   U2 (CC §9.4 I4) a unit touches the SPM only while it holds an op (its done means its SPM
  //      effects are complete): DMA / QACT (request in the accepting cycle allowed, as in M2), MVM
  //      strictly after its accept; an MVM SPM read response only while it holds the op.
  //      Weight-stream requests and responses (source WS of every arbiter) only while the MVM
  //      unit holds an op; DMA memory requests only while the DMA holds an op.
  //   U3 MVM done => quiescent and exact (docs/rtl-m3.md §3.0, §4, §6.4): in the cycle of
  //      mvm_done no WS request is offered on any channel, every WS beat requested has been
  //      received (counted at the arbiter's source port), no MVM SPM request or write is offered
  //      and no MVM SPM read is outstanding;
  //      channel c requested exactly rows_c x ROW_BEATS beats (rows_c = max(0, min(R, N - c*R)))
  //      and never more at any point; WS requests follow the §4 burst rule (len =
  //      min(64 - addr % 64, beats left), contiguous from w_off). An attention command (ATTQK / ATTPV)
  //      streams KV planes through the same source: for it only the unit-local rules hold (no traffic
  //      outside the unit's op, nothing outstanding at done, a response only after a request).
  //   U4 (W4-B) the VU (accepts up to 2 ops, CC §9.4 I3): done only while it holds an op, never more than
  //      2 in flight, no command with HAS_VU = 0, SPM access only while it holds an op (or in the accepting
  //      cycle).
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic dma_busy_q, qact_busy_q, mvm_busy_q;
  wire  dma_take  = dma_cmd_valid && dma_cmd_ready;
  wire  qact_take = qact_cmd_valid && qact_cmd_ready;
  wire  mvm_take  = mvm_cmd_valid && mvm_cmd_ready;
  wire  vu_take   = vu_cmd_valid && vu_cmd_ready;
  wire  vu_spm    = (|vu_rd_req_valid) || (|vu_wr_valid);
  logic [1:0] vu_cnt_q;                  // VU ops in flight (accepted, not done)
  wire  dma_spm   = spm_rd_req_valid[TSR_SPM_CL_DMA] || spm_wr_valid[TSR_SPM_CL_DMA];
  wire  qact_spm  = spm_rd_req_valid[TSR_SPM_CL_QACT] || spm_wr_valid[TSR_SPM_CL_QACT];
  wire  mvm_spm   = spm_rd_req_valid[TSR_SPM_CL_MVM] || spm_wr_valid[TSR_SPM_CL_MVM];
  wire  dma_mreq  = |d_rd_req_valid || |d_wr_req_valid;

  // Same arming scheme as tsr_spm / tsr_mem_arb: the initial block runs before Verilator's first
  // evaluation, which may already see a clock edge with X-randomised rst_n and state.
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_b_n[RB_CHK]) chk_on_q <= 1'b1;

  // U3 bookkeeping: per channel WS beats expected / requested / received in the current MVM,
  // next expected request address; MVM SPM reads outstanding.
  logic [NUM_CH-1:0][47:0] ws_exp_q, ws_req_q, ws_rsp_q;
  logic [NUM_CH-1:0][63:0] ws_addr_q;
  logic [7:0]              mvm_rd_out_q;
  logic                    att_q;                // the MVM unit holds an attention command (cmd_xc)
  wire                     mvm_rd_take = spm_rd_req_valid[TSR_SPM_CL_MVM] && spm_rd_req_ready[TSR_SPM_CL_MVM];

  function automatic logic [47:0] ws_beats_exp(logic [31:0] n, logic [31:0] r, logic [31:0] row_beats,
                                               int unsigned ch);
    logic [31:0] rows;
    rows = tsr_mvm_rows_c(n, r, 64'(ch) * 64'(r));
    return 48'(rows) * 48'(row_beats);
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_b_n[RB_CHK]) begin
      dma_busy_q   <= 1'b0;
      qact_busy_q  <= 1'b0;
      mvm_busy_q   <= 1'b0;
      mvm_rd_out_q <= '0;
      vu_cnt_q     <= '0;
      att_q        <= 1'b0;
      ws_exp_q     <= '0;
      ws_req_q     <= '0;
      ws_rsp_q     <= '0;
      ws_addr_q    <= '0;
    end else if (chk_on_q) begin
      if (dma_take)   dma_busy_q  <= 1'b1;
      if (dma_done)   dma_busy_q  <= 1'b0;
      if (qact_take)  qact_busy_q <= 1'b1;
      if (qact_done)  qact_busy_q <= 1'b0;
      if (mvm_take)   mvm_busy_q  <= 1'b1;
      if (mvm_take)   att_q       <= mvm_cmd_xc;
      if (mvm_done)   mvm_busy_q  <= 1'b0;
      vu_cnt_q <= vu_cnt_q + 2'(vu_take) - 2'(vu_done);
      // U4
      assert (!(vu_cmd_valid && !HAS_VU))
        else $error("tsr_core_dp: VU command offered with HAS_VU = 0");
      assert (!(vu_done && vu_cnt_q == 2'd0))
        else $error("tsr_core_dp: VU done without an op in flight");
      assert (!(vu_take && vu_cnt_q - 2'(vu_done) >= 2'd2))
        else $error("tsr_core_dp: VU accepted a third op (S - D <= 2, CC §9.4 I3)");
      assert (!(vu_spm && vu_cnt_q == 2'd0 && !vu_take))
        else $error("tsr_core_dp: SPM access by an idle VU");
      // U1
      assert (!(mvm_cmd_valid && !HAS_MVM))
        else $error("tsr_core_dp: MVM command offered with HAS_MVM = 0");
      assert (!(dma_done && !dma_busy_q) && !(qact_done && !qact_busy_q) && !(mvm_done && !mvm_busy_q))
        else $error("tsr_core_dp: unit done without an op in flight (dma %0b qact %0b mvm %0b)",
                    dma_done, qact_done, mvm_done);
      assert (!(mvm_cmd_ready && mvm_busy_q))
        else $error("tsr_core_dp: MVM unit ready while it holds an op");
      assert (perf_mvm_busy == mvm_busy_q && perf_busy[U_MVM] == mvm_busy_q)
        else $error("tsr_core_dp: perf_mvm_busy %0b / perf_busy[MVM] %0b while the MVM unit %s an op",
                    perf_mvm_busy, perf_busy[U_MVM], mvm_busy_q ? "holds" : "does not hold");
      // U2 (I4)
      assert (!(dma_spm && !dma_busy_q && !dma_take) && !(qact_spm && !qact_busy_q && !qact_take) &&
              !(mvm_spm && !mvm_busy_q))
        else $error("tsr_core_dp: SPM access by an idle unit (dma %0b qact %0b mvm %0b)", dma_spm, qact_spm, mvm_spm);
      assert (!(dma_mreq && !dma_busy_q && !dma_take))
        else $error("tsr_core_dp: DMA memory request while the DMA holds no op");
      assert (!(spm_rd_rsp_valid[TSR_SPM_CL_MVM] && !mvm_busy_q))
        else $error("tsr_core_dp: MVM SPM read response after the MVM unit's done");
      // U3: MVM SPM read accounting
      mvm_rd_out_q <= mvm_rd_out_q + 8'(mvm_rd_take) - 8'(spm_rd_rsp_valid[TSR_SPM_CL_MVM]);
      assert (!(spm_rd_rsp_valid[TSR_SPM_CL_MVM] && mvm_rd_out_q == '0 && !mvm_rd_take))
        else $error("tsr_core_dp: MVM SPM read response without a request");
      if (mvm_done) begin
        assert (!mvm_spm && mvm_rd_out_q == '0)
          else $error("tsr_core_dp: MVM done with SPM traffic (request %0b, reads outstanding %0d)",
                      mvm_spm, mvm_rd_out_q);
      end
      // U3: weight streams, per channel
      for (int c = 0; c < NUM_CH; c++) begin
        automatic logic req = s_rd_req_valid[c][TSR_SRC_WS] && s_rd_req_ready[c][TSR_SRC_WS];
        automatic logic rsp = s_rd_rsp_valid[c][TSR_SRC_WS] && s_rd_rsp_ready[c][TSR_SRC_WS];
        automatic logic [47:0] left = ws_exp_q[c] - ws_req_q[c];
        automatic logic [47:0] want_len = (left < 48'(64 - ws_addr_q[c][5:0])) ? left : 48'(64 - ws_addr_q[c][5:0]);
        assert (!((s_rd_req_valid[c][TSR_SRC_WS] || s_rd_rsp_valid[c][TSR_SRC_WS]) && !mvm_busy_q))
          else $error("tsr_core_dp: weight-stream traffic on channel %0d while no MVM is in flight", c);
        assert (s_rd_rsp_ready[c][TSR_SRC_WS] || !HAS_MVM)
          else $error("tsr_core_dp: WS rd_rsp_ready low on channel %0d (rtl-m3 §4: constant 1)", c);
        if (mvm_take) begin
          // an attention command (cmd_xc: ATTQK / ATTPV) streams its KV planes through the same source with
          // its own request pattern (AT §8); only the unit-local rules are checked for it (att_q)
          ws_exp_q[c]  <= mvm_cmd_xc ? 48'd0 : ws_beats_exp(mvm_cmd.n, mvm_cmd.rows_per_ch, mvm_cmd.row_beats, c);
          ws_req_q[c]  <= '0;
          ws_rsp_q[c]  <= '0;
          ws_addr_q[c] <= mvm_cmd_xc ? 64'd0 : 64'(mvm_cmd.w_off);
        end else begin
          if (req) begin
            ws_req_q[c]  <= ws_req_q[c] + 48'(s_rd_req_len[c][TSR_SRC_WS]);
            ws_addr_q[c] <= ws_addr_q[c] + 64'(s_rd_req_len[c][TSR_SRC_WS]);
            assert (att_q ||
                    (64'(s_rd_req_addr[c][TSR_SRC_WS]) == ws_addr_q[c] &&
                     48'(s_rd_req_len[c][TSR_SRC_WS]) == want_len && want_len != '0))
              else $error("tsr_core_dp: WS request ch%0d addr %0d len %0d, expected addr %0d len %0d (rtl-m3 §4)",
                          c, s_rd_req_addr[c][TSR_SRC_WS], s_rd_req_len[c][TSR_SRC_WS], ws_addr_q[c], want_len);
            assert (s_rd_req_len[c][TSR_SRC_WS] != '0)
              else $error("tsr_core_dp: WS request ch%0d with length 0", c);
          end
          if (rsp) ws_rsp_q[c] <= ws_rsp_q[c] + 48'd1;
        end
        assert (!(rsp && ws_rsp_q[c] >= ws_req_q[c]))
          else $error("tsr_core_dp: WS response beat on channel %0d without a request", c);
        if (mvm_done) begin
          assert (!s_rd_req_valid[c][TSR_SRC_WS] && (att_q || ws_req_q[c] == ws_exp_q[c]) && ws_rsp_q[c] == ws_req_q[c])
            else $error("tsr_core_dp: MVM done on channel %0d with WS request %0b, %0d beats requested, %0d received, %0d expected (rows_c x ROW_BEATS)",
                        c, s_rd_req_valid[c][TSR_SRC_WS], ws_req_q[c], ws_rsp_q[c], ws_exp_q[c]);
        end
      end
    end
  end
`endif

endmodule
