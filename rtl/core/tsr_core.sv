// tsr_core.sv — the Tesseract core: the RISC-V control subsystem (tsr_ctrl: Ibex, debug module,
// TCMs, control window, every clock-domain crossing; rtl/ctrl) and the core-clock datapath
// (tsr_core_dp: issue path, CSR block, units, SPM, memory arbiters) (docs/design/04-control-core.md
// "CC" §2, §13.1-§13.2).
//
// Owner: integration (W4-A). The control core is mandatory (D14): the host loads a firmware image
// through the CSR port and the RISC-V issues every tile op; there is no host command path.
//
// Core boundary (spec Core boundary interface; EX §9.3, CC §13.1):
//   - clk / rst_n: the core clock and its synchronous, active-low reset (already synchronised to
//     clk). clk_rv / rst_rv_n: the control core's clock (ap_clk_2, 166.7 MHz planning value, CC
//     §11.1) and its reset (ap_rst_n_2), AND-ed only into the Ibex and debug-module resets; the
//     clk_rv domain's own reset is rst_n through tsr_ctrl's synchroniser (CC §12.2).
//   - CSR port: csr_valid/csr_ready/csr_we/csr_addr/csr_wdata/csr_rdata (tsr_csr contract, EX §5).
//   - memory port per channel c (packed [NUM_CH], index c): read request {addr_beat, len_beats, id},
//     read response {data, id, last}, write request {addr_beat, len_beats, id}, write data
//     {data, word mask, last}, write response {id}; all valid/ready (docs/rtl-m2.md §4). Every output is
//     driven from a register inside tsr_mem_arb.
//   - plat_err_i (the platform's error level, STATUS.PLAT_ERR) and host_irq_o = CTRL.IRQ_EN ∧
//     (LCQ_WR ≠ LCQ_RD) (EX §5.2; not connected on the U200).
// Units: DMA (LD / ST; KVST with HAS_ATT), QACT and, with HAS_MVM = 1 (default: the U200 configuration),
//   MVM (ATTQK / ATTPV with HAS_ATT) and, with HAS_VU, the vector unit (tsr_core_dp header). The DB block
//   (HAS_DB): tsr_dbg and the trace writer in tsr_core_dp, tsr_dbg_rv here (both clocks: the completion
//   records, RV_SAMPLE, the RV watchpoints on Ibex's data port, DBG_FLAGS and tsr_ctrl's DB window).
// Parameters: the datapath's (tsr_core_dp; results identical for every legal implementation value,
//   tests vary them: SPM_RD_LAT, MVM_* (docs/rtl-m3.md §1), RST_PIPE (reset register stages,
//   docs/rtl-m3.md §14; the control subsystem's core-domain reset is one of its copies)), the queue
//   depths QD_DMA / QD_QACT / QD_MVM / QD_SYNC (CC §7.4; tsr_ctrl's Q_DEPTH is built from the same
//   values, so the RISC-V's UNIT_OF and the dispatcher agree), ITCM_BYTES / DTCM_BYTES (CC-3) and
//   CTRL_SYNC (1: clk_rv is clk, the crossings keep their structure, CC §11.1).
// Verification: rtl/lint.sh (every configuration); the block benches (tb/chk, tb/issue, tb/csr,
//   tb/spm, tb/dma, tb/qact, tb/mvm*, tb/ctrl); tb/core (firmware on the RTL Ibex pushing ops
//   through the issue path, results against the simulator); the simulation-only integration checks
//   are in tsr_core_dp, tsr_issue and tsr_ctrl.

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
  parameter int unsigned     SPM_RD_LAT     = TSR_DEF_SPM_RD_LAT,
  // MVM unit implementation parameters (HAS_MVM = 1; docs/rtl-m3.md §1)
  parameter int unsigned     MVM_WS_FIFO    = TSR_DEF_WS_FIFO_BEATS,
  parameter int unsigned     MVM_BC_PIPE    = TSR_DEF_MVM_BC_PIPE,
  parameter bit              DDR_LOCAL_PIPE = 1'b0,
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
  // Reset register stages inside the core (implementation; docs/rtl-m3.md §14): 0 = rst_n
  // straight to every block (test builds); U200: 3 (the measured OOC closure, §14.2)
  parameter int unsigned     RST_PIPE       = 0,
  // issue path and control core (CC §7.4, §13.1)
  parameter int unsigned     QD_DMA         = tsr_tables_pkg::TSR_QDEPTH_U200_DMA,
  parameter int unsigned     QD_QACT        = tsr_tables_pkg::TSR_QDEPTH_U200_QACT,
  parameter int unsigned     QD_MVM         = tsr_tables_pkg::TSR_QDEPTH_U200_MVM,
  parameter int unsigned     QD_SYNC        = tsr_tables_pkg::TSR_QDEPTH_U200_SYNC,
  parameter int unsigned     ITCM_BYTES     = tsr_tables_pkg::TSR_ITCM_BYTES,
  parameter int unsigned     DTCM_BYTES     = tsr_tables_pkg::TSR_DTCM_BYTES,
  parameter bit              CTRL_SYNC      = 1'b0,
  parameter bit              RV_LOCAL_TRANSPORT = 1'b0,   // narrow links to a relocated control subsystem
  // W4-B (tsr_core_dp): vector unit (queue depth QD_VU; lane knobs VU_EXP_LANES / VU_BACK_LANES, VU §7.10),
  // attention, TOPK / DQROW (M8 RTL, off in the U200 M7 build), the DB block and its sizes (DB §10.3)
  parameter bit              HAS_VU         = 1'b1,
  parameter bit              HAS_ATT        = HAS_MVM,           // needs HAS_VU, HAS_MVM and MVM_MAC_LAT >= 3
  parameter bit              HAS_TOPK       = 1'b0,
  parameter bit              HAS_DQROW      = 1'b0,
  parameter bit              HAS_DB         = 1'b1,
  parameter int unsigned     QD_VU          = tsr_tables_pkg::TSR_QDEPTH_U200_VU,
  parameter int unsigned     VU_FRONT_LANES = tsr_tables_pkg::TSR_VU_FRONT_LANES,
  parameter int unsigned     VU_EXP_LANES   = tsr_tables_pkg::TSR_VU_FRONT_LANES,
  parameter int unsigned     VU_BACK_LANES  = tsr_tables_pkg::TSR_VU_BACK_LANES,
  parameter int unsigned     VU_LANE_PIPE   = 0,                 // register stages on each side of the VU lanes (SLR crossing)
  parameter int unsigned     HIST_DEPTH     = tsr_tables_pkg::TSR_HIST_DEPTH,
  parameter bit              HAS_TRW        = 1'b1,
  parameter int unsigned     NTBP           = tsr_tables_pkg::TSR_DBG_NTBP,
  parameter int unsigned     NRVWP          = tsr_tables_pkg::TSR_DBG_NRVWP,
  parameter bit              HAS_REC        = 1'b1,
  parameter int unsigned     REC_DEPTH      = tsr_tables_pkg::TSR_REC_DEPTH
) (
  input  logic                                   clk,
  // rst_n: synchronous in the core domain, and (RST_PIPE = 0) also the asynchronous input of the
  // clk_rv domain's reset synchroniser inside tsr_ctrl (CC §12.2)
  /* verilator lint_off SYNCASYNCNET */
  input  logic                                   rst_n,
  /* verilator lint_on SYNCASYNCNET */
  input  logic                                   clk_rv,
  input  logic                                   rst_rv_n,

  // ---- CSR port ----
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

  // ---- platform (EX §9.3) ----
  input  logic                                   plat_err_i,
  output logic                                   host_irq_o
);

  // The issue path's unit map (tsr_core_dp): the MVM queue only with the unit, the VU queue only with
  // the VU, unit 4 (AT) reserved (its ops run on the DMA, MVM and VU queues). tsr_ctrl gets the same depths
  // and CAPS (its UNIT_OF, CC §7.5); the formulas are tsr_core_dp's.
  localparam logic [31:0] CAPS     = (HAS_VU    ? tsr_tables_pkg::TSR_CAPS_VU    : 32'd0) |
                                     (HAS_ATT   ? tsr_tables_pkg::TSR_CAPS_ATT   : 32'd0) |
                                     (HAS_TOPK  ? tsr_tables_pkg::TSR_CAPS_TOPK  : 32'd0) |
                                     (HAS_DQROW ? tsr_tables_pkg::TSR_CAPS_DQROW : 32'd0) |
                                     (LOCAL_INTERMEDIATES ? tsr_tables_pkg::TSR_CAPS_ATT_LOCAL : 32'd0) |
                                     (LOCAL_SCALE ? tsr_tables_pkg::TSR_CAPS_MVM_LOCAL_SCALE : 32'd0) |
                                     (LOCAL_SOFTCAP ? tsr_tables_pkg::TSR_CAPS_MVM_LOCAL_SOFTCAP : 32'd0);
  localparam int unsigned QD_MVM_E = HAS_MVM ? QD_MVM : 0;
  localparam int unsigned QD_VU_E  = HAS_VU ? QD_VU : 0;
  localparam logic [5:0][4:0] Q_DEPTH = {5'(QD_SYNC), 5'd0, 5'(QD_VU_E), 5'(QD_MVM_E), 5'(QD_QACT), 5'(QD_DMA)};

  // ---- core-domain side of tsr_ctrl (CC §11.2) ----
  logic                                    rst_ctrl_n;
  logic                                    if_valid, if_ready, issue_valid, issue_ready;
  logic [tsr_tables_pkg::TSR_IFIFO_BITS-1:0] if_data, issue_entry;
  logic [5:0]                              if_count;
  logic [5:0][TSR_SEQ_W-1:0]               issued_seq, s_seq, d_seq;
  logic [5:0]                              err_valid, err_ack;
  logic                                    rec_resync;
  logic [5:0][127:0]                       err_rec;
  logic [3:0]                              q_hold;
  logic [5:0]                              q_err, q_nonempty;
  logic                                    spm_req_valid, spm_req_we, spm_rsp_valid, spm_rsp_err;
  logic                                    spm_bridge_req_valid, spm_bridge_rsp_valid;
  logic                                    spm_bridge_rsp_ready, spm_bridge_rsp_err;
  logic [25:0]                             spm_req_addr;
  logic [3:0]                              spm_req_be;
  logic [31:0]                             spm_req_wdata;
  logic [TSR_LINE_BITS-1:0]                spm_rsp_line, spm_bridge_rsp_line;
  logic                                    cwin_req_valid, cwin_rsp_valid;
  logic [11:0]                             cwin_req_off;
  logic [31:0]                             cwin_rsp_rdata;
  logic                                    lq_req_valid, lq_req_we, lq_rsp_valid, lq_rsp_fault;
  logic [11:0]                             lq_req_off;
  logic [31:0]                             lq_req_wdata, lq_rsp_rdata;
  logic [7:0][31:0]                        lq_req_lcq_w;
  logic                                    lq_csr_req_valid, lq_csr_req_we;
  logic [11:0]                             lq_csr_req_off;
  logic [31:0]                             lq_csr_req_wdata;
  logic [7:0][31:0]                        lq_csr_req_lcq_w;
  logic                                    log_valid, log_ready;
  logic [31:0]                             log_data, rt_state, rt_info;
  logic                                    ld_ready, ld_req, ld_we, ld_rsp_valid, ld_rsp_err, ld_rsp_timeout;
  logic [31:0]                             ld_addr, ld_wdata, ld_rsp_rdata;
  logic                                    dmi_ready, dmi_req, dmi_rsp_valid, dmi_rsp_timeout;
  logic [6:0]                              dmi_addr;
  logic [1:0]                              dmi_op, dmi_rsp_resp;
  logic [31:0]                             dmi_wdata, dmi_rsp_rdata;
  logic [63:0]                             mtime;
  logic [15:0]                             rv_cyc;
  logic [31:0]                             perf_push_stall, perf_bridge_acc, perf_bridge_wait, rv_boot;
  logic                                    push_timeout, bridge_timeout, rv_run, irq_lq, irq_lcq,
                                           perf_time_en, perf_evt_en, rv_halted, rv_sleep, rv_running,
                                           rv_reset_done, rv_clk_ok, rv_ndmreset, rv_dmactive, rv_halt_pend;
  // DB window of the control window (clk_rv) and the DB block's crossings (tsr_dbg_rv <-> tsr_core_dp)
  logic                                    db_req, db_we, db_irq4;
  logic [11:0]                             db_off;
  logic [31:0]                             db_wdata, db_rdata;
  logic                                    dbg_stop, dbg_halt_tgl, dbg_cancel_tgl, dbg_halt_on_boot;
  logic                                    rec_wr, rec_room, rec_all, smp_ready, smp_req, smp_local, smp_done,
                                           smp_timeout, wpa_ready, wpa_req, wpa_local, wpa_done, wpa_timeout,
                                           wph_v, wph_halt, dfault;
  logic [127:0]                            rec_wdata;
  logic [163:0]                            smp_data;
  logic [135:0]                            wpa_data;
  logic [1:0]                              wph_mask;
  logic [31:0]                             wph_addr;
  logic [4:0]                              dbg_flags;
  // Ibex taps (clk_rv) for tsr_dbg_rv
  logic                                    tap_data_req, tap_data_gnt, tap_data_we, tap_sleep, tap_halted,
                                           tap_in_reset, tap_dfault;
  logic [31:0]                             tap_data_addr, tap_cd_pc, tap_cd_npc, tap_cd_daddr, tap_cd_epc,
                                           tap_cd_eaddr;
  logic [3:0]                              tap_data_be;

  // -------------------------------------------------------------------------------------------
  // Control subsystem (rtl/ctrl/tsr_ctrl.sv; CC §2-§5, §11-§12)
  // -------------------------------------------------------------------------------------------
  tsr_ctrl #(
    .ITCM_BYTES(ITCM_BYTES), .DTCM_BYTES(DTCM_BYTES), .SPM_BYTES(SPM_BYTES), .CAPS(CAPS),
    .Q_DEPTH(Q_DEPTH), .CTRL_SYNC(CTRL_SYNC)
  ) u_ctrl (
    .clk, .rst_n(rst_ctrl_n), .clk_rv, .rst_rv_n,
    .if_valid_o(if_valid), .if_ready_i(if_ready), .if_data_o(if_data), .if_count_o(if_count),
    .issued_seq_o(issued_seq), .s_seq_i(s_seq), .d_seq_i(d_seq),
    .err_valid_i(err_valid), .err_rec_i(err_rec), .err_ack_o(err_ack),
    .rec_resync_o(rec_resync),
    .spm_req_valid_o(spm_req_valid), .spm_req_we_o(spm_req_we), .spm_req_addr_o(spm_req_addr),
    .spm_req_be_o(spm_req_be), .spm_req_wdata_o(spm_req_wdata),
    .spm_rsp_valid_i(spm_rsp_valid), .spm_rsp_line_i(spm_rsp_line), .spm_rsp_err_i(spm_rsp_err),
    .cwin_req_valid_o(cwin_req_valid), .cwin_req_off_o(cwin_req_off),
    .cwin_rsp_valid_i(cwin_rsp_valid), .cwin_rsp_rdata_i(cwin_rsp_rdata),
    .lq_req_valid_o(lq_req_valid), .lq_req_off_o(lq_req_off), .lq_req_we_o(lq_req_we),
    .lq_req_wdata_o(lq_req_wdata), .lq_req_lcq_w_o(lq_req_lcq_w),
    .lq_rsp_valid_i(lq_rsp_valid), .lq_rsp_rdata_i(lq_rsp_rdata), .lq_rsp_fault_i(lq_rsp_fault),
    .log_valid_o(log_valid), .log_data_o(log_data), .log_ready_i(log_ready),
    .rt_state_o(rt_state), .rt_info_o(rt_info),
    .ld_ready_o(ld_ready), .ld_req_i(ld_req), .ld_we_i(ld_we), .ld_addr_i(ld_addr), .ld_wdata_i(ld_wdata),
    .ld_rsp_valid_o(ld_rsp_valid), .ld_rsp_rdata_o(ld_rsp_rdata), .ld_rsp_err_o(ld_rsp_err),
    .ld_rsp_timeout_o(ld_rsp_timeout),
    .dmi_ready_o(dmi_ready), .dmi_req_i(dmi_req), .dmi_addr_i(dmi_addr), .dmi_op_i(dmi_op),
    .dmi_wdata_i(dmi_wdata), .dmi_rsp_valid_o(dmi_rsp_valid), .dmi_rsp_rdata_o(dmi_rsp_rdata),
    .dmi_rsp_resp_o(dmi_rsp_resp), .dmi_rsp_timeout_o(dmi_rsp_timeout),
    .mtime_i(mtime), .rv_cyc_o(rv_cyc), .perf_push_stall_o(perf_push_stall),
    .perf_bridge_acc_o(perf_bridge_acc), .perf_bridge_wait_o(perf_bridge_wait),
    .push_timeout_o(push_timeout), .bridge_timeout_o(bridge_timeout),
    .rv_run_i(rv_run), .rv_boot_i(rv_boot),
    // DB: the stop level, the halt request / cancel toggles, halt-on-boot (tsr_dbg through tsr_core_dp)
    .dbg_stop_i(dbg_stop), .dbg_halt_tgl_i(dbg_halt_tgl), .dbg_cancel_tgl_i(dbg_cancel_tgl),
    .dbg_halt_on_boot_i(dbg_halt_on_boot),
    .irq_lq_i(irq_lq), .irq_lcq_i(irq_lcq), .perf_time_en_i(perf_time_en), .perf_evt_en_i(perf_evt_en),
    .q_hold_i(q_hold), .q_err_i(q_err), .q_nonempty_i(q_nonempty),
    .rv_halted_o(rv_halted), .rv_sleep_o(rv_sleep), .rv_running_o(rv_running),
    .rv_reset_done_o(rv_reset_done), .rv_clk_ok_o(rv_clk_ok), .rv_ndmreset_o(rv_ndmreset),
    .rv_dmactive_o(rv_dmactive),
    .rv_halt_pend_o(rv_halt_pend),
    .db_req_o(db_req), .db_we_o(db_we), .db_off_o(db_off), .db_wdata_o(db_wdata), .db_rdata_i(db_rdata),
    .db_irq4_i(db_irq4), .db_irq6_i(1'b0),
    .tap_data_req_o(tap_data_req), .tap_data_gnt_o(tap_data_gnt), .tap_data_addr_o(tap_data_addr),
    .tap_data_we_o(tap_data_we), .tap_data_be_o(tap_data_be), .tap_cd_pc_o(tap_cd_pc),
    .tap_cd_npc_o(tap_cd_npc), .tap_cd_daddr_o(tap_cd_daddr), .tap_cd_epc_o(tap_cd_epc),
    .tap_cd_eaddr_o(tap_cd_eaddr), .tap_sleep_o(tap_sleep), .tap_halted_o(tap_halted),
    .tap_in_reset_o(tap_in_reset), .tap_dfault_o(tap_dfault)
  );

  // -------------------------------------------------------------------------------------------
  // DB block, both-clock part (tsr_dbg_rv: DB window registers, completion-record FIFO, RV_SAMPLE and
  // RVWP_APPLY handshakes, RV watchpoints, DBG_FLAGS; DB §10.2). Its core-domain side is tsr_dbg in
  // tsr_core_dp; its reset is tsr_ctrl's core-domain copy (also the input of its clk_rv synchroniser).
  // -------------------------------------------------------------------------------------------
  if (HAS_DB) begin : g_db
    tsr_dbg_rv #(.REC_DEPTH(REC_DEPTH), .HAS_RV_TAPS(1'b1), .CTRL_SYNC(CTRL_SYNC)) u_dbg_rv (
      .clk, .rst_n(rst_ctrl_n),
      .rec_wr, .rec_wdata, .rec_room, .rec_all, .smp_ready, .smp_req, .smp_local, .smp_done, .smp_timeout,
      .smp_data, .wpa_ready, .wpa_req, .wpa_local, .wpa_data, .wpa_done, .wpa_timeout,
      .wph_v, .wph_mask, .wph_addr, .wph_halt, .dfault, .flags(dbg_flags), .rv_run,
      .clk_rv, .db_req, .db_we, .db_off, .db_wdata, .db_rdata, .db_irq4,
      .ib_data_req(tap_data_req), .ib_data_gnt(tap_data_gnt), .ib_data_addr(tap_data_addr),
      .ib_data_we(tap_data_we), .ib_data_be(tap_data_be),
      .ib_cd_pc(tap_cd_pc), .ib_cd_npc(tap_cd_npc), .ib_cd_daddr(tap_cd_daddr), .ib_cd_epc(tap_cd_epc),
      .ib_cd_eaddr(tap_cd_eaddr),
      .ib_sleep(tap_sleep), .ib_halted(tap_halted), .ib_in_reset(tap_in_reset), .ib_dfault(tap_dfault)
    );
  end else begin : g_nodb
    // HAS_DB = 0: no records, no samples, no watchpoints; the DB window reads 0
    assign db_rdata  = 32'h0;
    assign db_irq4   = 1'b0;
    assign rec_room  = 1'b0;
    assign rec_all   = 1'b0;
    assign smp_ready = 1'b0;
    assign smp_done  = 1'b0;
    assign smp_timeout = 1'b0;
    assign smp_data  = '0;
    assign wpa_ready = 1'b0;
    assign wpa_done  = 1'b0;
    assign wpa_timeout = 1'b0;
    assign wph_v     = 1'b0;
    assign wph_mask  = '0;
    assign wph_addr  = '0;
    assign wph_halt  = 1'b0;
    assign dfault    = 1'b0;
    wire unused_nodb = &{1'b0, db_req, db_we, db_off, db_wdata, rec_wr, rec_wdata, smp_req, smp_local, wpa_req,
                         wpa_local, wpa_data, dbg_flags, tap_data_req, tap_data_gnt, tap_data_addr, tap_data_we,
                         tap_data_be, tap_cd_pc, tap_cd_npc, tap_cd_daddr, tap_cd_epc, tap_cd_eaddr, tap_sleep,
                         tap_halted, tap_in_reset, tap_dfault};
  end

  // The CDC stays local to u_ctrl. The segmented links run on clk, with
  // the same reset on both ends. Each endpoint is a placement unit.
  if (RV_LOCAL_TRANSPORT) begin : g_rv_local
    logic         spm_v, spm_err, issue_v, issue_ack, lq_v, lq_we, lq_end;
    logic [127:0] spm_data, lq_data;
    logic [128:0] issue_data;
    logic [11:0]  lq_off;

    (* keep_hierarchy = "yes" *) tsr_rv_spm_tx u_spm_tx (
      .clk, .rst_n(rst_ctrl_n), .req_valid(spm_req_valid), .bridge_req_valid(spm_bridge_req_valid),
      .bridge_rsp_valid(spm_bridge_rsp_valid), .bridge_rsp_ready(spm_bridge_rsp_ready),
      .bridge_rsp_line(spm_bridge_rsp_line), .bridge_rsp_err(spm_bridge_rsp_err),
      .link_valid(spm_v), .link_data(spm_data), .link_err(spm_err)
    );
    (* keep_hierarchy = "yes" *) tsr_rv_spm_rx u_spm_rx (
      .clk, .rst_n(rst_ctrl_n), .link_valid(spm_v), .link_data(spm_data), .link_err(spm_err),
      .rsp_valid(spm_rsp_valid), .rsp_line(spm_rsp_line), .rsp_err(spm_rsp_err)
    );
    (* keep_hierarchy = "yes" *) tsr_rv_issue_tx u_issue_tx (
      .clk, .rst_n(rst_ctrl_n), .fifo_valid(if_valid), .fifo_entry(if_data), .fifo_ready(if_ready),
      .link_valid(issue_v), .link_data(issue_data), .link_accepted(issue_ack)
    );
    (* keep_hierarchy = "yes" *) tsr_rv_issue_rx u_issue_rx (
      .clk, .rst_n(rst_ctrl_n), .link_valid(issue_v), .link_data(issue_data), .link_accepted(issue_ack),
      .issue_valid, .issue_entry, .issue_ready
    );
    (* keep_hierarchy = "yes" *) tsr_rv_lq_tx u_lq_tx (
      .clk, .rst_n(rst_ctrl_n), .req_valid(lq_req_valid), .req_off(lq_req_off), .req_we(lq_req_we),
      .req_wdata(lq_req_wdata), .req_lcq_w(lq_req_lcq_w), .link_valid(lq_v), .link_data(lq_data),
      .link_off(lq_off), .link_we(lq_we), .link_end(lq_end)
    );
    (* keep_hierarchy = "yes" *) tsr_rv_lq_rx u_lq_rx (
      .clk, .rst_n(rst_ctrl_n), .link_valid(lq_v), .link_data(lq_data), .link_off(lq_off),
      .link_we(lq_we), .link_end(lq_end), .req_valid(lq_csr_req_valid), .req_off(lq_csr_req_off),
      .req_we(lq_csr_req_we), .req_wdata(lq_csr_req_wdata), .req_lcq_w(lq_csr_req_lcq_w)
    );
  end else begin : g_rv_direct
    assign spm_bridge_req_valid = spm_req_valid;
    assign spm_bridge_rsp_ready = 1'b1;
    assign spm_rsp_valid        = spm_bridge_rsp_valid;
    assign spm_rsp_line         = spm_bridge_rsp_line;
    assign spm_rsp_err          = spm_bridge_rsp_err;
    assign issue_valid          = if_valid;
    assign issue_entry          = if_data;
    assign if_ready             = issue_ready;
    assign lq_csr_req_valid     = lq_req_valid;
    assign lq_csr_req_off       = lq_req_off;
    assign lq_csr_req_we        = lq_req_we;
    assign lq_csr_req_wdata     = lq_req_wdata;
    assign lq_csr_req_lcq_w     = lq_req_lcq_w;
  end

  // -------------------------------------------------------------------------------------------
  // Core-clock datapath (CC §13.2)
  // -------------------------------------------------------------------------------------------
  tsr_core_dp #(
    .NUM_CH(NUM_CH), .BEAT_BITS(BEAT_BITS), .LANES(LANES), .T_MAX(T_MAX), .K_MAX(K_MAX),
    .SPM_BYTES(SPM_BYTES), .VL(VL), .NSREG(NSREG), .RD_OUTSTANDING(RD_OUTSTANDING),
    .CH_BYTES(CH_BYTES), .ADDR_BEAT_W(ADDR_BEAT_W), .HAS_MVM(HAS_MVM), .SPM_RD_LAT(SPM_RD_LAT),
    .LOCAL_INTERMEDIATES(LOCAL_INTERMEDIATES), .LOCAL_SCALE(LOCAL_SCALE), .LOCAL_SOFTCAP(LOCAL_SOFTCAP),
    .MVM_WS_FIFO(MVM_WS_FIFO), .MVM_BC_PIPE(MVM_BC_PIPE), .MVM_RES_PIPE(MVM_RES_PIPE), .MVM_GATHER_PAIR(MVM_GATHER_PAIR), .MVM_GATHER_FLITS(MVM_GATHER_FLITS), .DDR_LOCAL_PIPE(DDR_LOCAL_PIPE),
    .MVM_ACT_RD_LAT(MVM_ACT_RD_LAT), .MVM_MAC_LAT(MVM_MAC_LAT), .MVM_MUL_LAT(MVM_MUL_LAT),
    .MVM_ADD_LAT(MVM_ADD_LAT), .MVM_WS_RD_LAT(MVM_WS_RD_LAT), .RST_PIPE(RST_PIPE),
    .RV_LOCAL_TRANSPORT(RV_LOCAL_TRANSPORT),
    .QD_DMA(QD_DMA), .QD_QACT(QD_QACT), .QD_MVM(QD_MVM), .QD_SYNC(QD_SYNC),
    .ITCM_BYTES(ITCM_BYTES), .DTCM_BYTES(DTCM_BYTES),
    .HAS_VU(HAS_VU), .HAS_ATT(HAS_ATT), .HAS_TOPK(HAS_TOPK), .HAS_DQROW(HAS_DQROW), .HAS_DB(HAS_DB),
    .QD_VU(QD_VU), .VU_FRONT_LANES(VU_FRONT_LANES), .VU_EXP_LANES(VU_EXP_LANES), .VU_BACK_LANES(VU_BACK_LANES), .VU_LANE_PIPE(VU_LANE_PIPE),
    .HIST_DEPTH(HIST_DEPTH),
    .HAS_TRW(HAS_TRW), .NTBP(NTBP), .NRVWP(NRVWP), .HAS_REC(HAS_REC)
  ) u_dp (
    .clk, .rst_n,
    .csr_valid, .csr_ready, .csr_we, .csr_addr, .csr_wdata, .csr_rdata,
    .m_rd_req_valid, .m_rd_req_ready, .m_rd_req_addr, .m_rd_req_len, .m_rd_req_id,
    .m_rd_rsp_valid, .m_rd_rsp_ready, .m_rd_rsp_data, .m_rd_rsp_id, .m_rd_rsp_last,
    .m_wr_req_valid, .m_wr_req_ready, .m_wr_req_addr, .m_wr_req_len, .m_wr_req_id,
    .m_wr_dat_valid, .m_wr_dat_ready, .m_wr_dat_data, .m_wr_dat_mask, .m_wr_dat_last,
    .m_wr_rsp_valid, .m_wr_rsp_ready, .m_wr_rsp_id,
    .rst_ctrl_n,
    .ifq_valid(issue_valid), .ifq_ready(issue_ready), .ifq_entry(issue_entry), .ifq_count(if_count),
    .issued_seq, .s_seq, .d_seq, .err_valid, .err_rec, .err_ack, .rec_resync, .q_hold, .q_err, .q_nonempty,
    .spm_req_valid(spm_bridge_req_valid), .spm_req_we, .spm_req_addr, .spm_req_be, .spm_req_wdata,
    .spm_rsp_valid(spm_bridge_rsp_valid), .spm_rsp_ready(spm_bridge_rsp_ready),
    .spm_rsp_line(spm_bridge_rsp_line), .spm_rsp_err(spm_bridge_rsp_err),
    .cwin_req_valid, .cwin_req_off, .cwin_rsp_valid, .cwin_rsp_rdata,
    .lq_req_valid(lq_csr_req_valid), .lq_req_off(lq_csr_req_off), .lq_req_we(lq_csr_req_we),
    .lq_req_wdata(lq_csr_req_wdata), .lq_req_lcq_w(lq_csr_req_lcq_w),
    .lq_rsp_valid, .lq_rsp_rdata, .lq_rsp_fault,
    .log_valid, .log_data, .log_ready, .rt_state, .rt_info,
    .ld_ready, .ld_req, .ld_we, .ld_addr, .ld_wdata, .ld_rsp_valid, .ld_rsp_rdata, .ld_rsp_err, .ld_rsp_timeout,
    .dmi_ready, .dmi_req, .dmi_addr, .dmi_op, .dmi_wdata, .dmi_rsp_valid, .dmi_rsp_rdata, .dmi_rsp_resp,
    .dmi_rsp_timeout,
    .mtime, .rv_cyc, .perf_push_stall, .perf_bridge_acc, .perf_bridge_wait, .push_timeout, .bridge_timeout,
    .rv_run, .rv_boot, .irq_lq, .irq_lcq, .perf_time_en, .perf_evt_en,
    .rv_halted, .rv_sleep, .rv_running, .rv_reset_done, .rv_clk_ok, .rv_ndmreset, .rv_dmactive,
    .plat_err_i, .host_irq_o,
    .rv_halt_pend, .dbg_stop, .dbg_halt_tgl, .dbg_cancel_tgl, .dbg_halt_on_boot,
    .rec_wr, .rec_wdata, .rec_room, .rec_all, .smp_ready, .smp_req, .smp_local, .smp_done, .smp_timeout,
    .smp_data, .wpa_ready, .wpa_req, .wpa_local, .wpa_data, .wpa_done, .wpa_timeout,
    .wph_v, .wph_mask, .wph_addr, .wph_halt, .dfault, .flags(dbg_flags)
  );

endmodule
