// tsr_ctrl.sv — the RISC-V control subsystem as tsr_core instantiates it (CC §2, §13.1-§13.2):
// tsr_ctrl_rv (clk_rv: Ibex, debug module, TCMs, bus, control window) + tsr_ctrl_cdc (every
// crossing) + the clk_rv reset synchronisers. The ports are the core-domain side of tsr_ctrl_cdc
// (tsr_core_dp connects them), the two clocks and resets, and the DB window of the control window
// (clk_rv, W4-B).
//
// Owner: ctrl agent (W4 item 3). Integration into tsr_core (the issue, CSR and SPM blocks are other
// owners'): see the "Integration" notes below; the full contract is the header of tsr_ctrl_cdc.sv.
//
// Resets (CC §12.2): the clk_rv domain reset is rst_n (ap_rst_n, core domain) through tsr_rst_sync
// into clk_rv (asynchronous assert, synchronous deassert after 2 cycles); rst_rv_n (ap_rst_n_2)
// is synchronised the same way and AND-ed only into the Ibex / DM resets, never into a CDC
// endpoint. Minimum rst_n low: max(RST_PIPE + 1 core cycles, 3 clk_rv cycles).
//
// Integration (tsr_core / tsr_core_dp):
//   - if_*: the dispatcher's issue-entry stream (CC §8.3), entry layout TSR_IFIFO_* (CC §7.2).
//   - issued_seq_o / s_seq_i / d_seq_i: ISSUED_SEQ copies for Q_ISSUED and QIDLE, and the
//     core-domain S[u] (accepts + non-running completions) and D[u] counters, each stepping by
//     at most 1 per cycle.
//   - err_valid_i / err_rec_i / err_ack_o: per unit, whenever the error record of unit u changes
//     (first error captured, cleared by QCLEAR / SOFT_RESET): raise err_valid_i[u] with the record
//     (tsr_ctrl_pkg::err_rec_t) and hold both until err_ack_o[u]; D[u] must not pass ERR_SEQ before
//     that acknowledge (I2), and a RISC-V QCLEAR's own D increment waits for the acknowledges of the
//     cleared records (CC §9.5).
//   - rec_resync_o: one pulse when clk_rv becomes OK again (a handshake is answered locally while it
//     is not, so clk_rv's copy of a record can be stale): the core marks all six records changed.
//   - spm_*: tsr_spm_bridge (SPM client 3): spm_req_valid_o is a level, request fields stable while
//     it is 1; answer once with spm_rsp_valid_i (read: the whole line holding spm_req_addr_o; write:
//     word spm_req_addr_o[5:2] with byte enables spm_req_be_o, answered after tsr_spm accepted it).
//     spm_req_valid_o falls in the cycle after the answer. With rtl/core/tsr_spm_bridge.sv
//     (req_valid / req_ready, rsp_valid / rsp_ready): req_valid = spm_req_valid_o, req_we,
//     req_waddr = spm_req_addr_o[WAW+1:2], req_be, req_wdata; rsp_ready = 1; spm_rsp_valid_i =
//     rsp_valid, spm_rsp_line_i = rsp_line, spm_rsp_err_i = rsp_err (the bridge is busy, so
//     req_ready = 0, until its answer, and idle again only after the valid fell).
//   - cwin_*: tsr_csr read requester 2 (CC §6.6): side-effect-free read of CSR offset
//     cwin_req_off_o (never 0x0C4 or 0xC00-0xDFC; never touches the PERF snapshot).
//   - lq_*: tsr_csr's LQ / LCQ port: lq_req_off_o is the control-window offset (0x200-0x2FC).
//     Reads return the core-domain register; for LOG_FREE (0x284) answer LOG_RD + (words the ring
//     dropped since the hardware reset, modulo 2^32): the clk_rv side computes LOG_FREE from it.
//     Writes LQ_POP (0x228) / LCQ_POST (0x260, lq_req_lcq_w_o = LCQ_W0-7, lq_req_wdata_o bit0
//     ANSWER) commit in the core domain; answer lq_rsp_fault_i = 1 for EX §4.5's faults.
//   - log_*: words of LOG_PUSH, in order, for the log ring (drop-new when full, EX §4.10).
//   - rt_state_o / rt_info_o: the host's RT_STATE / RT_INFO (0x038 / 0x03C).
//   - ld_*: tsr_csr's loader (LOAD_DATA writes and reads, VERIFY): one word per request at a
//     RISC-V byte address; ld_rsp_err_o = refused (the hart is not held in reset, or the address is
//     not in a TCM); ld_rsp_timeout_o -> LOAD_TIMEOUT. tsr_csr must not raise rv_run_i while a load
//     request is outstanding (the load port owns the TCM write port only while the hart is held).
//   - dmi_*: the DMI window (0xC00 + 4r, gated by DBG_CFG.DMI_EN in tsr_csr); dmi_op_i 1 read,
//     2 write; dmi_rsp_timeout_o -> STATUS.DMI_TIMEOUT.
//   - mtime_i: MTIME (core cycles since reset, +1 per cycle); rv_cyc_o: clk_rv cycle count for
//     RV_FREQ; perf_*_o: PERF X + 7 / X + 8 / X + 9 (32-bit, to be extended to 64 bits);
//     push_timeout_o / bridge_timeout_o: pulses for STATUS.PUSH_TIMEOUT / BRIDGE_TIMEOUT.
//   - rv_run_i (CTRL.RV_RUN), rv_boot_i (RV_BOOT), dbg_stop_i, dbg_halt_tgl_i / dbg_cancel_tgl_i
//     (DB §3.3 toggles), dbg_halt_on_boot_i, irq_lq_i (LQ_AVAIL != 0 and not ACTIVE), irq_lcq_i
//     (LCQ_FREE != 0), perf_time_en_i (ACTIVE and not QHALT), perf_evt_en_i (ACTIVE), q_hold_i
//     ({debug-halt, RISC-V, host, any} = QSTATUS[4:1]), q_err_i, q_nonempty_i.
//   - rv_halted_o, rv_sleep_o, rv_running_o, rv_reset_done_o, rv_clk_ok_o (STATUS bits 1, 2, 0
//     with RV_RUN, 4, 3), rv_ndmreset_o (the core applies QABORT at its rise, like RV_RUN 1 -> 0),
//     rv_halt_pend_o (DBG_STATUS.RV_HALT_PEND).
// Implementation (Vivado; gate T1 flow rtl/ctrl/viv/ctrl_ooc.tcl): rtl/ctrl/viv/tsr_ctrl_cdc.xdc
//   (plain XDC for every crossing of the design, tsr_dbg_rv's too: ASYNC_REG, max delay / bus skew between
//   the clocks; packaged into the kernel IP) and rtl/ctrl/viv/tsr_cdc_waivers.tcl (the report_cdc waivers
//   of the handshake data, FIFO LUTRAM and gray buses); rtl/ctrl/viv/tsr_ctrl_tcm.xdc at synthesis
//   (TCMs in block RAM 32K x 1 columns: RAM_STYLE BLOCK, CASCADE_HEIGHT 1; the U200
//   rtl/tech/xilinx/tsr_sram_1r1w.sv alone would map them to URAM; every core / kernel flow reads it) and
//   set_power_opt -exclude_cells on
//   the TCM BRAMs before opt_design (BRAM power optimisation gates the BRAM enables with the
//   fetch address).

module tsr_ctrl
  import tsr_tables_pkg::*;
  import tsr_ctrl_pkg::*;
#(
  parameter int unsigned        ITCM_BYTES     = TSR_ITCM_BYTES,
  parameter int unsigned        DTCM_BYTES     = TSR_DTCM_BYTES,
  parameter int unsigned        SPM_BYTES      = 4 * 1024 * 1024,
  parameter logic [31:0]        CAPS           = TSR_CAPS_VU | TSR_CAPS_ATT,
  parameter logic [5:0][4:0]    Q_DEPTH        = {5'd4, 5'd0, 5'd16, 5'd4, 5'd4, 5'd8},
  parameter int unsigned        FIFO_DEPTH     = TSR_ISSUE_FIFO_DEPTH,
  parameter int unsigned        LOG_DEPTH      = TSR_LOG_DEPTH,
  parameter int unsigned        ISSUE_WAIT_MAX = TSR_ISSUE_WAIT_MAX,
  parameter int unsigned        HS_TIMEOUT     = 4096,
  parameter bit                 CTRL_SYNC      = 1'b0,
  // RV32MFast: CC §3.2's fallback 1, taken by gate T1 (the single-cycle multiplier is maxperf's
  // clk_rv limiter at 166.7 MHz: 0.20 ns margin against 0.31 ns with RV32MFast; rtl/ctrl/VENDOR.md)
  parameter ibex_pkg::rv32m_e   RV32M          = ibex_pkg::RV32MFast,
  parameter ibex_pkg::regfile_e REGFILE        = ibex_pkg::RegFileFPGA,
  parameter bit                 WB_STAGE       = 1'b1,
  parameter bit                 BT_ALU         = 1'b1
) (
  input  logic                      clk,
  // rst_n: the core domain's synchronous reset, also the asynchronous input of the clk_rv domain's
  // reset synchroniser (CC §12.2).
  /* verilator lint_off SYNCASYNCNET */
  input  logic                      rst_n,
  /* verilator lint_on SYNCASYNCNET */
  input  logic                      clk_rv,
  input  logic                      rst_rv_n,
  // issue FIFO, read side
  output logic                      if_valid_o,
  input  logic                      if_ready_i,
  output logic [TSR_IFIFO_BITS-1:0] if_data_o,
  output logic [5:0]                if_count_o,   // issue-FIFO entries waiting (Q_FIFO; W4-A integrate)
  // sequences and error records
  output logic [5:0][27:0]          issued_seq_o,
  input  logic [5:0][27:0]          s_seq_i,
  input  logic [5:0][27:0]          d_seq_i,
  input  logic [5:0]                err_valid_i,
  input  err_rec_t [5:0]            err_rec_i,
  output logic [5:0]                err_ack_o,
  output logic                      rec_resync_o,
  // SPM bridge
  output logic                      spm_req_valid_o,
  output logic                      spm_req_we_o,
  output logic [25:0]               spm_req_addr_o,
  output logic [3:0]                spm_req_be_o,
  output logic [31:0]               spm_req_wdata_o,
  input  logic                      spm_rsp_valid_i,
  input  logic [511:0]              spm_rsp_line_i,
  input  logic                      spm_rsp_err_i,
  // core window
  output logic                      cwin_req_valid_o,
  output logic [11:0]               cwin_req_off_o,
  input  logic                      cwin_rsp_valid_i,
  input  logic [31:0]               cwin_rsp_rdata_i,
  // LQ / LCQ / log registers
  output logic                      lq_req_valid_o,
  output logic [11:0]               lq_req_off_o,
  output logic                      lq_req_we_o,
  output logic [31:0]               lq_req_wdata_o,
  output logic [7:0][31:0]          lq_req_lcq_w_o,
  input  logic                      lq_rsp_valid_i,
  input  logic [31:0]               lq_rsp_rdata_i,
  input  logic                      lq_rsp_fault_i,
  // log words, RT mirrors
  output logic                      log_valid_o,
  output logic [31:0]               log_data_o,
  input  logic                      log_ready_i,
  output logic [31:0]               rt_state_o,
  output logic [31:0]               rt_info_o,
  // loader, DMI
  output logic                      ld_ready_o,
  input  logic                      ld_req_i,
  input  logic                      ld_we_i,
  input  logic [31:0]               ld_addr_i,
  input  logic [31:0]               ld_wdata_i,
  output logic                      ld_rsp_valid_o,
  output logic [31:0]               ld_rsp_rdata_o,
  output logic                      ld_rsp_err_o,
  output logic                      ld_rsp_timeout_o,
  output logic                      dmi_ready_o,
  input  logic                      dmi_req_i,
  input  logic [6:0]                dmi_addr_i,
  input  logic [1:0]                dmi_op_i,
  input  logic [31:0]               dmi_wdata_i,
  output logic                      dmi_rsp_valid_o,
  output logic [31:0]               dmi_rsp_rdata_o,
  output logic [1:0]                dmi_rsp_resp_o,
  output logic                      dmi_rsp_timeout_o,
  // time, counters, events
  input  logic [63:0]               mtime_i,
  output logic [15:0]               rv_cyc_o,
  output logic [31:0]               perf_push_stall_o,
  output logic [31:0]               perf_bridge_acc_o,
  output logic [31:0]               perf_bridge_wait_o,
  output logic                      push_timeout_o,
  output logic                      bridge_timeout_o,
  // levels
  input  logic                      rv_run_i,
  input  logic [31:0]               rv_boot_i,
  input  logic                      dbg_stop_i,
  input  logic                      dbg_halt_tgl_i,
  input  logic                      dbg_cancel_tgl_i,
  input  logic                      dbg_halt_on_boot_i,
  input  logic                      irq_lq_i,
  input  logic                      irq_lcq_i,
  input  logic                      perf_time_en_i,
  input  logic                      perf_evt_en_i,
  input  logic [3:0]                q_hold_i,
  input  logic [5:0]                q_err_i,
  input  logic [5:0]                q_nonempty_i,
  output logic                      rv_halted_o,
  output logic                      rv_sleep_o,
  output logic                      rv_running_o,
  output logic                      rv_reset_done_o,
  output logic                      rv_clk_ok_o,
  output logic                      rv_ndmreset_o,
  output logic                      rv_dmactive_o,
  output logic                      rv_halt_pend_o,
  // DB window of the control window (clk_rv; W4-B): 0x160-0x17C except 0x178, 0x400-0x7FC.
  // db_rdata_i is sampled in the cycle after db_req_o.
  output logic                      db_req_o,
  output logic                      db_we_o,
  output logic [11:0]               db_off_o,
  output logic [31:0]               db_wdata_o,
  input  logic [31:0]               db_rdata_i,
  input  logic                      db_irq4_i,
  input  logic                      db_irq6_i,
  // Ibex taps for the DB block (clk_rv; tsr_dbg_rv, W4-B integrate): see tsr_ctrl_rv
  output logic                      tap_data_req_o,
  output logic                      tap_data_gnt_o,
  output logic [31:0]               tap_data_addr_o,
  output logic                      tap_data_we_o,
  output logic [3:0]                tap_data_be_o,
  output logic [31:0]               tap_cd_pc_o,
  output logic [31:0]               tap_cd_npc_o,
  output logic [31:0]               tap_cd_daddr_o,
  output logic [31:0]               tap_cd_epc_o,
  output logic [31:0]               tap_cd_eaddr_o,
  output logic                      tap_sleep_o,
  output logic                      tap_halted_o,
  output logic                      tap_in_reset_o,
  output logic                      tap_dfault_o
);

  localparam int unsigned FAW = $clog2(FIFO_DEPTH);
  localparam int unsigned LAW = $clog2(16);

  logic rst_rvd_n, rst_ext_n;
  // u_rst_rv starts from a reset flop of its own: a launch flop that also drove core-domain logic
  // would be a fan-out to the destination clock (report_cdc CDC-11). One core cycle later.
  (* keep = "true" *) logic rst_rvsrc_q;
  always_ff @(posedge clk) rst_rvsrc_q <= rst_n;
  tsr_rst_sync #(.STAGES(2)) u_rst_rv  (.clk(clk_rv), .rst_ni(rst_rvsrc_q), .rst_no(rst_rvd_n));
  tsr_rst_sync #(.STAGES(2)) u_rst_ext (.clk(clk_rv), .rst_ni(rst_rv_n), .rst_no(rst_ext_n));

  // clk_rv side wires
  logic                      if_wr, if_empty, log_wr, x_ready, x_req, x_done, x_timeout;
  logic                      lq_ready, lq_req, lq_done, lq_timeout, rt_wr, ld_valid, ld_rsp;
  logic                      dmi_valid, dmi_rsp, push_ev, bridge_ev;
  logic [TSR_IFIFO_BITS-1:0] if_wdata;
  logic [FAW:0]              if_free;
  logic [LAW:0]              log_free;
  logic [31:0]               log_wdata, rt_state, rt_info, p_stall, p_bacc, p_bwait;
  logic [5:0][27:0]          issued_seq, s_seq, d_seq;
  err_rec_t [5:0]            err_rec;
  xreq_t                     x_req_data;
  logic [XRSP_W-1:0]         x_rsp;
  lqreq_t                    lq_req_data;
  logic [LQRSP_W-1:0]        lq_rsp;
  ldreq_t                    ld_req;
  logic [LDRSP_W-1:0]        ld_rsp_data;
  logic [DMIREQ_W-1:0]       dmi_req;
  logic [DMIRSP_W-1:0]       dmi_rsp_data;
  logic [63:0]               mtime;
  logic                      rv_run, dbg_stop, dbg_halt_req, dbg_halt_cancel, dbg_halt_on_boot;
  logic                      irq_lq, irq_lcq, perf_time_en, perf_evt_en;
  logic [31:0]               rv_boot;
  logic [3:0]                q_hold;
  logic [5:0]                q_err, q_nonempty;
  logic                      rv_halted, rv_sleep, rv_running, rv_reset_done, rv_ndmreset, rv_halt_pend, rv_dmactive;

  tsr_ctrl_rv #(
    .ITCM_BYTES(ITCM_BYTES), .DTCM_BYTES(DTCM_BYTES), .SPM_BYTES(SPM_BYTES), .CAPS(CAPS),
    .Q_DEPTH(Q_DEPTH), .FIFO_DEPTH(FIFO_DEPTH), .LOG_DEPTH(LOG_DEPTH), .LOG_FIFO_DEPTH(16),
    .ISSUE_WAIT_MAX(ISSUE_WAIT_MAX), .RV32M(RV32M), .REGFILE(REGFILE), .WB_STAGE(WB_STAGE),
    .BT_ALU(BT_ALU)
  ) u_rv (
    .clk               (clk_rv),
    .rst_n             (rst_rvd_n),
    .rst_ext_n         (rst_ext_n),
    .rv_run            (rv_run),
    .rv_boot           (rv_boot),
    .dbg_stop          (dbg_stop),
    .dbg_halt_req      (dbg_halt_req),
    .dbg_halt_cancel   (dbg_halt_cancel),
    .dbg_halt_on_boot  (dbg_halt_on_boot),
    .irq_lq            (irq_lq),
    .irq_lcq           (irq_lcq),
    .perf_time_en      (perf_time_en),
    .perf_evt_en       (perf_evt_en),
    .q_hold            (q_hold),
    .q_err             (q_err),
    .q_nonempty        (q_nonempty),
    .mtime             (mtime),
    .s_seq             (s_seq),
    .d_seq             (d_seq),
    .err_rec           (err_rec),
    .issued_seq        (issued_seq),
    .rv_halted         (rv_halted),
    .rv_sleep          (rv_sleep),
    .rv_reset_done     (rv_reset_done),
    .rv_running        (rv_running),
    .rv_ndmreset       (rv_ndmreset),
    .rv_dmactive       (rv_dmactive),
    .rv_halt_pend      (rv_halt_pend),
    .push_timeout_ev   (push_ev),
    .bridge_timeout_ev (bridge_ev),
    .perf_push_stall   (p_stall),
    .perf_bridge_acc   (p_bacc),
    .perf_bridge_wait  (p_bwait),
    .if_wr             (if_wr),
    .if_wdata          (if_wdata),
    .if_free           (if_free),
    .if_empty          (if_empty),
    .log_wr            (log_wr),
    .log_wdata         (log_wdata),
    .log_free          (log_free),
    .x_ready           (x_ready),
    .x_req             (x_req),
    .x_req_data        (x_req_data),
    .x_done            (x_done),
    .x_timeout         (x_timeout),
    .x_rsp             (x_rsp),
    .lq_ready          (lq_ready),
    .lq_req            (lq_req),
    .lq_req_data       (lq_req_data),
    .lq_done           (lq_done),
    .lq_timeout        (lq_timeout),
    .lq_rsp            (lq_rsp),
    .rt_state          (rt_state),
    .rt_info           (rt_info),
    .rt_wr             (rt_wr),
    .ld_valid          (ld_valid),
    .ld_req            (ld_req),
    .ld_rsp            (ld_rsp),
    .ld_rsp_data       (ld_rsp_data),
    .dmi_valid         (dmi_valid),
    .dmi_req           (dmi_req),
    .dmi_rsp           (dmi_rsp),
    .dmi_rsp_data      (dmi_rsp_data),
    .db_req            (db_req_o),
    .db_we             (db_we_o),
    .db_off            (db_off_o),
    .db_wdata          (db_wdata_o),
    .db_rdata          (db_rdata_i),
    .db_irq4           (db_irq4_i),
    .db_irq6           (db_irq6_i),
    .tap_data_req      (tap_data_req_o),
    .tap_data_gnt      (tap_data_gnt_o),
    .tap_data_addr     (tap_data_addr_o),
    .tap_data_we       (tap_data_we_o),
    .tap_data_be       (tap_data_be_o),
    .tap_cd_pc         (tap_cd_pc_o),
    .tap_cd_npc        (tap_cd_npc_o),
    .tap_cd_daddr      (tap_cd_daddr_o),
    .tap_cd_epc        (tap_cd_epc_o),
    .tap_cd_eaddr      (tap_cd_eaddr_o),
    .tap_sleep         (tap_sleep_o),
    .tap_halted        (tap_halted_o),
    .tap_in_reset      (tap_in_reset_o),
    .tap_dfault        (tap_dfault_o)
  );

  tsr_ctrl_cdc #(
    .FIFO_DEPTH(FIFO_DEPTH), .LOG_FIFO_DEPTH(16), .HS_TIMEOUT(HS_TIMEOUT), .CTRL_SYNC(CTRL_SYNC)
  ) u_cdc (
    .clk                 (clk),
    .rst_n               (rst_n),
    .if_valid_o          (if_valid_o),
    .if_ready_i          (if_ready_i),
    .if_data_o           (if_data_o),
    .if_count_o          (if_count_o),
    .issued_seq_o        (issued_seq_o),
    .s_seq_i             (s_seq_i),
    .d_seq_i             (d_seq_i),
    .err_valid_i         (err_valid_i),
    .err_rec_i           (err_rec_i),
    .err_ack_o           (err_ack_o),
    .rec_resync_o        (rec_resync_o),
    .spm_req_valid_o     (spm_req_valid_o),
    .spm_req_we_o        (spm_req_we_o),
    .spm_req_addr_o      (spm_req_addr_o),
    .spm_req_be_o        (spm_req_be_o),
    .spm_req_wdata_o     (spm_req_wdata_o),
    .spm_rsp_valid_i     (spm_rsp_valid_i),
    .spm_rsp_line_i      (spm_rsp_line_i),
    .spm_rsp_err_i       (spm_rsp_err_i),
    .cwin_req_valid_o    (cwin_req_valid_o),
    .cwin_req_off_o      (cwin_req_off_o),
    .cwin_rsp_valid_i    (cwin_rsp_valid_i),
    .cwin_rsp_rdata_i    (cwin_rsp_rdata_i),
    .lq_req_valid_o      (lq_req_valid_o),
    .lq_req_off_o        (lq_req_off_o),
    .lq_req_we_o         (lq_req_we_o),
    .lq_req_wdata_o      (lq_req_wdata_o),
    .lq_req_lcq_w_o      (lq_req_lcq_w_o),
    .lq_rsp_valid_i      (lq_rsp_valid_i),
    .lq_rsp_rdata_i      (lq_rsp_rdata_i),
    .lq_rsp_fault_i      (lq_rsp_fault_i),
    .log_valid_o         (log_valid_o),
    .log_data_o          (log_data_o),
    .log_ready_i         (log_ready_i),
    .rt_state_o          (rt_state_o),
    .rt_info_o           (rt_info_o),
    .ld_ready_o          (ld_ready_o),
    .ld_req_i            (ld_req_i),
    .ld_we_i             (ld_we_i),
    .ld_addr_i           (ld_addr_i),
    .ld_wdata_i          (ld_wdata_i),
    .ld_rsp_valid_o      (ld_rsp_valid_o),
    .ld_rsp_rdata_o      (ld_rsp_rdata_o),
    .ld_rsp_err_o        (ld_rsp_err_o),
    .ld_rsp_timeout_o    (ld_rsp_timeout_o),
    .dmi_ready_o         (dmi_ready_o),
    .dmi_req_i           (dmi_req_i),
    .dmi_addr_i          (dmi_addr_i),
    .dmi_op_i            (dmi_op_i),
    .dmi_wdata_i         (dmi_wdata_i),
    .dmi_rsp_valid_o     (dmi_rsp_valid_o),
    .dmi_rsp_rdata_o     (dmi_rsp_rdata_o),
    .dmi_rsp_resp_o      (dmi_rsp_resp_o),
    .dmi_rsp_timeout_o   (dmi_rsp_timeout_o),
    .mtime_i             (mtime_i),
    .rv_cyc_o            (rv_cyc_o),
    .perf_push_stall_o   (perf_push_stall_o),
    .perf_bridge_acc_o   (perf_bridge_acc_o),
    .perf_bridge_wait_o  (perf_bridge_wait_o),
    .push_timeout_o      (push_timeout_o),
    .bridge_timeout_o    (bridge_timeout_o),
    .rv_run_i            (rv_run_i),
    .rv_boot_i           (rv_boot_i),
    .dbg_stop_i          (dbg_stop_i),
    .dbg_halt_tgl_i      (dbg_halt_tgl_i),
    .dbg_cancel_tgl_i    (dbg_cancel_tgl_i),
    .dbg_halt_on_boot_i  (dbg_halt_on_boot_i),
    .irq_lq_i            (irq_lq_i),
    .irq_lcq_i           (irq_lcq_i),
    .perf_time_en_i      (perf_time_en_i),
    .perf_evt_en_i       (perf_evt_en_i),
    .q_hold_i            (q_hold_i),
    .q_err_i             (q_err_i),
    .q_nonempty_i        (q_nonempty_i),
    .rv_halted_o         (rv_halted_o),
    .rv_sleep_o          (rv_sleep_o),
    .rv_running_o        (rv_running_o),
    .rv_reset_done_o     (rv_reset_done_o),
    .rv_clk_ok_o         (rv_clk_ok_o),
    .rv_ndmreset_o       (rv_ndmreset_o),
    .rv_dmactive_o       (rv_dmactive_o),
    .rv_halt_pend_o      (rv_halt_pend_o),
    .clk_rv              (clk_rv),
    .rst_rv_n            (rst_rvd_n),
    .r_if_wr             (if_wr),
    .r_if_wdata          (if_wdata),
    .r_if_free           (if_free),
    .r_if_empty          (if_empty),
    .r_issued_seq        (issued_seq),
    .r_s_seq             (s_seq),
    .r_d_seq             (d_seq),
    .r_err_rec           (err_rec),
    .r_x_ready           (x_ready),
    .r_x_req             (x_req),
    .r_x_req_data        (x_req_data),
    .r_x_done            (x_done),
    .r_x_timeout         (x_timeout),
    .r_x_rsp             (x_rsp),
    .r_lq_ready          (lq_ready),
    .r_lq_req            (lq_req),
    .r_lq_req_data       (lq_req_data),
    .r_lq_done           (lq_done),
    .r_lq_timeout        (lq_timeout),
    .r_lq_rsp            (lq_rsp),
    .r_log_wr            (log_wr),
    .r_log_wdata         (log_wdata),
    .r_log_free          (log_free),
    .r_rt_state          (rt_state),
    .r_rt_info           (rt_info),
    .r_rt_wr             (rt_wr),
    .r_ld_valid          (ld_valid),
    .r_ld_req            (ld_req),
    .r_ld_rsp            (ld_rsp),
    .r_ld_rsp_data       (ld_rsp_data),
    .r_dmi_valid         (dmi_valid),
    .r_dmi_req           (dmi_req),
    .r_dmi_rsp           (dmi_rsp),
    .r_dmi_rsp_data      (dmi_rsp_data),
    .r_mtime             (mtime),
    .r_perf_push_stall   (p_stall),
    .r_perf_bridge_acc   (p_bacc),
    .r_perf_bridge_wait  (p_bwait),
    .r_push_timeout_ev   (push_ev),
    .r_bridge_timeout_ev (bridge_ev),
    .r_rv_run            (rv_run),
    .r_rv_boot           (rv_boot),
    .r_dbg_stop          (dbg_stop),
    .r_dbg_halt_req      (dbg_halt_req),
    .r_dbg_halt_cancel   (dbg_halt_cancel),
    .r_dbg_halt_on_boot  (dbg_halt_on_boot),
    .r_irq_lq            (irq_lq),
    .r_irq_lcq           (irq_lcq),
    .r_perf_time_en      (perf_time_en),
    .r_perf_evt_en       (perf_evt_en),
    .r_q_hold            (q_hold),
    .r_q_err             (q_err),
    .r_q_nonempty        (q_nonempty),
    .r_rv_halted         (rv_halted),
    .r_rv_sleep          (rv_sleep),
    .r_rv_running        (rv_running),
    .r_rv_reset_done     (rv_reset_done),
    .r_rv_ndmreset       (rv_ndmreset),
    .r_rv_dmactive       (rv_dmactive),
    .r_rv_halt_pend      (rv_halt_pend)
  );

endmodule
