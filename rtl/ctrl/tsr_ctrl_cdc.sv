// tsr_ctrl_cdc.sv — every clock-domain crossing between the core clock (clk = ap_clk) and clk_rv
// (CC §11.2; docs/ctrl-riscv.md §7.1-§7.2). One side of each structure faces tsr_ctrl_rv (clk_rv),
// the other the core datapath (tsr_core_dp: tsr_issue, tsr_csr, tsr_spm_bridge).
//
// Spec: CC §2.2 `tsr_ctrl_cdc`.
//
// Crossings (direction, structure, CC §11.2):
//   issue FIFO          RV -> core  tsr_cdc_fifo, 32 x 387 b (CC §7.2 layout), gray pointers
//   ISSUED_SEQ[u]       RV -> core  tsr_cdc_gray 28 b x 6
//   S[u], D[u]          core -> RV  tsr_cdc_gray 28 b x 12 (each steps by <= 1 per core cycle: asserted)
//   error records       core -> RV  tsr_cdc_hs per unit, 128 b (I2: err_ack only once clk_rv shows it)
//   SPM bridge + core   RV <-> core tsr_cdc_hs, 64 b / 513 b, one access at a time (shared slot)
//     window
//   LQ / LCQ registers  RV <-> core tsr_cdc_hs (own slot), 301 b / 33 b
//   log words           RV -> core  tsr_cdc_fifo, 16 x 32 b
//   RT_STATE / RT_INFO  RV -> core  tsr_cdc_hs on each write (the latest value always arrives)
//   LOAD_*              core -> RV  tsr_cdc_hs into the TCM ports (load port), 65 b / 33 b
//   DMI                 core <-> RV tsr_cdc_hs, 41 b / 34 b (dm::dmi_req_t / dm::dmi_resp_t)
//   MTIME               core -> RV  tsr_cdc_gray 64 b
//   clk_rv cycle count  RV -> core  tsr_cdc_gray 16 b (RV_FREQ); PERF PUSH_STALL / BRIDGE_ACC /
//                                   BRIDGE_WAIT 3 x 32 b
//   sticky events       RV -> core  toggles (PUSH_TIMEOUT, BRIDGE_TIMEOUT)
//   levels              both        tsr_sync (RV_RUN, holds, error / non-empty bits, ACTIVE-derived
//                                   PERF gates, irq_fast[2] / [3] levels, dbg_stop, HALT_ON_BOOT;
//                                   RV_HALTED, RV_SLEEP, RV_RUNNING, ndmreset, dmactive, DB's pend,
//                                   RV_RESET_DONE two stages behind the FIFO write pointer,
//                                   "clk_rv out of reset"); toggles for DB's halt request / cancel
//   RV_BOOT             core -> RV  quasi-static bus (written while the hart is held in reset)
// Bounds (CC §11.2 rules): the core -> RV requests (LOAD, DMI, error records) end after HS_TIMEOUT
//   core cycles (ld_rsp_timeout / dmi_rsp_timeout; an error record is then acknowledged locally),
//   and are answered locally at once while clk_rv is not OK (rv_clk_ok = 0: "nothing crosses into
//   a domain in reset"); the RV -> core requests end after HS_TIMEOUT clk_rv cycles (the window
//   answers with an access fault and BRIDGE_TIMEOUT). Late answers are discarded by tag.
// RV_CLK_OK (EX §5.3 STATUS bit3): the clk_rv domain is out of reset and its cycle count moved
//   within the last 256 core cycles.
// Resets: rst_n (core, synchronous) and rst_rv_n (the clk_rv domain reset: ap_rst_n through
//   tsr_rst_sync) come from the same hardware reset; nothing here is reset by a RISC-V reset.
// CTRL_SYNC = 1 (clk_rv = clk, CC §11.1): every synchronizer becomes one register stage.

module tsr_ctrl_cdc
  import tsr_tables_pkg::*;
  import tsr_ctrl_pkg::*;
#(
  parameter  int unsigned FIFO_DEPTH     = TSR_ISSUE_FIFO_DEPTH,
  parameter  int unsigned LOG_FIFO_DEPTH = 16,
  parameter  int unsigned HS_TIMEOUT     = 4096,
  parameter  bit          CTRL_SYNC      = 1'b0,
  localparam int unsigned FAW            = $clog2(FIFO_DEPTH),
  localparam int unsigned LAW            = $clog2(LOG_FIFO_DEPTH)
) (
  // ================================================================ core clock domain
  input  logic                      clk,
  input  logic                      rst_n,
  // issue FIFO, read side (tsr_issue's dispatcher, CC §8.3)
  output logic                      if_valid_o,
  input  logic                      if_ready_i,
  output logic [TSR_IFIFO_BITS-1:0] if_data_o,
  output logic [5:0]                if_count_o,   // entries waiting for the dispatcher (Q_FIFO, saturating)
  // sequences
  output logic [5:0][27:0]          issued_seq_o,
  input  logic [5:0][27:0]          s_seq_i,
  input  logic [5:0][27:0]          d_seq_i,
  // error records (CC §9.4 I2): hold err_valid_i[u] with err_rec_i[u] until err_ack_o[u]
  input  logic [5:0]                err_valid_i,
  input  err_rec_t [5:0]            err_rec_i,
  output logic [5:0]                err_ack_o,
  output logic                      rec_resync_o, // clk_rv is OK again: the core re-sends all six records
  // SPM bridge (tsr_spm_bridge, client 3): one access at a time; answer with one spm_rsp_valid_i
  // pulse (a write after tsr_spm accepted it, CC §10.5)
  output logic                      spm_req_valid_o,
  output logic                      spm_req_we_o,
  output logic [25:0]               spm_req_addr_o,
  output logic [3:0]                spm_req_be_o,
  output logic [31:0]               spm_req_wdata_o,
  input  logic                      spm_rsp_valid_i,
  input  logic [511:0]              spm_rsp_line_i,
  input  logic                      spm_rsp_err_i,
  // core window (tsr_csr read requester 2, CC §6.6)
  output logic                      cwin_req_valid_o,
  output logic [11:0]               cwin_req_off_o,
  input  logic                      cwin_rsp_valid_i,
  input  logic [31:0]               cwin_rsp_rdata_i,
  // LQ / LCQ / log registers (tsr_csr's third port, EX §4.1, CC-30)
  output logic                      lq_req_valid_o,
  output logic [11:0]               lq_req_off_o,
  output logic                      lq_req_we_o,
  output logic [31:0]               lq_req_wdata_o,
  output logic [7:0][31:0]          lq_req_lcq_w_o,
  input  logic                      lq_rsp_valid_i,
  input  logic [31:0]               lq_rsp_rdata_i,
  input  logic                      lq_rsp_fault_i,
  // log words
  output logic                      log_valid_o,
  output logic [31:0]               log_data_o,
  input  logic                      log_ready_i,
  // RT_* mirrors (host 0x038 / 0x03C)
  output logic [31:0]               rt_state_o,
  output logic [31:0]               rt_info_o,
  // loader (EX §3.3): ld_req_i only while ld_ready_o; one ld_rsp_valid_o pulse per request
  output logic                      ld_ready_o,
  input  logic                      ld_req_i,
  input  logic                      ld_we_i,
  input  logic [31:0]               ld_addr_i,
  input  logic [31:0]               ld_wdata_i,
  output logic                      ld_rsp_valid_o,
  output logic [31:0]               ld_rsp_rdata_o,
  output logic                      ld_rsp_err_o,
  output logic                      ld_rsp_timeout_o,
  // DMI (DB §3.2): dmi_req_i only while dmi_ready_o; one dmi_rsp_valid_o pulse per request
  output logic                      dmi_ready_o,
  input  logic                      dmi_req_i,
  input  logic [6:0]                dmi_addr_i,
  input  logic [1:0]                dmi_op_i,
  input  logic [31:0]               dmi_wdata_i,
  output logic                      dmi_rsp_valid_o,
  output logic [31:0]               dmi_rsp_rdata_o,
  output logic [1:0]                dmi_rsp_resp_o,
  output logic                      dmi_rsp_timeout_o,
  // time and counters
  input  logic [63:0]               mtime_i,
  output logic [15:0]               rv_cyc_o,
  output logic [31:0]               perf_push_stall_o,
  output logic [31:0]               perf_bridge_acc_o,
  output logic [31:0]               perf_bridge_wait_o,
  output logic                      push_timeout_o,
  output logic                      bridge_timeout_o,
  // levels core -> RV
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
  // levels RV -> core
  output logic                      rv_halted_o,
  output logic                      rv_sleep_o,
  output logic                      rv_running_o,
  output logic                      rv_reset_done_o,
  output logic                      rv_clk_ok_o,
  output logic                      rv_ndmreset_o,
  output logic                      rv_dmactive_o,
  output logic                      rv_halt_pend_o,

  // ================================================================ clk_rv domain (tsr_ctrl_rv)
  input  logic                      clk_rv,
  input  logic                      rst_rv_n,
  input  logic                      r_if_wr,
  input  logic [TSR_IFIFO_BITS-1:0] r_if_wdata,
  output logic [FAW:0]              r_if_free,
  output logic                      r_if_empty,
  input  logic [5:0][27:0]          r_issued_seq,
  output logic [5:0][27:0]          r_s_seq,
  output logic [5:0][27:0]          r_d_seq,
  output err_rec_t [5:0]            r_err_rec,
  output logic                      r_x_ready,
  input  logic                      r_x_req,
  input  xreq_t                     r_x_req_data,
  output logic                      r_x_done,
  output logic                      r_x_timeout,
  output logic [XRSP_W-1:0]         r_x_rsp,
  output logic                      r_lq_ready,
  input  logic                      r_lq_req,
  input  lqreq_t                    r_lq_req_data,
  output logic                      r_lq_done,
  output logic                      r_lq_timeout,
  output logic [LQRSP_W-1:0]        r_lq_rsp,
  input  logic                      r_log_wr,
  input  logic [31:0]               r_log_wdata,
  output logic [LAW:0]              r_log_free,
  input  logic [31:0]               r_rt_state,
  input  logic [31:0]               r_rt_info,
  input  logic                      r_rt_wr,
  output logic                      r_ld_valid,
  output ldreq_t                    r_ld_req,
  input  logic                      r_ld_rsp,
  input  logic [LDRSP_W-1:0]        r_ld_rsp_data,
  output logic                      r_dmi_valid,
  output logic [DMIREQ_W-1:0]       r_dmi_req,
  input  logic                      r_dmi_rsp,
  input  logic [DMIRSP_W-1:0]       r_dmi_rsp_data,
  output logic [63:0]               r_mtime,
  input  logic [31:0]               r_perf_push_stall,
  input  logic [31:0]               r_perf_bridge_acc,
  input  logic [31:0]               r_perf_bridge_wait,
  input  logic                      r_push_timeout_ev,
  input  logic                      r_bridge_timeout_ev,
  output logic                      r_rv_run,
  output logic [31:0]               r_rv_boot,
  output logic                      r_dbg_stop,
  output logic                      r_dbg_halt_req,
  output logic                      r_dbg_halt_cancel,
  output logic                      r_dbg_halt_on_boot,
  output logic                      r_irq_lq,
  output logic                      r_irq_lcq,
  output logic                      r_perf_time_en,
  output logic                      r_perf_evt_en,
  output logic [3:0]                r_q_hold,
  output logic [5:0]                r_q_err,
  output logic [5:0]                r_q_nonempty,
  input  logic                      r_rv_halted,
  input  logic                      r_rv_sleep,
  input  logic                      r_rv_running,
  input  logic                      r_rv_reset_done,
  input  logic                      r_rv_ndmreset,
  input  logic                      r_rv_dmactive,
  input  logic                      r_rv_halt_pend
);

  localparam int unsigned ST = CTRL_SYNC ? 1 : 2;   // synchronizer stages

  logic rv_ok;   // core domain: clk_rv is OK (requests may cross)

  // ---------------------------------------------------------------- issue FIFO, log FIFO
  logic [FAW:0] if_cnt;
  logic [LAW:0] log_rd_count;
  tsr_cdc_fifo #(.W(TSR_IFIFO_BITS), .DEPTH(FIFO_DEPTH), .STAGES(ST)) u_ififo (
    .wclk(clk_rv), .wrst_n(rst_rv_n), .wr_en(r_if_wr), .wr_data(r_if_wdata), .wr_free(r_if_free),
    .wr_empty(r_if_empty),
    .rclk(clk), .rrst_n(rst_n), .rd_valid(if_valid_o), .rd_data(if_data_o), .rd_ready(if_ready_i),
    .rd_count(if_cnt)
  );
  // Q_FIFO (EX §5.4) is 6 bits wide at tsr_csr; a deeper FIFO saturates there.
  assign if_count_o = (32'(if_cnt) > 32'd63) ? 6'd63 : 6'(if_cnt);

  logic log_wr_empty;
  tsr_cdc_fifo #(.W(32), .DEPTH(LOG_FIFO_DEPTH), .STAGES(ST)) u_lfifo (
    .wclk(clk_rv), .wrst_n(rst_rv_n), .wr_en(r_log_wr), .wr_data(r_log_wdata), .wr_free(r_log_free),
    .wr_empty(log_wr_empty),
    .rclk(clk), .rrst_n(rst_n), .rd_valid(log_valid_o), .rd_data(log_data_o), .rd_ready(log_ready_i),
    .rd_count(log_rd_count)
  );
  wire unused_log_rd_count = &{1'b0, log_rd_count};

  // ---------------------------------------------------------------- gray counters
  for (genvar u = 0; u < 6; u++) begin : g_seq
    tsr_cdc_gray #(.W(28), .STAGES(ST)) u_issued (
      .src_clk(clk_rv), .src_rst_n(rst_rv_n), .src_cnt(r_issued_seq[u]),
      .dst_clk(clk), .dst_rst_n(rst_n), .dst_cnt(issued_seq_o[u])
    );
    tsr_cdc_gray #(.W(28), .STAGES(ST)) u_s (
      .src_clk(clk), .src_rst_n(rst_n), .src_cnt(s_seq_i[u]),
      .dst_clk(clk_rv), .dst_rst_n(rst_rv_n), .dst_cnt(r_s_seq[u])
    );
    tsr_cdc_gray #(.W(28), .STAGES(ST)) u_d (
      .src_clk(clk), .src_rst_n(rst_n), .src_cnt(d_seq_i[u]),
      .dst_clk(clk_rv), .dst_rst_n(rst_rv_n), .dst_cnt(r_d_seq[u])
    );
  end

  tsr_cdc_gray #(.W(64), .STAGES(ST)) u_mtime (
    .src_clk(clk), .src_rst_n(rst_n), .src_cnt(mtime_i),
    .dst_clk(clk_rv), .dst_rst_n(rst_rv_n), .dst_cnt(r_mtime)
  );

  logic [15:0] rv_cyc_q;
  always_ff @(posedge clk_rv) begin
    if (!rst_rv_n) rv_cyc_q <= '0;
    else           rv_cyc_q <= rv_cyc_q + 1'b1;
  end
  tsr_cdc_gray #(.W(16), .STAGES(ST)) u_rvcyc (
    .src_clk(clk_rv), .src_rst_n(rst_rv_n), .src_cnt(rv_cyc_q),
    .dst_clk(clk), .dst_rst_n(rst_n), .dst_cnt(rv_cyc_o)
  );
  tsr_cdc_gray #(.W(32), .STAGES(ST)) u_pstall (
    .src_clk(clk_rv), .src_rst_n(rst_rv_n), .src_cnt(r_perf_push_stall),
    .dst_clk(clk), .dst_rst_n(rst_n), .dst_cnt(perf_push_stall_o)
  );
  tsr_cdc_gray #(.W(32), .STAGES(ST)) u_pbacc (
    .src_clk(clk_rv), .src_rst_n(rst_rv_n), .src_cnt(r_perf_bridge_acc),
    .dst_clk(clk), .dst_rst_n(rst_n), .dst_cnt(perf_bridge_acc_o)
  );
  tsr_cdc_gray #(.W(32), .STAGES(ST)) u_pbwait (
    .src_clk(clk_rv), .src_rst_n(rst_rv_n), .src_cnt(r_perf_bridge_wait),
    .dst_clk(clk), .dst_rst_n(rst_n), .dst_cnt(perf_bridge_wait_o)
  );

  // ---------------------------------------------------------------- error records (core -> RV)
  for (genvar u = 0; u < 6; u++) begin : g_err
    logic       a_ready, a_done, a_timeout, b_valid;
    logic [0:0] a_rsp;
    logic [ERR_REC_W-1:0] b_req;
    tsr_cdc_hs #(.REQ_W(ERR_REC_W), .RSP_W(1), .TIMEOUT(HS_TIMEOUT), .STAGES(ST)) u_hs (
      .clk_a(clk), .rst_a_n(rst_n), .a_ready(a_ready),
      .a_req(err_valid_i[u] && a_ready && !a_done), .a_local(!rv_ok),
      .a_req_data(err_rec_i[u]), .a_done(a_done), .a_timeout(a_timeout), .a_rsp_data(a_rsp),
      .clk_b(clk_rv), .rst_b_n(rst_rv_n), .b_valid(b_valid), .b_req_data(b_req),
      .b_rsp(b_valid), .b_rsp_data(1'b0)
    );
    // The record register the control window shows: written in the cycle the answer leaves, so
    // it is visible before the acknowledge reaches the core (I2).
    always_ff @(posedge clk_rv) begin
      if (!rst_rv_n)    r_err_rec[u] <= '0;
      else if (b_valid) r_err_rec[u] <= b_req;
    end
    assign err_ack_o[u] = a_done;   // a registered pulse; the core drops err_valid_i[u] after it
    wire unused_err = &{1'b0, a_rsp, a_timeout};
  end

  // While clk_rv is not OK a record handshake is answered locally (a_local), so clk_rv's copy goes
  // stale if the record changes in that time (a SOFT_RESET clearing it, a new error). When clk_rv is
  // OK again, one pulse makes the core send all six records once more (tsr_issue rec_resync).
  logic rv_ok_d_q;
  always_ff @(posedge clk) begin
    if (!rst_n) rv_ok_d_q <= 1'b0;
    else        rv_ok_d_q <= rv_ok;
  end
  assign rec_resync_o = rv_ok && !rv_ok_d_q;

  // ---------------------------------------------------------------- bridge + core window (RV -> core)
  logic  x_bvalid, x_brsp;
  xreq_t x_breq;
  logic [XRSP_W-1:0] x_brsp_data;
  tsr_cdc_hs #(.REQ_W(XREQ_W), .RSP_W(XRSP_W), .TIMEOUT(HS_TIMEOUT), .STAGES(ST)) u_hs_x (
    .clk_a(clk_rv), .rst_a_n(rst_rv_n), .a_ready(r_x_ready), .a_req(r_x_req), .a_local(1'b0),
    .a_req_data(r_x_req_data), .a_done(r_x_done), .a_timeout(r_x_timeout), .a_rsp_data(r_x_rsp),
    .clk_b(clk), .rst_b_n(rst_n), .b_valid(x_bvalid), .b_req_data(x_breq),
    .b_rsp(x_brsp), .b_rsp_data(x_brsp_data)
  );
  assign spm_req_valid_o  = x_bvalid && x_breq.kind != XK_CORE_RD;
  assign spm_req_we_o     = x_breq.kind == XK_SPM_WR;
  assign spm_req_addr_o   = x_breq.addr;
  assign spm_req_be_o     = x_breq.be;
  assign spm_req_wdata_o  = x_breq.wdata;
  assign cwin_req_valid_o = x_bvalid && x_breq.kind == XK_CORE_RD;
  assign cwin_req_off_o   = x_breq.addr[11:0];
  assign x_brsp           = (spm_req_valid_o && spm_rsp_valid_i) || (cwin_req_valid_o && cwin_rsp_valid_i);
  assign x_brsp_data      = cwin_req_valid_o ? {1'b0, 480'h0, cwin_rsp_rdata_i} : {spm_rsp_err_i, spm_rsp_line_i};

  // ---------------------------------------------------------------- LQ / LCQ (RV -> core)
  logic   lq_bvalid, lq_brsp;
  lqreq_t lq_breq;
  tsr_cdc_hs #(.REQ_W(LQREQ_W), .RSP_W(LQRSP_W), .TIMEOUT(HS_TIMEOUT), .STAGES(ST)) u_hs_lq (
    .clk_a(clk_rv), .rst_a_n(rst_rv_n), .a_ready(r_lq_ready), .a_req(r_lq_req), .a_local(1'b0),
    .a_req_data(r_lq_req_data), .a_done(r_lq_done), .a_timeout(r_lq_timeout), .a_rsp_data(r_lq_rsp),
    .clk_b(clk), .rst_b_n(rst_n), .b_valid(lq_bvalid), .b_req_data(lq_breq),
    .b_rsp(lq_brsp), .b_rsp_data({lq_rsp_fault_i, lq_rsp_rdata_i})
  );
  assign lq_req_valid_o = lq_bvalid;
  assign lq_req_off_o   = lq_breq.off;
  assign lq_req_we_o    = lq_breq.we;
  assign lq_req_wdata_o = lq_breq.wdata;
  assign lq_req_lcq_w_o = lq_breq.lcq_w;
  assign lq_brsp        = lq_bvalid && lq_rsp_valid_i;

  // ---------------------------------------------------------------- RT mirrors (RV -> core)
  logic rt_ready, rt_done, rt_dirty_q, rt_bvalid, rt_timeout;
  logic [0:0]  rt_rsp;
  logic [63:0] rt_breq;
  logic        rt_send;
  assign rt_send = rt_dirty_q && rt_ready && !rt_done;
  always_ff @(posedge clk_rv) begin
    if (!rst_rv_n)    rt_dirty_q <= 1'b0;
    else if (r_rt_wr) rt_dirty_q <= 1'b1;
    else if (rt_send) rt_dirty_q <= 1'b0;
  end
  tsr_cdc_hs #(.REQ_W(64), .RSP_W(1), .TIMEOUT(0), .STAGES(ST)) u_hs_rt (
    .clk_a(clk_rv), .rst_a_n(rst_rv_n), .a_ready(rt_ready), .a_req(rt_send), .a_local(1'b0),
    .a_req_data({r_rt_info, r_rt_state}), .a_done(rt_done), .a_timeout(rt_timeout), .a_rsp_data(rt_rsp),
    .clk_b(clk), .rst_b_n(rst_n), .b_valid(rt_bvalid), .b_req_data(rt_breq),
    .b_rsp(rt_bvalid), .b_rsp_data(1'b0)
  );
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rt_state_o <= '0;
      rt_info_o  <= '0;
    end else if (rt_bvalid) begin
      rt_state_o <= rt_breq[31:0];
      rt_info_o  <= rt_breq[63:32];
    end
  end

  // ---------------------------------------------------------------- LOAD_* (core -> RV)
  logic [LDRSP_W-1:0] ld_arsp;
  logic               ld_done;
  tsr_cdc_hs #(.REQ_W(LDREQ_W), .RSP_W(LDRSP_W), .TIMEOUT(HS_TIMEOUT), .STAGES(ST)) u_hs_ld (
    .clk_a(clk), .rst_a_n(rst_n), .a_ready(ld_ready_o), .a_req(ld_req_i), .a_local(!rv_ok),
    .a_req_data({ld_we_i, ld_addr_i, ld_wdata_i}), .a_done(ld_done), .a_timeout(ld_rsp_timeout_o),
    .a_rsp_data(ld_arsp),
    .clk_b(clk_rv), .rst_b_n(rst_rv_n), .b_valid(r_ld_valid), .b_req_data(r_ld_req),
    .b_rsp(r_ld_rsp), .b_rsp_data(r_ld_rsp_data)
  );
  assign ld_rsp_valid_o = ld_done;
  assign ld_rsp_rdata_o = ld_rsp_timeout_o ? '0 : ld_arsp[31:0];
  assign ld_rsp_err_o   = !ld_rsp_timeout_o && ld_arsp[32];

  // ---------------------------------------------------------------- DMI (core -> RV)
  logic [DMIRSP_W-1:0] dmi_arsp;
  tsr_cdc_hs #(.REQ_W(DMIREQ_W), .RSP_W(DMIRSP_W), .TIMEOUT(HS_TIMEOUT), .STAGES(ST)) u_hs_dmi (
    .clk_a(clk), .rst_a_n(rst_n), .a_ready(dmi_ready_o), .a_req(dmi_req_i), .a_local(!rv_ok),
    .a_req_data({dmi_addr_i, dmi_op_i, dmi_wdata_i}), .a_done(dmi_rsp_valid_o),
    .a_timeout(dmi_rsp_timeout_o), .a_rsp_data(dmi_arsp),
    .clk_b(clk_rv), .rst_b_n(rst_rv_n), .b_valid(r_dmi_valid), .b_req_data(r_dmi_req),
    .b_rsp(r_dmi_rsp), .b_rsp_data(r_dmi_rsp_data)
  );
  assign dmi_rsp_rdata_o = dmi_rsp_timeout_o ? '0 : dmi_arsp[33:2];
  assign dmi_rsp_resp_o  = dmi_rsp_timeout_o ? 2'b00 : dmi_arsp[1:0];

  // ---------------------------------------------------------------- sticky events (RV -> core)
  logic [1:0] ev_tgl_q, ev_tgl_s, ev_tgl_d;
  always_ff @(posedge clk_rv) begin
    if (!rst_rv_n) ev_tgl_q <= '0;
    else           ev_tgl_q <= ev_tgl_q ^ {r_bridge_timeout_ev, r_push_timeout_ev};
  end
  tsr_sync #(.W(2), .STAGES(ST), .PER_BIT(1'b1)) u_sync_ev (.clk(clk), .rst_n(rst_n), .d(ev_tgl_q), .q(ev_tgl_s));
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ev_tgl_d         <= '0;
      push_timeout_o   <= 1'b0;
      bridge_timeout_o <= 1'b0;
    end else begin
      ev_tgl_d         <= ev_tgl_s;
      push_timeout_o   <= ev_tgl_s[0] != ev_tgl_d[0];
      bridge_timeout_o <= ev_tgl_s[1] != ev_tgl_d[1];
    end
  end

  // ---------------------------------------------------------------- levels core -> RV
  localparam int unsigned LV_W = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 4 + 6 + 6 + 2;
  logic [LV_W-1:0] lv_s;
  (* dont_touch = "true" *) logic [LV_W-1:0] lv_q;   // not merged with tsr_dbg_rv's copy of rv_run / dbg_stop (CDC-11)
  // The core's level outputs are combinational functions of several registers (holds, queue states,
  // DB sources): a glitch of one could be sampled by the synchronizer. Every bit therefore enters
  // it from a flop that drives nothing else (report_cdc CDC-10), one core cycle later.
  always_ff @(posedge clk) begin
    if (!rst_n) lv_q <= '0;
    else        lv_q <= {dbg_cancel_tgl_i, dbg_halt_tgl_i, q_nonempty_i, q_err_i, q_hold_i, perf_evt_en_i,
                         perf_time_en_i, irq_lcq_i, irq_lq_i, dbg_halt_on_boot_i, dbg_stop_i, rv_run_i};
  end
  tsr_sync #(.W(LV_W), .STAGES(ST), .PER_BIT(1'b1)) u_sync_lv (
    .clk(clk_rv), .rst_n(rst_rv_n), .d(lv_q), .q(lv_s)
  );
  assign {r_q_nonempty, r_q_err, r_q_hold, r_perf_evt_en, r_perf_time_en, r_irq_lcq, r_irq_lq,
          r_dbg_halt_on_boot, r_dbg_stop, r_rv_run} = lv_s[LV_W-3:0];

  logic [1:0] dtgl_d;
  always_ff @(posedge clk_rv) begin
    if (!rst_rv_n) begin
      dtgl_d            <= '0;
      r_dbg_halt_req    <= 1'b0;
      r_dbg_halt_cancel <= 1'b0;
    end else begin
      dtgl_d            <= lv_s[LV_W-1:LV_W-2];
      r_dbg_halt_req    <= lv_s[LV_W-2] != dtgl_d[0];
      r_dbg_halt_cancel <= lv_s[LV_W-1] != dtgl_d[1];
    end
  end

  // RV_BOOT: quasi-static (the host writes it while the hart is held in reset, EX §3.3; the
  // sequencer samples it two cycles after the synchronized RV_RUN).
  tsr_sync #(.W(32), .STAGES(ST)) u_sync_boot (.clk(clk_rv), .rst_n(rst_rv_n), .d(rv_boot_i), .q(r_rv_boot));

  // ---------------------------------------------------------------- levels RV -> core
  logic rv_up_q;            // the clk_rv domain is out of reset
  logic [1:0] rdone_dly_q;  // RV_RESET_DONE two stages behind the FIFO write pointer (CC §11.2)
  always_ff @(posedge clk_rv) begin
    if (!rst_rv_n) begin
      rv_up_q     <= 1'b0;
      rdone_dly_q <= 2'b11;
    end else begin
      rv_up_q     <= 1'b1;
      rdone_dly_q <= {rdone_dly_q[0], r_rv_reset_done};
    end
  end
  logic [7:0] rl_s;
  tsr_sync #(.W(8), .STAGES(ST), .RST_VAL(8'b0_0_001000), .PER_BIT(1'b1)) u_sync_rl (
    .clk(clk), .rst_n(rst_n),
    .d({r_rv_dmactive, rv_up_q, r_rv_halt_pend, r_rv_ndmreset, rdone_dly_q[1], r_rv_running, r_rv_sleep, r_rv_halted}),
    .q(rl_s)
  );
  assign {rv_halt_pend_o, rv_ndmreset_o, rv_reset_done_o, rv_running_o, rv_sleep_o, rv_halted_o} = rl_s[5:0];
  assign rv_dmactive_o = rl_s[7];

  // RV_CLK_OK: out of reset and the clk_rv cycle count moved recently.
  logic [15:0] cyc_prev_q;
  logic [8:0]  still_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cyc_prev_q <= '0;
      still_q    <= '1;
    end else begin
      cyc_prev_q <= rv_cyc_o;
      if (rv_cyc_o != cyc_prev_q) still_q <= '0;
      else if (!still_q[8])       still_q <= still_q + 1'b1;
    end
  end
  assign rv_ok       = rl_s[6] && !still_q[8];
  assign rv_clk_ok_o = rv_ok;

  wire unused_ok = &{1'b0, log_wr_empty, rt_rsp, rt_timeout, lv_s};

endmodule
