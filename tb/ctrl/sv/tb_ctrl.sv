// tb_ctrl.sv — top of the tb/ctrl bench (CC §15.2): tsr_ctrl (tsr_ctrl_rv +
// tsr_ctrl_cdc) with the core-domain stub tb_ctrl_core. Testbench only. In tb/ctrl/sv, not
// tb/ctrl/hdl: rtl/lint.sh lints tb/*/hdl with rtl/filelist.f alone; rtl/ctrl/lint.sh lints it.
//
// cocotb drives the two clocks and resets, the core-side controls (rv_run, rv_boot, the DB levels
// and toggles, the hold / error / non-empty levels, the interrupt levels), the loader and DMI
// requests and the error records; the stub answers the issue FIFO, the SPM bridge, the core window
// and the LQ / LCQ port. Ibex's RVFI stream (built with RVFI defined) is written to rvfi.log, one
// line per retired instruction: order pc_rdata pc_wdata insn trap intr rd_addr rd_wdata mem_addr
// mem_rmask mem_wmask mem_wdata mem_rdata debug_mode (hex).

module tb_ctrl
  import tsr_tables_pkg::*;
  import tsr_ctrl_pkg::*;
#(
  parameter int unsigned ISSUE_WAIT_MAX = TSR_ISSUE_WAIT_MAX,
  parameter int unsigned HS_TIMEOUT     = 4096,
  parameter bit          CTRL_SYNC      = 1'b0,
  parameter bit          RV_LOCAL_TRANSPORT = 1'b0,
  parameter bit          XSLR_REG       = 1'b0    // the SPM link end's request stage (needs RV_LOCAL_TRANSPORT)
) (
  input  logic          clk,
  input  logic          rst_n,
  input  logic          clk_rv,
  input  logic          rst_rv_n,
  // stub knobs
  input  logic [31:0]   cfg_seed,
  input  logic [7:0]    cfg_lat,
  input  logic          cfg_pop_mute,
  input  logic          cfg_acc_mute,
  input  logic          cfg_spm_mute,
  input  logic          cfg_cwin_mute,
  input  logic          cfg_lq_mute,
  // core-side controls
  input  logic          rv_run,
  input  logic [31:0]   rv_boot,
  input  logic          dbg_stop,
  input  logic          dbg_halt_tgl,
  input  logic          dbg_cancel_tgl,
  input  logic          dbg_halt_on_boot,
  input  logic          irq_lq,
  input  logic          irq_lcq,
  input  logic          perf_time_en,
  input  logic          perf_evt_en,
  input  logic [3:0]    q_hold,
  input  logic [5:0]    q_err,
  input  logic [5:0]    q_nonempty,
  input  logic [5:0]    err_valid,
  input  logic [767:0]  err_rec,
  output logic [5:0]    err_ack,
  output logic          rec_resync,
  // loader and DMI
  output logic          ld_ready,
  input  logic          ld_req,
  input  logic          ld_we,
  input  logic [31:0]   ld_addr,
  input  logic [31:0]   ld_wdata,
  output logic          ld_rsp_valid,
  output logic [31:0]   ld_rsp_rdata,
  output logic          ld_rsp_err,
  output logic          ld_rsp_timeout,
  output logic          dmi_ready,
  input  logic          dmi_req,
  input  logic [6:0]    dmi_addr,
  input  logic [1:0]    dmi_op,
  input  logic [31:0]   dmi_wdata,
  output logic          dmi_rsp_valid,
  output logic [31:0]   dmi_rsp_rdata,
  output logic [1:0]    dmi_rsp_resp,
  output logic          dmi_rsp_timeout,
  // observed core-side outputs
  output logic [31:0]   rt_state,
  output logic [31:0]   rt_info,
  output logic [167:0]  issued_seq,
  output logic [167:0]  s_seq,
  output logic [167:0]  d_seq,
  output logic [15:0]   rv_cyc,
  output logic [31:0]   perf_push_stall,
  output logic [31:0]   perf_bridge_acc,
  output logic [31:0]   perf_bridge_wait,
  output logic          push_timeout,
  output logic          bridge_timeout,
  output logic          rv_halted,
  output logic          rv_sleep,
  output logic          rv_running,
  output logic          rv_reset_done,
  output logic          rv_clk_ok,
  output logic          rv_ndmreset,
  output logic          rv_dmactive,
  output logic          rv_halt_pend,
  output logic          if_valid,
  output logic [31:0]   stub_errors,
  output logic [31:0]   stub_pops,
  output logic [31:0]   rvfi_count
);

  logic                      if_ready;
  logic [TSR_IFIFO_BITS-1:0] if_data;
  logic [5:0][27:0]          issued_q, s_q, d_q;
  logic                      spm_req_valid, spm_req_we, spm_rsp_valid;
  logic [25:0]               spm_req_addr;
  logic [3:0]                spm_req_be;
  logic [31:0]               spm_req_wdata;
  logic [511:0]              spm_rsp_line;
  logic                      cwin_req_valid, cwin_rsp_valid;
  logic [11:0]               cwin_req_off;
  logic [31:0]               cwin_rsp_rdata;
  logic                      lq_req_valid, lq_req_we, lq_rsp_valid, lq_rsp_fault;
  logic [11:0]               lq_req_off;
  logic [31:0]               lq_req_wdata, lq_rsp_rdata;
  logic [7:0][31:0]          lq_req_lcq_w;
  logic                      log_valid, log_ready;
  logic [31:0]               log_data;
  logic [63:0]               mtime;
  logic [5:0]                if_count;          // Q_FIFO (tsr_csr's, not modeled by the stub)
  wire unused_if_count = &{1'b0, if_count};
  logic                      db_req, db_we;     // DB window: not modeled by the stub
  logic [11:0]               db_off;
  logic [31:0]               db_wdata;
  wire unused_db = &{1'b0, db_req, db_we, db_off, db_wdata};
  // Ibex taps for the DB block: not used by this bench
  logic                      tap_data_req, tap_data_gnt, tap_data_we, tap_sleep, tap_halted, tap_in_reset, tap_dfault;
  logic [31:0]               tap_data_addr, tap_cd_pc, tap_cd_npc, tap_cd_daddr, tap_cd_epc, tap_cd_eaddr;
  logic [3:0]                tap_data_be;
  wire unused_taps = &{1'b0, tap_data_req, tap_data_gnt, tap_data_we, tap_sleep, tap_halted, tap_in_reset,
                       tap_dfault, tap_data_addr, tap_cd_pc, tap_cd_npc, tap_cd_daddr, tap_cd_epc, tap_cd_eaddr,
                       tap_data_be};

  assign issued_seq = issued_q;
  assign s_seq      = s_q;
  assign d_seq      = d_q;

  tsr_ctrl #(.ISSUE_WAIT_MAX(ISSUE_WAIT_MAX), .HS_TIMEOUT(HS_TIMEOUT), .CTRL_SYNC(CTRL_SYNC)) u_dut (
    .clk                (clk),
    .rst_n              (rst_n),
    .clk_rv             (clk_rv),
    .rst_rv_n           (rst_rv_n),
    .if_valid_o         (if_valid),
    .if_ready_i         (if_ready),
    .if_data_o          (if_data),
    .if_count_o         (if_count),
    .issued_seq_o       (issued_q),
    .s_seq_i            (s_q),
    .d_seq_i            (d_q),
    .err_valid_i        (err_valid),
    .err_rec_i          (err_rec),
    .err_ack_o          (err_ack),
    .rec_resync_o       (rec_resync),
    .spm_req_valid_o    (spm_req_valid),
    .spm_req_we_o       (spm_req_we),
    .spm_req_addr_o     (spm_req_addr),
    .spm_req_be_o       (spm_req_be),
    .spm_req_wdata_o    (spm_req_wdata),
    .spm_rsp_valid_i    (spm_rsp_valid),
    .spm_rsp_line_i     (spm_rsp_line),
    .spm_rsp_err_i      (1'b0),
    .cwin_req_valid_o   (cwin_req_valid),
    .cwin_req_off_o     (cwin_req_off),
    .cwin_rsp_valid_i   (cwin_rsp_valid),
    .cwin_rsp_rdata_i   (cwin_rsp_rdata),
    .lq_req_valid_o     (lq_req_valid),
    .lq_req_off_o       (lq_req_off),
    .lq_req_we_o        (lq_req_we),
    .lq_req_wdata_o     (lq_req_wdata),
    .lq_req_lcq_w_o     (lq_req_lcq_w),
    .lq_rsp_valid_i     (lq_rsp_valid),
    .lq_rsp_rdata_i     (lq_rsp_rdata),
    .lq_rsp_fault_i     (lq_rsp_fault),
    .log_valid_o        (log_valid),
    .log_data_o         (log_data),
    .log_ready_i        (log_ready),
    .rt_state_o         (rt_state),
    .rt_info_o          (rt_info),
    .ld_ready_o         (ld_ready),
    .ld_req_i           (ld_req),
    .ld_we_i            (ld_we),
    .ld_addr_i          (ld_addr),
    .ld_wdata_i         (ld_wdata),
    .ld_rsp_valid_o     (ld_rsp_valid),
    .ld_rsp_rdata_o     (ld_rsp_rdata),
    .ld_rsp_err_o       (ld_rsp_err),
    .ld_rsp_timeout_o   (ld_rsp_timeout),
    .dmi_ready_o        (dmi_ready),
    .dmi_req_i          (dmi_req),
    .dmi_addr_i         (dmi_addr),
    .dmi_op_i           (dmi_op),
    .dmi_wdata_i        (dmi_wdata),
    .dmi_rsp_valid_o    (dmi_rsp_valid),
    .dmi_rsp_rdata_o    (dmi_rsp_rdata),
    .dmi_rsp_resp_o     (dmi_rsp_resp),
    .dmi_rsp_timeout_o  (dmi_rsp_timeout),
    .mtime_i            (mtime),
    .rv_cyc_o           (rv_cyc),
    .perf_push_stall_o  (perf_push_stall),
    .perf_bridge_acc_o  (perf_bridge_acc),
    .perf_bridge_wait_o (perf_bridge_wait),
    .push_timeout_o     (push_timeout),
    .bridge_timeout_o   (bridge_timeout),
    .rv_run_i           (rv_run),
    .rv_boot_i          (rv_boot),
    .dbg_stop_i         (dbg_stop),
    .dbg_halt_tgl_i     (dbg_halt_tgl),
    .dbg_cancel_tgl_i   (dbg_cancel_tgl),
    .dbg_halt_on_boot_i (dbg_halt_on_boot),
    .irq_lq_i           (irq_lq),
    .irq_lcq_i          (irq_lcq),
    .perf_time_en_i     (perf_time_en),
    .perf_evt_en_i      (perf_evt_en),
    .q_hold_i           (q_hold),
    .q_err_i            (q_err),
    .q_nonempty_i       (q_nonempty),
    .rv_halted_o        (rv_halted),
    .rv_sleep_o         (rv_sleep),
    .rv_running_o       (rv_running),
    .rv_reset_done_o    (rv_reset_done),
    .rv_clk_ok_o        (rv_clk_ok),
    .rv_ndmreset_o      (rv_ndmreset),
    .rv_dmactive_o      (rv_dmactive),
    .rv_halt_pend_o     (rv_halt_pend),
    .db_req_o           (db_req),
    .db_we_o            (db_we),
    .db_off_o           (db_off),
    .db_wdata_o         (db_wdata),
    .db_rdata_i         (32'h0),
    .db_irq4_i          (1'b0),
    .db_irq6_i          (1'b0),
    .tap_data_req_o     (tap_data_req),
    .tap_data_gnt_o     (tap_data_gnt),
    .tap_data_addr_o    (tap_data_addr),
    .tap_data_we_o      (tap_data_we),
    .tap_data_be_o      (tap_data_be),
    .tap_cd_pc_o        (tap_cd_pc),
    .tap_cd_npc_o       (tap_cd_npc),
    .tap_cd_daddr_o     (tap_cd_daddr),
    .tap_cd_epc_o       (tap_cd_epc),
    .tap_cd_eaddr_o     (tap_cd_eaddr),
    .tap_sleep_o        (tap_sleep),
    .tap_halted_o       (tap_halted),
    .tap_in_reset_o     (tap_in_reset),
    .tap_dfault_o       (tap_dfault)
  );

  logic issue_valid, issue_ready;
  logic [TSR_IFIFO_BITS-1:0] issue_entry;
  logic spm_bridge_req_valid, spm_bridge_rsp_valid, spm_bridge_rsp_ready;
  logic spm_bridge_req_we;
  logic [25:0] spm_bridge_req_addr;
  logic [3:0] spm_bridge_req_be;
  logic [31:0] spm_bridge_req_wdata;
  logic [511:0] spm_bridge_rsp_line;
  logic lq_csr_req_valid, lq_csr_req_we;
  logic [11:0] lq_csr_req_off;
  logic [31:0] lq_csr_req_wdata;
  logic [7:0][31:0] lq_csr_req_lcq_w;

  if (RV_LOCAL_TRANSPORT) begin : g_rv_local
    logic         spm_v, spm_err, issue_v, issue_ack, lq_v, lq_we, lq_end;
    logic [127:0] spm_data, lq_data;
    logic [128:0] issue_data;
    logic [11:0]  lq_off;
    logic         unused_spm_rsp_err;

    tsr_rv_spm_tx #(.XSLR_REG(XSLR_REG)) u_spm_tx (
      .clk, .rst_n(rst_n), .req_valid(spm_req_valid), .bridge_req_valid(spm_bridge_req_valid),
      .req_fields({spm_req_we, spm_req_addr, spm_req_be, spm_req_wdata}),
      .bridge_req_fields({spm_bridge_req_we, spm_bridge_req_addr, spm_bridge_req_be, spm_bridge_req_wdata}),
      .bridge_rsp_valid(spm_bridge_rsp_valid), .bridge_rsp_ready(spm_bridge_rsp_ready),
      .bridge_rsp_line(spm_bridge_rsp_line), .bridge_rsp_err(1'b0),
      .link_valid(spm_v), .link_data(spm_data), .link_err(spm_err)
    );
    tsr_rv_spm_rx u_spm_rx (
      .clk, .rst_n(rst_n), .link_valid(spm_v), .link_data(spm_data), .link_err(spm_err),
      .rsp_valid(spm_rsp_valid), .rsp_line(spm_rsp_line), .rsp_err(unused_spm_rsp_err)
    );
    tsr_rv_issue_tx u_issue_tx (
      .clk, .rst_n(rst_n), .fifo_valid(if_valid), .fifo_entry(if_data), .fifo_ready(if_ready),
      .link_valid(issue_v), .link_data(issue_data), .link_accepted(issue_ack)
    );
    tsr_rv_issue_rx u_issue_rx (
      .clk, .rst_n(rst_n), .link_valid(issue_v), .link_data(issue_data), .link_accepted(issue_ack),
      .issue_valid, .issue_entry, .issue_ready
    );
    tsr_rv_lq_tx u_lq_tx (
      .clk, .rst_n(rst_n), .req_valid(lq_req_valid), .req_off(lq_req_off), .req_we(lq_req_we),
      .req_wdata(lq_req_wdata), .req_lcq_w(lq_req_lcq_w), .link_valid(lq_v), .link_data(lq_data),
      .link_off(lq_off), .link_we(lq_we), .link_end(lq_end)
    );
    tsr_rv_lq_rx u_lq_rx (
      .clk, .rst_n(rst_n), .link_valid(lq_v), .link_data(lq_data), .link_off(lq_off),
      .link_we(lq_we), .link_end(lq_end), .req_valid(lq_csr_req_valid), .req_off(lq_csr_req_off),
      .req_we(lq_csr_req_we), .req_wdata(lq_csr_req_wdata), .req_lcq_w(lq_csr_req_lcq_w)
    );
  end else begin : g_rv_direct
    assign spm_bridge_req_valid = spm_req_valid;
    assign spm_bridge_req_we    = spm_req_we;
    assign spm_bridge_req_addr  = spm_req_addr;
    assign spm_bridge_req_be    = spm_req_be;
    assign spm_bridge_req_wdata = spm_req_wdata;
    assign spm_bridge_rsp_ready = 1'b1;
    assign spm_rsp_valid        = spm_bridge_rsp_valid;
    assign spm_rsp_line         = spm_bridge_rsp_line;
    assign issue_valid          = if_valid;
    assign issue_entry          = if_data;
    assign if_ready             = issue_ready;
    assign lq_csr_req_valid     = lq_req_valid;
    assign lq_csr_req_off       = lq_req_off;
    assign lq_csr_req_we        = lq_req_we;
    assign lq_csr_req_wdata     = lq_req_wdata;
    assign lq_csr_req_lcq_w     = lq_req_lcq_w;
  end

  tb_ctrl_core u_core (
    .clk(clk), .rst_n(rst_n), .cfg_seed(cfg_seed), .cfg_lat(cfg_lat), .cfg_pop_mute(cfg_pop_mute),
    .cfg_acc_mute(cfg_acc_mute), .cfg_spm_mute(cfg_spm_mute), .cfg_cwin_mute(cfg_cwin_mute),
    .cfg_lq_mute(cfg_lq_mute),
    .if_valid(issue_valid), .if_ready(issue_ready), .if_data(issue_entry), .issued_seq(issued_q),
    .s_seq(s_q), .d_seq(d_q),
    .spm_req_valid(spm_bridge_req_valid), .spm_req_we(spm_bridge_req_we), .spm_req_addr(spm_bridge_req_addr),
    .spm_req_be(spm_bridge_req_be), .spm_req_wdata(spm_bridge_req_wdata), .spm_rsp_valid(spm_bridge_rsp_valid), .spm_rsp_ready(spm_bridge_rsp_ready),
    .spm_rsp_line(spm_bridge_rsp_line),
    .cwin_req_valid(cwin_req_valid), .cwin_req_off(cwin_req_off), .cwin_rsp_valid(cwin_rsp_valid),
    .cwin_rsp_rdata(cwin_rsp_rdata),
    .lq_req_valid(lq_csr_req_valid), .lq_req_off(lq_csr_req_off), .lq_req_we(lq_csr_req_we),
    .lq_req_wdata(lq_csr_req_wdata), .lq_req_lcq_w(lq_csr_req_lcq_w), .lq_rsp_valid(lq_rsp_valid),
    .lq_rsp_rdata(lq_rsp_rdata), .lq_rsp_fault(lq_rsp_fault),
    .log_valid(log_valid), .log_data(log_data), .log_ready(log_ready),
    .mtime(mtime), .stub_errors(stub_errors), .stub_pops(stub_pops)
  );

  // RVFI log (RVFI is defined in this bench's build).
  integer fd_rvfi;
  initial fd_rvfi = $fopen("rvfi.log", "w");
  always_ff @(posedge clk_rv) begin
    if (!rst_n) begin
      rvfi_count <= '0;
    end else if (u_dut.u_rv.rvfi_valid) begin
      rvfi_count <= rvfi_count + 1;
      $fwrite(fd_rvfi, "%h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
              u_dut.u_rv.rvfi_order, u_dut.u_rv.rvfi_pc_rdata, u_dut.u_rv.rvfi_pc_wdata,
              u_dut.u_rv.rvfi_insn, u_dut.u_rv.rvfi_trap, u_dut.u_rv.rvfi_intr,
              u_dut.u_rv.rvfi_rd_addr, u_dut.u_rv.rvfi_rd_wdata, u_dut.u_rv.rvfi_mem_addr,
              u_dut.u_rv.rvfi_mem_rmask, u_dut.u_rv.rvfi_mem_wmask, u_dut.u_rv.rvfi_mem_wdata,
              u_dut.u_rv.rvfi_mem_rdata, u_dut.u_rv.rvfi_ext_debug_mode);
      $fflush(fd_rvfi);
    end
  end

endmodule
