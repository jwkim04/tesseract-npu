// tsr_csr_tb.sv — testbench wrapper (tb/csr): tsr_csr and tsr_issue connected as tsr_core_dp
// connects them (CC §13.2), every other port at the top: the host CSR port, the core-domain side of
// tsr_ctrl_cdc (core window, LQ / LCQ port, log words, loader, DMI, levels and counters), the issue
// FIFO, the external checker class, the unit command ports and the record handshake. The nets
// between the two blocks keep tsr_issue's port names, so the tb/issue environment
// (tb/issue/issue_tb.py) drives and monitors this wrapper unchanged.

module tsr_csr_tb
  import tsr_pkg::*;
#(
  parameter int unsigned     NUM_CH      = 4,
  parameter int unsigned     T_MAX       = 8,
  parameter int unsigned     K_MAX       = 32768,
  parameter int unsigned     SPM_BYTES   = 4 * 1024 * 1024,
  parameter longint unsigned CH_BYTES    = 64'd17179869184,
  parameter int unsigned     ADDR_BEAT_W = 28,
  parameter logic [31:0]     CAPS        = 32'd0,
  parameter logic [31:0]     CTL_ID      = 32'h0000_5D01,
  parameter int unsigned     QD_DMA      = 8,
  parameter int unsigned     QD_QACT     = 4,
  parameter int unsigned     QD_MVM      = 4,
  parameter int unsigned     QD_VU       = 16,
  parameter int unsigned     QD_ATT      = 0,
  parameter int unsigned     QD_SYNC     = 4,
  parameter int unsigned     SR_RV_MIN   = 64,
  parameter int unsigned     SR_RV_MAX   = 65536,
  parameter int unsigned     XC_CMD_W    = 256,
  parameter bit              FAST_CLK    = 1'b0
) (
  input  logic                        clk,
  input  logic                        rst_n,
  // host CSR port
  input  logic                        csr_valid,
  output logic                        csr_ready,
  input  logic                        csr_we,
  input  logic [11:0]                 csr_addr,
  input  logic [31:0]                 csr_wdata,
  output logic [31:0]                 csr_rdata,
  // tsr_ctrl_cdc, core-domain side
  input  logic                        cwin_req_valid,
  input  logic [11:0]                 cwin_req_off,
  output logic                        cwin_rsp_valid,
  output logic [31:0]                 cwin_rsp_rdata,
  input  logic                        lq_req_valid,
  input  logic [11:0]                 lq_req_off,
  input  logic                        lq_req_we,
  input  logic [31:0]                 lq_req_wdata,
  input  logic [7:0][31:0]            lq_req_lcq_w,
  output logic                        lq_rsp_valid,
  output logic [31:0]                 lq_rsp_rdata,
  output logic                        lq_rsp_fault,
  input  logic                        log_valid,
  input  logic [31:0]                 log_data,
  output logic                        log_ready,
  input  logic [31:0]                 rt_state,
  input  logic [31:0]                 rt_info,
  input  logic                        ld_ready,
  output logic                        ld_req,
  output logic                        ld_we,
  output logic [31:0]                 ld_addr,
  output logic [31:0]                 ld_wdata,
  input  logic                        ld_rsp_valid,
  input  logic [31:0]                 ld_rsp_rdata,
  input  logic                        ld_rsp_err,
  input  logic                        ld_rsp_timeout,
  input  logic                        dmi_en,
  input  logic                        dmi_ready,
  output logic                        dmi_req,
  output logic [6:0]                  dmi_addr,
  output logic [1:0]                  dmi_op,
  output logic [31:0]                 dmi_wdata,
  input  logic                        dmi_rsp_valid,
  input  logic [31:0]                 dmi_rsp_rdata,
  input  logic [1:0]                  dmi_rsp_resp,
  input  logic                        dmi_rsp_timeout,
  input  logic [31:0]                 dbg_rdata,
  input  logic                        db_stop_active,
  input  logic [15:0][31:0]           sreg,
  output logic [63:0]                 mtime,
  input  logic [15:0]                 rv_cyc,
  input  logic [31:0]                 perf_push_stall,
  input  logic [31:0]                 perf_bridge_acc,
  input  logic [31:0]                 perf_bridge_wait,
  input  logic                        push_timeout,
  input  logic                        bridge_timeout,
  output logic                        rv_run,
  output logic [31:0]                 rv_boot,
  output logic                        irq_lq,
  output logic                        irq_lcq,
  output logic                        perf_time_en,
  output logic                        perf_evt_en,
  input  logic                        rv_halted,
  input  logic                        rv_sleep,
  input  logic                        rv_running,
  input  logic                        rv_reset_done,
  input  logic                        rv_clk_ok,
  input  logic                        rv_ndmreset,
  input  logic [5:0][27:0]            issued_seq,
  input  logic [5:0]                  ifq_count,
  output logic                        sr_db_clear,
  output logic                        dbg_req,
  // issue FIFO, external class, units, records (tsr_issue)
  input  logic                        ifq_valid,
  output logic                        ifq_ready,
  input  tsr_ifent_t                  ifq_entry,
  output logic                        xc_req_valid,
  output tsr_insn_t                   xc_req_insn,
  input  logic                        xc_rsp_valid,
  input  logic [3:0]                  xc_rsp_code,
  input  logic [31:0]                 xc_rsp_info,
  input  tsr_ext_t [TSR_EXT_N-1:0]    xc_rsp_ext,
  input  logic [XC_CMD_W-1:0]         xc_rsp_cmd,
  output logic                        dma_cmd_valid,
  input  logic                        dma_cmd_ready,
  output tsr_dma_cmd_t                dma_cmd,
  output logic                        dma_cmd_xc,
  output logic [XC_CMD_W-1:0]         dma_xcmd,
  input  logic                        dma_done,
  output logic                        qact_cmd_valid,
  input  logic                        qact_cmd_ready,
  output tsr_qact_cmd_t               qact_cmd,
  input  logic                        qact_done,
  output logic                        mvm_cmd_valid,
  input  logic                        mvm_cmd_ready,
  output tsr_mvm_cmd_t                mvm_cmd,
  output logic                        mvm_cmd_xc,
  output logic [XC_CMD_W-1:0]         mvm_xcmd,
  input  logic                        mvm_done,
  output logic                        vu_cmd_valid,
  input  logic                        vu_cmd_ready,
  output logic [XC_CMD_W-1:0]         vu_cmd,
  input  logic                        vu_done,
  output logic                        att_cmd_valid,
  input  logic                        att_cmd_ready,
  output logic [XC_CMD_W-1:0]         att_cmd,
  input  logic                        att_done,
  input  logic [5:0]                  dbg_hold,
  input  logic [5:0]                  dbg_step,
  output logic [5:0]                  err_valid,
  output logic [5:0][127:0]           err_hs,
  input  logic [5:0]                  err_ack,
  input  logic [NUM_CH-1:0]           perf_ch_rd_beat,
  input  logic [NUM_CH-1:0]           perf_ch_wait,
  input  logic [63:0][3:0]            perf_ext,
  input  logic [5:0]                  unit_progress,
  input  logic                        plat_err_i,
  output logic                        host_irq_o,
  output logic                        unit_hung
);

  // nets between tsr_csr and tsr_issue (tsr_issue's port names)
  logic                       host_hold, halt_hold, qabort, rv_reset, host_clear, sr_busy;
  tsr_err_rec_t [5:0]         err_rec;
  logic [5:0][TSR_SEQ_W-1:0]  seq_s, seq_d;
  logic [31:0]                q_pidx;
  logic [5:0][8:0]            q_state;
  logic [5:0]                 q_nonempty;
  logic                       err_any, abort_active, qidle, qquiescent;
  logic [2:0]                 hold_src;
  logic [2:0]                 perf_retired;
  logic                       perf_push, perf_mvm_wait, perf_mvm_empty, perf_hold;
  logic [5:0]                 perf_busy;
  // taps (monitors)
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
  logic                       dbg_we, dbg_host;
  logic [11:0]                dbg_addr;
  logic [31:0]                dbg_wdata;

  tsr_csr #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CAPS(CAPS), .CTL_ID(CTL_ID),
    .QD_DMA(QD_DMA), .QD_QACT(QD_QACT), .QD_MVM(QD_MVM), .QD_VU(QD_VU), .QD_ATT(QD_ATT), .QD_SYNC(QD_SYNC),
    .SR_RV_MIN(SR_RV_MIN), .SR_RV_MAX(SR_RV_MAX), .FAST_CLK(FAST_CLK)
  ) u_csr (
    .clk, .rst_n,
    .csr_valid, .csr_ready, .csr_we, .csr_addr, .csr_wdata, .csr_rdata,
    .cwin_req_valid, .cwin_req_off, .cwin_rsp_valid, .cwin_rsp_rdata,
    .lq_req_valid, .lq_req_off, .lq_req_we, .lq_req_wdata, .lq_req_lcq_w, .lq_rsp_valid, .lq_rsp_rdata,
    .lq_rsp_fault,
    .log_valid, .log_data, .log_ready, .rt_state, .rt_info,
    .ld_ready, .ld_req, .ld_we, .ld_addr, .ld_wdata, .ld_rsp_valid, .ld_rsp_rdata, .ld_rsp_err, .ld_rsp_timeout,
    .dmi_en, .dmi_ready, .dmi_req, .dmi_addr, .dmi_op, .dmi_wdata, .dmi_rsp_valid, .dmi_rsp_rdata, .dmi_rsp_resp,
    .dmi_rsp_timeout,
    .dbg_req, .dbg_we, .dbg_addr, .dbg_wdata, .dbg_host, .dbg_rdata, .sr_db_clear, .db_stop_active,
    .sreg,
    .mtime, .rv_cyc, .perf_push_stall, .perf_bridge_acc, .perf_bridge_wait, .push_timeout, .bridge_timeout,
    .rv_run, .rv_boot, .irq_lq, .irq_lcq, .perf_time_en, .perf_evt_en,
    .rv_halted, .rv_sleep, .rv_running, .rv_reset_done, .rv_clk_ok, .rv_ndmreset, .issued_seq, .ifq_count,
    .host_hold, .halt_hold, .qabort, .rv_reset, .host_clear, .sr_busy,
    .qidle, .qquiescent, .err_any, .hold_src, .q_state, .seq_d, .err_rec, .q_pidx,
    .perf_retired, .perf_push, .perf_busy, .perf_mvm_wait, .perf_mvm_empty, .perf_hold,
    .perf_ch_rd_beat, .perf_ch_wait, .perf_ext, .unit_progress,
    .plat_err_i, .host_irq_o, .unit_hung
  );

  tsr_issue #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CH_BYTES(CH_BYTES),
    .ADDR_BEAT_W(ADDR_BEAT_W), .CAPS(CAPS), .QD_DMA(QD_DMA), .QD_QACT(QD_QACT), .QD_MVM(QD_MVM), .QD_VU(QD_VU),
    .QD_ATT(QD_ATT), .QD_SYNC(QD_SYNC), .XC_CMD_W(XC_CMD_W)
  ) u_issue (
    .clk, .rst_n,
    .ifq_valid, .ifq_ready, .ifq_entry,
    .xc_req_valid, .xc_req_insn, .xc_rsp_valid, .xc_rsp_code, .xc_rsp_info, .xc_rsp_ext, .xc_rsp_cmd,
    .dma_cmd_valid, .dma_cmd_ready, .dma_cmd, .dma_cmd_xc, .dma_xcmd, .dma_done,
    .qact_cmd_valid, .qact_cmd_ready, .qact_cmd, .qact_done,
    .mvm_cmd_valid, .mvm_cmd_ready, .mvm_cmd, .mvm_cmd_xc, .mvm_xcmd, .mvm_done,
    .vu_cmd_valid, .vu_cmd_ready, .vu_cmd, .vu_done,
    .att_cmd_valid, .att_cmd_ready, .att_cmd, .att_done,
    .host_hold, .halt_hold, .qabort, .rv_reset, .host_clear, .sr_busy, .dbg_hold, .dbg_step,
    .err_rec, .err_valid, .err_hs, .err_ack, .rec_resync(1'b0),
    .seq_s, .seq_d, .q_pidx, .q_state, .q_nonempty, .err_any, .abort_active, .hold_src, .qidle, .qquiescent,
    .perf_retired, .perf_push, .perf_busy, .perf_mvm_wait, .perf_mvm_empty, .perf_hold,
    .tap_enq, .tap_enq_unit, .tap_enq_seq, .tap_enq_pidx, .tap_enq_code, .tap_enq_info, .tap_enq_kind,
    .tap_enq_dep0, .tap_enq_dep1, .tap_enq_w, .tap_enq_rec, .tap_enq_pc, .tap_enq_ext,
    .tap_start, .tap_start_pidx, .tap_done, .tap_done_code, .tap_done_ran, .tap_done_pidx,
    .tap_mark, .tap_mark_value, .tap_mark_pidx, .tap_mark_pc,
    .head_valid, .head_pidx, .head_w0
  );

  // monitored by the bench (cocotb reads them by name)
  wire unused_ok = &{1'b0, q_nonempty, abort_active, dbg_we, dbg_host, dbg_addr, dbg_wdata, head_valid,
                     head_pidx, head_w0, seq_s, tap_enq, tap_enq_rec, tap_mark, tap_enq_unit, tap_enq_code,
                     tap_enq_seq, tap_enq_pidx, tap_enq_info, tap_enq_dep0, tap_enq_dep1, tap_enq_pc, tap_enq_kind,
                     tap_enq_w, tap_enq_ext, tap_start, tap_done, tap_done_ran, tap_start_pidx, tap_done_pidx,
                     tap_done_code, tap_mark_value, tap_mark_pidx, tap_mark_pc};

endmodule
