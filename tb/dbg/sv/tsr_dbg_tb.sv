// tsr_dbg_tb.sv — testbench wrapper (tb/dbg): tsr_csr, tsr_issue, the DB block (tsr_dbg,
// tsr_dbg_rv) and the trace writer's write mux (tsr_dbg_wrmux) connected as tsr_core_dp /
// tsr_core connect them (the reference wiring for the integrator, docs/design/05-debug.md §13),
// every other port at the top: the host CSR port, the core-domain side of tsr_ctrl_cdc (core
// window, LQ / LCQ port, log words, loader, DMI, levels and counters), tsr_ctrl's DB inputs and
// window (dbg_stop, the halt toggles, db_*), the Ibex taps of tsr_dbg_rv, the issue FIFO, the
// external checker class, the unit command ports, the record handshake, the DMA side of the
// write mux (a bench-driven write stream) and channel TRW_CH's arbiter write source (the bench's
// memory). The nets between the blocks keep tsr_issue's port names, so tb/issue's environment
// (tb/issue/issue_tb.py) drives and monitors this wrapper unchanged.
// Not a synthesis source; linted by tb/dbg/test_dbg.py (test_dbg_lint).

module tsr_dbg_tb
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
  parameter int unsigned     HIST_DEPTH  = 512,
  parameter bit              HAS_TRW     = 1'b1,
  parameter int unsigned     NTBP        = 4,
  parameter int unsigned     NRVWP       = 2,
  parameter bit              HAS_REC     = 1'b1,
  parameter bit              HAS_RV_TAPS = 1'b0,
  // XSLR_REG: the register slices of tsr_core_dp (g_tslice) between the trace writer and the write mux
  parameter bit              XSLR_REG    = 1'b0,
  parameter bit              FAST_CLK    = 1'b0
) (
  input  logic                        clk,
  input  logic                        rst_n,
  input  logic                        clk_rv,
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
  input  logic                        dmi_ready,
  output logic                        dmi_req,
  output logic [6:0]                  dmi_addr,
  output logic [1:0]                  dmi_op,
  output logic [31:0]                 dmi_wdata,
  input  logic                        dmi_rsp_valid,
  input  logic [31:0]                 dmi_rsp_rdata,
  input  logic [1:0]                  dmi_rsp_resp,
  input  logic                        dmi_rsp_timeout,
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
  input  logic                        rv_dmactive,
  input  logic [5:0][27:0]            issued_seq,
  input  logic [5:0]                  ifq_count,
  // tsr_ctrl's DB inputs and window (clk_rv), the unit watchdog level
  output logic                        dbg_stop,
  output logic                        dbg_halt_tgl,
  output logic                        dbg_cancel_tgl,
  output logic                        dbg_halt_on_boot,
  input  logic                        rv_halt_pend,
  input  logic                        unit_hung,
  input  logic                        db_req,
  input  logic                        db_we,
  input  logic [11:0]                 db_off,
  input  logic [31:0]                 db_wdata,
  output logic [31:0]                 db_rdata,
  output logic                        db_irq4,
  // Ibex taps (clk_rv)
  input  logic                        ib_data_req,
  input  logic                        ib_data_gnt,
  input  logic [31:0]                 ib_data_addr,
  input  logic                        ib_data_we,
  input  logic [3:0]                  ib_data_be,
  input  logic [4:0][31:0]            ib_cd,
  input  logic                        ib_sleep,
  input  logic                        ib_halted,
  input  logic                        ib_in_reset,
  input  logic                        ib_dfault,
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
  output logic [5:0]                  err_valid,
  output logic [5:0][127:0]           err_hs,
  input  logic [5:0]                  err_ack,
  input  logic [NUM_CH-1:0]           perf_ch_rd_beat,
  input  logic [NUM_CH-1:0]           perf_ch_wait,
  input  logic [5:0]                  unit_progress,
  input  logic                        plat_err_i,
  output logic                        host_irq_o,
  // DMA side of the write mux (bench write stream) and the arbiter write source (bench memory)
  input  logic                        d_wr_req_valid,
  output logic                        d_wr_req_ready,
  input  logic [ADDR_BEAT_W-1:0]      d_wr_req_addr,
  input  logic [TSR_LEN_W-1:0]        d_wr_req_len,
  input  logic                        d_wr_dat_valid,
  output logic                        d_wr_dat_ready,
  input  logic [511:0]                d_wr_dat_data,
  input  logic [15:0]                 d_wr_dat_mask,
  input  logic                        d_wr_dat_last,
  output logic                        d_wr_rsp_valid,
  output logic                        s_wr_req_valid,
  input  logic                        s_wr_req_ready,
  output logic [ADDR_BEAT_W-1:0]      s_wr_req_addr,
  output logic [TSR_LEN_W-1:0]        s_wr_req_len,
  output logic [TSR_SRC_ID_W-1:0]     s_wr_req_id,
  output logic                        s_wr_dat_valid,
  input  logic                        s_wr_dat_ready,
  output logic [511:0]                s_wr_dat_data,
  output logic [15:0]                 s_wr_dat_mask,
  output logic                        s_wr_dat_last,
  input  logic                        s_wr_rsp_valid,
  output logic                        s_wr_rsp_ready,
  input  logic [TSR_SRC_ID_W-1:0]     s_wr_rsp_id
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
  logic [5:0]                 dbg_hold, dbg_step;
  // taps
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
  // DBG block port of tsr_csr
  logic                       dbg_req, dbg_we, dbg_host, sr_db_clear, db_stop_active, dmi_en;
  logic [11:0]                dbg_addr;
  logic [31:0]                dbg_wdata, dbg_rdata;
  logic [63:0][3:0]           perf_ext;
  logic                       csr_unit_hung;       // the bench drives unit_hung (HUNG_STOP) itself
  // tsr_dbg <-> tsr_dbg_rv
  logic                       rec_wr, rec_room, rec_all, smp_ready, smp_req, smp_local, smp_done, smp_timeout;
  logic [127:0]               rec_wdata;
  logic [163:0]               smp_data;
  logic                       wpa_ready, wpa_req, wpa_local, wpa_done, wpa_timeout;
  logic [135:0]               wpa_data;
  logic                       wph_v, wph_halt, dfault;
  logic [1:0]                 wph_mask;
  logic [31:0]                wph_addr;
  logic [4:0]                 flags;
  // trace writer write source
  logic                       t_wr_req_valid, t_wr_req_ready, t_wr_dat_valid, t_wr_dat_ready, t_wr_dat_last;
  logic                       t_wr_rsp_valid;
  logic [ADDR_BEAT_W-1:0]     t_wr_req_addr;
  logic [TSR_LEN_W-1:0]       t_wr_req_len;
  logic [511:0]               t_wr_dat_data;
  // the trace writer's side of the slices
  logic                       tw_req_valid, tw_req_ready, tw_dat_valid, tw_dat_ready, tw_dat_last;
  logic                       tw_rsp_valid;
  logic [ADDR_BEAT_W-1:0]     tw_req_addr;
  logic [TSR_LEN_W-1:0]       tw_req_len;
  logic [511:0]               tw_dat_data;
  logic [TSR_SRC_ID_W-1:0]    d_wr_rsp_id;

  tsr_csr #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CAPS(CAPS), .CTL_ID(CTL_ID),
    .QD_DMA(QD_DMA), .QD_QACT(QD_QACT), .QD_MVM(QD_MVM), .QD_VU(QD_VU), .QD_ATT(QD_ATT), .QD_SYNC(QD_SYNC),
    .SR_RV_MIN(SR_RV_MIN), .SR_RV_MAX(SR_RV_MAX)
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
    .perf_ch_rd_beat, .perf_ch_rd_beat_b('0), .perf_ch_wait, .perf_ext, .unit_progress,
    .plat_err_i, .host_irq_o, .unit_hung(csr_unit_hung)
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

  tsr_dbg #(
    .NUM_CH(NUM_CH), .CH_BYTES(CH_BYTES), .ADDR_BEAT_W(ADDR_BEAT_W),
    .QD_DMA(QD_DMA), .QD_QACT(QD_QACT), .QD_MVM(QD_MVM), .QD_VU(QD_VU), .QD_ATT(QD_ATT), .QD_SYNC(QD_SYNC),
    .HIST_DEPTH(HIST_DEPTH), .HAS_TRW(HAS_TRW), .NTBP(NTBP), .NRVWP(NRVWP), .HAS_REC(HAS_REC), .FAST_CLK(FAST_CLK)
  ) u_dbg (
    .clk, .rst_n,
    .dbg_req, .dbg_we, .dbg_addr, .dbg_wdata, .dbg_host, .dbg_rdata, .sr_db_clear, .db_stop_active, .dmi_en,
    .perf_ext,
    .csr_valid, .csr_ready, .csr_we, .csr_addr, .csr_wdata, .dmi_rsp_valid, .dmi_rsp_resp, .dmi_rsp_timeout,
    .ifq_valid, .ifq_ready, .ifq_entry,
    .tap_enq, .tap_enq_unit, .tap_enq_seq, .tap_enq_pidx, .tap_enq_code, .tap_enq_kind, .tap_enq_w,
    .tap_enq_rec, .tap_enq_pc, .tap_enq_ext, .tap_start, .tap_start_pidx, .tap_done, .tap_done_code,
    .tap_done_ran, .tap_done_pidx, .tap_mark, .tap_mark_value, .tap_mark_pidx,
    .head_valid, .head_pidx, .q_pidx, .q_state, .hold_src, .seq_d, .err_rec, .dbg_hold, .dbg_step,
    .sr_busy, .rv_run, .rv_running, .rv_halted, .rv_sleep, .rv_ndmreset, .rv_dmactive, .rv_halt_pend, .rv_clk_ok, .unit_hung,
    .mtime,
    .dbg_stop, .dbg_halt_tgl, .dbg_cancel_tgl, .dbg_halt_on_boot,
    .rec_wr, .rec_wdata, .rec_room, .rec_all, .smp_ready, .smp_req, .smp_local, .smp_done, .smp_timeout, .smp_data,
    .wpa_ready, .wpa_req, .wpa_local, .wpa_data, .wpa_done, .wpa_timeout, .wph_v, .wph_mask, .wph_addr, .wph_halt,
    .dfault, .flags,
    .mt_rd_req('0), .mt_rd_last('0), .mt_wr_req('0), .mt_wr_rsp('0), .mt_rd_req_b('0), .mt_rd_last_b('0),
    .trw_wr_req_valid(tw_req_valid), .trw_wr_req_ready(tw_req_ready), .trw_wr_req_addr(tw_req_addr),
    .trw_wr_req_len(tw_req_len), .trw_wr_dat_valid(tw_dat_valid), .trw_wr_dat_ready(tw_dat_ready),
    .trw_wr_dat_data(tw_dat_data), .trw_wr_dat_last(tw_dat_last), .trw_wr_rsp_valid(tw_rsp_valid)
  );

  // The trace writer to the write mux: direct, or as tsr_core_dp wires it with XSLR_REG (g_tslice): a tsr_rslice each for
  // the request and the data, two flops for the burst response.
  if (XSLR_REG) begin : g_tslice
    tsr_rslice #(.W(ADDR_BEAT_W + TSR_LEN_W), .FORM(1)) u_tq (
      .clk, .rst_n,
      .s_valid(tw_req_valid), .s_ready(tw_req_ready), .s_data({tw_req_addr, tw_req_len}),
      .m_valid(t_wr_req_valid), .m_ready(t_wr_req_ready), .m_data({t_wr_req_addr, t_wr_req_len})
    );
    tsr_rslice #(.W(512 + 1), .FORM(1)) u_td (
      .clk, .rst_n,
      .s_valid(tw_dat_valid), .s_ready(tw_dat_ready), .s_data({tw_dat_data, tw_dat_last}),
      .m_valid(t_wr_dat_valid), .m_ready(t_wr_dat_ready), .m_data({t_wr_dat_data, t_wr_dat_last})
    );
    logic [1:0] rsp_q;
    always_ff @(posedge clk) begin
      if (!rst_n) rsp_q <= '0;
      else        rsp_q <= {rsp_q[0], t_wr_rsp_valid};
    end
    assign tw_rsp_valid = rsp_q[1];
  end else begin : g_tdirect
    assign t_wr_req_valid = tw_req_valid;
    assign tw_req_ready   = t_wr_req_ready;
    assign t_wr_req_addr  = tw_req_addr;
    assign t_wr_req_len   = tw_req_len;
    assign t_wr_dat_valid = tw_dat_valid;
    assign tw_dat_ready   = t_wr_dat_ready;
    assign t_wr_dat_data  = tw_dat_data;
    assign t_wr_dat_last  = tw_dat_last;
    assign tw_rsp_valid   = t_wr_rsp_valid;
  end

  tsr_dbg_rv #(.REC_DEPTH(64), .HAS_RV_TAPS(HAS_RV_TAPS)) u_dbg_rv (
    .clk, .rst_n,
    .rec_wr, .rec_wdata, .rec_room, .rec_all, .smp_ready, .smp_req, .smp_local, .smp_done, .smp_timeout, .smp_data,
    .wpa_ready, .wpa_req, .wpa_local, .wpa_data, .wpa_done, .wpa_timeout, .wph_v, .wph_mask, .wph_addr, .wph_halt,
    .dfault, .flags, .rv_run,
    .clk_rv, .db_req, .db_we, .db_off, .db_wdata, .db_rdata, .db_irq4,
    .ib_data_req, .ib_data_gnt, .ib_data_addr, .ib_data_we, .ib_data_be,
    .ib_cd_pc(ib_cd[0]), .ib_cd_npc(ib_cd[1]), .ib_cd_daddr(ib_cd[2]), .ib_cd_epc(ib_cd[3]), .ib_cd_eaddr(ib_cd[4]),
    .ib_sleep, .ib_halted, .ib_in_reset, .ib_dfault
  );

  tsr_dbg_wrmux #(.ADDR_BEAT_W(ADDR_BEAT_W)) u_wrmux (
    .clk, .rst_n,
    .d_wr_req_valid, .d_wr_req_ready, .d_wr_req_addr, .d_wr_req_len, .d_wr_req_id('0),
    .d_wr_dat_valid, .d_wr_dat_ready, .d_wr_dat_data, .d_wr_dat_mask, .d_wr_dat_last,
    .d_wr_rsp_valid, .d_wr_rsp_ready(1'b1), .d_wr_rsp_id,
    .t_wr_req_valid, .t_wr_req_ready, .t_wr_req_addr, .t_wr_req_len,
    .t_wr_dat_valid, .t_wr_dat_ready, .t_wr_dat_data, .t_wr_dat_last, .t_wr_rsp_valid,
    .s_wr_req_valid, .s_wr_req_ready, .s_wr_req_addr, .s_wr_req_len, .s_wr_req_id,
    .s_wr_dat_valid, .s_wr_dat_ready, .s_wr_dat_data, .s_wr_dat_mask, .s_wr_dat_last,
    .s_wr_rsp_valid, .s_wr_rsp_ready, .s_wr_rsp_id
  );

  // monitored by the bench (cocotb reads them by name)
  wire unused_ok = &{1'b0, q_nonempty, abort_active, head_w0, seq_s, tap_enq_info, tap_enq_dep0, tap_enq_dep1,
                     tap_mark_pc, d_wr_rsp_id, qquiescent, dbg_hold, dbg_step, csr_unit_hung};

endmodule
