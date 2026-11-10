// tsr_dbg.sv — the DB block, core-clock part (docs/design/05-debug.md "DB" §3.2-§3.3, §3.7, §3.9,
// §5, §6.2, §8.1-§8.4, §10.1; EX §5.1, §6): the DBG register block at core CSR 0x800-0xAFC (SREG
// excluded), the DMI gate, DB's halt requests into tsr_ctrl, the stop logic and tile breakpoints
// (tsr_dbg_stop), the shadow queues, unit probes and record serializer (tsr_dbg_shq), the dispatch
// history (tsr_dbg_hist), the trace writer (tsr_dbg_trw) and the core side of the clk_rv parts
// (tsr_dbg_rv: completion records, RV_SAMPLE, RV watchpoints, DBG_FLAGS).
//
// tsr_core_dp instantiates it next to tsr_issue and tsr_csr (wiring: tsr_core_dp's DB
// section); tb/dbg (tb/dbg/sv/tsr_dbg_tb.sv) wires it the same
// way. Oracle: the simulator's DBG model sw/sim/tsr_dbg.c (DB §9.1 "queue model, §5 semantics
// exactly"); tb/dbg compares register by register, stop by stop, record by record.
//
// Ports
//   CSR block (tsr_csr): dbg_req (one cycle) / dbg_we / dbg_addr / dbg_wdata / dbg_host for every
//     access to 0x800-0xAFC except SREG; dbg_rdata must be valid in the cycle after dbg_req
//     (tsr_csr A_DBG). dbg_addr is stable from the cycle before dbg_req (tsr_csr's A_DEC), so the
//     read mux runs continuously on it in two registered stages (DB §11 "pipelined"). Reads have
//     no side effects; every access completes in that fixed time (DB §7.1). sr_db_clear (SOFT_RESET
//     step 1) clears S, the match queue and a QSTEP; db_stop_active = STOP_ACTIVE (PERF
//     DBG_STOP_CYCLES, tsr_csr). dmi_en = DBG_CFG.DMI_EN gates tsr_csr's DMI window (DB-2).
//   Host CSR port snoop (csr_*: the transfer completes at csr_valid ∧ csr_ready): a DMI-window
//     write while DMI_EN = 0 sets DMI_BLOCKED (tsr_csr drops it). dmi_rsp_*: a response code other
//     than 0 sets DMI_RESP_ERR. DBG_RV_STATE.DMACTIVE is the debug module's own dmcontrol.dmactive
//     (rv_dmactive: riscv-dbg's dmactive_o through tsr_ctrl_cdc), not a shadow of host writes, so a
//     write that never reached the DM (DMI_EN = 0, a DMI timeout) or a DM reset shows as it is.
//   Issue FIFO snoop (ifq_valid / ifq_ready / ifq_entry): the deps of an entry as pushed (the
//     history records them, DB §5.6; the enqueue tap carries them as dispatched) and the
//     dispatcher state for DBG_QSTATE (a mirror: POP for the pop cycle, CHK until the enqueue for
//     an op or a QCLEAR).
//   tsr_issue: the enqueue tap, start / completion strobes, TRACE_MARK, head info, Q_PIDX, q_state,
//     hold_src, seq_d, err_rec; dbg_hold / dbg_step (registered levels; dbg_hold = 0 from the
//     third cycle of sr_busy, dbg_step = 0 while a host or RISC-V hold source is set).
//   tsr_csr / tsr_ctrl levels: sr_busy, rv_run, rv_running, rv_halted, rv_sleep, rv_ndmreset,
//     rv_halt_pend, rv_clk_ok, unit_hung (STATUS.UNIT_HUNG), mtime.
//   tsr_ctrl (DB §3.3, §3.9): dbg_stop (a stop or the debug-halt hold in effect: suspends
//     ISSUE_WAIT_MAX), dbg_halt_tgl / dbg_cancel_tgl (toggles: one halt request per change;
//     RV_HALT_CANCEL), dbg_halt_on_boot (HALT_ON_BOOT ∧ RV_HALT_EN).
//   tsr_dbg_rv (core-domain side): rec_* completion records into the REC FIFO, smp_* the RV_SAMPLE
//     handshake, wpa_* the RVWP_APPLY handshake (both tsr_cdc_hs A sides, bounded by 4,096
//     cycles), wph_* RV-watchpoint hit events (one per watched access, a mask of the watchpoints
//     hit), dfault, flags (DBG_FLAGS, DB §10.2).
//   Memory taps (DBG_RDOUT / DBG_WROUT, DB §5.5): per channel the memory-port handshakes.
//   Trace writer write source: to tsr_dbg_wrmux on channel TRW_CH.
//   perf_ext: PERF increments of DB's slots (61 TRW_RECORDS, 62 TRW_DROPS, X + 10 REC_DROPPED);
//     tsr_core_dp adds them to the other areas' perf_ext (slot 63 is tsr_csr's, from
//     db_stop_active).
// Enqueue pipeline: tap (cycle t: shadow queue write) -> e1 (t + 1: registered tap, history main
//   write, matcher) -> m2 (t + 2: match result, HITS, BEFORE push) -> m3 (t + 3: AFTER push,
//   history INFO side write). See tsr_dbg_stop for why holds still apply in time.
// Register map (DB §10.1; reset values in brackets; nothing here is reset by QABORT, QCLEAR, the
//   RISC-V reset or SOFT_RESET except the stop state of sr_db_clear): see the table in DB §10.1.
//   Decisions where the simulator is followed: DBG_STOP_PIDX shows the T of the last activation
//   while no stop is active; DBG_STOP_AT reads S even when not valid; unlisted offsets read 0.
//   Model values (DB §9.1, not compared): DBG_Uu_*, DBG_QSTATE, DBG_RDOUT / WROUT, the RV sample.
// Not implemented here: SBA_REFUSED (needs a refusal strobe from
//   tsr_rv_bus), and DBG_CTRL writes do not wait for RV_SAMPLE / RVWP_APPLY to complete (tsr_csr's
//   DBG access has a fixed latency; a host reads after its own round trip, and the handshakes end
//   within 4,096 cycles).

module tsr_dbg
  import tsr_pkg::*;
#(
  parameter int unsigned     NUM_CH      = TSR_DEF_NUM_CH,
  parameter longint unsigned CH_BYTES    = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter int unsigned     QD_DMA      = tsr_tables_pkg::TSR_QDEPTH_U200_DMA,
  parameter int unsigned     QD_QACT     = tsr_tables_pkg::TSR_QDEPTH_U200_QACT,
  parameter int unsigned     QD_MVM      = tsr_tables_pkg::TSR_QDEPTH_U200_MVM,
  parameter int unsigned     QD_VU       = tsr_tables_pkg::TSR_QDEPTH_U200_VU,
  parameter int unsigned     QD_ATT      = tsr_tables_pkg::TSR_QDEPTH_U200_ATT,
  parameter int unsigned     QD_SYNC     = tsr_tables_pkg::TSR_QDEPTH_U200_SYNC,
  // DB §10.3
  parameter int unsigned     HIST_DEPTH  = tsr_tables_pkg::TSR_HIST_DEPTH,
  parameter bit              HAS_TRW     = 1'b1,
  parameter int unsigned     NTBP        = tsr_tables_pkg::TSR_DBG_NTBP,
  parameter int unsigned     NRVWP       = tsr_tables_pkg::TSR_DBG_NRVWP,
  parameter bit              HAS_REC     = 1'b1,
  parameter bit              FAST_CLK    = 1'b0    // the 250 MHz stages of the DB block (tsr_dbg_trw)
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // ---- tsr_csr: DBG block, DMI gate, SOFT_RESET, PERF ----
  input  logic                          dbg_req,
  input  logic                          dbg_we,
  input  logic [11:0]                   dbg_addr,
  input  logic [31:0]                   dbg_wdata,
  input  logic                          dbg_host,
  output logic [31:0]                   dbg_rdata,
  input  logic                          sr_db_clear,
  output logic                          db_stop_active,
  output logic                          dmi_en,
  output logic [63:0][3:0]              perf_ext,

  // ---- host CSR port and DMI response snoop ----
  input  logic                          csr_valid,
  input  logic                          csr_ready,
  input  logic                          csr_we,
  input  logic [TSR_CSR_AW-1:0]         csr_addr,
  input  logic [31:0]                   csr_wdata,
  input  logic                          dmi_rsp_valid,
  input  logic [1:0]                    dmi_rsp_resp,
  input  logic                          dmi_rsp_timeout,

  // ---- issue FIFO snoop ----
  input  logic                          ifq_valid,
  input  logic                          ifq_ready,
  input  tsr_ifent_t                    ifq_entry,

  // ---- tsr_issue ----
  input  logic                          tap_enq,
  input  logic [3:0]                    tap_enq_unit,
  input  logic [TSR_SEQ_W-1:0]          tap_enq_seq,
  input  logic [31:0]                   tap_enq_pidx,
  input  logic [3:0]                    tap_enq_code,
  input  logic [1:0]                    tap_enq_kind,
  input  tsr_insn_t                     tap_enq_w,
  input  logic                          tap_enq_rec,
  input  logic [31:0]                   tap_enq_pc,
  input  tsr_ext_t [TSR_EXT_N-1:0]      tap_enq_ext,
  input  logic [5:0]                    tap_start,
  input  logic [5:0][31:0]              tap_start_pidx,
  input  logic [5:0]                    tap_done,
  input  logic [5:0][3:0]               tap_done_code,
  input  logic [5:0]                    tap_done_ran,
  input  logic [5:0][31:0]              tap_done_pidx,
  input  logic                          tap_mark,
  input  logic [31:0]                   tap_mark_value,
  input  logic [31:0]                   tap_mark_pidx,
  input  logic [5:0]                    head_valid,
  input  logic [5:0][31:0]              head_pidx,
  input  logic [31:0]                   q_pidx,
  input  logic [5:0][8:0]               q_state,
  input  logic [2:0]                    hold_src,          // {debug halt, RISC-V, host}
  input  logic [5:0][TSR_SEQ_W-1:0]     seq_d,
  input  tsr_err_rec_t [5:0]            err_rec,
  output logic [5:0]                    dbg_hold,
  output logic [5:0]                    dbg_step,

  // ---- tsr_csr / tsr_ctrl levels ----
  input  logic                          sr_busy,
  input  logic                          rv_run,
  input  logic                          rv_running,
  input  logic                          rv_halted,
  input  logic                          rv_sleep,
  input  logic                          rv_ndmreset,
  input  logic                          rv_dmactive,    // the debug module's dmcontrol.dmactive (synchronized)
  input  logic                          rv_halt_pend,
  input  logic                          rv_clk_ok,
  input  logic                          unit_hung,
  input  logic [63:0]                   mtime,

  // ---- tsr_ctrl: DB's halt requests (DB §3.3) ----
  output logic                          dbg_stop,
  output logic                          dbg_halt_tgl,
  output logic                          dbg_cancel_tgl,
  output logic                          dbg_halt_on_boot,

  // ---- tsr_dbg_rv (core-domain side) ----
  output logic                          rec_wr,
  output logic [127:0]                  rec_wdata,
  input  logic                          rec_room,          // the REC FIFO has a free entry
  input  logic                          rec_all,           // REC_CFG.ALL
  input  logic                          smp_ready,
  output logic                          smp_req,
  output logic                          smp_local,
  input  logic                          smp_done,
  input  logic                          smp_timeout,
  input  logic [163:0]                  smp_data,          // {state[3:0], eaddr, epc, daddr, npc, pc}
  input  logic                          wpa_ready,
  output logic                          wpa_req,
  output logic                          wpa_local,
  output logic [135:0]                  wpa_data,          // {wp1 {ctl[3:0], hi, lo}, wp0}
  input  logic                          wpa_done,
  input  logic                          wpa_timeout,
  input  logic                          wph_v,
  input  logic [1:0]                    wph_mask,
  input  logic [31:0]                   wph_addr,
  input  logic                          wph_halt,
  input  logic                          dfault,
  output logic [4:0]                    flags,             // DBG_FLAGS (DB §10.2)

  // ---- memory-port taps (DBG_RDOUT / DBG_WROUT) ----
  input  logic [NUM_CH-1:0]             mt_rd_req,         // read request handshake
  input  logic [NUM_CH-1:0]             mt_rd_last,        // read response last-beat handshake
  input  logic [NUM_CH-1:0]             mt_rd_req_b,       // the same on the read-only port B (MVM_DUAL; 0 otherwise)
  input  logic [NUM_CH-1:0]             mt_rd_last_b,
  input  logic [NUM_CH-1:0]             mt_wr_req,         // write request handshake
  input  logic [NUM_CH-1:0]             mt_wr_rsp,         // write response handshake

  // ---- trace writer write source (tsr_dbg_wrmux on channel TRW_CH) ----
  output logic                          trw_wr_req_valid,
  input  logic                          trw_wr_req_ready,
  output logic [ADDR_BEAT_W-1:0]        trw_wr_req_addr,
  output logic [TSR_LEN_W-1:0]          trw_wr_req_len,
  output logic                          trw_wr_dat_valid,
  input  logic                          trw_wr_dat_ready,
  output logic [511:0]                  trw_wr_dat_data,
  output logic                          trw_wr_dat_last,
  input  logic                          trw_wr_rsp_valid
);

  if (NRVWP > 2 || NTBP > 4 || NUM_CH < 1 || NUM_CH > 16) begin : g_chk
    $fatal(1, "tsr_dbg: NRVWP in [0, 2], NTBP in [0, 4], NUM_CH in [1, 16] (DB §10.3)");
  end

  import tsr_tables_pkg::*;

  localparam int unsigned X      = 5 + 2 * NUM_CH;
  localparam int unsigned LOG2_H = (HIST_DEPTH > 1) ? $clog2(HIST_DEPTH) : 0;
  localparam logic [31:0] DBG_ID_VAL =
      {8'd0, 4'(LOG2_H), 3'd0, 2'(NRVWP), 3'(NTBP), 1'(HAS_REC), 1'(NRVWP != 0), 1'(HAS_TRW),
       1'(HIST_DEPTH != 0), 8'(TSR_DBG_VERSION)};
  localparam logic [31:0] STICKY_MASK = TSR_DBG_STATUS_RVWP_HIT | TSR_DBG_STATUS_TRW_DROP |
      TSR_DBG_STATUS_REC_DROP | TSR_DBG_STATUS_SBA_REFUSED | TSR_DBG_STATUS_DMI_BLOCKED |
      TSR_DBG_STATUS_RV_SAMPLE_TIMEOUT | TSR_DBG_STATUS_RV_DFAULT | TSR_DBG_STATUS_PTR_ERR |
      TSR_DBG_STATUS_RVWP_TIMEOUT | TSR_DBG_STATUS_MATCHQ_OVF | TSR_DBG_STATUS_DMI_RESP_ERR;
  localparam logic [31:0] TBP_CTL_MASK = 32'hFFFF_01FF;

  // ---- register state ----
  logic [11:0]      cfg_q;
  logic [31:0]      sticky_q;
  logic [15:0]      step_n_q;
  logic [31:0]      hist_idx_q;
  logic [3:0][31:0] tbp_ctl_q, tbp_a_q, tbp_b_q, tbp_hits_q;
  logic [1:0][3:0]  wp_ctl_q;
  logic [1:0][31:0] wp_lo_q, wp_hi_q, wp_hits_q;
  logic [31:0]      wp_addr_q;
  logic [31:0]      trw_base_q, trw_lines_q, trw_rptr_q;
  logic [10:0]      trw_mask_q;
  logic [4:0][31:0] smp_q;
  logic             smp_valid_q;
  logic [7:0]       smp_seq_q;
  logic [15:0]      dbg_entries_q;
  logic             halted_prev_q;
  logic             halt_tgl_q, cancel_tgl_q;

  wire c_dmi_en     = cfg_q[0];
  wire c_serial     = cfg_q[1];
  wire c_err_stop   = cfg_q[2];
  wire c_stop_halt  = cfg_q[3];
  wire c_hist_en    = cfg_q[4];
  wire c_hist_ferr  = cfg_q[5];
  wire c_hist_frz   = cfg_q[6];
  wire c_hung_stop  = cfg_q[7];
  wire c_hob        = cfg_q[8];
  wire c_halt_en    = cfg_q[9];
  wire c_trw_en     = cfg_q[10] && HAS_TRW;
  wire c_trw_full   = cfg_q[11];

  // ---- register writes (one per cycle) ----
  wire        wr  = dbg_req && dbg_we && dbg_host;
  wire [11:0] A   = {dbg_addr[11:2], 2'b00};
  wire        wal = (dbg_addr[1:0] == 2'b00);
  wire        w_ctrl = wr && wal && A == TSR_CSR_DBG_CTRL;
  wire [31:0] wd  = dbg_wdata;
  wire        ev_qstep    = w_ctrl && wd[2];
  wire        ev_release  = w_ctrl && !wd[2] && wd[1];
  wire        ev_stop_now = w_ctrl && !wd[2] && !wd[1] && wd[0];
  wire        ev_hclear   = w_ctrl && wd[4];
  wire        ev_sample   = w_ctrl && wd[5];
  wire        ev_apply    = w_ctrl && wd[6];
  wire        ev_flush    = w_ctrl && wd[7];
  wire        ev_cancel   = w_ctrl && wd[8];
  wire        ev_mqclr    = w_ctrl && wd[9];
  wire        ev_stop_at  = wr && wal && A == TSR_CSR_DBG_STOP_AT;

  // =============================================================================================
  // Enqueue pipeline stage e1, pushed deps
  // =============================================================================================
  logic [31:0] pdep0_q, pdep1_q;
  always_ff @(posedge clk) begin
    if (ifq_valid && ifq_ready) begin
      pdep0_q <= ifq_entry.dep0;
      pdep1_q <= ifq_entry.dep1;
    end
  end
  wire tap_qclr = (tap_enq_kind == TSR_QK_QCLEAR);

  logic                     e1_v_q, e1_rec_q;
  logic [2:0]               e1_unit_q;
  logic [TSR_SEQ_W-1:0]     e1_seq_q;
  logic [31:0]              e1_pidx_q, e1_dep0_q, e1_dep1_q, e1_pc_q, e1_time_q;
  logic [3:0]               e1_code_q;
  tsr_insn_t                e1_w_q;
  tsr_ext_t [TSR_EXT_N-1:0] e1_ext_q;
  always_ff @(posedge clk) begin
    if (!rst_n) e1_v_q <= 1'b0;
    else        e1_v_q <= tap_enq;
  end
  always_ff @(posedge clk) begin
    if (tap_enq) begin
      e1_unit_q <= tap_enq_unit[2:0];
      e1_seq_q  <= tap_enq_seq;
      e1_pidx_q <= tap_enq_pidx;
      e1_code_q <= tap_enq_code;
      e1_w_q    <= tap_enq_w;
      e1_ext_q  <= tap_enq_ext;
      e1_rec_q  <= tap_enq_rec;
      e1_pc_q   <= tap_enq_pc;
      e1_time_q <= mtime[31:0];
      e1_dep0_q <= tap_qclr ? TSR_TICKET_NONE : pdep0_q;
      e1_dep1_q <= tap_qclr ? TSR_TICKET_NONE : pdep1_q;
    end
  end

  // =============================================================================================
  // Shadow queues, stop logic, history, trace writer
  // =============================================================================================
  logic [5:0]                out_v;
  logic [5:0][31:0]          out_pidx;
  logic [5:0][TSR_SEQ_W-1:0] out_seq;
  logic [5:0][7:0]           out_n;
  logic [5:0][1:0]           out_acc;
  logic [5:0][63:0]          out_t;
  logic [5:0]                st_hold, st_step;
  logic                      st_active, st_sv, st_drained, st_busy, st_mqovf;
  logic [31:0]               st_t, st_s, st_x, st_last;
  logic [4:0]                st_cause;
  logic [3:0]                st_bp;
  logic [6:0]                st_mqn;
  logic                      m2_v, h3_v, h3_stop, act, act_tbp_halt;
  logic [2:0]                m2_unit;
  logic [3:0]                m2_hit, h3_hit;
  logic [31:0]               h3_pidx;
  logic                      srel_v, sact_v;
  logic [31:0]               srel_t, sact_t;
  logic [4:0]                srel_cause, sact_cause;

  tsr_dbg_stop #(.NUM_CH(NUM_CH), .NTBP(NTBP)) u_stop (
    .clk, .rst_n,
    .cfg_serial(c_serial), .cfg_err_stop(c_err_stop), .cfg_hung_stop(c_hung_stop),
    .tbp_ctl(tbp_ctl_q), .tbp_a(tbp_a_q), .tbp_b(tbp_b_q), .step_n(step_n_q),
    .ev_stop_now, .ev_release, .ev_qstep, .ev_mq_clear(ev_mqclr), .ev_stop_at, .ev_stop_at_val(wd),
    .sr_clear(sr_db_clear),
    .next_pidx(q_pidx), .head_valid, .head_pidx, .halt_hold(hold_src[2]), .unit_hung,
    .out_v, .out_pidx,
    .e1_v(e1_v_q), .e1_unit(e1_unit_q), .e1_seq(e1_seq_q), .e1_pidx(e1_pidx_q), .e1_code(e1_code_q),
    .e1_w0(e1_w_q[0]), .e1_ext(e1_ext_q),
    .tap_start, .tap_start_pidx, .tap_done, .tap_done_code, .tap_done_pidx, .seq_d, .err_rec,
    .hold(st_hold), .step(st_step),
    .stop_active(st_active), .s_valid(st_sv), .drained(st_drained), .step_busy(st_busy),
    .stop_t(st_t), .s_val(st_s), .step_x(st_x), .last_t(st_last), .cause(st_cause), .bp_hit(st_bp),
    .mq_n(st_mqn), .mq_ovf(st_mqovf),
    .m2_v, .m2_unit, .m2_hit, .h3_v, .h3_pidx, .h3_hit, .h3_stop,
    .act, .act_tbp_halt, .srel_v, .srel_t, .srel_cause, .sact_v, .sact_t, .sact_cause
  );
  wire unused_m2u = &{1'b0, m2_unit, h3_pidx};

  logic         trw_v, trw_flush, trw_ready, trw_lost, rec_drop;
  logic [255:0] trw_rec;

  tsr_dbg_shq #(
    .QD_DMA(QD_DMA), .QD_QACT(QD_QACT), .QD_MVM(QD_MVM), .QD_VU(QD_VU), .QD_ATT(QD_ATT), .QD_SYNC(QD_SYNC)
  ) u_shq (
    .clk, .rst_n, .mtime,
    .tap_enq, .tap_enq_unit, .tap_enq_seq, .tap_enq_pidx, .tap_enq_w0lo(tap_enq_w[0][15:0]), .tap_enq_rec,
    .tap_start, .tap_done, .tap_done_code, .tap_done_ran, .tap_mark, .tap_mark_value, .tap_mark_pidx,
    .m2_v, .m2_hit,
    .srel_v, .srel_t, .srel_cause, .sact_v, .sact_t, .sact_cause,
    .rv_halted_lvl(rv_run && rv_halted && rv_running),
    .trw_en(c_trw_en), .trw_mask(trw_mask_q), .rec_en(HAS_REC), .rec_all,
    .out_v, .out_pidx, .out_seq, .out_n, .out_acc, .out_t,
    .trw_v, .trw_rec, .trw_flush, .trw_ready, .trw_lost,
    .rec_v(rec_wr), .rec_data(rec_wdata), .rec_room, .rec_drop
  );

  logic [31:0]  h_head, h_info, h_pidx, h_ticket, h_dep0, h_dep1, h_time, h_pc;
  logic         h_frozen;
  tsr_insn_t    h_w;
  tsr_dbg_hist #(.DEPTH(HIST_DEPTH)) u_hist (
    .clk, .rst_n,
    .cfg_en(c_hist_en), .cfg_freeze(c_hist_frz), .cfg_freeze_err(c_hist_ferr), .clear(ev_hclear),
    .unit_hung, .next_pidx(q_pidx),
    .e1_v(e1_v_q), .e1_pidx(e1_pidx_q), .e1_unit(e1_unit_q), .e1_seq(e1_seq_q), .e1_code(e1_code_q),
    .e1_rec(e1_rec_q), .e1_dep0(e1_dep0_q), .e1_dep1(e1_dep1_q), .e1_w(e1_w_q), .e1_pc(e1_pc_q),
    .e1_time(e1_time_q),
    .h3_v, .h3_hit, .h3_stop,
    .tap_done, .tap_done_code, .tap_done_pidx,
    .idx(hist_idx_q), .head(h_head), .frozen(h_frozen),
    .rd_info(h_info), .rd_pidx(h_pidx), .rd_ticket(h_ticket), .rd_dep0(h_dep0), .rd_dep1(h_dep1),
    .rd_time(h_time), .rd_pc(h_pc), .rd_w(h_w)
  );

  logic [31:0] trw_wptr, trw_drops;
  logic        trw_drop_ev;
  logic [1:0]  perf_trec, perf_tdrop;
  if (HAS_TRW) begin : g_trw
    tsr_dbg_trw #(.ADDR_BEAT_W(ADDR_BEAT_W), .CH_BYTES(CH_BYTES), .TRW_CH(TSR_TRW_CH), .FAST_CLK(FAST_CLK)) u_trw (
      .clk, .rst_n, .mtime,
      .cfg_en(c_trw_en), .cfg_stop_full(c_trw_full), .base(trw_base_q), .lines(trw_lines_q),
      .rptr(trw_rptr_q), .flush(ev_flush),
      .in_v(trw_v), .in_rec(trw_rec), .in_flush(trw_flush), .in_ready(trw_ready), .in_lost(trw_lost),
      .wptr(trw_wptr), .drops(trw_drops), .drop_ev(trw_drop_ev), .perf_rec(perf_trec), .perf_drop(perf_tdrop),
      .wr_req_valid(trw_wr_req_valid), .wr_req_ready(trw_wr_req_ready), .wr_req_addr(trw_wr_req_addr),
      .wr_req_len(trw_wr_req_len), .wr_dat_valid(trw_wr_dat_valid), .wr_dat_ready(trw_wr_dat_ready),
      .wr_dat_data(trw_wr_dat_data), .wr_dat_last(trw_wr_dat_last), .wr_rsp_valid(trw_wr_rsp_valid)
    );
  end else begin : g_notrw
    assign trw_ready = 1'b1;
    assign trw_wptr = '0;
    assign trw_drops = '0;
    assign trw_drop_ev = 1'b0;
    assign perf_trec = '0;
    assign perf_tdrop = '0;
    assign trw_wr_req_valid = 1'b0;
    assign trw_wr_req_addr = '0;
    assign trw_wr_req_len = '0;
    assign trw_wr_dat_valid = 1'b0;
    assign trw_wr_dat_data = '0;
    assign trw_wr_dat_last = 1'b0;
    wire unused_trw = &{1'b0, trw_v, trw_rec, trw_flush, trw_lost, trw_wr_req_ready, trw_wr_dat_ready,
                        trw_wr_rsp_valid, ev_flush, trw_base_q, trw_lines_q, trw_rptr_q, c_trw_full};
  end

  // =============================================================================================
  // Holds, halt requests, flags
  // =============================================================================================
  // dbg_hold is forced to 0 while SR_BUSY (DB §5.2), from SR_BUSY's third cycle: SOFT_RESET step 1
  // raises SR_BUSY and its QABORT pulse together and tsr_issue applies a QABORT one cycle after it
  // sees it (in a cycle without decisions), so a held head is released only after the abort put
  // its unit in error state (it then completes ABORTED, as in the simulator, instead of starting)
  logic [1:0] srb_q;
  always_ff @(posedge clk) begin
    if (!rst_n) srb_q <= '0;
    else        srb_q <= {srb_q[0], sr_busy};
  end
  assign dbg_hold = (sr_busy && srb_q == 2'b11) ? 6'd0 : st_hold;
  assign dbg_step = (hold_src[0] || hold_src[1]) ? 6'd0 : st_step;
  assign db_stop_active = st_active;
  assign dbg_stop = st_active || hold_src[2];
  assign dbg_halt_on_boot = c_hob && c_halt_en;
  assign dbg_halt_tgl = halt_tgl_q;
  assign dbg_cancel_tgl = cancel_tgl_q;
  assign dmi_en = c_dmi_en;
  assign flags = {c_hob, c_dmi_en, c_trw_en, c_serial, st_active || hold_src[2]};

  // DB §3.3: a request while RV_HALT_EN and the hart is not halted; tsr_ctrl_rv's pend makes it
  // one halt (cleared by RV_HALTED, RV_HALT_CANCEL and the reset release)
  wire halt_req = c_halt_en && !(rv_run && rv_halted) &&
                  ((act && (act_tbp_halt || c_stop_halt)) || (wph_v && wph_halt));

  // =============================================================================================
  // Register state
  // =============================================================================================
  wire smp_ok  = smp_done && !smp_timeout;
  wire tbp_wr  = wr && wal && A >= TSR_CSR_DBG_TBP_CTL_BASE && A < 12'h8C0;
  wire wp_wr   = wr && wal && A >= TSR_CSR_DBG_RVWP_CTL_BASE && A < 12'h880;
  wire [1:0] tbp_i = A[5:4];
  wire       wp_j  = A[4];
  wire [1:0] tbp_f = A[3:2];
  wire [1:0] wp_f  = A[3:2];
  wire dmi_wr_host = csr_valid && csr_ready && csr_we && csr_addr >= 12'hC00 && csr_addr < 12'hE00 &&
                     csr_addr[1:0] == 2'b00;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cfg_q         <= 12'(TSR_CSR_DBG_CFG_RESET);
      sticky_q      <= '0;
      step_n_q      <= 16'(TSR_CSR_DBG_STEP_N_RESET);
      hist_idx_q    <= '0;
      tbp_ctl_q     <= '0;
      tbp_a_q       <= '0;
      tbp_b_q       <= '0;
      tbp_hits_q    <= '0;
      wp_ctl_q      <= '0;
      wp_lo_q       <= '0;
      wp_hi_q       <= '0;
      wp_hits_q     <= '0;
      wp_addr_q     <= '0;
      trw_base_q    <= '0;
      trw_lines_q   <= '0;
      trw_rptr_q    <= '0;
      trw_mask_q    <= 11'(TSR_CSR_DBG_TRW_MASK_RESET);
      smp_q         <= '0;
      smp_valid_q   <= 1'b0;
      smp_seq_q     <= '0;
      dbg_entries_q <= '0;
      halted_prev_q <= 1'b0;
      halt_tgl_q    <= 1'b0;
      cancel_tgl_q  <= 1'b0;
    end else begin
      // ---- host writes ----
      if (wr && wal) begin
        case (A)
          TSR_CSR_DBG_CFG:       cfg_q <= wd[11:0];
          TSR_CSR_DBG_STATUS:    sticky_q <= sticky_q & ~(wd & STICKY_MASK);
          TSR_CSR_DBG_STEP_N:    step_n_q <= wd[15:0];
          TSR_CSR_DBG_TRW_BASE:  trw_base_q <= {wd[31:4], 4'd0};
          TSR_CSR_DBG_TRW_LINES: if (wd >= 32'd16 && (wd & (wd - 32'd1)) == 32'd0) trw_lines_q <= wd;
          TSR_CSR_DBG_TRW_RPTR:  if ((wd - trw_rptr_q) <= (trw_wptr - trw_rptr_q)) trw_rptr_q <= wd;
          TSR_CSR_DBG_TRW_MASK:  trw_mask_q <= wd[10:0];
          TSR_CSR_DBG_HIST_IDX:  hist_idx_q <= wd;
          default: ;
        endcase
      end
      if (tbp_wr && int'(tbp_i) < int'(NTBP)) begin
        case (tbp_f)
          2'd0: tbp_ctl_q[tbp_i] <= wd & TBP_CTL_MASK;
          2'd1: tbp_a_q[tbp_i]   <= wd;
          2'd2: tbp_b_q[tbp_i]   <= wd;
          default: ;
        endcase
      end
      if (wp_wr && int'(wp_j) < int'(NRVWP)) begin
        case (wp_f)
          2'd0: wp_ctl_q[wp_j] <= wd[3:0];
          2'd1: wp_lo_q[wp_j]  <= wd;
          2'd2: wp_hi_q[wp_j]  <= wd;
          default: ;
        endcase
      end
      // ---- TBP and RV-watchpoint hit counters (saturating; a write clears) ----
      for (int i = 0; i < 4; i++) begin
        if (tbp_wr && tbp_i == 2'(i) && tbp_f == 2'd3) tbp_hits_q[i] <= '0;
        else if (m2_v && m2_hit[i] && tbp_hits_q[i] != 32'hFFFF_FFFF) tbp_hits_q[i] <= tbp_hits_q[i] + 32'd1;
      end
      for (int j = 0; j < 2; j++) begin
        if (wp_wr && wp_j == 1'(j) && wp_f == 2'd3) wp_hits_q[j] <= '0;
        else if (wph_v && wph_mask[j] && wp_hits_q[j] != 32'hFFFF_FFFF) wp_hits_q[j] <= wp_hits_q[j] + 32'd1;
      end
      if (wph_v) wp_addr_q <= wph_addr;
      // ---- sticky bits (events win over a W1C of the same cycle) ----
      if (wph_v && wph_mask[0]) sticky_q[13] <= 1'b1;
      if (wph_v && wph_mask[1]) sticky_q[14] <= 1'b1;
      if (trw_drop_ev) sticky_q[20] <= 1'b1;
      if (rec_drop)    sticky_q[21] <= 1'b1;
      if (dmi_wr_host && !c_dmi_en) sticky_q[23] <= 1'b1;
      if (smp_done && smp_timeout) sticky_q[24] <= 1'b1;
      if (dfault)      sticky_q[25] <= 1'b1;
      if ((wr && wal && A == TSR_CSR_DBG_TRW_LINES && !(wd >= 32'd16 && (wd & (wd - 32'd1)) == 32'd0)) ||
          (wr && wal && A == TSR_CSR_DBG_TRW_RPTR && !((wd - trw_rptr_q) <= (trw_wptr - trw_rptr_q))))
        sticky_q[26] <= 1'b1;
      if (wpa_done && wpa_timeout) sticky_q[27] <= 1'b1;
      if (st_mqovf)    sticky_q[28] <= 1'b1;
      if (dmi_rsp_valid && !dmi_rsp_timeout && dmi_rsp_resp != 2'd0) sticky_q[29] <= 1'b1;
      // ---- RV sample (DB §6.2) ----
      if (smp_done) begin
        smp_valid_q <= !smp_timeout;
        if (smp_ok) begin
          smp_q     <= smp_data[159:0];
          smp_seq_q <= smp_seq_q + 8'd1;
        end
      end
      // ---- RISC-V view ----
      halted_prev_q <= rv_run && rv_halted;
      if (rv_run && rv_halted && !halted_prev_q) dbg_entries_q <= dbg_entries_q + 16'd1;
      if (halt_req)  halt_tgl_q   <= !halt_tgl_q;
      if (ev_cancel) cancel_tgl_q <= !cancel_tgl_q;
    end
  end

  // the clk_rv handshakes (a request while one is in flight is dropped)
  assign smp_req   = ev_sample && smp_ready;
  assign smp_local = !rv_clk_ok;
  assign wpa_req   = ev_apply && wpa_ready && (NRVWP != 0);
  assign wpa_local = !rv_clk_ok;
  assign wpa_data  = {wp_ctl_q[1], wp_hi_q[1], wp_lo_q[1], wp_ctl_q[0], wp_hi_q[0], wp_lo_q[0]};
  wire unused_smp  = &{1'b0, smp_data[163:160]};

  // =============================================================================================
  // Probes
  // =============================================================================================
  // memory taps: outstanding read / write bursts per channel (8 bits each, modulo)
  logic [15:0][7:0] rdout_q, wrout_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rdout_q <= '0;
      wrout_q <= '0;
    end else begin
      for (int c = 0; c < NUM_CH; c++) begin
        rdout_q[c] <= rdout_q[c] + 8'(mt_rd_req[c]) + 8'(mt_rd_req_b[c]) - 8'(mt_rd_last[c]) - 8'(mt_rd_last_b[c]);
        wrout_q[c] <= wrout_q[c] + 8'(mt_wr_req[c]) - 8'(mt_wr_rsp[c]);
      end
    end
  end

  // dispatcher mirror (DBG_QSTATE.DISP): POP in the pop cycle, CHK until the entry's enqueue
  logic [2:0] disp_q;
  logic       pop_q, pop_tk_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      disp_q <= TSR_DBG_QSTATE_DISP_IDLE;
      pop_q  <= 1'b0;
    end else begin
      pop_q <= ifq_valid && ifq_ready;
      if (ifq_valid && ifq_ready) begin
        disp_q   <= TSR_DBG_QSTATE_DISP_POP;
        pop_tk_q <= (ifq_entry.kind == TSR_IFIFO_KIND_OP) ||
                    (ifq_entry.w[0][3:0] == TSR_IFIFO_CTRL_CODE_QCLEAR);
      end else if (pop_q) begin
        disp_q <= pop_tk_q ? TSR_DBG_QSTATE_DISP_CHK : TSR_DBG_QSTATE_DISP_IDLE;
      end
      if (tap_enq) disp_q <= TSR_DBG_QSTATE_DISP_IDLE;
    end
  end

  // per-unit probe words (registered)
  logic [5:0][31:0] u_state_q, u_old_q, u_opidx_q, u_age_q;
  logic [7:0]       outsum_q;
  always_ff @(posedge clk) begin
    logic [9:0] sum;
    sum = '0;
    for (int u = 0; u < 6; u++) begin
      automatic logic [7:0]  qs = q_state[u][7:0];
      automatic logic [2:0]  hs = TSR_DBG_U_STATE_HEAD_EMPTY;
      automatic logic [3:0]  hsrc = '0;
      automatic logic [63:0] age = mtime - out_t[u];
      if (qs[4:0] != 5'd0) begin
        if (qs[7])      hs = TSR_DBG_U_STATE_HEAD_HELD;
        else if (qs[6]) hs = TSR_DBG_U_STATE_HEAD_WAIT_DEP;
        else if (qs[5]) hs = TSR_DBG_U_STATE_HEAD_WAIT_UNIT;
        else            hs = TSR_DBG_U_STATE_HEAD_LOAD;
      end
      if (out_n[u] != 8'(out_acc[u])) hsrc = {dbg_hold[u], hold_src};
      u_state_q[u] <= {12'd0, out_n[u], hsrc, 1'b0, hs, 2'd0, out_acc[u]};
      u_old_q[u]   <= out_v[u] ? {1'b0, 3'(u), out_seq[u]} : TSR_TICKET_NONE;
      u_opidx_q[u] <= out_v[u] ? out_pidx[u] : 32'd0;
      u_age_q[u]   <= !out_v[u] ? 32'd0 : (age[63:32] != 32'd0 ? 32'hFFFF_FFFF : age[31:0]);
      sum = sum + 10'(out_n[u]);
    end
    outsum_q <= (sum > 10'd255) ? 8'd255 : sum[7:0];
  end

  // =============================================================================================
  // Read mux: stage 1 per group (0x8xx, 0x9xx, 0xAxx) on dbg_addr, stage 2 the group
  // =============================================================================================
  wire [31:0] status_v = (sticky_q & STICKY_MASK) |
      (st_active ? (TSR_DBG_STATUS_STOP_ACTIVE | {23'd0, st_cause, 4'd0}) : 32'd0) |
      (st_drained ? TSR_DBG_STATUS_DRAINED : 32'd0) |
      (st_busy ? TSR_DBG_STATUS_STEP_BUSY : 32'd0) |
      ((dbg_hold != 6'd0) ? TSR_DBG_STATUS_DB_HELD : 32'd0) |
      {19'd0, st_bp, 9'd0} |
      (rv_halt_pend ? TSR_DBG_STATUS_RV_HALT_PEND : 32'd0) |
      ((!wpa_ready && NRVWP != 0) ? TSR_DBG_STATUS_RVWP_BUSY : 32'd0) |
      (h_frozen ? TSR_DBG_STATUS_HIST_FROZEN : 32'd0) |
      ((rv_run && rv_halted) ? TSR_DBG_STATUS_RV_HALTED : 32'd0) |
      ((rv_run && rv_sleep) ? TSR_DBG_STATUS_RV_SLEEP : 32'd0) |
      (st_sv ? TSR_DBG_STATUS_S_VALID : 32'd0);
  wire [31:0] rv_state_v = {dbg_entries_q, smp_seq_q, rv_halt_pend, rv_ndmreset, rv_dmactive, smp_valid_q,
                            !rv_run, rv_run && rv_sleep, 1'b0, rv_run && rv_halted};

  wire [11:0] RA = {dbg_addr[11:2], 2'b00};
  logic [31:0] g8_d, g9_d, ga_d;
  always_comb begin
    g8_d = '0;
    case (RA)
      TSR_CSR_DBG_ID:        g8_d = DBG_ID_VAL;
      TSR_CSR_DBG_CFG:       g8_d = {20'd0, cfg_q};
      TSR_CSR_DBG_STATUS:    g8_d = status_v;
      TSR_CSR_DBG_STOP_PIDX: g8_d = st_active ? st_t : st_last;
      TSR_CSR_DBG_STOP_AT:   g8_d = st_s;
      TSR_CSR_DBG_STEP_N:    g8_d = {16'd0, step_n_q};
      TSR_CSR_DBG_NEXT_PIDX: g8_d = q_pidx;
      TSR_CSR_DBG_STEP_PIDX: g8_d = st_x;
      TSR_CSR_DBG_MATCHQ:    g8_d = {25'd0, st_mqn};
      TSR_CSR_DBG_MTIME_LO:  g8_d = mtime[31:0];
      TSR_CSR_DBG_MTIME_HI:  g8_d = mtime[63:32];
      TSR_CSR_DBG_QSTATE:    g8_d = {8'd0, outsum_q, 5'd0, disp_q, 8'd0};
      TSR_CSR_DBG_RV_STATE:  g8_d = rv_state_v;
      TSR_CSR_DBG_RV_PC:     g8_d = smp_q[0];
      TSR_CSR_DBG_RV_NPC:    g8_d = smp_q[1];
      TSR_CSR_DBG_RV_DADDR:  g8_d = smp_q[2];
      TSR_CSR_DBG_RV_EPC:    g8_d = smp_q[3];
      TSR_CSR_DBG_RV_EADDR:  g8_d = smp_q[4];
      TSR_CSR_DBG_RV_WPADDR: g8_d = (NRVWP != 0) ? wp_addr_q : 32'd0;
      default: ;
    endcase
    if (RA >= 12'h860 && RA < 12'h880 && int'(RA[4]) < int'(NRVWP)) begin
      case (RA[3:2])
        2'd0: g8_d = {28'd0, wp_ctl_q[RA[4]]};
        2'd1: g8_d = wp_lo_q[RA[4]];
        2'd2: g8_d = wp_hi_q[RA[4]];
        default: g8_d = wp_hits_q[RA[4]];
      endcase
    end
    if (RA >= 12'h880 && RA < 12'h8C0 && int'(RA[5:4]) < int'(NTBP)) begin
      case (RA[3:2])
        2'd0: g8_d = tbp_ctl_q[RA[5:4]];
        2'd1: g8_d = tbp_a_q[RA[5:4]];
        2'd2: g8_d = tbp_b_q[RA[5:4]];
        default: g8_d = tbp_hits_q[RA[5:4]];
      endcase
    end
    if (RA >= 12'h8C0 && RA < 12'h8E0) begin
      for (int b = 0; b < 4; b++) begin
        // channel {RA[3:2], b}; a concatenation, not integer arithmetic (tsr_mvm_pv_lsel: Vivado 2024.2 drops index bits)
        automatic logic [3:0] c = {RA[3:2], 2'(b)};
        if (int'(c) < int'(NUM_CH)) g8_d[8*b +: 8] = RA[4] ? wrout_q[c] : rdout_q[c];
      end
    end
  end
  always_comb begin
    automatic logic [2:0] u = RA[6:4];
    g9_d = '0;
    if (RA < 12'h980) begin
      if (u < 3'd6) begin
        case (RA[3:2])
          2'd0: g9_d = u_state_q[u];
          2'd1: g9_d = u_old_q[u];
          2'd2: g9_d = u_opidx_q[u];
          default: g9_d = u_age_q[u];
        endcase
      end else if (RA[3:2] == 2'd1) begin
        g9_d = TSR_TICKET_NONE;                // units 6, 7: OLDEST reads "none"
      end
    end else begin
      case (RA)
        TSR_CSR_DBG_TRW_BASE:  g9_d = HAS_TRW ? trw_base_q : 32'd0;
        TSR_CSR_DBG_TRW_LINES: g9_d = HAS_TRW ? trw_lines_q : 32'd0;
        TSR_CSR_DBG_TRW_WPTR:  g9_d = trw_wptr;
        TSR_CSR_DBG_TRW_RPTR:  g9_d = HAS_TRW ? trw_rptr_q : 32'd0;
        TSR_CSR_DBG_TRW_DROPS: g9_d = trw_drops;
        TSR_CSR_DBG_TRW_MASK:  g9_d = {21'd0, trw_mask_q};
        default: ;
      endcase
    end
  end
  always_comb begin
    ga_d = '0;
    case (RA)
      TSR_CSR_DBG_HIST_HEAD:   ga_d = h_head;
      TSR_CSR_DBG_HIST_IDX:    ga_d = hist_idx_q;
      TSR_CSR_DBG_HIST_INFO:   ga_d = h_info;
      TSR_CSR_DBG_HIST_PIDX:   ga_d = h_pidx;
      TSR_CSR_DBG_HIST_TICKET: ga_d = h_ticket;
      TSR_CSR_DBG_HIST_DEP0:   ga_d = h_dep0;
      TSR_CSR_DBG_HIST_DEP1:   ga_d = h_dep1;
      TSR_CSR_DBG_HIST_TIME:   ga_d = h_time;
      TSR_CSR_DBG_HIST_PC:     ga_d = h_pc;
      default: ;
    endcase
    if (RA >= 12'hA40 && RA < 12'hA60) ga_d = h_w[RA[4:2]];
  end

  logic [31:0] g8_q, g9_q, ga_q, rdata_q;
  logic [1:0]  gsel_q;
  always_ff @(posedge clk) begin
    g8_q   <= g8_d;
    g9_q   <= g9_d;
    ga_q   <= ga_d;
    gsel_q <= (dbg_addr[1:0] != 2'b00) ? 2'd3 : (RA[11:8] == 4'h8 ? 2'd0 : (RA[11:8] == 4'h9 ? 2'd1 :
              (RA[11:8] == 4'hA ? 2'd2 : 2'd3)));
    case (gsel_q)
      2'd0:    rdata_q <= g8_q;
      2'd1:    rdata_q <= g9_q;
      2'd2:    rdata_q <= ga_q;
      default: rdata_q <= '0;
    endcase
  end
  assign dbg_rdata = rdata_q;

  // =============================================================================================
  // PERF (EX §6, DB §8.1): TRW_RECORDS, TRW_DROPS, REC_DROPPED
  // =============================================================================================
  always_comb begin
    perf_ext = '0;
    perf_ext[TSR_PERF_TRW_RECORDS] = 4'(perf_trec);
    perf_ext[TSR_PERF_TRW_DROPS]   = 4'(perf_tdrop);
    perf_ext[X + TSR_PERF_XOFF_REC_DROPPED] = 4'(rec_drop);
  end

  wire unused_ok = &{1'b0, dbg_host, ifq_entry, tap_enq_unit[3], csr_wdata, q_state[0][8], q_state[1][8],
                     q_state[2][8], q_state[3][8], q_state[4][8], q_state[5][8]};

endmodule
