// tsr_csr.sv — the host CSR block: the whole EX §5 map (identification, CTRL / STATUS, the
// firmware loader with VERIFY, the launch queue LQ, the completion ring LCQ, the log ring, the
// watermarks, PERF, Q_*, SREG, the DMI window), the SOFT_RESET sequencer, the RISC-V reset
// actions of the core domain, MTIME, RV_FREQ and the unit watchdog (docs/design/01-execution-
// model.md "EX" §3-§6, §8.5-§8.6; docs/design/04-control-core.md "CC" §6, §12).
//
// Owner: issue path (CC §2.2, §13.2: `tsr_csr` rewritten; W4 item 1). Instantiated by tsr_core_dp
// next to tsr_issue; the clk_rv side is tsr_ctrl / tsr_ctrl_cdc (W4 item 3), whose core-domain
// ports these are. Oracle: the simulator's host CSR port and control window (sw/sim/tsr_sim.c
// csr_read / csr_write / tsr_sim_lq_cw_access, tsr_q.c); tb/csr compares every register, ring,
// fault and counter with it (cycle-agnostic), tb/issue the issue-path side.
//
// Requesters (CC §6.6): the host CSR port and the core window (cwin_*, tsr_ctrl_cdc's requester 2)
//   share one access pipeline, round robin per access; the LQ / LCQ registers of the RISC-V
//   (lq_*) have their own port, so a core-window read never delays a pop or a post. Only host LO
//   reads of PERF touch the snapshot; core-window reads of LOAD_DATA (0x0C4) and of the DMI window
//   return 0 without access (CC §5.3).
// Host CSR port (core boundary; protocol unchanged, docs/rtl-m2.md §6): the master holds
//   csr_valid / addr / we / wdata until csr_ready, a registered one-cycle strobe with csr_rdata.
//   Access pipeline: capture (A_IDLE) -> decode, register writes and read selection (A_DEC) ->
//   respond (A_RSP); 4 cycles per transfer including the handshake edge. LOAD_DATA, DMI and the DBG
//   block take extra states (A_LDW / A_LDR / A_DMI / A_DBG), each bounded: the loader and DMI
//   handshakes end within 4,096 core cycles + 1 (tsr_ctrl_cdc's HS_TIMEOUT; answered locally
//   while clk_rv is not running), and a LOAD_DATA write waits at most LD_WAIT_MAX cycles for the
//   loader port before it is dropped with LOAD_TIMEOUT. Unmapped or unaligned offsets read 0,
//   writes are ignored (EX §5.1).
// Core-domain side of tsr_ctrl_cdc (CC §11.2):
//   cwin_*  req_valid level with req_off stable until the one-cycle rsp_valid (rsp_rdata)
//   lq_*    the same protocol; writes LQ_POP (0x228) and LCQ_POST (0x260: lq_req_lcq_w = LCQ_W0-7,
//           wdata bit0 ANSWER) commit here and answer lq_rsp_fault for EX §4.5's faults (other
//           offsets are ignored and answered); reads return the core-domain value; a LOG_FREE (0x284) read returns
//           LOG_RD + (words the ring dropped since the hardware reset) mod 2^32, from which the
//           clk_rv side computes LOG_FREE with its own push count (EX §4.1)
//   log_*   LOG_PUSH words in order (log_ready = 1: drop-new when the ring is full, EX §4.10)
//   ld_*    one loader word per request while ld_ready; ld_rsp_valid with rdata / err (no TCM
//           there, or the hart not held) / timeout (-> LOAD_TIMEOUT)
//   dmi_*   one DMI access per request while dmi_ready (op 1 read, 2 write), gated by dmi_en
//           (DB's DBG_CFG.DMI_EN); dmi_rsp_timeout -> DMI_TIMEOUT
//   rt_state / rt_info (mirrors), rv_cyc (clk_rv cycle count), perf_push_stall / bridge_acc /
//   bridge_wait (32-bit counts, gated on the clk_rv side, extended to 64 bits here),
//   push_timeout / bridge_timeout (pulses), rv_halted / rv_sleep / rv_running / rv_reset_done /
//   rv_clk_ok / rv_ndmreset (levels), issued_seq (ISSUED_SEQ copies), ifq_count (issue-FIFO
//   entries visible to the dispatcher, Q_FIFO); outputs mtime, rv_run, rv_boot, irq_lq, irq_lcq,
//   perf_time_en (ACTIVE ∧ ¬QHALT), perf_evt_en (ACTIVE).
//   rv_run is held low while a loader request is outstanding (the loader owns the TCM ports only
//   while the hart is held, tsr_ctrl); a VERIFY stops when CTRL.RV_RUN rises.
// Issue path (tsr_issue): host_hold, halt_hold (RV_RUN ∧ RV_HALTED ∧ HOLD_ON_HALT), qabort
//   (pulse: CTRL.QABORT, an LQ_CANCEL covering the running launch, the RISC-V reset, SOFT_RESET),
//   rv_reset (pulse: RV_RUN 1 -> 0, ndmreset rising, SOFT_RESET step 1), host_clear (pulse: an
//   accepted CTRL.QCLEAR, SOFT_RESET step 3), sr_busy; inputs qidle, qquiescent, err_any, hold_src,
//   q_state, seq_d, err_rec, q_pidx and the PERF events.
// CTRL (EX §5.2): level bits first, then the pulses in bit order (QABORT, QCLEAR, SOFT_RESET).
//   RV_RUN changes are refused while SR_BUSY (RV_RUN_REFUSED). RV_RUN 1 -> 0 (and ndmreset rising):
//   QABORT at once, ANSWERED := TAKEN (EX §8.5). QCLEAR is accepted only with QIDLE and the RISC-V
//   quiet (RV_RESET_DONE, or RV_HALTED), else QCLEAR_REFUSED.
// SOFT_RESET sequencer (EX §8.5, CC §12.2, CC-31): a CTRL.SOFT_RESET write sets SR_BUSY and returns.
//   Step 1: RV_RUN := 0 with the RISC-V reset actions (QABORT even if RV_RUN was 0), sr_db_clear
//   (DB's stop state; dbg_hold is ignored while sr_busy), then wait for RV_RESET_DONE: at least
//   SR_RV_MIN cycles (the synchronised level of the clk_rv side must have seen RV_RUN drop) and at
//   most SR_RV_MAX = 2^16, else SR_RV_TIMEOUT and on (no clk_rv). Step 2: wait for QIDLE (no entry
//   queued or running, issue FIFO empty). Step 3: TAKEN := PUSHED, ANSWERED := TAKEN, LCQ_RD :=
//   LCQ_WR, LOG_RD := LOG_WR, both watermark VALID bits := 0, the host QCLEAR effects, PERF := 0
//   and the snapshot dropped, the sticky bits and LOG_DROPPED cleared (SR_RV_TIMEOUT set by this
//   sequence is kept, so the host can see it). Step 4: SR_BUSY := 0.
// Loader (EX §3.3): LOAD_ADDR (word aligned) restarts LOAD_CRC; a LOAD_DATA write posts the word
//   to the loader port, folds it into the receive CRC (zlib crc32, bytes in address order) and
//   advances LOAD_ADDR by 4; a host LOAD_DATA read returns the word at LOAD_ADDR (0 on err /
//   timeout). VERIFY reads LOAD_COUNT words from LOAD_ADDR (LOAD_ADDR unchanged), recomputes
//   LOAD_CRC, LOAD_STATUS.VERIFY_BUSY meanwhile, stops at a timeout (LOAD_TIMEOUT). Refused while
//   RV_RUN = 1 or SR_BUSY (LOAD_REFUSED for writes and VERIFY; reads return 0): and while VERIFY
//   runs (the sequencer owns the port).
// PERF (EX §6, CC §6.3): 64 slots of 64 bits; time slots count with ACTIVE ∧ ¬QHALT (QHALT =
//   RV_RUN ∧ RV_HALTED ∧ QQUIESCENT), event slots with ACTIVE, DBG_STOP_CYCLES (63) with ACTIVE ∧
//   (RV_HALTED ∨ db_stop_active), the three clk_rv counters (X + 7, 8, 9) by their 32-bit deltas;
//   perf_ext adds the other areas' per-slot increments (VU, AT, SPM, DB) with the slot's gate. A
//   host LO read snapshots the counter's HI half for the next HI read of that counter (the
//   simulator's rule: the snapshot stays until the next LO read or SOFT_RESET).
// RV_FREQ: clk_rv cycles in the last 2^16 core cycles (16 windows of 4,096, F_RV < 16 F_CORE).
// Watchdog (EX §8.6): per datapath unit, a counter runs while the unit holds an op and restarts
//   on unit_progress[u] (any SPM handshake, memory beat or done of the unit; tsr_core_dp ORs them);
//   at 2^WDOG_CFG.L: Q_STATE[u].HUNG (until the next progress) and STATUS.UNIT_HUNG; L = 0 off.

module tsr_csr
  import tsr_pkg::*;
#(
  parameter int unsigned NUM_CH      = TSR_DEF_NUM_CH,
  parameter int unsigned LANES       = TSR_DEF_LANES,
  parameter int unsigned T_MAX       = TSR_DEF_T_MAX,
  parameter int unsigned K_MAX       = TSR_DEF_K_MAX,
  parameter int unsigned SPM_BYTES   = TSR_DEF_SPM_BYTES,
  parameter int unsigned VL          = TSR_DEF_VL,
  parameter int unsigned NSREG       = TSR_DEF_NSREG,
  parameter logic [31:0] CAPS        = 32'd0,
  // CTL_ID (EX §5.1): control-core revision 1, debug module, SPM bridge, log ring, watchdog,
  // SPM line-buffered alias (the simulator's TSR_SIM_CTL_ID_FULL)
  parameter logic [31:0] CTL_ID      = 32'h0000_5D01,
  parameter logic [31:0] CFG3        = 32'd0,
  parameter int unsigned ITCM_BYTES  = tsr_tables_pkg::TSR_ITCM_BYTES,
  parameter int unsigned DTCM_BYTES  = tsr_tables_pkg::TSR_DTCM_BYTES,
  parameter int unsigned QD_DMA      = tsr_tables_pkg::TSR_QDEPTH_U200_DMA,
  parameter int unsigned QD_QACT     = tsr_tables_pkg::TSR_QDEPTH_U200_QACT,
  parameter int unsigned QD_MVM      = tsr_tables_pkg::TSR_QDEPTH_U200_MVM,
  parameter int unsigned QD_VU       = tsr_tables_pkg::TSR_QDEPTH_U200_VU,
  parameter int unsigned QD_ATT      = tsr_tables_pkg::TSR_QDEPTH_U200_ATT,
  parameter int unsigned QD_SYNC     = tsr_tables_pkg::TSR_QDEPTH_U200_SYNC,
  parameter bit          HAS_VU      = 1'b1,       // SREG0-15 present (EX §5.1)
  parameter int unsigned SR_RV_MIN   = 64,
  parameter int unsigned SR_RV_MAX   = tsr_tables_pkg::TSR_SR_RV_WAIT_CYCLES,
  parameter int unsigned LD_WAIT_MAX = tsr_tables_pkg::TSR_LOAD_TIMEOUT_CYCLES,
  parameter int unsigned FREQ_WIN    = 4096,
  // FAST_CLK (docs/platform-u200.md §8.9): the PERF counters take their clear, enable and increment from registers (1 cycle later,
  // all alike, so the counts equal the single-stage counts shifted by one cycle)
  parameter bit          FAST_CLK    = 1'b0
) (
  input  logic                       clk,
  input  logic                       rst_n,

  // ---- host CSR port (core boundary) ----
  input  logic                       csr_valid,
  output logic                       csr_ready,
  input  logic                       csr_we,
  input  logic [TSR_CSR_AW-1:0]      csr_addr,
  input  logic [31:0]                csr_wdata,
  output logic [31:0]                csr_rdata,

  // ---- core window (tsr_ctrl_cdc requester 2) ----
  input  logic                       cwin_req_valid,
  input  logic [11:0]                cwin_req_off,
  output logic                       cwin_rsp_valid,
  output logic [31:0]                cwin_rsp_rdata,

  // ---- LQ / LCQ registers of the RISC-V (EX §4.5) ----
  input  logic                       lq_req_valid,
  input  logic [11:0]                lq_req_off,
  input  logic                       lq_req_we,
  input  logic [31:0]                lq_req_wdata,
  input  logic [7:0][31:0]           lq_req_lcq_w,
  output logic                       lq_rsp_valid,
  output logic [31:0]                lq_rsp_rdata,
  output logic                       lq_rsp_fault,

  // ---- log words, RT mirrors ----
  input  logic                       log_valid,
  input  logic [31:0]                log_data,
  output logic                       log_ready,
  input  logic [31:0]                rt_state,
  input  logic [31:0]                rt_info,

  // ---- loader port (TCMs) ----
  input  logic                       ld_ready,
  output logic                       ld_req,
  output logic                       ld_we,
  output logic [31:0]                ld_addr,
  output logic [31:0]                ld_wdata,
  input  logic                       ld_rsp_valid,
  input  logic [31:0]                ld_rsp_rdata,
  input  logic                       ld_rsp_err,
  input  logic                       ld_rsp_timeout,

  // ---- DMI (DB §3.2) ----
  input  logic                       dmi_en,
  input  logic                       dmi_ready,
  output logic                       dmi_req,
  output logic [6:0]                 dmi_addr,
  output logic [1:0]                 dmi_op,
  output logic [31:0]                dmi_wdata,
  input  logic                       dmi_rsp_valid,
  input  logic [31:0]                dmi_rsp_rdata,
  input  logic [1:0]                 dmi_rsp_resp,
  input  logic                       dmi_rsp_timeout,

  // ---- DBG block (DB §10.1, W4-B): 0x800-0xAFC except SREG; dbg_rdata the cycle after dbg_req ----
  output logic                       dbg_req,
  output logic                       dbg_we,
  output logic [11:0]                dbg_addr,
  output logic [31:0]                dbg_wdata,
  output logic                       dbg_host,
  input  logic [31:0]                dbg_rdata,
  output logic                       sr_db_clear,
  input  logic                       db_stop_active,

  // ---- s bank (VU) ----
  input  logic [15:0][31:0]          sreg,

  // ---- clk_rv side (tsr_ctrl_cdc) ----
  output logic [63:0]                mtime,
  input  logic [15:0]                rv_cyc,
  input  logic [31:0]                perf_push_stall,
  input  logic [31:0]                perf_bridge_acc,
  input  logic [31:0]                perf_bridge_wait,
  input  logic                       push_timeout,
  input  logic                       bridge_timeout,
  output logic                       rv_run,
  output logic [31:0]                rv_boot,
  output logic                       irq_lq,
  output logic                       irq_lcq,
  output logic                       perf_time_en,
  output logic                       perf_evt_en,
  input  logic                       rv_halted,
  input  logic                       rv_sleep,
  input  logic                       rv_running,
  input  logic                       rv_reset_done,
  input  logic                       rv_clk_ok,
  input  logic                       rv_ndmreset,
  input  logic [5:0][TSR_SEQ_W-1:0]  issued_seq,
  input  logic [5:0]                 ifq_count,

  // ---- issue path (tsr_issue) ----
  output logic                       host_hold,
  output logic                       halt_hold,
  output logic                       qabort,
  output logic                       rv_reset,
  output logic                       host_clear,
  output logic                       sr_busy,
  input  logic                       qidle,
  input  logic                       qquiescent,
  input  logic                       err_any,
  input  logic [2:0]                 hold_src,
  input  logic [5:0][8:0]            q_state,
  input  logic [5:0][TSR_SEQ_W-1:0]  seq_d,
  input  tsr_err_rec_t [5:0]         err_rec,
  input  logic [31:0]                q_pidx,
  input  logic [2:0]                 perf_retired,
  input  logic                       perf_push,
  input  logic [5:0]                 perf_busy,
  input  logic                       perf_mvm_wait,
  input  logic                       perf_mvm_empty,
  input  logic                       perf_hold,

  // ---- datapath PERF events and the other areas' slots ----
  input  logic [NUM_CH-1:0]          perf_ch_rd_beat,
  input  logic [NUM_CH-1:0]          perf_ch_wait,
  input  logic [63:0][3:0]           perf_ext,
  input  logic [5:0]                 unit_progress,

  // ---- platform (EX §9.3) ----
  input  logic                       plat_err_i,
  output logic                       host_irq_o,

  // STATUS.UNIT_HUNG (the sticky bit, EX §8.6) for the DB block: HUNG_STOP and the history freeze (DB §5.4)
  output logic                       unit_hung
);

  // -------------------------------------------------------------------------------------------
  // Constants and offsets (generated table, EX-30)
  // -------------------------------------------------------------------------------------------
  localparam int unsigned LQD  = tsr_tables_pkg::TSR_LQ_DEPTH;      // 8
  localparam int unsigned LCQD = tsr_tables_pkg::TSR_LCQ_DEPTH;     // 8
  localparam int unsigned LOGD = tsr_tables_pkg::TSR_LOG_DEPTH;     // 256
  localparam int unsigned X    = 5 + 2 * NUM_CH;                    // first PERF extension slot

  localparam logic [31:0] VERSION_VAL = tsr_tables_pkg::TSR_VERSION_VALUE;
  localparam logic [31:0] CFG0_VAL    = {8'd0, 4'(T_MAX), 12'(LANES), 8'(NUM_CH)};
  localparam logic [31:0] CFG1_VAL    = 32'(SPM_BYTES / TSR_LINE_BYTES);
  localparam logic [31:0] CFG2_VAL    = {16'(K_MAX / 256), 8'(NSREG), 8'(VL)};
  localparam logic [31:0] RV_CFG_VAL  = {16'(DTCM_BYTES / 1024), 16'(ITCM_BYTES / 1024)};
  localparam logic [31:0] LQ_CFG_VAL  = {4'd0, 12'(LOGD), 8'(LCQD), 8'(LQD)};
  localparam logic [31:0] PINFO_VAL   = {16'd0, 8'(X), 8'd64};
  localparam logic [31:0] QDEPTH_VAL  = {2'd0, 5'(QD_SYNC), 5'(QD_ATT), 5'(QD_VU), 5'(QD_MVM), 5'(QD_QACT), 5'(QD_DMA)};
  localparam logic [5:0]  UNITS       = {QD_SYNC != 0, QD_ATT != 0, QD_VU != 0, QD_MVM != 0, QD_QACT != 0, QD_DMA != 0};

  // CTRL / STATUS bits (EX §5.2-§5.3)
  localparam int unsigned C_RV_RUN = 0, C_HOST_HOLD = 1, C_HOLD_ON_HALT = 2, C_IRQ_EN = 3;
  localparam int unsigned C_QABORT = 8, C_QCLEAR = 9, C_SOFT_RESET = 10;
  localparam int unsigned S_LOAD_REFUSED = 16, S_LOAD_TIMEOUT = 17, S_DMI_TIMEOUT = 18, S_QCLEAR_REFUSED = 19,
                          S_PUSH_TIMEOUT = 20, S_BRIDGE_TIMEOUT = 21, S_LQ_OVERFLOW = 22, S_RING_RD_BAD = 23,
                          S_LOG_DROP = 24, S_UNIT_HUNG = 25, S_SR_RV_TIMEOUT = 26, S_RV_RUN_REFUSED = 27;

  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX || X + 22 > 61) begin : g_chk_ch
    $fatal(1, "tsr_csr: NUM_CH must be in [1, 16] (PERF slots, EX §6)");
  end

  // PERF slot gate classes (EX §6)
  localparam logic [2:0] PC_NONE = 3'd0, PC_TIME = 3'd1, PC_EVT = 3'd2, PC_RV = 3'd3, PC_STOP = 3'd4;
  function automatic logic [2:0] pclass(int unsigned i);
    if (i == 0 || i == 2 || i == 3) return PC_TIME;
    if (i == 1) return PC_EVT;
    if (i >= 5 && i < 5 + NUM_CH) return PC_EVT;
    if (i >= 5 + NUM_CH && i < X) return PC_TIME;
    if (i >= X && i < X + 22) begin
      case (i - X)
        6, 10, 11, 15, 19: return PC_EVT;
        7, 8, 9:           return PC_RV;
        default:           return PC_TIME;
      endcase
    end
    if (i == 61 || i == 62) return PC_EVT;
    if (i == 63) return PC_STOP;
    return PC_NONE;
  endfunction

  // -------------------------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------------------------
  logic [3:0]  ctrl_q;                    // {IRQ_EN, HOLD_ON_HALT, HOST_HOLD, RV_RUN}
  logic [27:16] sticky_q;
  logic [31:8] boot_q;
  logic [4:0]  wdog_q;
  logic [31:0] lq_w_q [8];
  logic [31:0] lq_ring [LQD][8];
  logic [31:0] pushed_q, taken_q, answered_q;
  logic [31:0] cancel_q, stop_q;
  logic        cancel_v_q, stop_v_q;
  logic [31:0] lcq_ring [LCQD][8];
  logic [31:0] lcq_wr_q, lcq_rd_q;
  logic [31:0] log_ring [LOGD];
  logic [31:0] log_wr_q, log_rd_q, log_dropped_q, log_drops_q;
  logic [7:0]  log_cfg_q;
  logic [31:0] load_addr_q, load_crc_q, load_count_q;
  logic [63:0] perf_q [64];
  logic        snap_v_q;
  logic [5:0]  snap_idx_q;
  logic [31:0] snap_hi_q;
  logic [63:0] mtime_q;
  logic        ndm_q;
  logic [31:0] rv_freq_q;
  logic [5:0]  hung_q;                    // Q_STATE.HUNG (watchdog)
  logic [5:0]  wdog_hit;

  wire active = (taken_q != answered_q);
  wire qhalt  = ctrl_q[C_RV_RUN] && rv_halted && qquiescent;

  // -------------------------------------------------------------------------------------------
  // CRC-32 (zlib): one word, bytes in address order, on the pre-final state
  // -------------------------------------------------------------------------------------------
  function automatic logic [31:0] crc_word(logic [31:0] st, logic [31:0] w);
    logic [31:0] c;
    c = st;
    for (int b = 0; b < 4; b++) begin
      c = c ^ {24'd0, w[8 * b +: 8]};
      for (int k = 0; k < 8; k++) c = (c >> 1) ^ (tsr_tables_pkg::TSR_CRC32_POLY & {32{c[0]}});
    end
    return c;
  endfunction

  // -------------------------------------------------------------------------------------------
  // Access pipeline (host + core window)
  // -------------------------------------------------------------------------------------------
  typedef enum logic [2:0] { A_IDLE, A_DEC, A_RSP, A_LDW, A_LDR, A_DMI, A_DBG } astate_e;
  astate_e     ast_q;
  logic        a_host_q, a_we_q, rr_q;
  logic [11:0] a_addr_q;
  logic [31:0] a_wdata_q, a_rdata_q;
  logic [15:0] a_wait_q;
  logic        a_sent_q;                  // A_LDR / A_DMI: the request is out
  logic        csr_ready_q, cw_rsp_q;

  wire host_rq = csr_valid && !csr_ready_q;
  wire cw_rq   = cwin_req_valid && !cw_rsp_q;
  wire pick_host = host_rq && (!cw_rq || !rr_q);
  wire pick_cw   = cw_rq && (!host_rq || rr_q);

  // ---- loader sequencer state ----
  logic        vf_busy_q, vf_out_q;       // VERIFY running / a VERIFY read in flight
  logic [31:0] vf_addr_q, vf_left_q;
  logic        ld_out_q;                  // a loader request in flight (any)
  logic        ld_rd_q;                   // it is a read for the access pipeline
  wire         load_refused = ctrl_q[C_RV_RUN] || sr_busy;

  // ---- decode of the captured access ----
  wire aligned = (a_addr_q[1:0] == 2'b00);
  wire [11:0] A = a_addr_q;
  wire in_perf  = (A >= 12'h100) && (A < 12'h300);
  wire [5:0] p_idx = 6'((A - 12'h100) >> 3);
  wire in_lcq   = (A >= 12'h300) && (A < 12'h400);
  wire in_log   = (A >= 12'h400) && (A < 12'h800);
  wire in_sreg  = (A >= 12'h9C0) && (A < 12'hA00);
  wire in_dbg   = (A >= 12'h800) && (A < 12'hB00) && !in_sreg;
  wire in_dmi   = (A >= 12'hC00) && (A < 12'hE00);
  wire in_q     = (A >= 12'hE00) && (A < 12'hF20);
  wire in_lqw   = (A >= 12'h040) && (A < 12'h060);

  logic [31:0] status_v;
  always_comb begin
    status_v = '0;
    status_v[0]  = rv_running;
    status_v[1]  = rv_halted;
    status_v[2]  = rv_sleep;
    status_v[3]  = rv_clk_ok;
    status_v[4]  = rv_reset_done;
    status_v[5]  = active;
    status_v[6]  = qidle;
    status_v[7]  = qquiescent;
    status_v[8]  = err_any;
    status_v[9]  = sr_busy;
    status_v[10] = (lcq_wr_q != lcq_rd_q);
    status_v[11] = (log_wr_q != log_rd_q);
    status_v[12] = plat_err_i;
    status_v[15:13] = hold_src;
    status_v[27:16] = sticky_q;
  end

  // Q_* (EX §5.4): unit slots 0..7, absent units read 0
  logic [31:0] q_val;
  always_comb begin
    logic [2:0] u;
    q_val = '0;
    u = 3'(A[4:2]);
    if (A >= 12'hE00 && A < 12'hE20) begin
      if (32'(A[4:2]) < 6 && UNITS[u]) q_val = 32'(seq_d[u]);
    end else if (A >= 12'hE20 && A < 12'hE40) begin
      if (32'(A[4:2]) < 6 && UNITS[u]) q_val = 32'(issued_seq[u]);
    end else if (A >= 12'hE40 && A < 12'hE60) begin
      if (32'(A[4:2]) < 6 && UNITS[u]) q_val = {22'd0, hung_q[u], q_state[u]};
    end else if (A == 12'hE60) begin
      q_val = 32'(ifq_count);
    end else if (A == 12'hE64) begin
      q_val = q_pidx;
    end else if (A >= 12'hE80 && A < 12'hF00) begin
      u = 3'(A[6:4]);
      if (32'(A[6:4]) < 6 && UNITS[u]) begin
        case (A[3:2])
          2'd0: q_val = 32'(err_rec[u].seq);
          2'd1: q_val = 32'(err_rec[u].code);
          2'd2: q_val = err_rec[u].info;
          default: q_val = err_rec[u].w0;
        endcase
      end
    end else if (A >= 12'hF00 && A < 12'hF20) begin
      if (32'(A[4:2]) < 6 && UNITS[u]) q_val = err_rec[u].pidx;
    end
  end

  // read value of every single-cycle register (A_DEC)
  logic [31:0] rd_val;
  logic        r_perf, r_hi;
  always_comb begin
    rd_val = '0;
    r_perf = 1'b0;
    r_hi   = A[2];
    case (A)
      12'h000: rd_val = tsr_tables_pkg::TSR_ID_MAGIC;
      12'h004: rd_val = VERSION_VAL;
      12'h008: rd_val = CFG0_VAL;
      12'h00C: rd_val = CFG1_VAL;
      12'h010: rd_val = CFG2_VAL;
      12'h014: rd_val = CAPS;
      12'h018: rd_val = CFG3;
      12'h01C: rd_val = CTL_ID;
      12'h020: rd_val = {28'd0, ctrl_q};
      12'h024: rd_val = status_v;
      12'h028: rd_val = {boot_q, 8'd0};
      12'h02C: rd_val = rv_freq_q;
      12'h030: rd_val = RV_CFG_VAL;
      12'h034: rd_val = {27'd0, wdog_q};
      12'h038: rd_val = rt_state;
      12'h03C: rd_val = rt_info;
      12'h064: rd_val = pushed_q;
      12'h068: rd_val = taken_q;
      12'h06C: rd_val = cancel_q;
      12'h070: rd_val = stop_q;
      12'h074: rd_val = lcq_wr_q;
      12'h078: rd_val = lcq_rd_q;
      12'h07C: rd_val = LQ_CFG_VAL | {cancel_v_q, stop_v_q, 30'd0};
      12'h080: rd_val = PINFO_VAL;
      12'h084: rd_val = answered_q;
      12'h088: rd_val = QDEPTH_VAL;
      12'h090: rd_val = log_wr_q;
      12'h094: rd_val = log_rd_q;
      12'h098: rd_val = {24'd0, log_cfg_q};
      12'h09C: rd_val = log_dropped_q;
      12'h0C0: rd_val = load_addr_q;
      12'h0C8: rd_val = ~load_crc_q;
      12'h0CC: rd_val = load_count_q;
      12'h0D4: rd_val = {31'd0, vf_busy_q};
      default: begin
        if (in_lqw)  rd_val = lq_w_q[A[4:2]];
        if (in_perf) r_perf = 1'b1;
        if (in_lcq)  rd_val = lcq_ring[A[7:5]][A[4:2]];
        if (in_log)  rd_val = log_ring[A[9:2]];
        if (in_sreg) rd_val = HAS_VU ? sreg[A[5:2]] : 32'd0;
        if (in_q)    rd_val = q_val;
      end
    endcase
    if (!aligned) begin
      rd_val = '0;
      r_perf = 1'b0;
    end
  end
  wire [63:0] p_val = perf_q[p_idx];

  // -------------------------------------------------------------------------------------------
  // LQ / LCQ port (one request at a time; answered the cycle after it is seen)
  // -------------------------------------------------------------------------------------------
  logic        lq_rsp_q, lq_fault_q;
  logic [31:0] lq_rdata_q;
  wire         lq_go = lq_req_valid && !lq_rsp_q;
  wire [31:0]  avail = pushed_q - taken_q;
  wire [31:0]  lcq_used = lcq_wr_q - lcq_rd_q;
  wire [31:0]  lcq_free = 32'(LCQD) - lcq_used;
  wire         lq_pop  = lq_go && lq_req_we && lq_req_off == 12'h228;
  wire         lq_post = lq_go && lq_req_we && lq_req_off == 12'h260;
  wire         pop_ok  = (avail != 32'd0) && !active;
  wire         post_ok = (lcq_free != 32'd0) && !(lq_req_wdata[0] && !active);
  logic [31:0] lq_rd_v;
  always_comb begin
    lq_rd_v = '0;
    if (lq_req_off >= 12'h200 && lq_req_off < 12'h220) begin
      if (avail != 32'd0) lq_rd_v = lq_ring[taken_q[$clog2(LQD)-1:0]][lq_req_off[4:2]];
    end else begin
      case (lq_req_off)
        12'h220: lq_rd_v = avail;
        12'h224: lq_rd_v = taken_q;
        12'h22C: lq_rd_v = cancel_q;
        12'h230: lq_rd_v = stop_q;
        12'h234: lq_rd_v = {28'd0, plat_err_i, active, stop_v_q, cancel_v_q};
        12'h264: lq_rd_v = lcq_free;
        12'h284: lq_rd_v = log_rd_q + log_drops_q;      // the clk_rv side computes LOG_FREE
        12'h288: lq_rd_v = {24'd0, log_cfg_q};
        default: ;
      endcase
    end
  end

  // -------------------------------------------------------------------------------------------
  // SOFT_RESET sequencer state
  // -------------------------------------------------------------------------------------------
  typedef enum logic [1:0] { SR_IDLE, SR_RV, SR_DRAIN, SR_CLR } srstate_e;
  srstate_e    sr_q;
  logic [16:0] sr_cnt_q;
  logic        sr_to_q;                   // this sequence went on without RV_RESET_DONE

  // -------------------------------------------------------------------------------------------
  // Main register block: host / core-window accesses, LQ port, log words, loader, SR, resets
  // -------------------------------------------------------------------------------------------
  logic        qabort_q, rv_reset_q, host_clear_q, sr_db_clear_q;
  logic        ld_req_q, ld_we_q, dmi_req_q, dbg_req_q;
  logic [31:0] ld_addr_q, ld_wdata_q;

  // decode helpers for writes (A_DEC, host only)
  wire wr = (ast_q == A_DEC) && a_host_q && a_we_q && aligned;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ast_q         <= A_IDLE;
      rr_q          <= 1'b0;
      csr_ready_q   <= 1'b0;
      cw_rsp_q      <= 1'b0;
      a_sent_q      <= 1'b0;
      ctrl_q        <= '0;
      sticky_q      <= '0;
      boot_q        <= '0;
      wdog_q        <= 5'(tsr_tables_pkg::TSR_WDOG_L_RESET);
      pushed_q      <= '0;
      taken_q       <= '0;
      answered_q    <= '0;
      cancel_q      <= '0;
      stop_q        <= '0;
      cancel_v_q    <= 1'b0;
      stop_v_q      <= 1'b0;
      lcq_wr_q      <= '0;
      lcq_rd_q      <= '0;
      log_wr_q      <= '0;
      log_rd_q      <= '0;
      log_dropped_q <= '0;
      log_drops_q   <= '0;
      log_cfg_q     <= 8'(tsr_tables_pkg::TSR_CSR_LOG_CFG_RESET);
      load_addr_q   <= '0;
      load_crc_q    <= '1;
      load_count_q  <= '0;
      snap_v_q      <= 1'b0;
      lq_rsp_q      <= 1'b0;
      qabort_q      <= 1'b0;
      rv_reset_q    <= 1'b0;
      host_clear_q  <= 1'b0;
      sr_db_clear_q <= 1'b0;
      ld_req_q      <= 1'b0;
      ld_out_q      <= 1'b0;
      ld_rd_q       <= 1'b0;
      vf_busy_q     <= 1'b0;
      vf_out_q      <= 1'b0;
      dmi_req_q     <= 1'b0;
      dbg_req_q     <= 1'b0;
      sr_q          <= SR_IDLE;
      sr_busy       <= 1'b0;
      sr_to_q       <= 1'b0;
      ndm_q         <= 1'b0;
      for (int i = 0; i < 8; i++) lq_w_q[i] <= '0;
    end else begin
      // one-cycle pulses
      csr_ready_q   <= 1'b0;
      cw_rsp_q      <= 1'b0;
      lq_rsp_q      <= 1'b0;
      qabort_q      <= 1'b0;
      rv_reset_q    <= 1'b0;
      host_clear_q  <= 1'b0;
      sr_db_clear_q <= 1'b0;
      ld_req_q      <= 1'b0;
      dmi_req_q     <= 1'b0;
      dbg_req_q     <= 1'b0;
      ndm_q         <= rv_ndmreset;

      // ---- sticky events from the clk_rv side ----
      if (push_timeout)   sticky_q[S_PUSH_TIMEOUT]   <= 1'b1;
      if (bridge_timeout) sticky_q[S_BRIDGE_TIMEOUT] <= 1'b1;

      // ---- log words (EX §4.10: drop-new) ----
      if (log_valid) begin
        if ((log_wr_q - log_rd_q) < 32'(LOGD)) begin
          log_ring[log_wr_q[7:0]] <= log_data;
          log_wr_q <= log_wr_q + 32'd1;
        end else begin
          if (log_dropped_q != '1) log_dropped_q <= log_dropped_q + 32'd1;
          log_drops_q <= log_drops_q + 32'd1;
          sticky_q[S_LOG_DROP] <= 1'b1;
        end
      end

      // ---- LQ / LCQ port of the RISC-V (EX §4.5) ----
      if (lq_go) begin
        lq_rsp_q   <= 1'b1;
        lq_rdata_q <= lq_req_we ? 32'd0 : lq_rd_v;
        lq_fault_q <= (lq_pop && !pop_ok) || (lq_post && !post_ok);
        if (lq_pop && pop_ok) taken_q <= taken_q + 32'd1;
        if (lq_post && post_ok) begin
          for (int i = 0; i < 8; i++) lcq_ring[lcq_wr_q[$clog2(LCQD)-1:0]][i] <= lq_req_lcq_w[i];
          lcq_wr_q <= lcq_wr_q + 32'd1;
          if (lq_req_wdata[0]) answered_q <= answered_q + 32'd1;
        end
      end

      // ---- access pipeline ----
      case (ast_q)
        A_IDLE: begin
          if (pick_host || pick_cw) begin
            a_host_q  <= pick_host;
            a_we_q    <= pick_host ? csr_we : 1'b0;
            a_addr_q  <= pick_host ? csr_addr : cwin_req_off;
            a_wdata_q <= csr_wdata;
            rr_q      <= pick_host;            // the other requester first next time
            ast_q     <= A_DEC;
          end
        end
        A_DEC: begin
          a_rdata_q <= a_we_q ? 32'd0 : rd_val;        // writes answer 0 (docs/rtl-m2.md §6)
          a_wait_q  <= '0;
          a_sent_q  <= 1'b0;
          ast_q     <= A_RSP;
          if (r_perf && !a_we_q) begin
            if (!r_hi) begin
              a_rdata_q <= p_val[31:0];
              if (a_host_q) begin                // host LO read: snapshot for the next HI read
                snap_v_q   <= 1'b1;
                snap_idx_q <= p_idx;
                snap_hi_q  <= p_val[63:32];
              end
            end else begin
              a_rdata_q <= (a_host_q && snap_v_q && snap_idx_q == p_idx) ? snap_hi_q : p_val[63:32];
            end
          end
          if (aligned && in_dbg) begin
            ast_q <= A_DBG;
            dbg_req_q <= !(a_we_q && !a_host_q);
          end
          if (aligned && in_dmi) begin
            a_rdata_q <= '0;
            if (a_host_q && dmi_en) ast_q <= A_DMI;
          end
          if (aligned && A == 12'h0C4) begin
            a_rdata_q <= '0;
            if (a_host_q && !load_refused && !vf_busy_q) ast_q <= a_we_q ? A_LDW : A_LDR;
            else if (a_host_q && a_we_q) sticky_q[S_LOAD_REFUSED] <= 1'b1;
          end
          // ---- host writes (EX §5.1-§5.3) ----
          if (wr) begin
            case (A)
              12'h020: begin                    // CTRL: levels, then pulses in bit order
                logic run_n;
                run_n = ctrl_q[C_RV_RUN];
                if (a_wdata_q[C_RV_RUN] != ctrl_q[C_RV_RUN]) begin
                  if (sr_busy) sticky_q[S_RV_RUN_REFUSED] <= 1'b1;
                  else         run_n = a_wdata_q[C_RV_RUN];
                end
                ctrl_q <= {a_wdata_q[C_IRQ_EN], a_wdata_q[C_HOLD_ON_HALT], a_wdata_q[C_HOST_HOLD], run_n};
                if (ctrl_q[C_RV_RUN] && !run_n) begin          // RISC-V reset (EX §8.5)
                  qabort_q   <= 1'b1;
                  rv_reset_q <= 1'b1;
                  answered_q <= taken_q;
                end
                if (a_wdata_q[C_QABORT]) qabort_q <= 1'b1;
                if (a_wdata_q[C_QCLEAR]) begin
                  if (qidle && (rv_reset_done || rv_halted)) host_clear_q <= 1'b1;
                  else sticky_q[S_QCLEAR_REFUSED] <= 1'b1;
                end
                if (a_wdata_q[C_SOFT_RESET] && sr_q == SR_IDLE) begin  // SOFT_RESET step 1
                  sr_q          <= SR_RV;
                  sr_busy       <= 1'b1;
                  sr_cnt_q      <= '0;
                  sr_to_q       <= 1'b0;
                  ctrl_q[C_RV_RUN] <= 1'b0;
                  qabort_q      <= 1'b1;
                  rv_reset_q    <= 1'b1;
                  sr_db_clear_q <= 1'b1;
                  answered_q    <= taken_q;
                end
              end
              12'h024: sticky_q <= sticky_q & ~a_wdata_q[27:16];
              // RV_BOOT is taken when RV_RUN rises (and again after a debug ndmreset): a write while
              // RV_RUN = 1 is ignored, so the bus that crosses to clk_rv is quasi-static (CC §11.2,
              // W4-A review). The simulator ignores it too (tsr_sim.c csr_write).
              12'h028: if (!ctrl_q[C_RV_RUN]) boot_q <= a_wdata_q[31:8];
              12'h034: wdog_q   <= a_wdata_q[4:0];
              12'h060: begin                    // LQ_PUSH
                if (avail >= 32'(LQD)) sticky_q[S_LQ_OVERFLOW] <= 1'b1;
                else begin
                  for (int i = 0; i < 8; i++) lq_ring[pushed_q[$clog2(LQD)-1:0]][i] <= lq_w_q[i];
                  pushed_q <= pushed_q + 32'd1;
                end
              end
              12'h06C: begin                    // LQ_CANCEL (EX §4.9): hard cancel
                cancel_q   <= a_wdata_q;
                cancel_v_q <= 1'b1;
                if (active && ((a_wdata_q - answered_q) < 32'h8000_0000)) qabort_q <= 1'b1;
              end
              12'h070: begin
                stop_q   <= a_wdata_q;
                stop_v_q <= 1'b1;
              end
              12'h078: begin                    // LCQ_RD (EX §4.2 write rule)
                if ((a_wdata_q - lcq_rd_q) <= lcq_used) lcq_rd_q <= a_wdata_q;
                else sticky_q[S_RING_RD_BAD] <= 1'b1;
              end
              12'h094: begin
                if ((a_wdata_q - log_rd_q) <= (log_wr_q - log_rd_q)) log_rd_q <= a_wdata_q;
                else sticky_q[S_RING_RD_BAD] <= 1'b1;
              end
              12'h098: log_cfg_q <= {a_wdata_q[7:4], 2'b00, a_wdata_q[1:0]};
              12'h0C0: begin
                // A write restarts LOAD_CRC, which VERIFY is folding words into: refused while
                // VERIFY_BUSY (like LOAD_DATA and a second VERIFY), else the CRC would cover a suffix
                // of the segment and nothing would flag it (W4-A review). LOAD_COUNT is only sampled
                // when VERIFY starts, so it stays writable.
                if (vf_busy_q) sticky_q[S_LOAD_REFUSED] <= 1'b1;
                else begin
                  load_addr_q <= {a_wdata_q[31:2], 2'b00};
                  load_crc_q  <= '1;
                end
              end
              12'h0CC: load_count_q <= a_wdata_q;
              12'h0D0: begin                    // LOAD_CTRL.VERIFY
                if (a_wdata_q[0]) begin
                  if (load_refused || vf_busy_q) sticky_q[S_LOAD_REFUSED] <= 1'b1;
                  else begin
                    vf_busy_q  <= (load_count_q != 32'd0);
                    vf_addr_q  <= load_addr_q;
                    vf_left_q  <= load_count_q;
                    load_crc_q <= '1;
                  end
                end
              end
              default: if (in_lqw) lq_w_q[A[4:2]] <= a_wdata_q;
            endcase
          end
        end
        A_LDW: begin                            // LOAD_DATA write: post to the loader port
          a_wait_q <= a_wait_q + 16'd1;
          if (ld_ready && !ld_out_q) begin
            ld_req_q    <= 1'b1;
            ld_we_q     <= 1'b1;
            ld_addr_q   <= load_addr_q;
            ld_wdata_q  <= a_wdata_q;
            ld_out_q    <= 1'b1;
            ld_rd_q     <= 1'b0;
            load_crc_q  <= crc_word(load_crc_q, a_wdata_q);
            load_addr_q <= load_addr_q + 32'd4;
            ast_q       <= A_RSP;
          end else if (32'(a_wait_q) >= LD_WAIT_MAX) begin
            sticky_q[S_LOAD_TIMEOUT] <= 1'b1;     // dropped
            ast_q <= A_RSP;
          end
        end
        A_LDR: begin                            // LOAD_DATA read
          if (!a_sent_q) begin
            a_wait_q <= a_wait_q + 16'd1;
            if (ld_ready && !ld_out_q) begin
              ld_req_q  <= 1'b1;
              ld_we_q   <= 1'b0;
              ld_addr_q <= load_addr_q;
              ld_out_q  <= 1'b1;
              ld_rd_q   <= 1'b1;
              a_sent_q  <= 1'b1;
            end else if (32'(a_wait_q) >= LD_WAIT_MAX) begin
              sticky_q[S_LOAD_TIMEOUT] <= 1'b1;
              ast_q <= A_RSP;
            end
          end else if (ld_rsp_valid) begin
            a_rdata_q <= (ld_rsp_err || ld_rsp_timeout) ? 32'd0 : ld_rsp_rdata;
            if (ld_rsp_timeout) sticky_q[S_LOAD_TIMEOUT] <= 1'b1;
            ast_q <= A_RSP;
          end
        end
        A_DMI: begin
          if (!a_sent_q) begin
            if (dmi_ready) begin
              dmi_req_q <= 1'b1;
              a_sent_q  <= 1'b1;
            end
          end else if (dmi_rsp_valid) begin
            a_rdata_q <= a_we_q ? 32'd0 : dmi_rsp_rdata;
            if (dmi_rsp_timeout) sticky_q[S_DMI_TIMEOUT] <= 1'b1;
            ast_q <= A_RSP;
          end
        end
        A_DBG: begin                            // dbg_rdata is valid the cycle after dbg_req
          a_sent_q <= 1'b1;
          if (a_sent_q) begin
            a_rdata_q <= a_we_q ? 32'd0 : dbg_rdata;
            ast_q     <= A_RSP;
          end
        end
        A_RSP: begin
          if (a_host_q) csr_ready_q <= 1'b1;
          else          cw_rsp_q    <= 1'b1;
          ast_q <= A_IDLE;
        end
        default: ast_q <= A_IDLE;
      endcase

      // ---- loader port: responses, VERIFY sequencer ----
      if (ld_rsp_valid && ld_out_q) begin
        ld_out_q <= 1'b0;
        if (!ld_rd_q && ld_rsp_timeout) sticky_q[S_LOAD_TIMEOUT] <= 1'b1;   // a posted write timed out
        if (vf_out_q) begin
          vf_out_q <= 1'b0;
          if (ld_rsp_timeout) begin
            sticky_q[S_LOAD_TIMEOUT] <= 1'b1;
            vf_busy_q <= 1'b0;                  // VERIFY stops; the CRC is then wrong
          end else begin
            load_crc_q <= crc_word(load_crc_q, ld_rsp_err ? 32'd0 : ld_rsp_rdata);
            if (vf_left_q == 32'd0) vf_busy_q <= 1'b0;
          end
        end
      end
      if (vf_busy_q && !vf_out_q && !ld_out_q && ast_q != A_LDW && ast_q != A_LDR) begin
        if (ctrl_q[C_RV_RUN] || sr_busy) begin
          vf_busy_q <= 1'b0;                    // the hart leaves reset: VERIFY stops
        end else if (vf_left_q != 32'd0 && ld_ready) begin
          ld_req_q  <= 1'b1;
          ld_we_q   <= 1'b0;
          ld_addr_q <= vf_addr_q;
          ld_out_q  <= 1'b1;
          ld_rd_q   <= 1'b0;
          vf_out_q  <= 1'b1;
          vf_addr_q <= vf_addr_q + 32'd4;
          vf_left_q <= vf_left_q - 32'd1;
        end
      end

      // ---- ndmreset rising: the RISC-V reset actions of the core (CC §12.2 step 1) ----
      if (rv_ndmreset && !ndm_q) begin
        qabort_q   <= 1'b1;
        rv_reset_q <= 1'b1;
        answered_q <= taken_q;
      end

      // ---- SOFT_RESET steps 1-4 ----
      case (sr_q)
        SR_RV: begin
          sr_cnt_q <= sr_cnt_q + 17'd1;
          if (32'(sr_cnt_q) >= SR_RV_MIN && rv_reset_done) begin
            sr_q <= SR_DRAIN;
          end else if (32'(sr_cnt_q) >= SR_RV_MAX) begin
            sr_to_q <= 1'b1;
            sticky_q[S_SR_RV_TIMEOUT] <= 1'b1;
            sr_q <= SR_DRAIN;
          end
        end
        SR_DRAIN: if (qidle && !qabort_q) sr_q <= SR_CLR;
        SR_CLR: begin
          taken_q       <= pushed_q;
          answered_q    <= pushed_q;
          lcq_rd_q      <= lcq_wr_q + 32'(lq_post && post_ok);
          log_rd_q      <= log_wr_q + 32'(log_valid && ((log_wr_q - log_rd_q) < 32'(LOGD)));
          cancel_v_q    <= 1'b0;
          stop_v_q      <= 1'b0;
          host_clear_q  <= 1'b1;
          snap_v_q      <= 1'b0;
          sticky_q      <= '0;
          sticky_q[S_SR_RV_TIMEOUT] <= sr_to_q;
          log_dropped_q <= '0;
          sr_q          <= SR_IDLE;
          sr_busy       <= 1'b0;
        end
        default: ;
      endcase
      if (wdog_hit != '0) sticky_q[S_UNIT_HUNG] <= 1'b1;
    end
  end

  // loader / DMI / DBG request outputs
  assign ld_req    = ld_req_q;
  assign ld_we     = ld_we_q;
  assign ld_addr   = ld_addr_q;
  assign ld_wdata  = ld_wdata_q;
  assign dmi_req   = dmi_req_q;
  assign dmi_addr  = a_addr_q[8:2];
  assign dmi_op    = a_we_q ? 2'd2 : 2'd1;
  assign dmi_wdata = a_wdata_q;
  assign dbg_req   = dbg_req_q;
  assign dbg_we    = a_we_q;
  assign dbg_addr  = a_addr_q;
  assign dbg_wdata = a_wdata_q;
  assign dbg_host  = a_host_q;

  // -------------------------------------------------------------------------------------------
  // PERF counters (EX §6)
  // -------------------------------------------------------------------------------------------
  logic [31:0] rvc_now [3];                // the clk_rv counts of X + 7, 8, 9
  assign rvc_now[0] = perf_push_stall;
  assign rvc_now[1] = perf_bridge_acc;
  assign rvc_now[2] = perf_bridge_wait;

  logic [63:0][3:0] p_int;                  // internal per-slot increments
  always_comb begin
    p_int = '0;
    p_int[0] = 4'd1;
    p_int[1] = 4'(perf_retired);
    p_int[2] = 4'(perf_busy[2]);
    p_int[3] = 4'(perf_busy[3]);
    for (int c = 0; c < NUM_CH; c++) begin
      p_int[5 + c]          = 4'(perf_ch_rd_beat[c]);
      p_int[5 + NUM_CH + c] = 4'(perf_ch_wait[c]);
    end
    p_int[X + 0]  = 4'(perf_busy[0]);
    p_int[X + 1]  = 4'(perf_busy[1]);
    p_int[X + 2]  = 4'(perf_busy[4]);
    p_int[X + 3]  = 4'(perf_mvm_wait);
    p_int[X + 4]  = 4'(perf_mvm_empty);
    p_int[X + 5]  = 4'(perf_hold);
    p_int[X + 6]  = 4'(perf_push);
    p_int[X + 11] = 4'(lq_post && post_ok && lq_req_wdata[0]);
    p_int[X + 12] = 4'(ctrl_q[C_RV_RUN] && rv_sleep);
    p_int[63]     = 4'(ctrl_q[C_RV_RUN] && rv_halted || db_stop_active);
  end

  for (genvar i = 0; i < 64; i++) begin : g_perf
    localparam logic [2:0] CL = pclass(i);
    if (CL == PC_NONE) begin : g_none
      assign perf_q[i] = 64'd0;
      wire unused_p = &{1'b0, perf_ext[i], p_int[i]};
    end else if (CL == PC_RV) begin : g_rv
      localparam int unsigned K = i - X - 7;
      logic [31:0] last_q;                  // the count at the previous cycle: add the delta
      wire  [31:0] rv_delta = rvc_now[K] - last_q;   // modulo 2^32
      if (FAST_CLK) begin : g_pipe
        (* keep = "true" *) logic clr_q;
        logic [31:0] delta_q;
        always_ff @(posedge clk) begin
          clr_q <= !rst_n || (sr_q == SR_CLR);
          if (!rst_n) last_q <= '0;
          else        last_q <= rvc_now[K];
          delta_q <= rv_delta;
        end
        always_ff @(posedge clk) begin
          if (clr_q) perf_q[i] <= '0;
          else       perf_q[i] <= perf_q[i] + {32'd0, delta_q};
        end
      end else begin : g_flat
        always_ff @(posedge clk) begin
          if (!rst_n) begin
            perf_q[i] <= '0;
            last_q    <= '0;
          end else begin
            last_q <= rvc_now[K];
            if (sr_q == SR_CLR) perf_q[i] <= '0;
            else                perf_q[i] <= perf_q[i] + {32'd0, rv_delta};
          end
        end
      end
      wire unused_p = &{1'b0, perf_ext[i], p_int[i]};
    end else begin : g_cnt
      wire en = (CL == PC_TIME) ? (active && !qhalt) : active;
      wire [4:0] inc = 5'(p_int[i]) + 5'(perf_ext[i]);
      if (FAST_CLK) begin : g_pipe
        // a register copy of the clear, the enable and the increment per counter: 64 loads each instead of 4,096 / 1,345
        (* keep = "true" *) logic clr_q;
        (* keep = "true" *) logic en_q;
        logic [4:0] inc_q;
        always_ff @(posedge clk) begin
          clr_q <= !rst_n || (sr_q == SR_CLR);
          en_q  <= en;
          inc_q <= inc;
        end
        always_ff @(posedge clk) begin
          if (clr_q)     perf_q[i] <= '0;
          else if (en_q) perf_q[i] <= perf_q[i] + 64'(inc_q);
        end
      end else begin : g_flat
        always_ff @(posedge clk) begin
          if (!rst_n)               perf_q[i] <= '0;
          else if (sr_q == SR_CLR)  perf_q[i] <= '0;
          else if (en)              perf_q[i] <= perf_q[i] + 64'(inc);
        end
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // MTIME, RV_FREQ, watchdog
  // -------------------------------------------------------------------------------------------
  logic [11:0] fw_q;
  logic [3:0]  fn_q;
  logic [15:0] cyc_last_q;
  logic [31:0] facc_q;
  logic [31:0] wd_q [5];
  wire  [15:0] cyc_delta = rv_cyc - cyc_last_q;      // clk_rv cycles in this window, modulo 2^16
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      mtime_q    <= '0;
      fw_q       <= '0;
      fn_q       <= '0;
      cyc_last_q <= '0;
      facc_q     <= '0;
      rv_freq_q  <= '0;
      hung_q     <= '0;
      for (int u = 0; u < 5; u++) wd_q[u] <= '0;
    end else begin
      mtime_q <= mtime_q + 64'd1;
      fw_q    <= fw_q + 12'd1;
      if (32'(fw_q) == FREQ_WIN - 1) begin
        fw_q       <= '0;
        cyc_last_q <= rv_cyc;
        fn_q       <= fn_q + 4'd1;
        if (fn_q == 4'd15) begin
          rv_freq_q <= facc_q + {16'd0, cyc_delta};
          facc_q    <= '0;
        end else begin
          facc_q <= facc_q + {16'd0, cyc_delta};
        end
      end
      for (int u = 0; u < 5; u++) begin
        if (!perf_busy[u] || unit_progress[u] || wdog_q == 5'd0) begin
          wd_q[u] <= '0;
          if (unit_progress[u] || !perf_busy[u]) hung_q[u] <= 1'b0;
        end else if (!wd_q[u][wdog_q]) begin
          wd_q[u] <= wd_q[u] + 32'd1;
        end
        if (wdog_hit[u]) hung_q[u] <= 1'b1;
      end
    end
  end
  always_comb begin
    wdog_hit = '0;
    for (int u = 0; u < 5; u++)
      wdog_hit[u] = perf_busy[u] && !unit_progress[u] && wdog_q != 5'd0 && wd_q[u][wdog_q] && !hung_q[u];
  end

  // -------------------------------------------------------------------------------------------
  // Outputs
  // -------------------------------------------------------------------------------------------
  assign csr_ready      = csr_ready_q;
  assign csr_rdata      = csr_ready_q ? a_rdata_q : 32'd0;
  assign cwin_rsp_valid = cw_rsp_q;
  assign cwin_rsp_rdata = a_rdata_q;
  assign lq_rsp_valid   = lq_rsp_q;
  assign lq_rsp_rdata   = lq_rdata_q;
  assign lq_rsp_fault   = lq_fault_q;
  assign log_ready      = 1'b1;
  assign mtime          = mtime_q;
  assign rv_run         = ctrl_q[C_RV_RUN] && !ld_out_q && !vf_busy_q;
  assign rv_boot        = {boot_q, 8'd0};
  assign irq_lq         = (avail != 32'd0) && !active;
  assign irq_lcq        = (lcq_free != 32'd0);
  assign perf_time_en   = active && !qhalt;
  assign perf_evt_en    = active;
  assign host_hold      = ctrl_q[C_HOST_HOLD];
  assign halt_hold      = ctrl_q[C_RV_RUN] && rv_halted && ctrl_q[C_HOLD_ON_HALT];
  assign qabort         = qabort_q;
  assign rv_reset       = rv_reset_q;
  assign host_clear     = host_clear_q;
  assign sr_db_clear    = sr_db_clear_q;
  assign host_irq_o     = ctrl_q[C_IRQ_EN] && (lcq_wr_q != lcq_rd_q);
  assign unit_hung      = sticky_q[S_UNIT_HUNG];

  wire unused_ok = &{1'b0, dmi_rsp_resp, perf_busy[5], unit_progress[5], hung_q[5], lq_req_off[1:0],
                     lq_req_wdata[31:1]};

`ifndef SYNTHESIS
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      assert ((taken_q - answered_q) <= 32'd1) else $error("tsr_csr: TAKEN - ANSWERED = %0d (EX §4.2)", taken_q - answered_q);
      assert ((pushed_q - taken_q) <= 32'(LQD)) else $error("tsr_csr: PUSHED - TAKEN above LQ_DEPTH");
      assert ((lcq_wr_q - lcq_rd_q) <= 32'(LCQD)) else $error("tsr_csr: LCQ_WR - LCQ_RD above LCQ_DEPTH");
      assert ((log_wr_q - log_rd_q) <= 32'(LOGD)) else $error("tsr_csr: LOG_WR - LOG_RD above LOG_DEPTH");
      assert (!(ld_rsp_valid && !ld_out_q)) else $error("tsr_csr: loader response without a request");
      assert (!(ld_req_q && !ld_ready)) else $error("tsr_csr: loader request while the port is not ready");
    end
  end
`endif

endmodule
