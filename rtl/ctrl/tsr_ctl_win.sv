// tsr_ctl_win.sv — control window, core window, SPM windows and line buffer of the RISC-V
// (CC §5, §7.1-§7.5, §10.5, §12.3; EX §4.5 for 0x200-0x2FC). clk_rv domain.
//
// Spec: CC §2.2 `tsr_ctl_win`, `tsr_spm_lb`. The simulator's model of the
// same registers is sw/sim/tsr_q.c (q_cw_read / q_cw_write / q_push / q_qctrl / q_trace_mark),
// sw/sim/tsr_sim.c (tsr_sim_lq_cw_access) and sw/sim/tsr_bus.c (windows, line buffer); tb/ctrl
// compares this block with them.
//
// Requests come from tsr_rv_bus one at a time (w_req pulse, the access already decoded and checked:
// word accesses only for the control and core windows, no core-window store, no SBA write to
// either window); each gets exactly one answer (w_rsp pulse, registered).
//
// Control window (offsets from 0x2000_0000; reads have no side effects; undecoded offsets read 0
// and ignore writes, as tsr_q.c):
//   0x000-0x01C ISSUE_W0-7, 0x020 / 0x024 ISSUE_DEP0 / 1, 0x028 ISSUE_PUSH (CC §7.1): unit u =
//     UNIT_OF(w0[7:0]) when the opcode is legal, its CAPS bits are built and unit u is built, else
//     SYNC (CC §7.4). The push waits while ISSUE_CREDIT[u] = 0 (the credit includes a free FIFO
//     entry); the wait counts clk_rv cycles except while dbg_stop = 1 (CC §7.3, DB §3.9) and after
//     ISSUE_WAIT_MAX cycles ends with a store access fault and PUSH_TIMEOUT (ISSUE_STATUS bit0, the
//     core's sticky STATUS bit by push_timeout_ev). A push that goes writes the issue-FIFO entry
//     {OP, w0..w7 with WMASK applied, DEP0, DEP1, REC, ticket {u, ISSUED_SEQ[u]}, push PC} (CC §7.2
//     layout, TSR_IFIFO_* of tsr_tables_pkg), ISSUED_SEQ[u] += 1, ISSUE_PIDX += 1,
//     ISSUE_TICKET := ticket, DEP0 / DEP1 := TICKET_NONE.
//   0x02C ISSUE_TICKET, 0x030 ISSUE_CREDIT (unit u at [4u+3:4u]: min(Q_DEPTH[u] -
//     (ISSUED_SEQ[u] - S_sync[u]), FIFO free entries, 15), 0 for an absent unit), 0x034 ISSUE_UNITS,
//     0x038 ISSUE_STATUS (W1C; bit1 ERR_IRQ set on a rising "some unit in error state"), 0x03C
//     ISSUE_PIDX, 0x040 + 4u DONE_SEQ (the synchronized D[u]), 0x080 + 4u ISSUED_SEQ, 0x0C0 + 16u
//     ERR_SEQ / ERR_CODE / ERR_INFO / ERR_W0 and 0x180 + 4u ERR_PIDX (the records received from the
//     core, CC §11.2), 0x140 QCTRL (each set bit of [3:0] one CTRL entry, in bit order, all or
//     nothing: waits until the FIFO has room for all of them and, for QCLEAR, a SYNC credit; QCLEAR
//     is ticketed like a push), 0x144 QSTATUS, 0x148 WAIT_TICKET, 0x14C WAIT_CREDIT,
//     0x178 TRACE_MARK (a CTRL entry with w1 = the value, DB §8.4), 0x1C4 SPM_LB_INV.
//   0x160-0x17C (except 0x178) and 0x400-0x7FC are DB's (DB §10.2): forwarded to the db_* port
//     (read data one cycle after db_req; tied to 0 until the DB block exists).
//   0x200-0x2FC (EX §4.5, CC-30): LQ_HEAD_W0-7, LQ_AVAIL, LQ_HEAD_SEQ, LQ_CANCEL, LQ_STOP, LQ_FLAGS,
//     LCQ_FREE, LOG_CFG and LOG_FREE are read by a handshake into the core domain (the core's value
//     at its answer); LQ_POP and LCQ_POST (with the eight staged words) commit and fault there;
//     LCQ_W0-7, RT_STATE and RT_INFO are clk_rv registers (RT_* mirrored to the core by rt_wr);
//     LOG_PUSH goes into the log CDC FIFO (a full FIFO stalls the store, at most LOG_WAIT_MAX
//     cycles, then the word is dropped: LOG_PUSH never faults); LOG_FREE = LOG_DEPTH - (LOG_WR_rv -
//     r), clamped at 0, where LOG_WR_rv counts the words this side put into the FIFO and r is the
//     core's answer (LOG_RD plus the words the ring dropped, see tsr_ctrl.sv "interfaces").
//   0x300 / 0x304 MTIME (the synchronized core cycle count), 0x308 / 0x30C MTIMECMP.
// Core window (0x2000_1000 + CSR offset, CC §5.3): a handshake read of tsr_csr's second requester;
//   0x0C4 (LOAD_DATA) and 0xC00-0xDFC (DMI) read 0 without a request.
// SPM windows (CC §10.5): exact window = one bridge transaction per access (read: the whole line,
//   the word picked here; write: the word with its byte enables). Line-buffered alias: a hart load
//   hits the 512-bit line buffer when its tag matches, else fills it with one bridge read. The
//   line buffer is invalidated by any store to either SPM alias (hart or SBA), any hart access to
//   the control or core window, a rising edge of any interrupt line, SPM_LB_INV (a control-window
//   access) and the RISC-V reset. SBA reads of either alias go to the SPM and leave the line buffer
//   alone (as tsr_bus.c).
// Bounds: every handshake (bridge, core window, LQ / LCQ) ends after the hs TIMEOUT (4,096 clk_rv
//   cycles, in tsr_ctrl_cdc) with an access fault and BRIDGE_TIMEOUT (bridge_timeout_ev).
// RISC-V reset (CC §12.2): rv_kill answers a waiting push / QCTRL / TRACE_MARK with an error
//   without writing the FIFO (no entry, no ticket, no ISSUE_PIDX step); hart_clr (held >= 3 cycles)
//   resets the staging words, DEP0 / DEP1, WAIT_TICKET, WAIT_CREDIT, ISSUE_STATUS, MTIMECMP,
//   LCQ_W0-7, RT_STATE / RT_INFO and the line buffer (tsr_q_rv_reset and rv_reset_seq of the
//   simulator). ISSUED_SEQ, ISSUE_PIDX and ISSUE_TICKET are reset by the hardware reset only.
// Interrupts (CC §12.3): irq_fast[0] WAIT_TICKET done, [1] ERR_IRQ, [2] LQ level, [3] LCQ level,
//   [4] / [6] DB, [5] WAIT_CREDIT, irq_timer MTIME >= MTIMECMP; others 0. Registered.
// PERF (EX §6, clk_rv counters): PUSH_STALL (cycles a push / QCTRL / TRACE_MARK waited, gated by
//   perf_time_en), BRIDGE_ACC (hart bridge transactions incl. line-buffer fills, gated by
//   perf_evt_en), BRIDGE_WAIT (cycles they waited, perf_time_en). 32-bit, crossing by gray code.

module tsr_ctl_win
  import tsr_tables_pkg::*;
  import tsr_ctrl_pkg::*;
#(
  parameter logic [31:0]     CAPS           = TSR_CAPS_VU | TSR_CAPS_ATT,
  parameter logic [5:0][4:0] Q_DEPTH        = {5'd4, 5'd0, 5'd16, 5'd4, 5'd4, 5'd8},  // [u], CC §7.4
  parameter int unsigned     SPM_BYTES      = 4 * 1024 * 1024,
  parameter int unsigned     FIFO_DEPTH     = TSR_ISSUE_FIFO_DEPTH,
  parameter int unsigned     LOG_DEPTH      = TSR_LOG_DEPTH,
  parameter int unsigned     LOG_FIFO_DEPTH = 16,
  parameter int unsigned     ISSUE_WAIT_MAX = TSR_ISSUE_WAIT_MAX,
  parameter int unsigned     LOG_WAIT_MAX   = 4096,
  localparam int unsigned    FAW            = $clog2(FIFO_DEPTH),
  localparam int unsigned    LAW            = $clog2(LOG_FIFO_DEPTH)
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         hart_clr,
  input  logic                         rv_kill,
  // window port (tsr_rv_bus)
  input  logic                         w_req,
  input  master_e                      w_master,
  input  region_e                      w_region,
  input  logic                         w_we,
  input  logic [3:0]                   w_be,
  input  logic [31:0]                  w_addr,
  input  logic [31:0]                  w_wdata,
  input  logic [31:0]                  w_pc,
  output logic                         w_rsp,
  output logic [31:0]                  w_rdata,
  output logic                         w_err,
  // issue FIFO, write side
  output logic                         if_wr,
  output logic [TSR_IFIFO_BITS-1:0]    if_wdata,
  input  logic [FAW:0]                 if_free,
  input  logic                         if_empty,
  // sequences and core state (synchronized copies)
  output logic [5:0][27:0]             issued_seq,
  input  logic [5:0][27:0]             s_seq,
  input  logic [5:0][27:0]             d_seq,
  input  err_rec_t [5:0]               err_rec,
  input  logic [3:0]                   q_hold,      // QSTATUS [4:1]: {debug-halt, RISC-V, host, any}
  input  logic [5:0]                   q_err,
  input  logic [5:0]                   q_nonempty,
  input  logic [63:0]                  mtime,
  input  logic                         irq_lq,
  input  logic                         irq_lcq,
  input  logic                         dbg_stop,
  input  logic                         perf_time_en,
  input  logic                         perf_evt_en,
  // DB window (tie db_rdata / db_irq to 0 until then)
  output logic                         db_req,
  output logic                         db_we,
  output logic [11:0]                  db_off,
  output logic [31:0]                  db_wdata,
  input  logic [31:0]                  db_rdata,
  input  logic                         db_irq4,
  input  logic                         db_irq6,
  // bridge / core-window handshake (A side)
  input  logic                         x_ready,
  output logic                         x_req,
  output xreq_t                        x_req_data,
  input  logic                         x_done,
  input  logic                         x_timeout,
  input  logic [XRSP_W-1:0]            x_rsp,
  // LQ / LCQ handshake (A side)
  input  logic                         lq_ready,
  output logic                         lq_req,
  output lqreq_t                       lq_req_data,
  input  logic                         lq_done,
  input  logic                         lq_timeout,
  input  logic [LQRSP_W-1:0]           lq_rsp,
  // log FIFO, write side
  output logic                         log_wr,
  output logic [31:0]                  log_wdata,
  input  logic [LAW:0]                 log_free,
  // RT_* mirror source
  output logic [31:0]                  rt_state,
  output logic [31:0]                  rt_info,
  output logic                         rt_wr,
  // events, PERF, interrupts
  output logic                         push_timeout_ev,
  output logic                         bridge_timeout_ev,
  output logic [31:0]                  perf_push_stall,
  output logic [31:0]                  perf_bridge_acc,
  output logic [31:0]                  perf_bridge_wait,
  output logic                         irq_timer,
  output logic [14:0]                  irq_fast
);

  localparam logic [31:0] TICKET_NONE = TSR_TICKET_NONE;
  localparam logic [2:0]  USYNC       = 3'(TSR_UNIT_SYNC);
  localparam int unsigned SPM_AW      = $clog2(SPM_BYTES);
  localparam int unsigned NLINE_W     = SPM_AW - 6;

  if (FIFO_DEPTH > 32 || SPM_BYTES > (1 << 26) || (SPM_BYTES & (SPM_BYTES - 1)) != 0 && SPM_BYTES > (1 << 25))
  begin : g_chk
    $fatal(1, "tsr_ctl_win: FIFO_DEPTH <= 32, SPM_BYTES <= 64 MiB");
  end

  // ------------------------------------------------------------------ unit map (CC §7.4)
  function automatic logic built(input int unsigned u);
    return u < 6 && Q_DEPTH[u] != 5'd0;
  endfunction

  function automatic logic [2:0] unit_of(input logic [7:0] op);
    logic [3:0]  u;
    logic [31:0] need;
    u    = tsr_op_unit(op);
    need = tsr_op_caps(op);
    if (!tsr_op_legal(op) || ((CAPS & need) != need) || u > 4'd5 || Q_DEPTH[u[2:0]] == 5'd0)
      return 3'(TSR_UNIT_SYNC);
    return u[2:0];
  endfunction

  // ------------------------------------------------------------------ registers
  logic [7:0][31:0] iss_w_q;
  logic [31:0]      dep0_q, dep1_q, ticket_q, pidx_q, wait_ticket_q, wait_credit_q;
  logic [1:0]       iss_status_q;
  logic [63:0]      mtimecmp_q;
  logic [7:0][31:0] lcq_w_q;
  logic [31:0]      rt_state_q, rt_info_q;
  logic [27:0]      issued_q [6];
  logic [31:0]      log_wr_q;
  logic             q_err_any_q;   // some unit in error, last cycle (ISSUE_STATUS.ERR_IRQ edge)

  for (genvar u = 0; u < 6; u++) begin : g_iss
    assign issued_seq[u] = issued_q[u];
  end
  assign rt_state = rt_state_q;
  assign rt_info  = rt_info_q;

  // Credits, registered (CC §7.3). A push changes ISSUED_SEQ one cycle before the next request can
  // be evaluated (the bus answers, then Ibex issues), so the registered credit is never stale for a
  // decision.
  logic [5:0][3:0] credit_q;
  always_ff @(posedge clk) begin
    for (int unsigned u = 0; u < 6; u++) begin
      logic [27:0] used;
      logic [31:0] c;
      used = issued_q[u] - s_seq[u];
      if (!built(u) || used >= 28'(Q_DEPTH[u])) c = 0;
      else c = 32'(Q_DEPTH[u]) - 32'(used);
      if (c > 32'(if_free)) c = 32'(if_free);
      if (c > 15) c = 15;
      credit_q[u] <= rst_n ? c[3:0] : 4'd0;
    end
  end

  function automatic logic ticket_done(input logic [31:0] t);
    logic [3:0]  u;
    logic [27:0] d;
    u = t[31:28];
    if (u > 4'd5) return 1'b0;
    if (Q_DEPTH[u[2:0]] == 5'd0) return 1'b0;
    d = d_seq[u[2:0]] - t[27:0];
    return d >= 28'd1 && d < 28'h800_0000;
  endfunction

  logic [5:0] units_built;
  always_comb begin
    for (int unsigned u = 0; u < 6; u++) units_built[u] = built(u);
  end

  // QIDLE as the clk_rv side sees it (CC §5.1).
  logic qidle;
  always_comb begin
    qidle = if_empty;
    for (int unsigned u = 0; u < 6; u++)
      if (built(u) && issued_q[u] != d_seq[u]) qidle = 1'b0;
  end

  // ------------------------------------------------------------------ local control-window reads
  logic [11:0] off;
  assign off = w_addr[11:0];

  function automatic logic is_lq_hs_rd(input logic [11:0] o);
    return (o >= 12'h200 && o <= 12'h234 && o != TSR_CW_LQ_POP) || o == TSR_CW_LCQ_FREE ||
           o == TSR_CW_LOG_FREE || o == TSR_CW_LOG_CFG;
  endfunction
  function automatic logic is_db(input logic [11:0] o);
    return (o >= TSR_CW_REC_COUNT && o <= TSR_CW_REC_CFG && o != TSR_CW_TRACE_MARK) ||
           (o >= TSR_CW_DBG_FLAGS && o < 12'h800);
  endfunction

  logic [31:0] cw_rd;
  always_comb begin
    logic [2:0] u;
    cw_rd = '0;
    u = '0;
    if (off < 12'h020) begin
      cw_rd = iss_w_q[off[4:2]];
    end else if (off >= TSR_CW_DONE_SEQ_BASE && off < 12'h060) begin
      u = off[4:2];
      cw_rd = (u < 3'd6 && built(int'(u))) ? {4'h0, d_seq[u]} : '0;
    end else if (off >= TSR_CW_ISSUED_SEQ_BASE && off < 12'h0A0) begin
      u = off[4:2];
      cw_rd = (u < 3'd6 && built(int'(u))) ? {4'h0, issued_q[u]} : '0;
    end else if (off >= TSR_CW_ERR_SEQ_BASE && off < TSR_CW_QCTRL) begin
      u = 3'((off - TSR_CW_ERR_SEQ_BASE) >> 4);
      if (u < 3'd6 && built(int'(u))) begin
        unique case (off[3:2])
          2'd0:    cw_rd = {4'h0, err_rec[u].seq};
          2'd1:    cw_rd = {28'h0, err_rec[u].code};
          2'd2:    cw_rd = err_rec[u].info;
          default: cw_rd = err_rec[u].w0;
        endcase
      end
    end else if (off >= TSR_CW_ERR_PIDX_BASE && off < 12'h1A0) begin
      u = off[4:2];
      cw_rd = (u < 3'd6 && built(int'(u))) ? err_rec[u].pidx : '0;
    end else if (off >= TSR_CW_LCQ_W_BASE && off < TSR_CW_LCQ_POST) begin
      cw_rd = lcq_w_q[off[4:2]];
    end else begin
      unique case (off)
        TSR_CW_ISSUE_DEP0:   cw_rd = dep0_q;
        TSR_CW_ISSUE_DEP1:   cw_rd = dep1_q;
        TSR_CW_ISSUE_TICKET: cw_rd = ticket_q;
        TSR_CW_ISSUE_CREDIT: cw_rd = {8'h0, credit_q};
        TSR_CW_ISSUE_UNITS:  cw_rd = {26'h0, units_built};
        TSR_CW_ISSUE_STATUS: cw_rd = {30'h0, iss_status_q};
        TSR_CW_ISSUE_PIDX:   cw_rd = pidx_q;
        TSR_CW_QSTATUS:      cw_rd = {10'h0, q_nonempty, 2'b00, q_err, 3'b000, q_hold, qidle};
        TSR_CW_WAIT_TICKET:  cw_rd = wait_ticket_q;
        TSR_CW_WAIT_CREDIT:  cw_rd = wait_credit_q;
        TSR_CW_RT_STATE:     cw_rd = rt_state_q;
        TSR_CW_RT_INFO:      cw_rd = rt_info_q;
        TSR_CW_MTIME_LO:     cw_rd = mtime[31:0];
        TSR_CW_MTIME_HI:     cw_rd = mtime[63:32];
        TSR_CW_MTIMECMP_LO:  cw_rd = mtimecmp_q[31:0];
        TSR_CW_MTIMECMP_HI:  cw_rd = mtimecmp_q[63:32];
        default:             cw_rd = '0;
      endcase
    end
  end

  // ------------------------------------------------------------------ request classification
  logic hart;
  assign hart = w_master == MS_IBEX;

  logic [3:0] qbits;
  logic [2:0] qn;
  assign qbits = w_wdata[3:0];
  assign qn    = 3'(qbits[0]) + 3'(qbits[1]) + 3'(qbits[2]) + 3'(qbits[3]);

  // ------------------------------------------------------------------ FSM
  typedef enum logic [3:0] {
    S_IDLE, S_ISSUE, S_EMIT, S_X, S_LQ, S_LOG, S_DB, S_DB2
  } st_e;

  // QCTRL bits left after the lowest one (the entry being emitted).
  function automatic logic [3:0] qrest(input logic [3:0] b);
    return b & (b - 4'd1);
  endfunction
  typedef enum logic [1:0] { IK_PUSH, IK_QCTRL, IK_TRACE } ik_e;

  st_e          st_q;
  ik_e          ik_q;
  logic [31:0]  ival_q;       // the store's value (push word, QCTRL bits, marker)
  logic [31:0]  ipc_q;
  logic [2:0]   iu_q;         // push unit
  logic [3:0]   qrem_q;       // QCTRL bits left to emit
  logic [16:0]  wait_q;       // wait count (ISSUE_WAIT_MAX <= 65,536 fits 17 bits)
  logic [12:0]  lwait_q;
  xkind_e       xk_q;
  logic         x_lb_q;       // the transaction fills the line buffer
  logic         x_hart_q;
  logic [3:0]   x_word_q;
  logic [11:0]  lq_off_q;

  logic         rsp_q, rsp_err_q;
  logic [31:0]  rsp_rdata_q;
  assign w_rsp   = rsp_q;
  assign w_rdata = rsp_rdata_q;
  assign w_err   = rsp_err_q;

  logic         lb_valid_q;
  logic [NLINE_W-1:0] lb_line_q;
  logic [511:0] lb_data_q;

  logic [2:0] push_u;
  assign push_u = unit_of(iss_w_q[0][7:0]);

  // Issue condition of the waiting store.
  logic can_issue;
  always_comb begin
    unique case (ik_q)
      IK_PUSH:  can_issue = credit_q[iu_q] != 4'd0;
      IK_QCTRL: can_issue = (32'(if_free) >= 32'(qrem_q[0]) + 32'(qrem_q[1]) + 32'(qrem_q[2]) + 32'(qrem_q[3])) &&
                            (!qrem_q[3] || credit_q[USYNC] != 4'd0);
      default:  can_issue = if_free != '0;
    endcase
  end

  // Issue-FIFO entry assembly.
  logic        emit;
  logic [2:0]  emit_code;   // CTRL code of the next QCTRL entry
  logic [31:0] e_w [8];
  logic [31:0] e_dep0, e_dep1, e_ticket;
  logic        e_kind_ctrl, e_rec;
  always_comb begin
    emit_code = qrem_q[0] ? 3'(TSR_IFIFO_CTRL_CODE_QABORT) :
                qrem_q[1] ? 3'(TSR_IFIFO_CTRL_CODE_QHOLD_SET) :
                qrem_q[2] ? 3'(TSR_IFIFO_CTRL_CODE_QHOLD_CLR) : 3'(TSR_IFIFO_CTRL_CODE_QCLEAR);
    for (int unsigned i = 0; i < 8; i++) e_w[i] = '0;
    e_dep0 = TICKET_NONE; e_dep1 = TICKET_NONE; e_ticket = TICKET_NONE; e_kind_ctrl = 1'b1; e_rec = 1'b0;
    unique case (ik_q)
      IK_PUSH: begin
        for (int unsigned i = 0; i < 8; i++)
          e_w[i] = (i == 0 || ival_q[TSR_ISSUE_PUSH_WMASK_LSB + i]) ? iss_w_q[i] : '0;
        e_dep0 = dep0_q; e_dep1 = dep1_q; e_kind_ctrl = 1'b0; e_rec = ival_q[0];
        e_ticket = {1'b0, iu_q, issued_q[iu_q]};
      end
      IK_QCTRL: begin
        e_w[0] = {29'h0, emit_code};
        if (emit_code == 3'(TSR_IFIFO_CTRL_CODE_QCLEAR)) e_ticket = {4'(TSR_UNIT_SYNC), issued_q[USYNC]};
      end
      default: begin
        e_w[0] = 32'(TSR_IFIFO_CTRL_CODE_TRACE_MARK);
        e_w[1] = ival_q;
      end
    endcase
  end

  always_comb begin
    if_wdata = '0;
    if_wdata[TSR_IFIFO_KIND_LSB +: TSR_IFIFO_KIND_W] = e_kind_ctrl ? TSR_IFIFO_KIND_CTRL : TSR_IFIFO_KIND_OP;
    for (int unsigned i = 0; i < 8; i++) if_wdata[TSR_IFIFO_W_LSB + 32*i +: 32] = e_w[i];
    if_wdata[TSR_IFIFO_DEP0_LSB +: 32]   = e_dep0;
    if_wdata[TSR_IFIFO_DEP1_LSB +: 32]   = e_dep1;
    if_wdata[TSR_IFIFO_REC_LSB]          = e_rec;
    if_wdata[TSR_IFIFO_TICKET_LSB +: 32] = e_ticket;
    if_wdata[TSR_IFIFO_PC_LSB +: 32]     = ipc_q;
  end
  assign if_wr = emit;

  // SPM window helpers.
  logic [SPM_AW-1:0] spm_off;
  assign spm_off = (w_region == RG_SPM) ? SPM_AW'(w_addr - TSR_RV_SPM_BASE) : SPM_AW'(w_addr - TSR_RV_SPM_LB_BASE);
  logic lb_hit;
  assign lb_hit = lb_valid_q && lb_line_q == spm_off[SPM_AW-1:6];

  // Interrupt lines (registered) and their rising edges (line-buffer invalidation, CC §10.5).
  logic [14:0] irq_fast_q;
  logic        irq_timer_q;
  logic        irq_rise;
  assign irq_fast  = irq_fast_q;
  assign irq_timer = irq_timer_q;

  logic [31:0] lq_r;
  assign lq_r = lq_rsp[31:0];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q            <= S_IDLE;
      rsp_q           <= 1'b0;
      iss_w_q         <= '0;
      dep0_q          <= TICKET_NONE;
      dep1_q          <= TICKET_NONE;
      ticket_q        <= TICKET_NONE;
      pidx_q          <= '0;
      wait_ticket_q   <= TSR_CW_WAIT_TICKET_RESET;
      wait_credit_q   <= TSR_CW_WAIT_CREDIT_RESET;
      iss_status_q    <= '0;
      mtimecmp_q      <= '1;
      lcq_w_q         <= '0;
      rt_state_q      <= '0;
      rt_info_q       <= '0;
      for (int unsigned u = 0; u < 6; u++) issued_q[u] <= '0;
      log_wr_q        <= '0;
      lb_valid_q      <= 1'b0;
      emit            <= 1'b0;
      x_req           <= 1'b0;
      lq_req          <= 1'b0;
      log_wr          <= 1'b0;
      db_req          <= 1'b0;
      rt_wr           <= 1'b0;
      push_timeout_ev <= 1'b0;
      bridge_timeout_ev <= 1'b0;
      perf_push_stall <= '0;
      perf_bridge_acc <= '0;
      perf_bridge_wait <= '0;
    end else begin
      rsp_q             <= 1'b0;
      rsp_err_q         <= 1'b0;
      emit              <= 1'b0;
      x_req             <= 1'b0;
      lq_req            <= 1'b0;
      log_wr            <= 1'b0;
      db_req            <= 1'b0;
      rt_wr             <= 1'b0;
      push_timeout_ev   <= 1'b0;
      bridge_timeout_ev <= 1'b0;

      // ERR_IRQ: rising "some unit in error state" (CC §5.1).
      if (|q_err && !q_err_any_q) iss_status_q[1] <= 1'b1;
      if (irq_rise) lb_valid_q <= 1'b0;

      // An entry written in the previous cycle (emit is registered, so the FIFO write and the
      // counter steps happen together at this edge).
      if (emit) begin
        if (ik_q == IK_PUSH) begin
          issued_q[iu_q] <= issued_q[iu_q] + 1'b1;
          pidx_q   <= pidx_q + 1'b1;
          ticket_q <= e_ticket;
          dep0_q   <= TICKET_NONE;
          dep1_q   <= TICKET_NONE;
        end else if (ik_q == IK_QCTRL && emit_code == 3'(TSR_IFIFO_CTRL_CODE_QCLEAR)) begin
          issued_q[USYNC] <= issued_q[USYNC] + 1'b1;
          pidx_q   <= pidx_q + 1'b1;
          ticket_q <= e_ticket;
        end
        if (ik_q == IK_QCTRL) qrem_q <= qrest(qrem_q);
      end

      unique case (st_q)
        S_IDLE: if (w_req) begin
          // Hart accesses to the control or core window invalidate the line buffer; so does any
          // store to an SPM alias (hart or SBA).
          if (hart && (w_region == RG_CW || w_region == RG_CORE)) lb_valid_q <= 1'b0;
          if ((w_region == RG_SPM || w_region == RG_SPMLB) && w_we) lb_valid_q <= 1'b0;
          unique case (w_region)
            RG_CW: begin
              if (!w_we) begin
                if (is_lq_hs_rd(off)) begin
                  lq_req <= 1'b1;
                  lq_req_data <= '{lcq_w: lcq_w_q, wdata: '0, we: 1'b0, off: off};
                  lq_off_q <= off;
                  st_q <= S_LQ;
                end else if (is_db(off)) begin
                  db_req <= 1'b1; db_we <= 1'b0; db_off <= off; db_wdata <= '0;
                  st_q <= S_DB;
                end else begin
                  rsp_q <= 1'b1; rsp_rdata_q <= cw_rd;
                end
              end else begin
                rsp_rdata_q <= '0;
                if (off < 12'h020) begin
                  iss_w_q[off[4:2]] <= w_wdata; rsp_q <= 1'b1;
                end else if (off >= TSR_CW_LCQ_W_BASE && off < TSR_CW_LCQ_POST) begin
                  lcq_w_q[off[4:2]] <= w_wdata; rsp_q <= 1'b1;
                end else if (is_db(off)) begin
                  db_req <= 1'b1; db_we <= 1'b1; db_off <= off; db_wdata <= w_wdata;
                  st_q <= S_DB;
                end else begin
                  unique case (off)
                    TSR_CW_ISSUE_DEP0:   begin dep0_q <= w_wdata; rsp_q <= 1'b1; end
                    TSR_CW_ISSUE_DEP1:   begin dep1_q <= w_wdata; rsp_q <= 1'b1; end
                    TSR_CW_ISSUE_PUSH: begin
                      ik_q <= IK_PUSH; ival_q <= w_wdata; ipc_q <= w_pc; iu_q <= push_u;
                      wait_q <= '0; st_q <= S_ISSUE;
                    end
                    TSR_CW_QCTRL: begin
                      if (qn == 3'd0) begin
                        rsp_q <= 1'b1;
                      end else begin
                        ik_q <= IK_QCTRL; ival_q <= w_wdata; ipc_q <= w_pc; qrem_q <= qbits;
                        wait_q <= '0; st_q <= S_ISSUE;
                      end
                    end
                    TSR_CW_TRACE_MARK: begin
                      ik_q <= IK_TRACE; ival_q <= w_wdata; ipc_q <= w_pc; wait_q <= '0; st_q <= S_ISSUE;
                    end
                    TSR_CW_ISSUE_STATUS: begin iss_status_q <= iss_status_q & ~w_wdata[1:0]; rsp_q <= 1'b1; end
                    TSR_CW_WAIT_TICKET:  begin wait_ticket_q <= w_wdata; rsp_q <= 1'b1; end
                    TSR_CW_WAIT_CREDIT:  begin
                      wait_credit_q <= w_wdata & (TSR_WAIT_CREDIT_UNIT | TSR_WAIT_CREDIT_EN); rsp_q <= 1'b1;
                    end
                    TSR_CW_MTIMECMP_LO:  begin mtimecmp_q[31:0] <= w_wdata; rsp_q <= 1'b1; end
                    TSR_CW_MTIMECMP_HI:  begin mtimecmp_q[63:32] <= w_wdata; rsp_q <= 1'b1; end
                    TSR_CW_RT_STATE:     begin rt_state_q <= w_wdata; rt_wr <= 1'b1; rsp_q <= 1'b1; end
                    TSR_CW_RT_INFO:      begin rt_info_q <= w_wdata; rt_wr <= 1'b1; rsp_q <= 1'b1; end
                    TSR_CW_LQ_POP, TSR_CW_LCQ_POST: begin
                      lq_req <= 1'b1;
                      lq_req_data <= '{lcq_w: lcq_w_q, wdata: w_wdata, we: 1'b1, off: off};
                      lq_off_q <= off;
                      st_q <= S_LQ;
                    end
                    TSR_CW_LOG_PUSH: begin
                      ival_q <= w_wdata; lwait_q <= '0; st_q <= S_LOG;
                    end
                    default: rsp_q <= 1'b1;   // read-only, reserved, SPM_LB_INV: no state change
                  endcase
                end
              end
            end
            RG_CORE: begin
              if (off == TSR_CSR_LOAD_DATA || (off >= TSR_CSR_DMI_BASE && off < 12'hE00)) begin
                rsp_q <= 1'b1; rsp_rdata_q <= '0;
              end else begin
                x_req <= 1'b1;
                x_req_data <= '{kind: XK_CORE_RD, be: 4'hF, addr: 26'(off), wdata: '0};
                xk_q <= XK_CORE_RD; x_lb_q <= 1'b0; x_hart_q <= hart; x_word_q <= '0;
                st_q <= S_X;
              end
            end
            default: begin   // RG_SPM, RG_SPMLB
              if (!w_we && w_region == RG_SPMLB && hart && lb_hit) begin
                rsp_q <= 1'b1; rsp_rdata_q <= lb_data_q[32*spm_off[5:2] +: 32];
              end else begin
                x_req <= 1'b1;
                x_req_data <= '{kind: w_we ? XK_SPM_WR : XK_SPM_RD, be: w_be, addr: 26'(spm_off), wdata: w_wdata};
                xk_q <= w_we ? XK_SPM_WR : XK_SPM_RD;
                x_lb_q <= !w_we && w_region == RG_SPMLB && hart;
                x_hart_q <= hart;
                x_word_q <= spm_off[5:2];
                if (!w_we && w_region == RG_SPMLB && hart) begin
                  lb_line_q  <= spm_off[SPM_AW-1:6];
                  lb_valid_q <= 1'b0;          // a miss: refilled when the bridge answers
                end
                if (hart && perf_evt_en) perf_bridge_acc <= perf_bridge_acc + 1'b1;
                st_q <= S_X;
              end
            end
          endcase
        end

        S_ISSUE: begin
          if (rv_kill) begin
            rsp_q <= 1'b1; rsp_err_q <= 1'b1; st_q <= S_IDLE;
          end else if (can_issue) begin
            emit <= 1'b1;
            st_q <= (ik_q == IK_QCTRL) ? S_EMIT : S_IDLE;
            if (ik_q != IK_QCTRL) rsp_q <= 1'b1;
          end else begin
            if (perf_time_en) perf_push_stall <= perf_push_stall + 1'b1;
            if (!dbg_stop) begin
              if (32'(wait_q) + 1 >= ISSUE_WAIT_MAX) begin
                rsp_q <= 1'b1; rsp_err_q <= 1'b1;
                iss_status_q[0] <= 1'b1;
                push_timeout_ev <= 1'b1;
                st_q <= S_IDLE;
              end
              wait_q <= wait_q + 1'b1;
            end
          end
        end

        S_EMIT: begin
          // QCTRL: one entry per cycle (the bit emitted last cycle is cleared at this edge).
          if (qrest(qrem_q) == 4'd0) begin
            rsp_q <= 1'b1; st_q <= S_IDLE;
          end else begin
            emit <= 1'b1;
          end
        end

        S_X: begin
          if (xk_q != XK_CORE_RD && x_hart_q && perf_time_en) perf_bridge_wait <= perf_bridge_wait + 1'b1;
          if (x_done) begin
            rsp_q <= 1'b1;
            st_q  <= S_IDLE;
            if (x_timeout || x_rsp[512]) begin
              rsp_err_q <= 1'b1;
              rsp_rdata_q <= '0;
              if (x_timeout) bridge_timeout_ev <= 1'b1;
            end else begin
              rsp_rdata_q <= (xk_q == XK_CORE_RD) ? x_rsp[31:0] : (xk_q == XK_SPM_RD) ? x_rsp[32*x_word_q +: 32] : '0;
              if (x_lb_q) begin
                lb_data_q  <= x_rsp[511:0];
                lb_valid_q <= 1'b1;
              end
            end
          end
        end

        S_LQ: begin
          if (lq_done) begin
            rsp_q <= 1'b1;
            st_q  <= S_IDLE;
            if (lq_timeout || lq_rsp[32]) begin
              rsp_err_q <= 1'b1; rsp_rdata_q <= '0;
              if (lq_timeout) bridge_timeout_ev <= 1'b1;
            end else if (lq_off_q == TSR_CW_LOG_FREE) begin
              rsp_rdata_q <= (log_wr_q - lq_r > 32'(LOG_DEPTH)) ? '0 : 32'(LOG_DEPTH) - (log_wr_q - lq_r);
            end else begin
              rsp_rdata_q <= lq_req_data.we ? '0 : lq_r;
            end
          end
        end

        S_LOG: begin
          if (log_free != '0) begin
            log_wr <= 1'b1; log_wdata <= ival_q; log_wr_q <= log_wr_q + 1'b1;
            rsp_q <= 1'b1; rsp_rdata_q <= '0; st_q <= S_IDLE;
          end else if (32'(lwait_q) + 1 >= LOG_WAIT_MAX) begin
            rsp_q <= 1'b1; rsp_rdata_q <= '0; st_q <= S_IDLE;    // dropped, never faults (EX §4.5)
          end else begin
            lwait_q <= lwait_q + 1'b1;
          end
        end

        S_DB:  st_q <= S_DB2;   // db_rdata is valid in the cycle after db_req
        S_DB2: begin
          rsp_q <= 1'b1; rsp_rdata_q <= db_we ? '0 : db_rdata; st_q <= S_IDLE;
        end

        default: st_q <= S_IDLE;
      endcase

      // RISC-V reset, step 3 (CC §12.2; tsr_q_rv_reset, rv_reset_seq).
      if (hart_clr) begin
        iss_w_q       <= '0;
        dep0_q        <= TICKET_NONE;
        dep1_q        <= TICKET_NONE;
        wait_ticket_q <= TSR_CW_WAIT_TICKET_RESET;
        wait_credit_q <= TSR_CW_WAIT_CREDIT_RESET;
        iss_status_q  <= '0;
        mtimecmp_q    <= '1;
        lcq_w_q       <= '0;
        lb_valid_q    <= 1'b0;
        if (rt_state_q != '0 || rt_info_q != '0) rt_wr <= 1'b1;
        rt_state_q    <= '0;
        rt_info_q     <= '0;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) q_err_any_q <= 1'b0;
    else        q_err_any_q <= |q_err;
  end

  // Interrupt levels (CC §12.3).
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      irq_fast_q  <= '0;
      irq_timer_q <= 1'b0;
      irq_rise    <= 1'b0;
    end else begin
      logic [14:0] f;
      logic        t;
      logic [2:0]  wcu;
      f = '0;
      wcu = wait_credit_q[2:0];
      f[TSR_IRQ_FAST_WAIT_TICKET] = wait_ticket_q[31:28] != TSR_UNIT_NONE && ticket_done(wait_ticket_q);
      f[TSR_IRQ_FAST_ERR]         = iss_status_q[1];
      f[2]                        = irq_lq;
      f[3]                        = irq_lcq;
      f[4]                        = db_irq4;
      f[TSR_IRQ_FAST_CREDIT]      = wait_credit_q[31] && wcu < 3'd6 && credit_q[wcu] != 4'd0;
      f[6]                        = db_irq6;
      t = mtime >= mtimecmp_q;
      irq_fast_q  <= f;
      irq_timer_q <= t;
      irq_rise    <= |(f & ~irq_fast_q) || (t && !irq_timer_q);
    end
  end

`ifndef SYNTHESIS
  bit arm;   // armed once the reset has been seen
  initial arm = 1'b0;
  always @(posedge clk) begin   // not always_ff: the initial block writes arm too (IEEE 1800 9.2.2.4)
    if (!rst_n) arm <= 1'b1;
    if (rst_n && arm) begin
      if (w_req && st_q != S_IDLE) $error("tsr_ctl_win: request while busy");
      if (emit && if_free == '0) $error("tsr_ctl_win: issue FIFO write while full");
      if (emit && ik_q == IK_PUSH && credit_q[iu_q] == 4'd0) $error("tsr_ctl_win: push without a credit");
      if (x_req && !x_ready) $error("tsr_ctl_win: bridge request while the handshake is busy");
      if (lq_req && !lq_ready) $error("tsr_ctl_win: LQ request while the handshake is busy");
    end
  end
`endif

endmodule
