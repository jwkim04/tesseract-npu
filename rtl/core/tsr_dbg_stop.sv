// tsr_dbg_stop.sv — DB's stop logic and tile-breakpoint matcher (docs/design/05-debug.md "DB"
// §5.1-§5.4): the manual threshold S, the match queue M, the stop point T, STOP_ACTIVE, DRAINED,
// QSTEP with DBG_STEP_N, serial mode, ERR_STOP / HUNG_STOP and the per-unit dbg_hold / dbg_step
// levels into the queue heads (CC §9.2-§9.3).
//
// Owner: DB (W4-B item 6). Instantiated by tsr_dbg. Oracle: the simulator's DBG model
// (sw/sim/tsr_dbg.c stop_T, hold_mask, drained, stop_now, release, qstep, step_advance,
// tbp_match, on_enq / on_done); tb/dbg compares the registers and the stops it produces.
//
// Definitions (DB §5.2): push indices compare modularly, ge(a, b) <=> (a - b) mod 2^32 < 2^31.
//   T = the earlier of S (if S_VALID) and the head of M (if any); STOP_ACTIVE <=> T exists.
//   DRAINED = STOP_ACTIVE and ge(DBG_NEXT_PIDX, T) and every unit's oldest outstanding entry (the
//     shadow queues, tsr_dbg_shq) is at or after T.
//   dbg_hold[u] = head_valid[u] and ((STOP_ACTIVE and ge(head_pidx[u], T)) or (SERIAL and
//     head_pidx[u] != min_out)), min_out = the oldest outstanding entry over all units. Because
//     T is the earlier of S and M's head, ge(h, T) <=> ge(h, S) or ge(h, M head): the two
//     comparisons run in parallel. tsr_dbg forces it to 0 while SR_BUSY; tsr_issue makes it a
//     hard hold, except after a QABORT (starts only) (CC §9.2).
// Timing (DB §5.1, CC §7.6, §9.3): an entry enqueued in cycle t (tap) is decided at its head in
//   cycle t + 3 at the earliest. Its tile-breakpoint match runs from the registered tap (tsr_dbg's
//   stage e1, cycle t + 1) and is registered (m2, cycle t + 2): while it waits to enter M, the
//   pending thresholds (P for BEFORE, P + 1 for AFTER) take part in the hold directly, so the
//   registered hold (computed in t + 2) holds the entry in t + 3: no dispatch cycle is added. M
//   takes the BEFORE part in t + 2 and the AFTER part in t + 3 (the dispatcher enqueues at most
//   one entry every 3 cycles, tsr_issue D_IDLE -> D_POP -> D_ENQ, so consecutive entries never
//   collide). The hold also covers this cycle's stop events (STOP_AT, STOP_NOW, ERR / HUNG), so a
//   stop set by a register write holds every head from the next cycle on (the simulator applies a
//   write between two events: no head is decided "during" it); a release takes effect one cycle
//   later than the state shows it.
//   dbg_step[u] = a QSTEP's arm and head_pidx[u] = X, registered; tsr_dbg gates it with the host
//   and RISC-V hold sources (DB §5.2: a step never overrides them).
//   min_out comes from a two-stage tournament over the shadow fronts: a stale value only holds a
//   head longer (the oldest entry never becomes younger), never releases a wrong one.
//   DRAINED is evaluated from registered comparisons and T is re-checked when it is used, so a
//   QSTEP advances only from a DRAINED state of the current T.
// Events (one per cycle from the host; DB §5.2 table): STOP_NOW (and a QSTEP acting as it), ERR
//   and HUNG keep an earlier valid S (S <- the earlier of S and DBG_NEXT_PIDX, the cause bits
//   OR-ed: the simulator's stop_now); a DBG_STOP_AT write sets S unconditionally; RELEASE ends a
//   QSTEP, invalidates S and pops M's head if it is T; QSTEP with STOP_ACTIVE waits for DRAINED,
//   then X <- T, S <- X + 1 (cause STEP), M's head popped if it is X, dbg_step armed for X, and
//   repeats DBG_STEP_N (0 = 1) times, each step ending with X's completion; QSTEP without a stop
//   but with the debug-halt hold steps the oldest valid head; MATCHQ_CLEAR empties M (a QSTEP
//   ends if no stop is left); SOFT_RESET step 1 (sr_clear) clears S and M and ends a QSTEP.
//   ERR / HUNG coinciding with a host event or a QSTEP advance wait one cycle (pending flags).
// Records (DB §8.3 STOP): a release record (activated 0) in the cycle of RELEASE, MATCHQ_CLEAR or
//   sr_clear when a stop was active; an activation record (activated 1) one cycle after STOP_NOW,
//   STOP_AT, ERR, HUNG, a tile-breakpoint push (after the entry's last push), a RELEASE that leaves
//   M's next entry as T, and a QSTEP advance; all carry T and the cause of the state they show.
//   Activations of the first five kinds also raise act (RISC-V halt request candidate, tsr_dbg);
//   the RELEASE re-activation and the QSTEP advance do not (the simulator's trw_stop_rec).
// Tile breakpoints (DB §5.3), TBP i < NTBP, evaluated on the registered tap e1: KIND 0 PIDX
//   (pidx - A) mod 2^32 < (B - A) mod 2^32; 1 OPCODE ((w0[7:0] ^ A[7:0]) & A[15:8]) = 0,
//   ((w0[11:8] ^ A[19:16]) & A[23:20]) = 0, B[7:0] = 0 or B[unit]; 2 SPM / 3 DRAM: a valid
//   extent of that kind with a mode in {R, W} of CTL overlapping [A, B) (DRAM: on a channel of
//   CHMASK, the MVM weights on every channel), never for a poisoned entry; 4 TICKET ticket = A;
//   5..7 never. COUNT_ONLY counts without stopping; AFTER pushes P + 1 instead of P.
// M: 64 x {pidx, TBP bits} (LUTRAM), increasing push indices; a push equal to the tail merges
//   into it (DB §5.2: at most Σ Q_DEPTH <= 44 thresholds are pending, so 64 never overflows;
//   mq_ovf pulses if it would). BP_HIT[i] = some pending entry carries bit i (per-TBP counters).

module tsr_dbg_stop
  import tsr_pkg::*;
#(
  parameter int unsigned NUM_CH = TSR_DEF_NUM_CH,
  parameter int unsigned NTBP   = tsr_tables_pkg::TSR_DBG_NTBP
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // ---- configuration (DBG_CFG, DBG_TBPi_*, DBG_STEP_N) ----
  input  logic                     cfg_serial,
  input  logic                     cfg_err_stop,
  input  logic                     cfg_hung_stop,
  input  logic [3:0][31:0]         tbp_ctl,
  input  logic [3:0][31:0]         tbp_a,
  input  logic [3:0][31:0]         tbp_b,
  input  logic [15:0]              step_n,

  // ---- host events (tsr_dbg: DBG_CTRL priority QSTEP > RELEASE > STOP_NOW, one per write) ----
  input  logic                     ev_stop_now,
  input  logic                     ev_release,
  input  logic                     ev_qstep,
  input  logic                     ev_mq_clear,
  input  logic                     ev_stop_at,
  input  logic [31:0]              ev_stop_at_val,
  input  logic                     sr_clear,          // SOFT_RESET step 1 (tsr_csr sr_db_clear)

  // ---- core state ----
  input  logic [31:0]              next_pidx,         // Q_PIDX = DBG_NEXT_PIDX
  input  logic [5:0]               head_valid,
  input  logic [5:0][31:0]         head_pidx,
  input  logic                     halt_hold,         // HOLD_SRC.HALT (RV_RUN ∧ RV_HALTED ∧ HOLD_ON_HALT)
  input  logic                     unit_hung,         // STATUS.UNIT_HUNG (EX §8.6)

  // ---- shadow queues (tsr_dbg_shq): oldest outstanding entry of each unit ----
  input  logic [5:0]               out_v,
  input  logic [5:0][31:0]         out_pidx,

  // ---- enqueue tap, registered (tsr_dbg stage e1) ----
  input  logic                     e1_v,
  input  logic [2:0]               e1_unit,
  input  logic [27:0]              e1_seq,
  input  logic [31:0]              e1_pidx,
  input  logic [3:0]               e1_code,
  input  logic [31:0]              e1_w0,
  input  tsr_ext_t [TSR_EXT_N-1:0] e1_ext,

  // ---- start / completion strobes (tsr_issue) and error records ----
  input  logic [5:0]               tap_start,
  input  logic [5:0][31:0]         tap_start_pidx,
  input  logic [5:0]               tap_done,
  input  logic [5:0][3:0]          tap_done_code,
  input  logic [5:0][31:0]         tap_done_pidx,
  input  logic [5:0][TSR_SEQ_W-1:0] seq_d,
  input  tsr_err_rec_t [5:0]       err_rec,

  // ---- holds and steps (tsr_dbg gates them) ----
  output logic [5:0]               hold,
  output logic [5:0]               step,

  // ---- state for DBG_STATUS / DBG_STOP_PIDX / ... ----
  output logic                     stop_active,
  output logic                     s_valid,
  output logic                     drained,
  output logic                     step_busy,
  output logic [31:0]              stop_t,            // T (valid while stop_active)
  output logic [31:0]              s_val,             // S
  output logic [31:0]              step_x,
  output logic [31:0]              last_t,            // T of the last activation (STOP_PIDX while inactive)
  output logic [4:0]               cause,
  output logic [3:0]               bp_hit,
  output logic [6:0]               mq_n,
  output logic                     mq_ovf,            // pulse: a push found M full (never expected)

  // ---- per enqueued entry ----
  output logic                     m2_v,              // stage 2: the match result of an entry
  output logic [2:0]               m2_unit,
  output logic [3:0]               m2_hit,            // TBPs matched (COUNT_ONLY included): HITS, flags
  output logic                     h3_v,              // stage 3: history INFO side fields
  output logic [31:0]              h3_pidx,
  output logic [3:0]               h3_hit,
  output logic                     h3_stop,           // the entry is T right after its pushes

  // ---- activations (halt-request candidates) and STOP records ----
  output logic                     act,
  output logic                     act_tbp_halt,
  output logic                     srel_v,
  output logic [31:0]              srel_t,
  output logic [4:0]               srel_cause,
  output logic                     sact_v,
  output logic [31:0]              sact_t,
  output logic [4:0]               sact_cause
);

  if (NTBP > 4) begin : g_chk_ntbp
    $fatal(1, "tsr_dbg_stop: NTBP must be in [0, 4] (DB §10.3)");
  end

  localparam logic [4:0] C_NOW  = tsr_tables_pkg::TSR_DBG_CAUSE_NOW;
  localparam logic [4:0] C_BP   = tsr_tables_pkg::TSR_DBG_CAUSE_BP;
  localparam logic [4:0] C_STEP = tsr_tables_pkg::TSR_DBG_CAUSE_STEP;
  localparam logic [4:0] C_ERR  = tsr_tables_pkg::TSR_DBG_CAUSE_ERR;
  localparam logic [4:0] C_HUNG = tsr_tables_pkg::TSR_DBG_CAUSE_HUNG;
  localparam logic [15:0] CH_ALL = 16'((32'd1 << NUM_CH) - 32'd1);

  function automatic logic ge(logic [31:0] a, logic [31:0] b);
    return (a - b) < 32'h8000_0000;
  endfunction

  // -------------------------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------------------------
  logic [31:0] s_q, last_t_q;
  logic        sv_q;
  logic [4:0]  sc_q;
  // match queue
  localparam int unsigned MQD = tsr_tables_pkg::TSR_MATCHQ_DEPTH;      // 64
  logic [31:0] mq_p   [MQD];
  logic [3:0]  mq_bits[MQD];
  logic [5:0]  mrd_q, mwr_q;
  logic [6:0]  mn_q;
  logic [31:0] mh_q, mt_q;                 // head / tail push index (valid while mn_q != 0)
  logic [3:0]  mhb_q, mtb_q;               // head / tail TBP bits
  logic [3:0][6:0] bpc_q;                  // pending entries per TBP bit
  // QSTEP
  typedef enum logic [1:0] { P_IDLE, P_DRAIN, P_DONE } phase_e;
  phase_e      ph_q;
  logic        busy_q, arm_q;
  logic [16:0] left_q;
  logic [31:0] x_q;
  // pending ERR / HUNG (deferred one cycle when another event uses the cycle)
  logic        err_pend_q, hung_pend_q, hung_prev_q;
  // activation record pending
  logic        actp_q, acth_q, actt_q;

  // -------------------------------------------------------------------------------------------
  // T, cause, BP_HIT
  // -------------------------------------------------------------------------------------------
  wire        m_any  = (mn_q != 7'd0);
  wire        active = sv_q || m_any;
  wire        t_is_s = sv_q && !(m_any && ge(s_q, mh_q));
  wire [31:0] t_cur  = t_is_s ? s_q : mh_q;
  wire [4:0]  cause_cur = (sv_q ? sc_q : 5'd0) | (m_any ? C_BP : 5'd0);

  // -------------------------------------------------------------------------------------------
  // Tile-breakpoint matcher (stage 1 -> m2)
  // -------------------------------------------------------------------------------------------
  logic [3:0] hit_c, bef_c, aft_c;
  logic       halt_c;
  always_comb begin
    hit_c  = '0;
    bef_c  = '0;
    aft_c  = '0;
    halt_c = 1'b0;
    for (int i = 0; i < int'(NTBP); i++) begin
      automatic logic [31:0] A = tbp_a[i], B = tbp_b[i];
      automatic logic [8:0]  ctl = tbp_ctl[i][8:0];
      automatic logic [15:0] chm = tbp_ctl[i][31:16];
      automatic logic        m = 1'b0;
      automatic logic [7:0]  units = B[7:0];
      case (ctl[3:1])
        tsr_tables_pkg::TSR_DBG_TBP_CTL_KIND_PIDX:   m = ((e1_pidx - A) < (B - A));
        tsr_tables_pkg::TSR_DBG_TBP_CTL_KIND_OPCODE:
          m = (((e1_w0[7:0] ^ A[7:0]) & A[15:8]) == 8'd0) &&
              (((e1_w0[11:8] ^ A[19:16]) & A[23:20]) == 4'd0) &&
              (units == 8'd0 || units[e1_unit]);
        tsr_tables_pkg::TSR_DBG_TBP_CTL_KIND_SPM, tsr_tables_pkg::TSR_DBG_TBP_CTL_KIND_DRAM: begin
          automatic logic want_dram = (ctl[3:1] == tsr_tables_pkg::TSR_DBG_TBP_CTL_KIND_DRAM);
          if (e1_code == TSR_ERR_NONE) begin
            for (int unsigned k = 0; k < TSR_EXT_N; k++) begin
              automatic tsr_ext_t e = e1_ext[k];
              automatic logic [15:0] chs = (e.ch == TSR_EXT_CH_ALL) ? CH_ALL : (16'd1 << e.ch[3:0]);
              automatic logic mode = e.wr ? ctl[5] : ctl[4];
              automatic logic chok = !want_dram || ((chs & chm) != 16'd0 && (e.ch == TSR_EXT_CH_ALL || e.ch < 8'd16));
              if (e.v && (e.dram == want_dram) && mode && chok &&
                  (e.lo < 48'(B)) && (48'(A) < e.hi))
                m = 1'b1;
            end
          end
        end
        tsr_tables_pkg::TSR_DBG_TBP_CTL_KIND_TICKET: m = ({1'b0, e1_unit, e1_seq} == A);
        default: m = 1'b0;                                   // kinds 5..7 never match
      endcase
      if (ctl[0] && m) begin
        hit_c[i] = 1'b1;
        if (!ctl[8]) begin
          if (ctl[6]) aft_c[i] = 1'b1;
          else        bef_c[i] = 1'b1;
          if (ctl[7]) halt_c = 1'b1;
        end
      end
    end
  end

  logic        m2_v_q, m3_v_q, h3_v_q;
  logic [2:0]  m2_unit_q;
  logic [31:0] m2_p_q, m3_p_q, h3_p_q;
  logic [3:0]  m2_hit_q, m2_bef_q, m2_aft_q, m3_aft_q, h3_hit_q;
  logic        m2_halt_q, m3_halt_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      m2_v_q <= 1'b0;
      m3_v_q <= 1'b0;
      h3_v_q <= 1'b0;
    end else begin
      m2_v_q <= e1_v;
      m3_v_q <= m2_v_q;                 // stage 3: the AFTER push (if any) and the history side write
      h3_v_q <= m2_v_q;
    end
  end
  always_ff @(posedge clk) begin
    if (e1_v) begin
      m2_unit_q <= e1_unit;
      m2_p_q    <= e1_pidx;
      m2_hit_q  <= hit_c;
      m2_bef_q  <= bef_c;
      m2_aft_q  <= aft_c;
      m2_halt_q <= halt_c;
    end
    if (m2_v_q) begin
      m3_p_q    <= m2_p_q + 32'd1;
      m3_aft_q  <= m2_aft_q;
      m3_halt_q <= m2_halt_q;
      h3_p_q    <= m2_p_q;
      h3_hit_q  <= m2_hit_q;
    end
  end
  wire unused_in = &{1'b0, e1_w0[31:12], tbp_ctl[0][15:9], tbp_ctl[1][15:9], tbp_ctl[2][15:9], tbp_ctl[3][15:9]};
  wire pend_b = m2_v_q && (m2_bef_q != 4'd0);              // P waits to enter M (this cycle's push)
  wire pend_a = (m2_v_q && (m2_aft_q != 4'd0)) || (m3_v_q && (m3_aft_q != 4'd0));
  wire [31:0] pend_a_p = m3_v_q ? m3_p_q : (m2_p_q + 32'd1);

  // -------------------------------------------------------------------------------------------
  // Events of this cycle and the next state of S
  // -------------------------------------------------------------------------------------------
  // ERR_STOP (DB §5.2): the completion of the entry that put its unit in error state (its seq and
  // code equal the unit's error record; the record is set at the decision, before the strobe).
  logic err_now;
  always_comb begin
    err_now = 1'b0;
    for (int u = 0; u < 6; u++)
      if (tap_done[u] && tap_done_code[u] != TSR_ERR_NONE && err_rec[u].err && err_rec[u].seq == seq_d[u] &&
          err_rec[u].code == tap_done_code[u])
        err_now = 1'b1;
  end
  wire hung_rise = unit_hung && !hung_prev_q;

  wire host_ev  = ev_stop_now || ev_release || ev_qstep || ev_mq_clear || ev_stop_at;
  // QSTEP with STOP_ACTIVE: DRAINED of the current T (registered comparisons, T re-checked)
  logic        dr_ge_q;                    // ge(next_pidx, T) and every front at or after T, of t_snap_q
  logic [31:0] t_snap_q;
  logic        act_snap_q;
  wire         drained_c = act_snap_q && active && dr_ge_q && (t_snap_q == t_cur);
  wire         adv  = (ph_q == P_DRAIN) && drained_c && !host_ev && !sr_clear;
  wire         xdone = (ph_q == P_DONE) && (|(tap_done & ({
                         tap_done_pidx[5] == x_q, tap_done_pidx[4] == x_q, tap_done_pidx[3] == x_q,
                         tap_done_pidx[2] == x_q, tap_done_pidx[1] == x_q, tap_done_pidx[0] == x_q})));
  // ERR / HUNG apply in a cycle without a host event, a QSTEP advance or sr_clear
  wire         free = !host_ev && !adv && !sr_clear;
  wire         do_err  = free && (err_pend_q || (err_now && cfg_err_stop));
  wire         do_hung = free && !do_err && (hung_pend_q || (hung_rise && cfg_hung_stop));

  // QSTEP without a stop under the debug-halt hold: the oldest valid head (registered tournament)
  logic        hmin_v_q;
  logic [31:0] hmin_q;

  // stop_now(cause): S <- the earlier of S and DBG_NEXT_PIDX, cause bits OR-ed when S is kept
  logic        sv_n;
  logic [31:0] s_n;
  logic [4:0]  sc_n;
  logic        pop_head;                   // pop M's head (RELEASE / QSTEP advance: it is T)
  logic        do_stop_now;
  logic [4:0]  stop_now_cause;
  always_comb begin
    sv_n = sv_q;
    s_n  = s_q;
    sc_n = sc_q;
    pop_head = 1'b0;
    do_stop_now = 1'b0;
    stop_now_cause = C_NOW;
    if (sr_clear) begin
      sv_n = 1'b0;
    end else if (ev_stop_at) begin
      sv_n = 1'b1;
      s_n  = ev_stop_at_val;
      sc_n = C_NOW;
    end else if (ev_qstep) begin
      if (!active && !halt_hold) begin
        do_stop_now = 1'b1;                // QSTEP otherwise acts as STOP_NOW
      end
    end else if (ev_release) begin
      sv_n = 1'b0;
      pop_head = active && m_any && !t_is_s;
    end else if (ev_stop_now) begin
      do_stop_now = 1'b1;
    end else if (adv) begin
      sv_n = 1'b1;
      s_n  = t_snap_q + 32'd1;
      sc_n = C_STEP;
      pop_head = m_any && (mh_q == t_snap_q);
    end else if (do_err) begin
      do_stop_now = 1'b1;
      stop_now_cause = C_ERR;
    end else if (do_hung) begin
      do_stop_now = 1'b1;
      stop_now_cause = C_HUNG;
    end
    if (do_stop_now) begin
      if (sv_q && ge(next_pidx, s_q)) begin
        sc_n = sc_q | stop_now_cause;
      end else begin
        sv_n = 1'b1;
        s_n  = next_pidx;
        sc_n = stop_now_cause;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Match queue
  // -------------------------------------------------------------------------------------------
  // pushes: the BEFORE part in stage 2, the AFTER part in stage 3 (never both in one cycle)
  wire        mpush   = pend_b || (m3_v_q && (m3_aft_q != 4'd0));
  wire [31:0] mpush_p = pend_b ? m2_p_q : m3_p_q;
  wire [3:0]  mpush_b = pend_b ? m2_bef_q : m3_aft_q;
  wire        mclear  = sr_clear || ev_mq_clear;
  wire [6:0]  mn_pop  = mn_q - 7'(pop_head);
  wire        merge   = mpush && (mn_pop != 7'd0) && (mt_q == mpush_p);
  wire        mfull   = (mn_pop == 7'(MQD));
  wire        mnew    = mpush && !merge && !mfull;

  always_ff @(posedge clk) begin
    if (mnew)       begin mq_p[mwr_q] <= mpush_p; mq_bits[mwr_q] <= mpush_b; end
    else if (merge) mq_bits[mwr_q - 6'd1] <= mtb_q | mpush_b;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      mrd_q <= '0;
      mwr_q <= '0;
      mn_q  <= '0;
      bpc_q <= '0;
    end else if (mclear) begin
      mrd_q <= '0;
      mwr_q <= '0;
      mn_q  <= '0;
      bpc_q <= '0;
    end else begin
      if (pop_head) mrd_q <= mrd_q + 6'd1;
      if (mnew)     mwr_q <= mwr_q + 6'd1;
      mn_q <= mn_pop + 7'(mnew);
      for (int unsigned i = 0; i < 4; i++)
        bpc_q[i] <= bpc_q[i] - 7'(pop_head && mhb_q[i]) +
                    7'((mnew && mpush_b[i]) || (merge && mpush_b[i] && !mtb_q[i]));
    end
    // head and tail registers (meaningful while mn_q != 0)
    if (mnew || merge) begin
      mt_q  <= mpush_p;
      mtb_q <= merge ? (mtb_q | mpush_b) : mpush_b;
    end
    if (mn_pop == 7'd0) begin
      if (mnew) begin
        mh_q  <= mpush_p;
        mhb_q <= mpush_b;
      end
    end else if (pop_head) begin
      // the entry after the head (it is the tail when one entry is left: merged bits included)
      mh_q  <= mq_p[mrd_q + 6'd1];
      mhb_q <= (mn_pop == 7'd1 && merge) ? (mtb_q | mpush_b) : mq_bits[mrd_q + 6'd1];
    end else if (merge && mn_q == 7'd1) begin
      mhb_q <= mhb_q | mpush_b;
    end
  end
  assign mq_ovf = mpush && !merge && mfull && !mclear;

  // -------------------------------------------------------------------------------------------
  // S, QSTEP, pending ERR / HUNG, activation records
  // -------------------------------------------------------------------------------------------
  wire step_cancel = sr_clear || ev_release || (ev_mq_clear && m_any && !sv_q);
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s_q         <= '0;
      sv_q        <= 1'b0;
      sc_q        <= '0;
      last_t_q    <= '0;
      ph_q        <= P_IDLE;
      busy_q      <= 1'b0;
      arm_q       <= 1'b0;
      left_q      <= '0;
      x_q         <= '0;
      err_pend_q  <= 1'b0;
      hung_pend_q <= 1'b0;
      hung_prev_q <= 1'b0;
      actp_q      <= 1'b0;
      acth_q      <= 1'b0;
      actt_q      <= 1'b0;
    end else begin
      s_q  <= s_n;
      sv_q <= sv_n;
      sc_q <= sc_n;
      hung_prev_q <= unit_hung;
      // ERR / HUNG that could not apply in this cycle
      err_pend_q  <= !do_err && (err_pend_q || (err_now && cfg_err_stop)) && !sr_clear;
      hung_pend_q <= !do_hung && (hung_pend_q || (hung_rise && cfg_hung_stop)) && !sr_clear;
      // activation record of this cycle's event, written next cycle from the state it produced
      actp_q <= 1'b0;
      acth_q <= 1'b0;
      actt_q <= 1'b0;
      if (!sr_clear) begin
        if (ev_stop_at || do_stop_now) begin
          actp_q <= 1'b1;
          acth_q <= 1'b1;
        end
        if (ev_release && active && !mclear && (mn_pop != 7'd0 || mnew)) actp_q <= 1'b1;
        if (adv) actp_q <= 1'b1;
        // a tile-breakpoint activation after the entry's last push
        if ((m2_v_q && (m2_bef_q != 4'd0) && (m2_aft_q == 4'd0)) || (m3_v_q && (m3_aft_q != 4'd0))) begin
          actp_q <= 1'b1;
          acth_q <= 1'b1;
          actt_q <= (m2_v_q && (m2_aft_q == 4'd0)) ? m2_halt_q : m3_halt_q;
        end
      end
      if (actp_q && active) last_t_q <= t_cur;
      // QSTEP (DB §5.2)
      if (step_cancel) begin
        ph_q   <= P_IDLE;
        busy_q <= 1'b0;
        arm_q  <= 1'b0;
      end else if (ev_qstep) begin
        if (active) begin
          left_q <= (step_n == 16'd0) ? 17'd1 : 17'(step_n);
          ph_q   <= P_DRAIN;
          busy_q <= 1'b1;
        end else if (halt_hold && hmin_v_q) begin
          x_q    <= hmin_q;
          left_q <= 17'd1;
          ph_q   <= P_DONE;
          busy_q <= 1'b1;
          arm_q  <= 1'b1;
        end
      end else if (adv) begin
        x_q   <= t_snap_q;
        ph_q  <= P_DONE;
        arm_q <= 1'b1;
      end else if (xdone) begin
        arm_q <= 1'b0;
        if (left_q > 17'd1) begin
          left_q <= left_q - 17'd1;
          ph_q   <= P_DRAIN;
        end else begin
          ph_q   <= P_IDLE;
          busy_q <= 1'b0;
        end
      end
      // the step pulse was taken: X started (DB §5.2 "pulse dbg_step[u] once")
      if (arm_q && !ev_qstep && !adv) begin
        for (int u = 0; u < 6; u++)
          if (tap_start[u] && tap_start_pidx[u] == x_q) arm_q <= 1'b0;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // DRAINED (registered comparisons of the current T), min_out, the oldest head
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    logic ok;
    ok = ge(next_pidx, t_cur);
    for (int u = 0; u < 6; u++)
      if (out_v[u] && !ge(out_pidx[u], t_cur)) ok = 1'b0;
    dr_ge_q    <= ok;
    t_snap_q   <= t_cur;
    act_snap_q <= active;
  end

  // two-stage tournaments: [0] fronts (min_out), [1] heads (QSTEP under the debug-halt hold)
  logic [2:0]       mo1_v_q, hm1_v_q;
  logic [2:0][31:0] mo1_q, hm1_q;
  logic             mo_v_q;
  logic [31:0]      mo_q;
  always_ff @(posedge clk) begin
    for (int k = 0; k < 3; k++) begin
      automatic int a = 2 * k, b = 2 * k + 1;
      // the older of a pair (ge(x, y): y is at or before x)
      mo1_v_q[k] <= out_v[a] || out_v[b];
      mo1_q[k]   <= (out_v[a] && (!out_v[b] || ge(out_pidx[b], out_pidx[a]))) ? out_pidx[a] : out_pidx[b];
      hm1_v_q[k] <= head_valid[a] || head_valid[b];
      hm1_q[k]   <= (head_valid[a] && (!head_valid[b] || ge(head_pidx[b], head_pidx[a]))) ? head_pidx[a] : head_pidx[b];
    end
  end
  always_ff @(posedge clk) begin
    logic        v, hv;
    logic [31:0] m, hm;
    v = 1'b0;
    m = '0;
    hv = 1'b0;
    hm = '0;
    for (int k = 0; k < 3; k++) begin
      if (mo1_v_q[k] && (!v || ge(m, mo1_q[k]))) m = mo1_q[k];
      v = v || mo1_v_q[k];
      if (hm1_v_q[k] && (!hv || ge(hm, hm1_q[k]))) hm = hm1_q[k];
      hv = hv || hm1_v_q[k];
    end
    mo_v_q   <= v;
    mo_q     <= m;
    hmin_v_q <= hv;
    hmin_q   <= hm;
  end

  // -------------------------------------------------------------------------------------------
  // dbg_hold / dbg_step (registered)
  // -------------------------------------------------------------------------------------------
  // The S part of the hold covers this cycle's events with parallel comparisons (timing: no chain
  // through the next value of S): the S held so far, a DBG_STOP_AT value, and DBG_NEXT_PIDX for
  // any STOP_NOW-like request (STOP_NOW, a QSTEP acting as it, ERR / HUNG pending or arriving).
  // Against the exact next S it can only hold one cycle longer (a release, a QSTEP advance, S moved
  // later by a write, an ERR / HUNG deferred), never shorter.
  wire stopnow_rq = ev_stop_now || (ev_qstep && !active && !halt_hold) || err_pend_q || hung_pend_q ||
                    (err_now && cfg_err_stop) || (hung_rise && cfg_hung_stop);
  logic [5:0] hold_q, step_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      hold_q <= '0;
      step_q <= '0;
    end else begin
      for (int u = 0; u < 6; u++) begin
        automatic logic [31:0] h = head_pidx[u];
        hold_q[u] <= head_valid[u] &&
                     ((sv_q && ge(h, s_q)) || (ev_stop_at && ge(h, ev_stop_at_val)) ||
                      (stopnow_rq && ge(h, next_pidx)) ||
                      (m_any && !mclear && ge(h, mh_q)) ||
                      (pend_b && ge(h, m2_p_q)) ||
                      (pend_a && ge(h, pend_a_p)) ||
                      (cfg_serial && mo_v_q && h != mo_q));
        step_q[u] <= arm_q && !step_cancel && head_valid[u] && h == x_q;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Outputs
  // -------------------------------------------------------------------------------------------
  assign hold        = hold_q;
  assign step        = step_q;
  assign stop_active = active;
  assign s_valid     = sv_q;
  assign drained     = drained_c;
  assign step_busy   = busy_q;
  assign stop_t      = t_cur;
  assign s_val       = s_q;
  assign step_x      = x_q;
  assign last_t      = last_t_q;
  assign cause       = cause_cur;
  for (genvar i = 0; i < 4; i++) begin : g_bp
    assign bp_hit[i] = (bpc_q[i] != 7'd0);
  end
  assign mq_n        = mn_q;

  assign m2_v    = m2_v_q;
  assign m2_unit = m2_unit_q;
  assign m2_hit  = m2_hit_q;
  assign h3_v    = h3_v_q;
  assign h3_pidx = h3_p_q;
  assign h3_hit  = h3_hit_q;
  assign h3_stop = active && (t_cur == h3_p_q);

  assign act          = actp_q && acth_q && active;
  assign act_tbp_halt = actt_q;
  assign sact_v       = actp_q && active;
  assign sact_t       = t_cur;
  assign sact_cause   = cause_cur;
  // release records: RELEASE, MATCHQ_CLEAR that ends every stop, SOFT_RESET step 1
  assign srel_v     = (ev_release || sr_clear || (ev_mq_clear && m_any && !sv_q)) && active;
  assign srel_t     = t_cur;
  assign srel_cause = cause_cur;

endmodule
