// tsr_issue.sv — the core side of the issue path: dispatcher, checker, the six unit queues, their
// heads (dependencies, SYNC, holds, DB's hard hold), the per-unit counters and error states,
// tickets and push indices, and the I2 record handshake (docs/design/04-control-core.md "CC"
// §7-§9; EX §7-§8).
//
// Owner: issue path (CC §2.2, §13.2; W4 item 1). Instantiated by tsr_core_dp next to tsr_csr.
// Oracle: the simulator's queue model (sw/sim/tsr_q.c): for a schedule-independent push stream
// every entry's enqueue code and completion code, the per-unit completion order, the counters,
// the error records and the push indices equal the simulator's (tb/issue); timing is the RTL's.
//
// Ports
//   Issue FIFO (core side of tsr_cdc_fifo, CC §7.2): first-word-fall-through; ifq_valid = an
//     entry is available, ifq_entry its 387 bits (tsr_ifent_t), ifq_ready pops it (one cycle
//     pulse, only while ifq_valid). The RISC-V side assigns tickets {u, ISSUED_SEQ[u]} and push
//     indices (CC §7.1); the dispatcher asserts that every ticket equals its own count.
//   xc_*: external checker classes (VU / AT opcodes, W4-B), passed through to tsr_chk.
//   Units (docs/rtl-m2.md §3.0): <u>_cmd_valid / <u>_cmd_ready / <u>_cmd / <u>_done. cmd_valid
//     and cmd are registers (CC §9.1 cmd_q / cmd_v_q, R2: held until cmd_ready). DMA and MVM also
//     carry <u>_cmd_xc / <u>_xcmd for the AT ops of their queues (KVST, ATTQK, ATTPV; W4-B); VU and
//     AT carry only the external class's payload. Units have no error output (CC-11).
//   Control (from tsr_csr): host_hold (CTRL.HOST_HOLD), halt_hold (RV_RUN ∧ RV_HALTED ∧
//     HOLD_ON_HALT), qabort (pulse: host QABORT, LQ_CANCEL covering the running launch, the
//     RISC-V reset, SOFT_RESET; CC §9.5), rv_reset (pulse: the RISC-V reset sequence clears the
//     RISC-V hold source, CC §12.2), host_clear (pulse: the effects of an accepted host QCLEAR
//     and of SOFT_RESET step 3), sr_busy (SOFT_RESET in progress: the hard hold is off, from the cycle
//     the SOFT_RESET step 1 QABORT applies: sr_off_q below).
//   DB (W4-B): dbg_hold[u] / dbg_step[u] levels (CC §9.2-§9.3), head info, and the taps:
//     enqueue {unit, seq, pidx, code, info, kind, deps as dispatched, w, REC, push PC, extents},
//     per-unit start and completion strobes, TRACE_MARK.
//   Records (I2, CC §9.4, §11.2): err_rec[u] is the core-domain record (Q_ERR_*). Whenever it
//     changes (an error captured, QABORT, a clear), a copy goes to the clk_rv side through
//     tsr_ctrl_cdc's per-unit handshake: err_valid[u] with err_hs[u] (the 128-bit crossing format
//     of tsr_ctrl_pkg::err_rec_t: {pidx, w0, info, code[3:0], seq[27:0]}, code 0 = no error) held
//     until the one-cycle err_ack[u]; a change during the handshake sends the newer record after
//     it. The CDC acknowledges only once the clk_rv side shows the record, and locally when the
//     clk_rv domain is not running or after its bound (CC §11.2), so SOFT_RESET drains without
//     clk_rv. A completion that would move D[u] past ERR_SEQ[u] waits until the record is clean
//     (no change pending, no handshake in flight); so does a QCLEAR's own completion after it
//     cleared the records (CC §9.5). rec_resync (one pulse, from tsr_ctrl_cdc when the clk_rv domain
//     is OK again) marks every unit's record changed, so the clk_rv side catches up with what was
//     answered locally in the meantime.
//   State: seq_s / seq_d (S[u], D[u], 28 bits, at most one step per cycle each: gray-crossed by
//     tsr_ctrl_cdc for the RISC-V's credits and DONE_SEQ), q_pidx (Q_PIDX), q_state (Q_STATE[8:0]
//     of EX §5.4 without HUNG), q_nonempty, err_any, abort_active, hold_src, qidle, qquiescent.
//   PERF events (tsr_csr gates them, EX §6): completions per cycle, pushes (ticketed entries
//     enqueued; counted at dispatch, which equals the RISC-V's push count while ACTIVE, R6), the
//     per-unit "holds an op" levels, MVM_WAIT / MVM_EMPTY, "some head held".
//
// Dispatcher (CC §8.3): D_IDLE -> D_POP (entry registered) -> CTRL: QABORT / QHOLD / TRACE_MARK act
//   and return; QCLEAR builds a SYNC entry with the snapshot cnt[0..5]; OP: tsr_chk runs (D_CHK),
//   the BAD_DEP rule is evaluated from cnt[] and clear_pt[] at D_POP -> D_ENQ writes queue u
//   (run entry or poisoned entry), cnt[u]++, pidx++. BAD_DEP (v, s), v != 15: v in 6..14 or unit
//   v not built; (s - cnt[v]) mod 2^28 < 2^27 (not pushed yet, the op's own ticket included); or
//   (clear_pt[v] - s) mod 2^28 in [1, 2^27). A bad dep is replaced by none; BAD_DEP wins over
//   every code but ILLEGAL_OP, with ERR_INFO 0; a poisoned NOP / FENCE is a plain poisoned entry.
// Heads (CC §9.1-§9.3), per unit: LUTRAM queue -> head register (loaded when the previous head
//   leaves) -> evaluation flops (dep_ok, snap_ok, failed, hold; registered every cycle) ->
//   decision -> cmd_v_q / cmd_q. An entry is never decided in the cycle it becomes head. Decision
//   order (the simulator's q_head): deps (and a FENCE / QCLEAR snapshot) not done: stay; DB's
//   hard hold (dbg_hold ∧ ¬abort_active ∧ ¬SR_BUSY, unless dbg_step): stay; QCLEAR: clear;
//   poisoned: COMPLETE(code); unit in error: COMPLETE(ABORTED); a dep failed: COMPLETE(DEP_FAILED);
//   NOP / FENCE: COMPLETE(0); a hold (host, RISC-V, debug halt, dbg_hold; unless dbg_step): stay;
//   LD / ST with 0 lines: completes ran without the unit; else, when the unit's cmd_ready is 1
//   and it holds fewer than its accept limit (I3): issue. A non-running completion is decided
//   only while the unit holds no op (completions stay in sequence order) and takes S[u]++ at
//   once and D[u]++ one cycle later (or after the record handshake, I2); a running op takes S[u]++
//   at the cmd transfer and D[u]++ at its done.
// Errors (CC §9.5): the first nonzero completion records {seq, code, info (the checker's for a
//   poisoned entry, else 0), w0, pidx}. QABORT (any source) applies one cycle after it is seen,
//   in a cycle without head decisions: every built unit not in error enters error state with
//   ERR_SEQ = S[u] + (a command offered and not yet taken), ABORTED, and the w0 / pidx of that
//   first entry not started: the head, else the queue front, else the entry the dispatcher holds
//   for that unit; else w0 = 0 and pidx = Q_PIDX (CC §9.5 leaves this case open; the simulator
//   takes the first issue-FIFO entry of the unit, which the core cannot see). A QCLEAR (the
//   RISC-V's at its SYNC head, or the host's) clears every record and sets clear_pt[u] (snapshot,
//   resp. cnt[u]); it wins over an error recorded in the same cycle (the simulator evaluates the
//   SYNC head last). abort_active: set by QABORT, cleared by QCLEAR.
// Invariants asserted in every simulation (CC §9.4): I1 an entry is decided only at its head
//   with both deps (and its snapshot) done; I2 above; I3 S[u] - D[u] <= accept limit (tables
//   TSR_ACCEPT_MAX_*), done only while the unit holds an op, at most one D[u] step per cycle; and
//   the dispatcher's ticket / count check. (I4, unit traffic only while an op is held, is the
//   integration check of tsr_core.)
// Latency (CC §7.6, §9.3): issue-FIFO entry -> enqueue 12 cycles (pop 1, D_POP 1, tsr_chk 8, D_ENQ
//   1, + 1), enqueue -> head 1, head -> decision 2, decision -> cmd transfer 1; done of t -> start
//   of an idle dependent on another unit 3 cycles; done -> start of the next entry of the same
//   unit 1 cycle (the next head is already evaluated).

// One unit queue: a ring of D entries of W bits (LUTRAM), front read combinationally.
// Waiver: this helper lives in tsr_issue.sv so the issue path keeps one filelist entry per block
// (rtl/filelist.f is shared, docs/rtl-m2.md §1 "Files"), as tsr_rslice does in tsr_mem_arb.sv.
/* verilator lint_off DECLFILENAME */
module tsr_issue_q #(
  parameter int unsigned W = 8,
  parameter int unsigned D = 4
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         we,
  input  logic [W-1:0] wdata,
  input  logic         re,
  output logic [W-1:0] front,
  output logic [4:0]   count
);
  localparam int unsigned PW = (D > 1) ? $clog2(D) : 1;
  logic [W-1:0]  mem [D];
  logic [PW-1:0] rd_q, wr_q;
  logic [4:0]    n_q;

  function automatic logic [PW-1:0] inc(logic [PW-1:0] p);
    return (32'(p) == D - 1) ? '0 : p + 1'b1;
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_q <= '0;
      wr_q <= '0;
      n_q  <= '0;
    end else begin
      if (we) wr_q <= inc(wr_q);
      if (re) rd_q <= inc(rd_q);
      n_q <= n_q + 5'(we) - 5'(re);
    end
  end
  always_ff @(posedge clk) if (we) mem[wr_q] <= wdata;

  assign front = mem[rd_q];
  assign count = n_q;

`ifndef SYNTHESIS
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      assert (!(we && !re && 32'(n_q) >= D)) else $error("%m: queue overflow (credit accounting)");
      assert (!(re && n_q == '0)) else $error("%m: pop of an empty queue");
    end
  end
`endif
endmodule
/* verilator lint_on DECLFILENAME */

module tsr_issue
  import tsr_pkg::*;
#(
  parameter int unsigned     NUM_CH      = TSR_DEF_NUM_CH,
  parameter int unsigned     T_MAX       = TSR_DEF_T_MAX,
  parameter int unsigned     K_MAX       = TSR_DEF_K_MAX,
  parameter int unsigned     SPM_BYTES   = TSR_DEF_SPM_BYTES,
  parameter longint unsigned CH_BYTES    = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter logic [31:0]     CAPS        = 32'd0,
  // queue depths (CC §7.4; 0 = unit absent; U200 M6: 8 4 4 16 0 4); SYNC must be built
  parameter int unsigned     QD_DMA      = tsr_tables_pkg::TSR_QDEPTH_U200_DMA,
  parameter int unsigned     QD_QACT     = tsr_tables_pkg::TSR_QDEPTH_U200_QACT,
  parameter int unsigned     QD_MVM      = tsr_tables_pkg::TSR_QDEPTH_U200_MVM,
  parameter int unsigned     QD_VU       = tsr_tables_pkg::TSR_QDEPTH_U200_VU,
  parameter int unsigned     QD_ATT      = tsr_tables_pkg::TSR_QDEPTH_U200_ATT,
  parameter int unsigned     QD_SYNC     = tsr_tables_pkg::TSR_QDEPTH_U200_SYNC,
  parameter int unsigned     XC_CMD_W    = 256,
  // simulation only (ticket-wrap tests, CC §15.2): reset value of every sequence counter
  parameter logic [27:0]     SEQ_INIT    = 28'd0,
  parameter logic [31:0]     PIDX_INIT   = 32'd0
) (
  input  logic                        clk,
  input  logic                        rst_n,

  // ---- issue FIFO (core side) ----
  input  logic                        ifq_valid,
  output logic                        ifq_ready,
  input  tsr_ifent_t                  ifq_entry,

  // ---- external checker classes (VU / AT; W4-B) ----
  output logic                        xc_req_valid,
  output tsr_insn_t                   xc_req_insn,
  input  logic                        xc_rsp_valid,
  input  logic [3:0]                  xc_rsp_code,
  input  logic [31:0]                 xc_rsp_info,
  input  tsr_ext_t [TSR_EXT_N-1:0]    xc_rsp_ext,
  input  logic [XC_CMD_W-1:0]         xc_rsp_cmd,

  // ---- units ----
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

  // ---- control (tsr_csr) ----
  input  logic                        host_hold,
  input  logic                        halt_hold,
  input  logic                        qabort,
  input  logic                        rv_reset,
  input  logic                        host_clear,
  input  logic                        sr_busy,
  input  logic [5:0]                  dbg_hold,
  input  logic [5:0]                  dbg_step,

  // ---- error records and their handshake to clk_rv (I2) ----
  output tsr_err_rec_t [5:0]          err_rec,
  output logic [5:0]                  err_valid,
  output logic [5:0][127:0]           err_hs,
  input  logic [5:0]                  err_ack,
  input  logic                        rec_resync,

  // ---- state ----
  output logic [5:0][TSR_SEQ_W-1:0]   seq_s,
  output logic [5:0][TSR_SEQ_W-1:0]   seq_d,
  output logic [31:0]                 q_pidx,
  output logic [5:0][8:0]             q_state,
  output logic [5:0]                  q_nonempty,
  output logic                        err_any,
  output logic                        abort_active,
  output logic [2:0]                  hold_src,
  output logic                        qidle,
  output logic                        qquiescent,

  // ---- PERF events (EX §6; gated by tsr_csr) ----
  output logic [2:0]                  perf_retired,
  output logic                        perf_push,
  output logic [5:0]                  perf_busy,
  output logic                        perf_mvm_wait,
  output logic                        perf_mvm_empty,
  output logic                        perf_hold,

  // ---- DB taps (CC §16) ----
  output logic                        tap_enq,
  output logic [3:0]                  tap_enq_unit,
  output logic [TSR_SEQ_W-1:0]        tap_enq_seq,
  output logic [31:0]                 tap_enq_pidx,
  output logic [3:0]                  tap_enq_code,
  output logic [31:0]                 tap_enq_info,
  output logic [1:0]                  tap_enq_kind,
  output logic [31:0]                 tap_enq_dep0,
  output logic [31:0]                 tap_enq_dep1,
  output tsr_insn_t                   tap_enq_w,
  output logic                        tap_enq_rec,
  output logic [31:0]                 tap_enq_pc,
  output tsr_ext_t [TSR_EXT_N-1:0]    tap_enq_ext,
  output logic [5:0]                  tap_start,
  output logic [5:0][31:0]            tap_start_pidx,
  output logic [5:0]                  tap_done,
  output logic [5:0][3:0]             tap_done_code,
  output logic [5:0]                  tap_done_ran,
  output logic [5:0][31:0]            tap_done_pidx,
  output logic                        tap_mark,
  output logic [31:0]                 tap_mark_value,
  output logic [31:0]                 tap_mark_pidx,
  output logic [31:0]                 tap_mark_pc,
  output logic [5:0]                  head_valid,
  output logic [5:0][31:0]            head_pidx,
  output logic [5:0][31:0]            head_w0
);

  // -------------------------------------------------------------------------------------------
  // Configuration
  // -------------------------------------------------------------------------------------------
  localparam int unsigned U_DMA = 0, U_QACT = 1, U_MVM = 2, U_VU = 3, U_ATT = 4, U_SYNC = 5;
  localparam int unsigned QD [6] = '{QD_DMA, QD_QACT, QD_MVM, QD_VU, QD_ATT, QD_SYNC};
  localparam logic [5:0] UNITS = {QD_SYNC != 0, QD_ATT != 0, QD_VU != 0, QD_MVM != 0, QD_QACT != 0,
                                  QD_DMA != 0};
  localparam int unsigned AMAX [6] = '{tsr_tables_pkg::TSR_ACCEPT_MAX_DMA, tsr_tables_pkg::TSR_ACCEPT_MAX_QACT,
                                       tsr_tables_pkg::TSR_ACCEPT_MAX_MVM, tsr_tables_pkg::TSR_ACCEPT_MAX_VU,
                                       tsr_tables_pkg::TSR_ACCEPT_MAX_ATT, tsr_tables_pkg::TSR_ACCEPT_MAX_SYNC};
  localparam bit AT_OPS = (CAPS & tsr_tables_pkg::TSR_CAPS_ATT) != 32'd0;   // KVST, ATTQK, ATTPV

  if (QD_SYNC == 0) begin : g_chk_sync
    $fatal(1, "tsr_issue: the SYNC unit must be built (CC §7.4)");
  end
  if (QD_DMA > TSR_QD_MAX || QD_QACT > TSR_QD_MAX || QD_MVM > TSR_QD_MAX || QD_VU > TSR_QD_MAX ||
      QD_ATT > TSR_QD_MAX || QD_SYNC > TSR_QD_MAX) begin : g_chk_qd
    $fatal(1, "tsr_issue: queue depths are at most 31 (Q_DEPTH field, EX §5.1)");
  end

  localparam logic [27:0] HALF28 = 28'h800_0000;
  localparam logic [31:0] TK_NONE = tsr_tables_pkg::TSR_TICKET_NONE;

  // Command payload layout per unit: {xc flag, external payload, native struct}, each part
  // present only where the unit needs it (SYNC: the FENCE / QCLEAR snapshot cnt[0..5]).
  localparam int unsigned DMAW  = $bits(tsr_dma_cmd_t);
  localparam int unsigned QACTW = $bits(tsr_qact_cmd_t);
  localparam int unsigned MVMW  = $bits(tsr_mvm_cmd_t);
  localparam int unsigned SNAPW = 6 * TSR_SEQ_W;
  localparam int unsigned NATMAX = MVMW;          // >= every native width (294 >= 168, 133, 101)
  function automatic int unsigned nat_w(int unsigned u);
    case (u)
      U_DMA:   return DMAW;
      U_QACT:  return QACTW;
      U_MVM:   return MVMW;
      U_SYNC:  return SNAPW;
      default: return 0;
    endcase
  endfunction
  function automatic bit has_xc(int unsigned u);
    return (u == U_VU) || (u == U_ATT) || (AT_OPS && (u == U_DMA || u == U_MVM));
  endfunction

  // Common part of a queue entry.
  typedef struct packed {
    logic [3:0]  code;     // poison code; 0 = runs (or SYNC kind)
    logic [31:0] info;
    logic [1:0]  kind;     // TSR_QK_*
    logic        nul;      // LD / ST with 0 lines
    logic [31:0] dep0;     // as dispatched (BAD_DEP dep -> none)
    logic [31:0] dep1;
    logic [31:0] pidx;
    logic [31:0] w0;
  } qcom_t;
  localparam int unsigned QCOMW = $bits(qcom_t);

  // -------------------------------------------------------------------------------------------
  // Ticket helpers (CC §7.5)
  // -------------------------------------------------------------------------------------------
  function automatic logic [27:0] sub28(logic [27:0] a, logic [27:0] b);
    return a - b;
  endfunction
  // ticket (v, s) done iff (D[v] - s) mod 2^28 in [1, 2^27); none is done; a reserved or absent
  // unit is never done
  function automatic logic tk_done(logic [31:0] t, logic [5:0][27:0] d);
    logic [3:0]  v;
    logic [27:0] x;
    v = t[31:28];
    if (v == 4'd15) return 1'b1;
    if (v >= 4'd6 || !UNITS[v[2:0]]) return 1'b0;
    x = sub28(d[v[2:0]], t[27:0]);
    return (x != 28'd0) && (x < HALF28);
  endfunction
  // failed(t): unit v in error state and s at or after ERR_SEQ[v] (CC §9.2)
  function automatic logic tk_failed(logic [31:0] t, tsr_err_rec_t [5:0] r);
    logic [3:0] v;
    v = t[31:28];
    if (v >= 4'd6 || !UNITS[v[2:0]]) return 1'b0;
    return r[v[2:0]].err && (sub28(t[27:0], r[v[2:0]].seq) < HALF28);
  endfunction

  // -------------------------------------------------------------------------------------------
  // State shared by the dispatcher and the heads
  // -------------------------------------------------------------------------------------------
  logic [5:0][27:0] s_q, d_q, cnt_q, clr_q;
  tsr_err_rec_t [5:0] rec_q;
  logic [5:0]       rec_dirty_q, rec_infl_q;
  logic [5:0][127:0] hs_q;                      // record copy in the handshake
  logic [31:0]      pidx_q;
  logic             hold_rv_q, abort_act_q;
  logic             abort_rq_q, clear_rq_q;     // registered requests, applied the next cycle

  // per-unit head signals (written in the generate blocks)
  logic [5:0]       hd_v, ev_v, cmd_v, nr_pend;
  qcom_t            hd_com [6];
  qcom_t            qf_com [6];                 // queue front (common part)
  logic [5:0][4:0]  q_cnt;
  logic [5:0]       d_inc, s_inc;               // D[u]++ / S[u]++ this cycle
  logic [5:0]       dec_any;                    // a decision this cycle (head leaves)
  logic [5:0]       dec_err;                    // the decision enters error state
  logic [5:0][3:0]  dec_code;
  logic [5:0]       dec_clear;                  // SYNC: a QCLEAR decided (applies the clear)
  logic [5:0][2:0]  stay_q;                     // registered stay reason (Q_STATE / PERF)
  logic [5:0][1:0]  run_out;                    // running ops held by the unit
  logic [5:0]       u_done;
  logic [SNAPW-1:0] sync_snap;                  // the SYNC head's snapshot (FENCE / QCLEAR)
  logic [5:0]       nr_fire_v;                  // a nonrunning completion fires (D[u]++)

  // -------------------------------------------------------------------------------------------
  // Checker
  // -------------------------------------------------------------------------------------------
  logic                   chk_req, chk_busy, chk_rsp;
  logic [3:0]             chk_unit, chk_code;
  logic [31:0]            chk_info;
  logic [1:0]             chk_kind;
  logic                   chk_null, chk_xc;
  tsr_ext_t [TSR_EXT_N-1:0] chk_ext;
  tsr_dma_cmd_t           chk_dma;
  tsr_qact_cmd_t          chk_qact;
  tsr_mvm_cmd_t           chk_mvm;
  logic [XC_CMD_W-1:0]    chk_xcmd;
  tsr_ifent_t             ent_q;

  tsr_chk #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CH_BYTES(CH_BYTES),
    .ADDR_BEAT_W(ADDR_BEAT_W), .CAPS(CAPS), .UNITS(UNITS), .XC_CMD_W(XC_CMD_W)
  ) u_chk (
    .clk, .rst_n,
    .req_valid(chk_req), .req_insn(ent_q.w), .busy(chk_busy),
    .rsp_valid(chk_rsp), .rsp_unit(chk_unit), .rsp_code(chk_code), .rsp_info(chk_info),
    .rsp_kind(chk_kind), .rsp_null(chk_null), .rsp_xc(chk_xc), .rsp_ext(chk_ext),
    .rsp_dma(chk_dma), .rsp_qact(chk_qact), .rsp_mvm(chk_mvm), .rsp_xcmd(chk_xcmd),
    .xc_req_valid, .xc_req_insn, .xc_rsp_valid, .xc_rsp_code, .xc_rsp_info, .xc_rsp_ext, .xc_rsp_cmd
  );

  // -------------------------------------------------------------------------------------------
  // Dispatcher (CC §8.3)
  // -------------------------------------------------------------------------------------------
  typedef enum logic [2:0] { D_IDLE, D_POP, D_CHK, D_ENQ } dstate_e;
  dstate_e dst_q;

  // the entry being enqueued
  logic [2:0]          eq_unit_q;
  qcom_t               eq_com_q;
  logic [NATMAX-1:0]   eq_nat_q;
  logic [XC_CMD_W-1:0] eq_xcmd_q;
  // read only by queues that carry the external payload (VU, AT, CAPS.ATT on DMA / MVM); none may
  // exist (tsr_core_dp without VU / AT, W4-A integrate)
  wire unused_eq_xcmd = &{1'b0, eq_xcmd_q};
  logic                eq_xc_q;
  tsr_ext_t [TSR_EXT_N-1:0] eq_ext_q;
  logic [1:0]          bad_q;           // BAD_DEP per dep (evaluated at D_POP)
  logic [31:0]         tk_q;            // the entry's ticket

  wire        pop_ctrl  = (ent_q.kind == tsr_tables_pkg::TSR_IFIFO_KIND_CTRL);
  wire        pop_op    = (ent_q.kind == tsr_tables_pkg::TSR_IFIFO_KIND_OP);
  wire [3:0]  pop_code  = ent_q.w[0][3:0];
  wire        pop_qclr  = pop_ctrl && pop_code == tsr_tables_pkg::TSR_IFIFO_CTRL_CODE_QCLEAR;

  function automatic logic dep_bad(logic [31:0] t, logic [5:0][27:0] cnt, logic [5:0][27:0] clr);
    logic [3:0]  v;
    logic [27:0] c;
    v = t[31:28];
    if (v == 4'd15) return 1'b0;
    if (v >= 4'd6 || !UNITS[v[2:0]]) return 1'b1;
    if (sub28(t[27:0], cnt[v[2:0]]) < HALF28) return 1'b1;          // not pushed yet
    c = sub28(clr[v[2:0]], t[27:0]);
    return (c != 28'd0) && (c < HALF28);                             // before the last QCLEAR
  endfunction

  // the dispatcher's entry for unit u (QABORT record when the unit's queue is empty)
  logic       disp_has;
  logic [2:0] disp_unit, disp_unit_q;    // disp_unit_q: unit of the op in tsr_chk (from D_POP)
  logic [3:0] pop_uof;
  logic [31:0] pop_need;
  always_comb begin
    pop_uof   = tsr_tables_pkg::tsr_op_unit(ent_q.w[0][7:0]);
    pop_need  = tsr_tables_pkg::tsr_op_caps(ent_q.w[0][7:0]);
    disp_has  = 1'b0;
    disp_unit = 3'(U_SYNC);
    if (dst_q == D_POP && pop_qclr) begin
      disp_has = 1'b1;
    end else if (dst_q == D_POP && pop_op) begin
      disp_has = 1'b1;
      if (tsr_tables_pkg::tsr_op_legal(ent_q.w[0][7:0]) && ((CAPS & pop_need) == pop_need) &&
          UNITS[pop_uof[2:0]])
        disp_unit = pop_uof[2:0];
    end else if (dst_q == D_CHK) begin
      disp_has  = 1'b1;
      disp_unit = chk_rsp ? chk_unit[2:0] : disp_unit_q;
    end else if (dst_q == D_ENQ) begin
      disp_has  = 1'b1;
      disp_unit = eq_unit_q;
    end
  end
  always_ff @(posedge clk) if (dst_q == D_POP) disp_unit_q <= disp_unit;

  // enqueue (D_ENQ): the queue of eq_unit_q has room (it always has: credits, CC §7.3)
  logic [5:0] q_full;
  wire        enq_go = (dst_q == D_ENQ) && !q_full[eq_unit_q];
  logic [5:0] enq_we;
  always_comb begin
    enq_we = '0;
    if (enq_go) enq_we[eq_unit_q] = 1'b1;
  end

  assign ifq_ready = (dst_q == D_IDLE) && ifq_valid;
  assign chk_req   = (dst_q == D_POP) && pop_op;

  // merged result (D_CHK -> D_ENQ)
  logic [3:0]  m_code;
  logic [31:0] m_info;
  logic [1:0]  m_kind;
  always_comb begin
    m_code = chk_code;
    m_info = chk_info;
    if ((bad_q != 2'b00) && (chk_code == TSR_ERR_NONE ||
         tsr_tables_pkg::tsr_err_check_rank(chk_code) > tsr_tables_pkg::tsr_err_check_rank(TSR_ERR_BAD_DEP))) begin
      m_code = TSR_ERR_BAD_DEP;
      m_info = 32'd0;
    end
    m_kind = (m_code == TSR_ERR_NONE) ? chk_kind : TSR_QK_OP;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      dst_q       <= D_IDLE;
      hold_rv_q   <= 1'b0;
      abort_rq_q  <= 1'b0;
      clear_rq_q  <= 1'b0;
    end else begin
      abort_rq_q <= qabort || (dst_q == D_POP && pop_ctrl && pop_code == tsr_tables_pkg::TSR_IFIFO_CTRL_CODE_QABORT);
      clear_rq_q <= host_clear;
      if (rv_reset) hold_rv_q <= 1'b0;
      case (dst_q)
        D_IDLE: if (ifq_valid) dst_q <= D_POP;
        D_POP: begin
          if (pop_op) begin
            dst_q <= D_CHK;
          end else if (pop_qclr) begin
            dst_q <= D_ENQ;
          end else begin
            dst_q <= D_IDLE;
            if (pop_ctrl && pop_code == tsr_tables_pkg::TSR_IFIFO_CTRL_CODE_QHOLD_SET) hold_rv_q <= 1'b1;
            if (pop_ctrl && pop_code == tsr_tables_pkg::TSR_IFIFO_CTRL_CODE_QHOLD_CLR) hold_rv_q <= 1'b0;
          end
        end
        D_CHK: if (chk_rsp) dst_q <= D_ENQ;
        D_ENQ: if (enq_go) dst_q <= D_IDLE;
        default: dst_q <= D_IDLE;
      endcase
    end
  end

  // entry capture and enqueue payload
  always_ff @(posedge clk) begin
    if (ifq_ready) ent_q <= ifq_entry;
    if (dst_q == D_POP) begin
      bad_q <= {dep_bad(ent_q.dep1, cnt_q, clr_q), dep_bad(ent_q.dep0, cnt_q, clr_q)};
      tk_q  <= ent_q.ticket;
      if (pop_qclr) begin                      // QCLEAR: a SYNC entry with the snapshot cnt[0..5]
        eq_unit_q     <= 3'(U_SYNC);
        eq_com_q      <= '{code: TSR_ERR_NONE, info: 32'd0, kind: TSR_QK_QCLEAR, nul: 1'b0,
                           dep0: TK_NONE, dep1: TK_NONE, pidx: pidx_q, w0: ent_q.w[0]};
        eq_nat_q      <= NATMAX'(cnt_q);
        eq_xcmd_q     <= '0;
        eq_xc_q       <= 1'b0;
        eq_ext_q      <= '0;
      end
    end
    if (dst_q == D_CHK && chk_rsp) begin
      eq_unit_q <= chk_unit[2:0];
      eq_com_q  <= '{code: m_code, info: m_info, kind: m_kind,
                     nul: chk_null && m_code == TSR_ERR_NONE,
                     dep0: bad_q[0] ? TK_NONE : ent_q.dep0, dep1: bad_q[1] ? TK_NONE : ent_q.dep1,
                     pidx: pidx_q, w0: ent_q.w[0]};
      case (chk_unit)
        4'(U_DMA):  eq_nat_q <= NATMAX'(chk_dma);
        4'(U_QACT): eq_nat_q <= NATMAX'(chk_qact);
        4'(U_MVM):  eq_nat_q <= NATMAX'(chk_mvm);
        default:    eq_nat_q <= (m_kind == TSR_QK_FENCE) ? NATMAX'(cnt_q) : '0;
      endcase
      eq_xcmd_q <= chk_xcmd;
      eq_xc_q   <= chk_xc;
      eq_ext_q  <= (m_code == TSR_ERR_NONE) ? chk_ext : '0;
    end
  end

  // counters: cnt[u] / pidx at enqueue
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cnt_q  <= {6{SEQ_INIT}};
      pidx_q <= PIDX_INIT;
    end else if (enq_go) begin
      cnt_q[eq_unit_q] <= cnt_q[eq_unit_q] + 28'd1;
      pidx_q           <= pidx_q + 32'd1;
    end
  end

  // taps: enqueue, TRACE_MARK
  assign tap_enq      = enq_go;
  assign tap_enq_unit = {1'b0, eq_unit_q};
  assign tap_enq_seq  = cnt_q[eq_unit_q];
  assign tap_enq_pidx = eq_com_q.pidx;
  assign tap_enq_code = eq_com_q.code;
  assign tap_enq_info = eq_com_q.info;
  assign tap_enq_kind = eq_com_q.kind;
  assign tap_enq_dep0 = eq_com_q.dep0;
  assign tap_enq_dep1 = eq_com_q.dep1;
  assign tap_enq_w    = ent_q.w;
  assign tap_enq_rec  = ent_q.rec;
  assign tap_enq_pc   = ent_q.pc;
  assign tap_enq_ext  = eq_ext_q;
  assign tap_mark       = (dst_q == D_POP) && pop_ctrl && pop_code == tsr_tables_pkg::TSR_IFIFO_CTRL_CODE_TRACE_MARK;
  assign tap_mark_value = ent_q.w[1];
  assign tap_mark_pidx  = pidx_q;
  assign tap_mark_pc    = ent_q.pc;
  assign perf_push      = enq_go;

  // -------------------------------------------------------------------------------------------
  // Unit ports as arrays
  // -------------------------------------------------------------------------------------------
  logic [5:0]          u_rdy;
  logic [NATMAX-1:0]   u_nat [6];
  logic [XC_CMD_W-1:0] u_xcmd [6];
  logic [5:0]          u_xc;
  assign u_rdy  = {1'b0, att_cmd_ready, vu_cmd_ready, mvm_cmd_ready, qact_cmd_ready, dma_cmd_ready};
  assign u_done = {1'b0, att_done, vu_done, mvm_done, qact_done, dma_done};

  assign dma_cmd_valid  = cmd_v[U_DMA];
  assign dma_cmd        = tsr_dma_cmd_t'(u_nat[U_DMA][DMAW-1:0]);
  assign dma_cmd_xc     = u_xc[U_DMA];
  assign dma_xcmd       = u_xcmd[U_DMA];
  assign qact_cmd_valid = cmd_v[U_QACT];
  assign qact_cmd       = tsr_qact_cmd_t'(u_nat[U_QACT][QACTW-1:0]);
  assign mvm_cmd_valid  = cmd_v[U_MVM];
  assign mvm_cmd        = tsr_mvm_cmd_t'(u_nat[U_MVM][MVMW-1:0]);
  assign mvm_cmd_xc     = u_xc[U_MVM];
  assign mvm_xcmd       = u_xcmd[U_MVM];
  assign vu_cmd_valid   = cmd_v[U_VU];
  assign vu_cmd         = u_xcmd[U_VU];
  assign att_cmd_valid  = cmd_v[U_ATT];
  assign att_cmd        = u_xcmd[U_ATT];

  // -------------------------------------------------------------------------------------------
  // Heads (CC §9.1-§9.3), one per unit
  // -------------------------------------------------------------------------------------------
  localparam logic [2:0] ST_NONE = 3'd0, ST_DEP = 3'd1, ST_HARD = 3'd2, ST_HOLD = 3'd3, ST_BUSY = 3'd4;

  wire hold_any   = host_hold || hold_rv_q || halt_hold;
  wire no_dec     = abort_rq_q;          // the abort applies this cycle: no head decides

  // SOFT_RESET raises sr_busy and its QABORT in the same cycle, but a QABORT is registered here
  // (abort_rq_q) and applies one cycle later (abort_act_q, the unit error records): two cycles
  // after sr_busy rises. The hard hold must stay on until then, or a held NOP / FENCE / error-coded
  // head would complete with its own code in those two cycles while the simulator, which aborts
  // first (EX §8.5 step 1), gives ABORTED. sr_off_q is sr_busy delayed by that latency; the
  // reset still finishes because abort_act_q takes over from the cycle it applies and DB forces
  // dbg_hold to 0 from SR_BUSY's third cycle on (tsr_dbg).
  logic [1:0] sr_dly_q;
  wire        sr_off_q = sr_dly_q[1];
  always_ff @(posedge clk) begin
    if (!rst_n) sr_dly_q <= 2'b00;
    else        sr_dly_q <= {sr_dly_q[0], sr_busy};
  end

  for (genvar gu = 0; gu < 6; gu++) begin : g_u
    localparam int unsigned NW = nat_w(gu);
    localparam bit          HX = has_xc(gu);
    localparam int unsigned XW = HX ? XC_CMD_W : 0;
    localparam int unsigned XF = (HX && NW != 0) ? 1 : 0;
    localparam int unsigned CW = NW + XW + XF;
    localparam int unsigned PW = QCOMW + CW;

    if (QD[gu] == 0) begin : g_absent
      assign hd_v[gu]       = 1'b0;
      assign ev_v[gu]       = 1'b0;
      assign cmd_v[gu]      = 1'b0;
      assign nr_pend[gu]    = 1'b0;
      assign hd_com[gu]     = '0;
      assign qf_com[gu]     = '0;
      assign q_cnt[gu]      = '0;
      assign q_full[gu]     = 1'b1;
      assign d_inc[gu]      = 1'b0;
      assign nr_fire_v[gu]  = 1'b0;
      assign s_inc[gu]      = 1'b0;
      assign dec_any[gu]    = 1'b0;
      assign dec_err[gu]    = 1'b0;
      assign dec_code[gu]   = '0;
      assign dec_clear[gu]  = 1'b0;
      assign stay_q[gu]     = ST_NONE;
      assign run_out[gu]    = '0;
      assign u_nat[gu]      = '0;
      assign u_xcmd[gu]     = '0;
      assign u_xc[gu]       = 1'b0;
      assign tap_start[gu]      = 1'b0;
      assign tap_start_pidx[gu] = '0;
      assign tap_done[gu]       = 1'b0;
      assign tap_done_code[gu]  = '0;
      assign tap_done_ran[gu]   = 1'b0;
      assign tap_done_pidx[gu]  = '0;
    end else begin : g_q
      // ---- queue ----
      logic [PW-1:0] q_front, q_wdata;
      logic [CW-1:0] enq_cmd;
      logic          q_re;
      if (gu == U_SYNC) begin : g_cs
        assign enq_cmd = eq_nat_q[SNAPW-1:0];
      end else if (NW != 0 && HX) begin : g_cnx
        assign enq_cmd = {eq_xc_q, eq_xcmd_q, eq_nat_q[NW-1:0]};
      end else if (NW != 0) begin : g_cn
        assign enq_cmd = eq_nat_q[NW-1:0];
      end else begin : g_cx
        assign enq_cmd = eq_xcmd_q;
      end
      assign q_wdata = {eq_com_q, enq_cmd};

      tsr_issue_q #(.W(PW), .D(QD[gu])) u_q (
        .clk, .rst_n, .we(enq_we[gu]), .wdata(q_wdata), .re(q_re), .front(q_front), .count(q_cnt[gu])
      );
      assign q_full[gu] = (32'(q_cnt[gu]) >= QD[gu]);
      assign qf_com[gu] = qcom_t'(q_front[PW-1:CW]);

      // ---- head register, evaluation flops, command register ----
      logic          hd_v_q, ev_v_q, cmd_v_q, nr_q, nr_ran_q, nr_clr_q;
      logic [3:0]    nr_code_q;
      logic [31:0]   nr_pidx_q;
      qcom_t         hd_q;
      logic [CW-1:0] hd_cmd_q, cmd_q;
      logic          dep_ok_q, snap_ok_q, failed_q, hold_q;
      logic [2:0]    stay_r_q;
      logic [1:0]    run_q;
      logic [31:0]   inf_pidx_q [4];          // running ops: pidx in accept order (I3 <= 2)
      logic [1:0]    inf_rd_q, inf_wr_q;
      logic [31:0]   hd_pidx_issue_q;          // pidx of the offered command

      wire  xfer   = cmd_v_q && u_rdy[gu];
      wire  is_snap = (hd_q.kind == TSR_QK_FENCE || hd_q.kind == TSR_QK_QCLEAR);
      wire  [27:0] d_next = d_q[gu] + 28'(u_done[gu]);
      wire  drained = (s_q[gu] == d_next) && (run_q == 2'(u_done[gu]));
      wire  room    = (32'(run_q) - 32'(u_done[gu])) < AMAX[gu];
      wire  hard    = dbg_hold[gu] && !abort_act_q && !sr_off_q && !dbg_step[gu];
      wire  hold_s  = (hold_q || dbg_hold[gu]) && !dbg_step[gu];
      wire  can     = hd_v_q && ev_v_q && !cmd_v_q && !nr_q && !no_dec;

      // decision (CC §9.3, the simulator's q_head order)
      logic       d_issue, d_nr, d_ran, d_clr;
      logic [3:0] d_code;
      logic [2:0] d_stay;
      always_comb begin
        d_issue = 1'b0;
        d_nr    = 1'b0;
        d_ran   = 1'b0;
        d_clr   = 1'b0;
        d_code  = TSR_ERR_NONE;
        d_stay  = ST_NONE;
        if (!dep_ok_q || (is_snap && !snap_ok_q)) begin
          d_stay = ST_DEP;
        end else if (hard) begin
          d_stay = ST_HARD;
        end else if (hd_q.kind == TSR_QK_QCLEAR) begin
          d_nr  = drained;
          d_clr = drained;
        end else if (hd_q.code != TSR_ERR_NONE) begin
          d_nr   = drained;
          d_code = hd_q.code;
        end else if (rec_q[gu].err) begin
          d_nr   = drained;
          d_code = TSR_ERR_ABORTED;
        end else if (failed_q) begin
          d_nr   = drained;
          d_code = TSR_ERR_DEP_FAILED;
        end else if (hd_q.kind == TSR_QK_NOP || hd_q.kind == TSR_QK_FENCE) begin
          d_nr = drained;
        end else if (hold_s) begin
          d_stay = ST_HOLD;
        end else if (hd_q.nul) begin
          d_nr  = drained;
          d_ran = drained;
        end else if (u_rdy[gu] && room) begin
          d_issue = 1'b1;
        end
        if (!d_issue && !d_nr && d_stay == ST_NONE) d_stay = ST_BUSY;
        if (!can) begin
          d_issue = 1'b0;
          d_nr    = 1'b0;
          d_ran   = 1'b0;
          d_clr   = 1'b0;
        end
      end
      wire leave = d_issue || d_nr;
      // the head register refills from the queue when empty or leaving
      assign q_re = (q_cnt[gu] != 5'd0) && (!hd_v_q || leave);

      // nonrunning completion fires: one cycle after the decision, after the record handshake
      // when it is the error entry (I2) or a QCLEAR (CC §9.5)
      wire nr_block = (rec_q[gu].err && d_q[gu] == rec_q[gu].seq && (rec_dirty_q[gu] || rec_infl_q[gu])) ||
                      (nr_clr_q && ((rec_dirty_q | rec_infl_q) != 6'd0));
      wire nr_fire  = nr_q && !nr_block;

      always_ff @(posedge clk) begin
        if (!rst_n) begin
          hd_v_q   <= 1'b0;
          ev_v_q   <= 1'b0;
          cmd_v_q  <= 1'b0;
          nr_q     <= 1'b0;
          run_q    <= '0;
          inf_rd_q <= '0;
          inf_wr_q <= '0;
          stay_r_q <= ST_NONE;
        end else begin
          // head register
          if (q_re) begin
            hd_v_q   <= 1'b1;
            ev_v_q   <= 1'b0;
            hd_q     <= qcom_t'(q_front[PW-1:CW]);
            hd_cmd_q <= q_front[CW-1:0];
          end else if (leave) begin
            hd_v_q <= 1'b0;
            ev_v_q <= 1'b0;
          end else if (hd_v_q) begin
            ev_v_q <= 1'b1;
          end
          // command register
          if (d_issue) begin
            cmd_v_q <= 1'b1;
            cmd_q   <= hd_cmd_q;
          end else if (xfer) begin
            cmd_v_q <= 1'b0;
          end
          // nonrunning completion
          if (d_nr) begin
            nr_q      <= 1'b1;
            nr_ran_q  <= d_ran;
            nr_clr_q  <= d_clr;
            nr_code_q <= d_code;
            nr_pidx_q <= hd_q.pidx;
          end else if (nr_fire) begin
            nr_q <= 1'b0;
          end
          // running ops held
          run_q <= run_q + 2'(xfer) - 2'(u_done[gu]);
          if (xfer) begin
            inf_pidx_q[inf_wr_q] <= hd_pidx_issue_q;
            inf_wr_q <= inf_wr_q + 2'd1;
          end
          if (u_done[gu]) inf_rd_q <= inf_rd_q + 2'd1;
          stay_r_q <= (hd_v_q && ev_v_q && !cmd_v_q && !nr_q) ? d_stay : ST_NONE;
        end
      end
      always_ff @(posedge clk) if (d_issue) hd_pidx_issue_q <= hd_q.pidx;

      // evaluation flops (registered every cycle for the current head)
      if (gu == U_SYNC) begin : g_snap
        logic [5:0] sv;
        always_comb begin
          for (int v = 0; v < 6; v++)
            sv[v] = !UNITS[v] || (sub28(d_q[v], hd_cmd_q[v * TSR_SEQ_W +: TSR_SEQ_W]) < HALF28);
        end
        always_ff @(posedge clk) snap_ok_q <= &sv;
        assign sync_snap = hd_cmd_q[SNAPW-1:0];
      end else begin : g_nosnap
        always_ff @(posedge clk) snap_ok_q <= 1'b1;
      end
      always_ff @(posedge clk) begin
        dep_ok_q <= tk_done(hd_q.dep0, d_q) && tk_done(hd_q.dep1, d_q);
        failed_q <= tk_failed(hd_q.dep0, rec_q) || tk_failed(hd_q.dep1, rec_q);
        hold_q   <= hold_any;
      end

      // outputs
      assign hd_v[gu]      = hd_v_q;
      assign ev_v[gu]      = ev_v_q;
      assign cmd_v[gu]     = cmd_v_q;
      assign nr_pend[gu]   = nr_q;
      assign hd_com[gu]    = hd_q;
      assign d_inc[gu]     = u_done[gu] || nr_fire;
      assign nr_fire_v[gu] = nr_fire;
      assign s_inc[gu]     = xfer || d_nr;
      assign dec_any[gu]   = leave;
      assign dec_err[gu]   = d_nr && d_code != TSR_ERR_NONE;
      assign dec_code[gu]  = d_code;
      assign dec_clear[gu] = d_clr;
      assign stay_q[gu]    = stay_r_q;
      assign run_out[gu]   = run_q;
      if (NW != 0) begin : g_onat
        assign u_nat[gu] = NATMAX'(cmd_q[NW-1:0]);
      end else begin : g_onat0
        assign u_nat[gu] = '0;
      end
      if (HX) begin : g_ox
        assign u_xcmd[gu] = cmd_q[NW +: XC_CMD_W];
        assign u_xc[gu]   = (XF != 0) ? cmd_q[CW-1] : 1'b1;
      end else begin : g_ox0
        assign u_xcmd[gu] = '0;
        assign u_xc[gu]   = 1'b0;
      end
      assign tap_start[gu]      = xfer;
      assign tap_start_pidx[gu] = hd_pidx_issue_q;
      assign tap_done[gu]       = d_inc[gu];
      assign tap_done_code[gu]  = nr_fire ? nr_code_q : TSR_ERR_NONE;
      assign tap_done_ran[gu]   = nr_fire ? nr_ran_q : 1'b1;
      assign tap_done_pidx[gu]  = nr_fire ? nr_pidx_q : inf_pidx_q[inf_rd_q];
    end
  end

  // -------------------------------------------------------------------------------------------
  // Counters, error records, QCLEAR / QABORT, record handshake
  // -------------------------------------------------------------------------------------------
  // the QCLEAR decided at the SYNC head (its snapshot = the SYNC head's command)
  logic [5:0][27:0] clr_snap;
  assign clr_snap = sync_snap;

  // QABORT record of unit u: the first entry not started
  function automatic logic [63:0] abort_entry(int unsigned u, logic hv, logic [63:0] hc, logic [4:0] qn,
                                              logic [63:0] fc, logic dh, logic [2:0] du, logic [31:0] w0d,
                                              logic [31:0] pq);
    if (hv)                 return hc;
    if (qn != 5'd0)         return fc;
    if (dh && 32'(du) == u) return {w0d, pq};
    return {32'd0, pq};
  endfunction

  logic [5:0] err_set_dec;              // a decision enters error state (unit not in error)
  always_comb begin
    for (int u = 0; u < 6; u++) err_set_dec[u] = dec_err[u] && !rec_q[u].err;
  end
  wire clear_now = clear_rq_q || (dec_clear[U_SYNC]);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s_q         <= {6{SEQ_INIT}};
      d_q         <= {6{SEQ_INIT}};
      clr_q       <= {6{SEQ_INIT}};
      rec_q       <= '0;
      rec_dirty_q <= '0;
      rec_infl_q  <= '0;
      hs_q        <= '0;
      abort_act_q <= 1'b0;
    end else begin
      for (int u = 0; u < 6; u++) begin
        if (s_inc[u]) s_q[u] <= s_q[u] + 28'd1;
        if (d_inc[u]) d_q[u] <= d_q[u] + 28'd1;
        // record handshake: a pending change is copied into the shadow and sent; ack ends it
        if (!rec_infl_q[u] && rec_dirty_q[u]) begin
          rec_dirty_q[u] <= 1'b0;
          rec_infl_q[u]  <= 1'b1;
          hs_q[u]        <= {rec_q[u].pidx, rec_q[u].w0, rec_q[u].info,
                             rec_q[u].err ? rec_q[u].code : 4'd0, rec_q[u].seq};
        end
        if (rec_infl_q[u] && err_ack[u]) rec_infl_q[u] <= 1'b0;
      end
      // clk_rv is OK again: its copy of a record may be stale (the handshake was answered locally while
      // it was not), so send every unit's record once more. D is held at ERR_SEQ meanwhile (I2), which
      // is safe: the handshake completes, or is answered locally again.
      if (rec_resync) rec_dirty_q <= rec_dirty_q | UNITS;
      // errors from head decisions
      for (int u = 0; u < 6; u++) begin
        if (err_set_dec[u]) begin
          rec_q[u] <= '{err: 1'b1, seq: s_q[u], code: dec_code[u],
                        info: (hd_com[u].code != TSR_ERR_NONE) ? hd_com[u].info : 32'd0,
                        w0: hd_com[u].w0, pidx: hd_com[u].pidx};
          rec_dirty_q[u] <= 1'b1;
        end
      end
      // QABORT (applies in a cycle without decisions)
      if (abort_rq_q) begin
        abort_act_q <= 1'b1;
        for (int u = 0; u < 6; u++) begin
          if (UNITS[u] && !rec_q[u].err) begin
            automatic logic [63:0] ae = abort_entry(u, hd_v[u], {hd_com[u].w0, hd_com[u].pidx}, q_cnt[u],
                                                    {qf_com[u].w0, qf_com[u].pidx}, disp_has, disp_unit,
                                                    ent_q.w[0], pidx_q);
            rec_q[u] <= '{err: 1'b1, seq: s_q[u] + 28'(cmd_v[u]), code: TSR_ERR_ABORTED, info: 32'd0,
                          w0: ae[63:32], pidx: ae[31:0]};
            rec_dirty_q[u] <= 1'b1;
          end
        end
      end
      // QCLEAR (wins over an error recorded in the same cycle)
      if (clear_now) begin
        abort_act_q <= 1'b0;
        for (int u = 0; u < 6; u++) begin
          rec_q[u] <= '0;
          if (rec_q[u].err || err_set_dec[u] || (abort_rq_q && UNITS[u])) rec_dirty_q[u] <= 1'b1;
          clr_q[u] <= clear_rq_q ? (cnt_q[u] + 28'((enq_go && 32'(eq_unit_q) == u) ? 1 : 0))
                                 : clr_snap[u];
        end
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Outputs
  // -------------------------------------------------------------------------------------------
  assign err_rec   = rec_q;
  assign err_valid = rec_infl_q;
  assign err_hs    = hs_q;
  assign seq_s    = s_q;
  assign seq_d    = d_q;
  assign q_pidx   = pidx_q;
  assign abort_active = abort_act_q;
  assign hold_src = {halt_hold, hold_rv_q, host_hold};

  logic [5:0] busy_u;      // the unit holds something: a running op, an offered command, a pending completion
  always_comb begin
    err_any = 1'b0;
    for (int u = 0; u < 6; u++) begin
      busy_u[u]     = (run_out[u] != 2'd0) || cmd_v[u] || nr_pend[u];
      q_nonempty[u] = (q_cnt[u] != 5'd0) || hd_v[u];
      perf_busy[u]  = (run_out[u] != 2'd0);
      err_any       = err_any || rec_q[u].err;
      q_state[u]    = {rec_q[u].err, (stay_q[u] == ST_HOLD || stay_q[u] == ST_HARD), (stay_q[u] == ST_DEP),
                       (run_out[u] != 2'd0), 5'(q_cnt[u] + 5'(hd_v[u]))};
      head_valid[u] = hd_v[u];
      head_pidx[u]  = hd_com[u].pidx;
      head_w0[u]    = hd_com[u].w0;
    end
  end

  // QIDLE: nothing queued, running or in the issue FIFO; QQUIESCENT (CC §9.6): no unit holds
  // anything, the FIFO and the dispatcher are idle, and every non-empty queue's head is held or
  // waits for its deps (registered decision).
  logic qidle_q, qq_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      qidle_q <= 1'b1;
      qq_q    <= 1'b1;
    end else begin
      qidle_q <= !ifq_valid && dst_q == D_IDLE && (busy_u == 6'd0) && (q_nonempty == 6'd0);
      qq_q    <= !ifq_valid && dst_q == D_IDLE && (busy_u == 6'd0) && (d_inc == 6'd0);
      for (int u = 0; u < 6; u++)
        if (q_nonempty[u] && !(hd_v[u] && ev_v[u] && (stay_q[u] == ST_DEP || stay_q[u] == ST_HOLD ||
                                                      stay_q[u] == ST_HARD)))
          qq_q <= 1'b0;
    end
  end
  assign qidle      = qidle_q;
  assign qquiescent = qq_q;

  // PERF events
  always_comb begin
    perf_retired = '0;
    for (int u = 0; u < 6; u++) perf_retired = perf_retired + 3'(d_inc[u]);
  end
  assign perf_mvm_wait  = UNITS[U_MVM] && !busy_u[U_MVM] && hd_v[U_MVM] && ev_v[U_MVM] &&
                          (stay_q[U_MVM] == ST_DEP || stay_q[U_MVM] == ST_HOLD || stay_q[U_MVM] == ST_HARD);
  assign perf_mvm_empty = UNITS[U_MVM] && !busy_u[U_MVM] && !q_nonempty[U_MVM];
  always_comb begin
    perf_hold = 1'b0;
    for (int u = 0; u < 6; u++)
      if (hd_v[u] && (stay_q[u] == ST_HOLD || stay_q[u] == ST_HARD)) perf_hold = 1'b1;
  end

  // unused: absent units' bits, the xc flag where no unit has both a native and an external
  // command, the checker's busy (the dispatcher waits for rsp_valid)
  wire unused_ok = &{1'b0, chk_busy, eq_xc_q, eq_nat_q, pop_uof[3], enq_we, u_rdy, dbg_hold, dbg_step};

  // -------------------------------------------------------------------------------------------
  // Invariants (simulation only; CC §9.4)
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      // dispatcher: the ticket equals its own count (CC §7.2, §8.3)
      if (dst_q == D_CHK && chk_rsp)
        assert (tk_q == {chk_unit, cnt_q[chk_unit[2:0]]})
          else $error("tsr_issue: ticket %08h of an OP entry, dispatcher expects unit %0d seq %0d",
                      tk_q, chk_unit, cnt_q[chk_unit[2:0]]);
      if (dst_q == D_POP && pop_qclr)
        assert (ent_q.ticket == {4'(U_SYNC), cnt_q[U_SYNC]})
          else $error("tsr_issue: QCLEAR ticket %08h, dispatcher expects SYNC seq %0d", ent_q.ticket, cnt_q[U_SYNC]);
      if (dst_q == D_POP)
        assert (pop_op || (pop_ctrl && pop_code >= 4'd1 && pop_code <= 4'd5))
          else $error("tsr_issue: issue-FIFO entry kind %0d code %0d is reserved", ent_q.kind, pop_code);
      if (dst_q == D_ENQ)
        assert (!q_full[eq_unit_q]) else $error("tsr_issue: queue %0d full at enqueue (credits)", eq_unit_q);
      for (int u = 0; u < 6; u++) begin
        // I1: decided at the head only with both deps (and the snapshot) done
        if (dec_any[u])
          assert (tk_done(hd_com[u].dep0, d_q) && tk_done(hd_com[u].dep1, d_q))
            else $error("tsr_issue: I1: unit %0d entry pidx %0d decided before its deps", u, hd_com[u].pidx);
        if (dec_any[u] && u == U_SYNC && (hd_com[u].kind == TSR_QK_FENCE || hd_com[u].kind == TSR_QK_QCLEAR))
          for (int v = 0; v < 6; v++)
            assert (!UNITS[v] || sub28(d_q[v], sync_snap[v * TSR_SEQ_W +: TSR_SEQ_W]) < HALF28)
              else $error("tsr_issue: I1: FENCE / QCLEAR pidx %0d decided before unit %0d reached its snapshot",
                          hd_com[u].pidx, v);
        // I3
        assert (sub28(s_q[u], d_q[u]) <= 28'(AMAX[u]))
          else $error("tsr_issue: I3: unit %0d S - D = %0d above its accept limit %0d", u,
                      sub28(s_q[u], d_q[u]), AMAX[u]);
        assert (!(u_done[u] && run_out[u] == 2'd0))
          else $error("tsr_issue: I3: unit %0d done without an op held", u);
        assert (!(u_done[u] && nr_fire_v[u]))
          else $error("tsr_issue: two D steps of unit %0d in one cycle", u);
        // I2: D[u] passes ERR_SEQ[u] only with the record visible on the clk_rv side
        if (d_inc[u] && rec_q[u].err && d_q[u] == rec_q[u].seq)
          assert (!rec_dirty_q[u] && !rec_infl_q[u])
            else $error("tsr_issue: I2: unit %0d D passes ERR_SEQ %0d before the record handshake", u, rec_q[u].seq);
        assert (!(err_ack[u] && !rec_infl_q[u])) else $error("tsr_issue: err_ack of unit %0d without a handshake", u);
      end
    end
  end
`endif

endmodule
