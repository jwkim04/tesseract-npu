// tsr_dbg_hist.sv — DB's dispatch history ring (docs/design/05-debug.md "DB" §5.6, §10.1
// DBG_HIST_*): every ticketed entry the dispatcher enqueues (ops and QCLEAR, poisoned entries
// included) with its push index, ticket, deps as pushed, the eight words, the push PC, the
// dispatch time MTIME[31:0] and INFO, in slot pidx mod DEPTH.
//
// Owner: DB (W4-B item 6). Instantiated by tsr_dbg. Oracle: the simulator's DBG model
// (sw/sim/tsr_dbg.c hist_recording / hist_frozen / on_enq / hist_clear and the DBG_HIST_* reads).
//
// Recording (stage e1 of tsr_dbg, one cycle after the tap): while HIST_EN, not HIST_FREEZE, not
//   frozen by UNIT_HUNG, and not more than DEPTH / 2 entries after the first entry that completed
//   with an error since the last HIST_CLEAR (HIST_FREEZE_ERR); DBG_HIST_HEAD := pidx + 1.
//   INFO's tile-breakpoint bits and the stop-point bit come two cycles later (tsr_dbg_stop's
//   stage 3: after the entry's match-queue pushes) into a side array of the same slots.
// HIST_FROZEN = HIST_FREEZE, or the UNIT_HUNG freeze, or HIST_FREEZE_ERR with DBG_HIST_HEAD at or
//   beyond the first error entry + DEPTH / 2. HIST_CLEAR empties the ring (every slot not
//   present), ends both freezes and sets DBG_HIST_HEAD := DBG_NEXT_PIDX.
// Reading: the slot of DBG_HIST_IDX is read continuously (block RAM, one cycle) into registers;
//   INFO.PRESENT = the slot was written since the last HIST_CLEAR and holds that push index. A slot
//   that is not PRESENT reads 0 in every field (INFO included), as in the simulator: the RAM is not
//   cleared, so without the gate it would show a wrapped-over entry or the power-up content. The
//   gated fields are registered (one more cycle of read latency behind DBG_HIST_IDX).
// DEPTH = 0: no history (DBG_ID bit8 = 0, every field reads 0). Otherwise a power of two >= 2.
// Storage (U200, DEPTH 512): main RAM 512 x 456 b (pidx, seq, unit, code, REC, deps, w0-w7, PC,
//   time) = 7 RAMB36 at 72 bits, side RAM 512 x 8 b, DEPTH valid flops.

module tsr_dbg_hist
  import tsr_pkg::*;
#(
  parameter int unsigned DEPTH = tsr_tables_pkg::TSR_HIST_DEPTH
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // ---- configuration and events ----
  input  logic                  cfg_en,
  input  logic                  cfg_freeze,
  input  logic                  cfg_freeze_err,
  input  logic                  clear,             // DBG_CTRL.HIST_CLEAR
  input  logic                  unit_hung,         // STATUS.UNIT_HUNG
  input  logic [31:0]           next_pidx,

  // ---- the registered enqueue tap (tsr_dbg stage e1) ----
  input  logic                  e1_v,
  input  logic [31:0]           e1_pidx,
  input  logic [2:0]            e1_unit,
  input  logic [TSR_SEQ_W-1:0]  e1_seq,
  input  logic [3:0]            e1_code,
  input  logic                  e1_rec,
  input  logic [31:0]           e1_dep0,
  input  logic [31:0]           e1_dep1,
  input  tsr_insn_t             e1_w,
  input  logic [31:0]           e1_pc,
  input  logic [31:0]           e1_time,

  // ---- stage 3 of tsr_dbg_stop: INFO's TBP and stop-point bits ----
  input  logic                  h3_v,
  input  logic [3:0]            h3_hit,
  input  logic                  h3_stop,

  // ---- completions (the first error since HIST_CLEAR) ----
  input  logic [5:0]            tap_done,
  input  logic [5:0][3:0]       tap_done_code,
  input  logic [5:0][31:0]      tap_done_pidx,

  // ---- reading ----
  input  logic [31:0]           idx,               // DBG_HIST_IDX
  output logic [31:0]           head,              // DBG_HIST_HEAD
  output logic                  frozen,            // DBG_STATUS.HIST_FROZEN
  output logic [31:0]           rd_info,
  output logic [31:0]           rd_pidx,
  output logic [31:0]           rd_ticket,
  output logic [31:0]           rd_dep0,
  output logic [31:0]           rd_dep1,
  output logic [31:0]           rd_time,
  output logic [31:0]           rd_pc,
  output tsr_insn_t             rd_w
);

  if (DEPTH != 0 && (DEPTH < 2 || (DEPTH & (DEPTH - 1)) != 0)) begin : g_chk
    $fatal(1, "tsr_dbg_hist: DEPTH must be 0 or a power of two >= 2 (DB §10.3)");
  end

  function automatic logic ge(logic [31:0] a, logic [31:0] b);
    return (a - b) < 32'h8000_0000;
  endfunction

  if (DEPTH == 0) begin : g_none
    assign head      = '0;
    assign frozen    = 1'b0;
    assign rd_info   = '0;
    assign rd_pidx   = '0;
    assign rd_ticket = '0;
    assign rd_dep0   = '0;
    assign rd_dep1   = '0;
    assign rd_time   = '0;
    assign rd_pc     = '0;
    assign rd_w      = '0;
    wire unused_h = &{1'b0, clk, rst_n, cfg_en, cfg_freeze, cfg_freeze_err, clear, unit_hung, next_pidx, e1_v,
                      e1_pidx, e1_unit, e1_seq, e1_code, e1_rec, e1_dep0, e1_dep1, e1_w, e1_pc, e1_time, h3_v,
                      h3_hit, h3_stop, tap_done, tap_done_code, tap_done_pidx, idx};
  end else begin : g_hist
    localparam int unsigned AW   = $clog2(DEPTH);
    localparam int unsigned HALF = DEPTH / 2;
    // main RAM word: {time, pc, w7..w0, dep1, dep0, rec, code, unit, seq, pidx} (452 b + 4 pad)
    localparam int unsigned MW = 456;
    typedef struct packed {
      logic [3:0]           pad;
      logic [31:0]          time_;
      logic [31:0]          pc;
      tsr_insn_t            w;
      logic [31:0]          dep1;
      logic [31:0]          dep0;
      logic                 rec;
      logic [3:0]           code;
      logic [2:0]           unit;
      logic [TSR_SEQ_W-1:0] seq;
      logic [31:0]          pidx;
    } hw_t;

    logic [DEPTH-1:0] valid_q;
    logic [31:0]      head_q, err_pidx_q;
    logic             err_q, hung_q, hung_prev_q;
    logic             r2_q, r3_q;                   // the entry of stage 2 / 3 was recorded
    logic [AW-1:0]    s2_q, s3_q;

    wire stop_rec = cfg_freeze || hung_q;
    wire err_lim  = err_q && cfg_freeze_err;
    wire rec_now  = e1_v && cfg_en && !stop_rec && !(err_lim && ge(e1_pidx, err_pidx_q + 32'(HALF)));
    wire [AW-1:0] slot = e1_pidx[AW-1:0];

    // the first completion with an error since HIST_CLEAR
    logic        err_now;
    logic [31:0] err_now_pidx;
    always_comb begin
      err_now = 1'b0;
      err_now_pidx = '0;
      for (int u = 5; u >= 0; u--) begin
        if (tap_done[u] && tap_done_code[u] != TSR_ERR_NONE) begin
          err_now = 1'b1;
          err_now_pidx = tap_done_pidx[u];
        end
      end
    end

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        valid_q     <= '0;
        head_q      <= '0;
        err_q       <= 1'b0;
        err_pidx_q  <= '0;
        hung_q      <= 1'b0;
        hung_prev_q <= 1'b0;
        r2_q        <= 1'b0;
        r3_q        <= 1'b0;
      end else begin
        hung_prev_q <= unit_hung;
        r2_q <= rec_now && !clear;
        r3_q <= r2_q && !clear;
        if (clear) begin
          valid_q <= '0;
          err_q   <= 1'b0;
          hung_q  <= 1'b0;
          head_q  <= next_pidx;
        end else begin
          if (rec_now) begin
            valid_q[slot] <= 1'b1;
            head_q        <= e1_pidx + 32'd1;
          end
          if (err_now && !err_q) begin
            err_q      <= 1'b1;
            err_pidx_q <= err_now_pidx;
          end
          if (unit_hung && !hung_prev_q) hung_q <= 1'b1;
        end
      end
    end
    always_ff @(posedge clk) begin
      s2_q <= slot;
      s3_q <= s2_q;
    end

    // main RAM: written at stage e1, read at DBG_HIST_IDX
    hw_t main_w, main_r;
    assign main_w = '{pad: 4'd0, time_: e1_time, pc: e1_pc, w: e1_w, dep1: e1_dep1, dep0: e1_dep0, rec: e1_rec,
                      code: e1_code, unit: e1_unit, seq: e1_seq, pidx: e1_pidx};
    logic [MW-1:0] main_rd;
    tsr_sram_1r1w #(.DEPTH(DEPTH), .WIDTH(MW), .RD_LAT(1)) u_main (
      .clk, .we(rec_now), .waddr(slot), .wdata(main_w), .wbe('1),
      .re(1'b1), .raddr(idx[AW-1:0]), .rdata(main_rd)
    );
    assign main_r = hw_t'(main_rd);
    // side RAM: {stop, TBPs} written at stage 3
    logic [7:0] side_rd;
    tsr_sram_1r1w #(.DEPTH(DEPTH), .WIDTH(8), .RD_LAT(1)) u_side (
      .clk, .we(r3_q && h3_v), .waddr(s3_q), .wdata({3'd0, h3_stop, h3_hit}), .wbe('1),
      .re(1'b1), .raddr(idx[AW-1:0]), .rdata(side_rd)
    );

    // registered read outputs (the RAM outputs are one cycle behind idx: compare with that idx)
    logic [31:0] idx_q;
    logic        val_q;
    always_ff @(posedge clk) begin
      idx_q <= idx;
      val_q <= valid_q[idx[AW-1:0]];
    end
    wire present = val_q && (main_r.pidx == idx_q);
    // A slot that is not PRESENT reads 0 in every field (the RAM is not cleared, so what it holds is a
    // wrapped-over entry or the power-up content; the simulator reads 0). The gated fields are registered
    // here, one more cycle behind DBG_HIST_IDX than the RAM output (the read mux behind sees only flops).
    always_ff @(posedge clk) begin
      rd_info   <= present ? {14'd0, side_rd[4], side_rd[3:0], main_r.rec, main_r.code, 1'b0, main_r.unit, 3'd0, 1'b1} : '0;
      rd_pidx   <= present ? main_r.pidx : '0;
      rd_ticket <= present ? {1'b0, main_r.unit, main_r.seq} : '0;
      rd_dep0   <= present ? main_r.dep0 : '0;
      rd_dep1   <= present ? main_r.dep1 : '0;
      rd_time   <= present ? main_r.time_ : '0;
      rd_pc     <= present ? main_r.pc : '0;
      rd_w      <= present ? main_r.w : '0;
    end
    assign head      = head_q;
    assign frozen    = stop_rec || (err_lim && ge(head_q, err_pidx_q + 32'(HALF)));
    wire unused_h = &{1'b0, main_r.pad, side_rd[7:5], idx[31:AW]};
  end

endmodule
