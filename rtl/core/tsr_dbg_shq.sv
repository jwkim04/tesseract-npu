// tsr_dbg_shq.sv — DB's per-unit shadow queues and the record serializer (docs/design/05-debug.md
// "DB" §5.5, §8.2-§8.4): the oldest outstanding entry of every unit (DRAINED, serial mode's
// min_out, the unit probes DBG_Uu_*), the times of every entry (dispatch, head, start, end) and
// the OP / MARK / STOP / RVHALT records for the trace writer and the completion records (REC)
// for the RISC-V.
//
// Instantiated by tsr_dbg. Oracle: the simulator's DBG model
// (sw/sim/tsr_dbg.c shq / on_enq / on_start / on_done / on_mark / trw_* / rec_push).
//
// Shadow queue of unit u (DB §8.3: it mirrors CC's queue from the enqueue tap and the start /
// completion strobes): a ring of SD = 2^ceil(log2(QD + 1 + ACCEPT_MAX + 4)) entries with three
// pointers, rrd <= rd <= wr. [rd, wr) are the outstanding entries (dispatched, not completed; at
// most QD + 1 + ACCEPT_MAX by the credits), the first nst of them started (accepted by the unit);
// [rrd, rd) are completed entries whose records wait for the serializer. Entries are written at
// the tap cycle itself (tsr_issue's enqueue strobe), so DRAINED never misses an entry that
// Q_PIDX already counts. Fields (one LUTRAM per write source): A at enqueue {pidx, seq, w0[15:0],
// REC, head-at-enqueue, t_dispatch}; B the tile-breakpoint flag (stage 2 of tsr_dbg_stop); H
// t_head; S t_start; D at completion {code, ran, t_end}. Times are MTIME[31:0]: the record fields
// are differences modulo 2^32 (DB §8.3 saturates them at 32 bits: they differ only for waits
// beyond 2^32 cycles, 15.5 s at 276 MHz) and t_end's upper word is rebuilt from MTIME when the
// record is written (a record waits far less than 2^32 cycles).
// Head time (the simulator's rule, sh_next_head): an entry becomes the head when every earlier
//   entry of its unit left the queue (started, or completed without running): at its enqueue if
//   none is left unstarted before it (t_head = t_dispatch), else when the entry before it starts
//   or completes without running. A completion with ran = 1 that never started (an LD / ST of 0
//   lines, completed without the unit) counts as starting at its completion (RUN 0).
// Serializer: one source per cycle, priority STOP release, STOP activation, RVHALT, MARK, then
//   the units round robin (completed entries in completion order per unit; a unit's waiting
//   record is read into a register first, so a unit is served at most every second cycle). A pick writes the TRW
//   record (DBG_CFG.TRW_EN and its DBG_TRW_MASK class; it waits while the trace writer is not
//   ready) and the REC record (an entry with REC or REC_CFG.ALL, every marker; dropped and counted
//   when the REC FIFO is full: DB §8.2 "records never stall execution"). The STOP / RVHALT / MARK
//   slots hold one event each (markers come at most one per 3 cycles, stop events at most two per
//   host access); an event finding its slot full replaces the older one, which is lost (trw_lost,
//   counted as a trace drop).
//   A unit whose ring is full at an enqueue drops its oldest waiting record the same way (the
//   serializer serves a unit with waiting records at least every 8 cycles, so this needs more
//   than 4 records per unit within that time).
// Records (DB §8.3, 8 words, word 0 bits [3:0] KIND): OP {unit, code, ran, REC, TBP, opcode,
//   w0[15:8] | pidx | ticket | t_end lo / hi | RUN | QWAIT | HWAIT}; MARK {next push index, value,
//   dispatch time}; STOP {T, cause, time, activated}; RVHALT {0, halted, time}. REC (DB §8.2):
//   {ticket, t_end[31:0], RUN, code | ran << 4 | opcode << 8 | pidx[15:0] << 16}; markers
//   {0xE000_0000, dispatch time, value, 1 << 5 | pidx[15:0] << 16}.

module tsr_dbg_shq
  import tsr_pkg::*;
#(
  parameter int unsigned QD_DMA  = tsr_tables_pkg::TSR_QDEPTH_U200_DMA,
  parameter int unsigned QD_QACT = tsr_tables_pkg::TSR_QDEPTH_U200_QACT,
  parameter int unsigned QD_MVM  = tsr_tables_pkg::TSR_QDEPTH_U200_MVM,
  parameter int unsigned QD_VU   = tsr_tables_pkg::TSR_QDEPTH_U200_VU,
  parameter int unsigned QD_ATT  = tsr_tables_pkg::TSR_QDEPTH_U200_ATT,
  parameter int unsigned QD_SYNC = tsr_tables_pkg::TSR_QDEPTH_U200_SYNC
) (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic [63:0]               mtime,

  // ---- tsr_issue: enqueue tap, start / completion strobes, markers ----
  input  logic                      tap_enq,
  input  logic [3:0]                tap_enq_unit,
  input  logic [TSR_SEQ_W-1:0]      tap_enq_seq,
  input  logic [31:0]               tap_enq_pidx,
  input  logic [15:0]               tap_enq_w0lo,
  input  logic                      tap_enq_rec,
  input  logic [5:0]                tap_start,
  input  logic [5:0]                tap_done,
  input  logic [5:0][3:0]           tap_done_code,
  input  logic [5:0]                tap_done_ran,
  input  logic                      tap_mark,
  input  logic [31:0]               tap_mark_value,
  input  logic [31:0]               tap_mark_pidx,

  // ---- tile-breakpoint flag of the entry enqueued two cycles earlier (tsr_dbg_stop m2) ----
  input  logic                      m2_v,
  input  logic [3:0]                m2_hit,

  // ---- STOP records (tsr_dbg_stop) and the RISC-V halt level ----
  input  logic                      srel_v,
  input  logic [31:0]               srel_t,
  input  logic [4:0]                srel_cause,
  input  logic                      sact_v,
  input  logic [31:0]               sact_t,
  input  logic [4:0]                sact_cause,
  input  logic                      rv_halted_lvl,     // RV_RUN ∧ STATUS.RV_HALTED ∧ RV_RUNNING

  // ---- configuration ----
  input  logic                      trw_en,
  input  logic [10:0]               trw_mask,
  input  logic                      rec_en,            // HAS_REC
  input  logic                      rec_all,           // REC_CFG.ALL (clk_rv, synchronized)

  // ---- probes ----
  output logic [5:0]                out_v,
  output logic [5:0][31:0]          out_pidx,
  output logic [5:0][TSR_SEQ_W-1:0] out_seq,
  output logic [5:0][7:0]           out_n,             // outstanding entries (saturating)
  output logic [5:0][1:0]           out_acc,           // started and not completed
  output logic [5:0][63:0]          out_t,             // time the oldest became the oldest

  // ---- trace writer ----
  output logic                      trw_v,
  output logic [255:0]              trw_rec,
  output logic                      trw_flush,         // with a LAUNCH_END marker: flush after it
  input  logic                      trw_ready,
  output logic                      trw_lost,          // pulse: a record lost before the writer

  // ---- REC (tsr_dbg_rv's FIFO) ----
  output logic                      rec_v,
  output logic [127:0]              rec_data,
  input  logic                      rec_room,
  output logic                      rec_drop            // pulse: REC FIFO full
);

  localparam int unsigned QD   [6] = '{QD_DMA, QD_QACT, QD_MVM, QD_VU, QD_ATT, QD_SYNC};
  localparam int unsigned AMAX [6] = '{tsr_tables_pkg::TSR_ACCEPT_MAX_DMA, tsr_tables_pkg::TSR_ACCEPT_MAX_QACT,
                                       tsr_tables_pkg::TSR_ACCEPT_MAX_MVM, tsr_tables_pkg::TSR_ACCEPT_MAX_VU,
                                       tsr_tables_pkg::TSR_ACCEPT_MAX_ATT, tsr_tables_pkg::TSR_ACCEPT_MAX_SYNC};
  function automatic int unsigned sd_of(logic [2:0] u);
    int unsigned need, d;
    need = QD[u] + 1 + AMAX[u] + 4;
    d = 4;
    while (d < need) d = d * 2;
    return d;
  endfunction

  localparam logic [3:0] K_OP     = tsr_tables_pkg::TSR_TRW_W0_KIND_OP;
  localparam logic [3:0] K_MARK   = tsr_tables_pkg::TSR_TRW_W0_KIND_MARK;
  localparam logic [3:0] K_STOP   = tsr_tables_pkg::TSR_TRW_W0_KIND_STOP;
  localparam logic [3:0] K_RVHALT = tsr_tables_pkg::TSR_TRW_W0_KIND_RVHALT;
  localparam logic [31:0] MARK_TICKET = {tsr_tables_pkg::TSR_UNIT_MARKER, 28'd0};   // DB §8.2: unit 14

  wire [31:0] now = mtime[31:0];

  // t_end's upper word from MTIME (the record waited less than 2^32 cycles): MTIME of the previous
  // cycle and its upper word minus 1, registered (a record is read at least one cycle after its end)
  logic [63:0] mt_q;
  logic [31:0] mt_hi_m1_q;
  always_ff @(posedge clk) begin
    mt_q       <= mtime;
    mt_hi_m1_q <= mtime[63:32] - 32'd1;
  end
  function automatic logic [63:0] full_time(logic [31:0] lo, logic [63:0] t, logic [31:0] hi_m1);
    return {(lo > t[31:0]) ? hi_m1 : t[63:32], lo};
  endfunction

  // tap pipeline for the B (TBP flag) write two cycles later: the entry's unit and ring index
  logic [2:0] tu1_q, tu2_q;
  logic [5:0] ti1_q, ti2_q;
  logic [5:0][5:0] wr_idx;                   // per unit: write index of this cycle's enqueue
  wire unused_ti = &{1'b0, ti2_q, tap_enq_unit[3]};
  always_ff @(posedge clk) begin
    tu1_q <= tap_enq_unit[2:0];
    ti1_q <= wr_idx[tap_enq_unit[2:0]];
    tu2_q <= tu1_q;
    ti2_q <= ti1_q;
  end

  // ---------------------------------------------------------------------------------------------
  // Per-unit shadow queues
  // ---------------------------------------------------------------------------------------------
  logic [5:0]        has_rec;                // completed entries wait for the serializer
  logic [5:0]        pick_u;                 // the serializer takes unit u's record this cycle
  logic [5:0]        lost_u;                 // a waiting record was dropped (ring full)
  logic [5:0][255:0] op_rec;                 // OP record of the waiting entry (rrd)
  logic [5:0][127:0] op_rr;                  // its REC record
  logic [5:0]        op_rq;                  // it wants a REC record (REC flag of the push)

  for (genvar gu = 0; gu < 6; gu++) begin : g_u
    localparam int unsigned SD = sd_of(3'(gu));
    localparam int unsigned AW = $clog2(SD);
    if (QD[gu] == 0) begin : g_absent
      assign has_rec[gu]  = 1'b0;
      assign lost_u[gu]   = 1'b0;
      assign op_rec[gu]   = '0;
      assign op_rr[gu]    = '0;
      assign op_rq[gu]    = 1'b0;
      assign out_v[gu]    = 1'b0;
      assign out_pidx[gu] = '0;
      assign out_seq[gu]  = '0;
      assign out_n[gu]    = '0;
      assign out_acc[gu]  = '0;
      assign out_t[gu]    = '0;
      assign wr_idx[gu]   = '0;
      wire unused_u = &{1'b0, pick_u[gu], tap_start[gu], tap_done[gu], tap_done_code[gu], tap_done_ran[gu]};
    end else begin : g_q
      // LUTRAM arrays (distributed RAM: one write port each, asynchronous reads):
      //   A {pidx 32, seq 28, w0lo 16, rec 1, hae 1, t_disp 32} at the enqueue, B the TBP flag,
      //   H t_head, S t_start, D {code 4, ran 1, t_end 32} at the completion
      localparam int unsigned AWID = 110, DWID = 37;
      logic [AWID-1:0] a_mem [SD];
      logic            b_mem [SD];
      logic [31:0]     h_mem [SD];
      logic [31:0]     s_mem [SD];
      logic [DWID-1:0] d_mem [SD];
      logic [AW:0] wr_q, rd_q, rrd_q;
      logic [1:0]  nst_q;
      logic [63:0] t_old_q;

      wire         enq   = tap_enq && (tap_enq_unit == 4'(gu));
      wire         st    = tap_start[gu];
      wire         dn    = tap_done[gu];
      wire [AW:0]  n     = wr_q - rd_q;               // outstanding
      wire [AW:0]  tot   = wr_q - rrd_q;              // outstanding + waiting records
      wire [AW:0]  unst  = n - (AW + 1)'(nst_q);      // outstanding, not started
      wire         dn_ns = dn && (nst_q == 2'd0);     // the unstarted head completes
      wire         full  = (tot == (AW + 1)'(SD));
      wire         drop  = enq && full && (rrd_q != rd_q);
      wire [AW-1:0] hd   = AW'(rd_q + (AW + 1)'(nst_q));   // the head (first unstarted entry)
      wire [AW:0]  unst_after = unst - (AW + 1)'(st) - (AW + 1)'(dn_ns);

      assign wr_idx[gu] = 6'(wr_q[AW-1:0]);

      // write ports: H = the entry after the leaving head becomes the head; S = the start of the
      // head (an unstarted completion starts and ends at once)
      wire          h_we = (st && unst >= (AW + 1)'(2)) || (dn_ns && n >= (AW + 1)'(2));
      wire [AW-1:0] h_wa = st ? AW'(hd + AW'(1)) : AW'(rd_q[AW-1:0] + AW'(1));
      wire          s_we = st || dn_ns;
      wire [AW-1:0] s_wa = st ? hd : rd_q[AW-1:0];
      wire          b_we = m2_v && (tu2_q == 3'(gu));
      always_ff @(posedge clk) if (enq) a_mem[wr_q[AW-1:0]] <= {tap_enq_pidx, tap_enq_seq, tap_enq_w0lo, tap_enq_rec,
                                                               (unst_after == '0), now};
      always_ff @(posedge clk) if (b_we) b_mem[ti2_q[AW-1:0]] <= (m2_hit != 4'd0);
      always_ff @(posedge clk) if (h_we) h_mem[h_wa] <= now;
      always_ff @(posedge clk) if (s_we) s_mem[s_wa] <= now;
      always_ff @(posedge clk) if (dn)   d_mem[rd_q[AW-1:0]] <= {tap_done_code[gu], tap_done_ran[gu], now};

      always_ff @(posedge clk) begin
        if (!rst_n) begin
          wr_q    <= '0;
          rd_q    <= '0;
          rrd_q   <= '0;
          nst_q   <= '0;
          t_old_q <= '0;
        end else begin
          if (enq) wr_q <= wr_q + 1'b1;
          if (dn)  rd_q <= rd_q + 1'b1;
          if (pick_u[gu] || drop) rrd_q <= rrd_q + 1'b1;
          nst_q <= nst_q + 2'(st) - 2'(dn && nst_q != 2'd0);
          if (dn || (enq && n == '0)) t_old_q <= mtime;
        end
      end

      // the waiting record (rrd)
      wire [AW-1:0]    ri = rrd_q[AW-1:0];
      wire [AWID-1:0]  ra = a_mem[ri];
      wire [DWID-1:0]  rdd = d_mem[ri];
      wire [31:0]      r_pidx = ra[109:78];
      wire [27:0]      r_seq  = ra[77:50];
      wire [15:0]      r_w0lo = ra[49:34];
      wire             r_rec  = ra[33];
      wire             r_hae  = ra[32];
      wire [31:0]      r_tdsp = ra[31:0];
      wire [3:0]       r_code = rdd[36:33];
      wire             r_ran  = rdd[32];
      wire [31:0]      r_tend = rdd[31:0];
      wire [31:0]      rh = r_hae ? r_tdsp : h_mem[ri];
      wire [31:0]      rs = s_mem[ri];
      wire [63:0]      tend  = full_time(r_tend, mt_q, mt_hi_m1_q);
      wire [31:0]      run   = r_ran ? (r_tend - rs) : 32'd0;
      wire [31:0]      qwait = rh - r_tdsp;
      wire [31:0]      hwait = r_ran ? (rs - rh) : (r_tend - rh);
      // registered copy of the waiting record (timing: LUTRAM read and time arithmetic in one
      // cycle, the serializer's select in the next); it is fresh when the entry at rrd was already
      // completed and rrd did not move at the edge that captured it
      logic [255:0] rec_q;
      logic [127:0] rr_q;
      logic         rq_q, fresh_q;
      always_ff @(posedge clk) begin
        rec_q <= {hwait, qwait, run, tend[63:32], tend[31:0], {4'(gu), r_seq}, r_pidx,
                  r_w0lo[15:8], r_w0lo[7:0], 1'b0, b_mem[ri], r_rec, r_ran, r_code, 4'(gu), K_OP};
        rr_q  <= {r_pidx[15:0], r_w0lo[7:0], 3'd0, r_ran, r_code, run, r_tend, {4'(gu), r_seq}};
        rq_q  <= r_rec;
      end
      always_ff @(posedge clk) begin
        if (!rst_n) fresh_q <= 1'b0;
        else        fresh_q <= (rrd_q != rd_q) && !(pick_u[gu] || drop);
      end
      assign has_rec[gu] = (rrd_q != rd_q) && fresh_q;
      assign lost_u[gu]  = drop;
      assign op_rq[gu]   = rq_q;
      assign op_rec[gu]  = rec_q;
      assign op_rr[gu]   = rr_q;

      // probes: the oldest outstanding entry (front)
      wire [AWID-1:0] fa = a_mem[rd_q[AW-1:0]];
      wire unused_fa = &{1'b0, fa[49:0]};
      assign out_v[gu]    = (n != '0);
      assign out_pidx[gu] = fa[109:78];
      assign out_seq[gu]  = fa[77:50];
      assign out_n[gu]    = 8'(n);
      assign out_acc[gu]  = nst_q;
      assign out_t[gu]    = t_old_q;

`ifndef SYNTHESIS
      logic chk_on_q;
      initial chk_on_q = 1'b0;
      always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
      always_ff @(posedge clk) begin
        if (rst_n && chk_on_q) begin
          assert (!(enq && full && rrd_q == rd_q))
            else $error("%m: shadow queue of unit %0d overflows with outstanding entries", gu);
          assert (!(dn && n == '0)) else $error("%m: completion of unit %0d with no outstanding entry", gu);
          assert (!(st && unst == '0)) else $error("%m: start of unit %0d with no unstarted entry", gu);
        end
      end
`endif
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Event slots: STOP release / activation, RVHALT, MARK
  // ---------------------------------------------------------------------------------------------
  logic         rel_v_q, sa_v_q, rh_v_q, mk_v_q, halted_q;
  logic [255:0] rel_q, sa_q, rh_q, mk_q;
  logic [127:0] mk_rr_q;
  logic         mk_end_q;
  logic         take_rel, take_sa, take_rh, take_mk;
  wire          rh_edge = (rv_halted_lvl != halted_q);
  logic         slot_lost;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rel_v_q  <= 1'b0;
      sa_v_q   <= 1'b0;
      rh_v_q   <= 1'b0;
      mk_v_q   <= 1'b0;
      halted_q <= 1'b0;
    end else begin
      halted_q <= rv_halted_lvl;
      if (take_rel) rel_v_q <= 1'b0;
      if (take_sa)  sa_v_q  <= 1'b0;
      if (take_rh)  rh_v_q  <= 1'b0;
      if (take_mk)  mk_v_q  <= 1'b0;
      if (srel_v) rel_v_q <= 1'b1;
      if (sact_v) sa_v_q  <= 1'b1;
      if (rh_edge) rh_v_q <= 1'b1;
      if (tap_mark) mk_v_q <= 1'b1;
    end
  end
  always_ff @(posedge clk) begin
    if (srel_v) rel_q <= {64'd0, 32'd0, mtime[63:32], mtime[31:0], 27'd0, srel_cause, srel_t, 28'd0, K_STOP};
    if (sact_v) sa_q  <= {64'd0, 32'd1, mtime[63:32], mtime[31:0], 27'd0, sact_cause, sact_t, 28'd0, K_STOP};
    if (rh_edge) rh_q <= {96'd0, mtime[63:32], mtime[31:0], 31'd0, rv_halted_lvl, 32'd0, 28'd0, K_RVHALT};
    if (tap_mark) begin
      mk_q     <= {96'd0, mtime[63:32], mtime[31:0], tap_mark_value, tap_mark_pidx, 28'd0, K_MARK};
      mk_rr_q  <= {tap_mark_pidx[15:0], 8'd0, 2'd0, 1'b1, 1'b0, 4'd0, tap_mark_value, mtime[31:0], MARK_TICKET};
      mk_end_q <= (tap_mark_value[31:28] == tsr_tables_pkg::TSR_MARK_CLASS_LAUNCH_END);
    end
  end
  // an event that finds its slot still full replaces the older one, which is lost (counted as a
  // drop when it would have been recorded)
  assign slot_lost = (srel_v && rel_v_q && !take_rel && trw_en && trw_mask[9]) ||
                     (sact_v && sa_v_q && !take_sa && trw_en && trw_mask[9]) ||
                     (rh_edge && rh_v_q && !take_rh && trw_en && trw_mask[10]) ||
                     (tap_mark && mk_v_q && !take_mk && trw_en && trw_mask[8]);

  // ---------------------------------------------------------------------------------------------
  // Serializer
  // ---------------------------------------------------------------------------------------------
  logic         ov_q;                        // output register holds a TRW record
  logic [255:0] orec_q;
  logic         oflush_q;
  logic         rv_q;
  logic [127:0] rdat_q;
  logic         rdrop_q;
  logic [2:0]   rr_q;                        // round robin: the unit served last
  wire          can = !ov_q || trw_ready;    // the output register is free at the next edge

  // the next unit after rr_q with a waiting record
  logic         u_any;
  logic [2:0]   u_sel;
  always_comb begin
    u_any = 1'b0;
    u_sel = '0;
    for (int k = 1; k <= 6; k++) begin
      automatic logic [2:0] c = 3'((int'(rr_q) + k) % 6);
      if (!u_any && has_rec[c]) begin
        u_any = 1'b1;
        u_sel = 3'(c);
      end
    end
  end

  logic         p_trw, p_rec, p_flush;
  logic [255:0] p_rec_w;
  logic [127:0] p_rr;
  always_comb begin
    take_rel = 1'b0;
    take_sa  = 1'b0;
    take_rh  = 1'b0;
    take_mk  = 1'b0;
    pick_u   = '0;
    p_trw    = 1'b0;
    p_rec    = 1'b0;
    p_flush  = 1'b0;
    p_rec_w  = '0;
    p_rr     = '0;
    if (can) begin
      if (rel_v_q) begin
        take_rel = 1'b1;
        p_trw    = trw_en && trw_mask[9];
        p_rec_w  = rel_q;
      end else if (sa_v_q) begin
        take_sa = 1'b1;
        p_trw   = trw_en && trw_mask[9];
        p_rec_w = sa_q;
      end else if (rh_v_q) begin
        take_rh = 1'b1;
        p_trw   = trw_en && trw_mask[10];
        p_rec_w = rh_q;
      end else if (mk_v_q) begin
        take_mk = 1'b1;
        p_trw   = trw_en && trw_mask[8];
        p_flush = p_trw && mk_end_q;
        p_rec_w = mk_q;
        p_rec   = rec_en;
        p_rr    = mk_rr_q;
      end else if (u_any) begin
        pick_u[u_sel] = 1'b1;
        p_trw   = trw_en && trw_mask[{1'b0, u_sel}];
        p_rec_w = op_rec[u_sel];
        p_rec   = rec_en && (op_rq[u_sel] || rec_all);
        p_rr    = op_rr[u_sel];
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ov_q    <= 1'b0;
      rv_q    <= 1'b0;
      rdrop_q <= 1'b0;
      rr_q    <= 3'd5;
    end else begin
      if (can) ov_q <= p_trw;
      rv_q    <= p_rec && rec_room;
      rdrop_q <= p_rec && !rec_room;
      if (|pick_u) rr_q <= u_sel;
    end
  end
  always_ff @(posedge clk) begin
    if (can && p_trw) begin
      orec_q   <= p_rec_w;
      oflush_q <= p_flush;
    end
    if (p_rec) rdat_q <= p_rr;
  end

  assign trw_v     = ov_q;
  assign trw_rec   = orec_q;
  assign trw_flush = oflush_q;
  assign trw_lost  = slot_lost || (|(lost_u & trw_mask[5:0])) && trw_en;
  assign rec_v     = rv_q;
  assign rec_data  = rdat_q;
  assign rec_drop  = rdrop_q;

endmodule
