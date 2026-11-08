// tsr_mvm_att.sv — attention helpers of the MVM unit (AT §8.1, §8.2; W4-B at): the per-cluster
// ATTQK score assembler (tsr_mvm_aqk) and the row-vector emitter (tsr_mvm_rvec) that writes ATTQK's
// MAXOUT and ATTPV's LSUM words (AT §5.2, §5.3: one f32 per pass row, rows of other passes and the
// words past R untouched, i.e. byte-enabled lines).
//
// Owner: AT (W4-B "at"). Oracle: tsr_ref_attqk / tsr_ref_attpv through the simulator's ATTQK /
// ATTPV (tb/at). Both modules produce gather jobs in the format of tsr_mvm_gather's job pipes
// {last, line, own, tail, y} (docs/rtl-m3.md §6.3): own = the words written (byte enables), tail =
// a tag for the central part (tsr_mvm_gather: ATT jobs are written with own only; bit 15 marks an
// ATTQK score line, bits [6:0] its pass row for the row-maximum tracker).

/* verilator lint_off DECLFILENAME */
// ================================================================================================
// tsr_mvm_aqk — ATTQK score assembler of one cluster (AT §8.1 "Gather": a line buffer per pass row,
// -inf outside the bounds, full-line writes).
//   Input: the back end's SEG stream (seg_valid / seg_y, no backpressure), in stream order: per
//   group of the channel (lambda = 0 .. ng - 1, group u = u_first + lambda C), per cell j = 0..15,
//   per head segment g = 0 .. nseg - 1, the T_MAX slot values (slot t < tact; slot t is query q(t),
//   head g G + hg0 + i(t)). Pass row e = g tact + t (< prows = nseg tact <= 128).
//   Line buffer: 16 banks of 2 x 128 words (half = lambda & 1). Word (e, j) of row e's line lives in
//   bank (e + j) mod 16 at {half, e} (skewed): the tact words of one SEG (fixed j, consecutive e) hit
//   tact different banks, and a line (fixed e, j = 0..15) is one read of all banks at one address,
//   rotated by e mod 16. One SEG per cycle and one line per cycle.
//   Drain: once a group's last SEG is written, its rows e = 0 .. prows - 1 are read in order and
//   each becomes one job: line = rowbase[e] + (u - u0), every word (own = FFFF), word j = -inf
//   (0xFF800000) unless lo_q <= 16 (u - u0) + j < hi_q (q = q(e)), tail = {1, e} (score line).
//   The cluster lets group lambda's first beat in only while groups_in - groups_done < 2 (its half
//   is free); groups_done counts drained groups (all its reads issued: a read latches the RAM
//   word in its cycle, so later writes of that half do not change it).
//   rowbase[e] = dst + r(e) rstride comes from the cluster's table builder (tb_*), complete before
//   the first drain (tb_done).
// ================================================================================================
module tsr_mvm_aqk
  import tsr_pkg::*;
#(
  parameter  int unsigned T_MAX   = TSR_DEF_T_MAX,
  parameter  int unsigned NUM_CH  = TSR_DEF_NUM_CH,
  parameter  int unsigned SPM_AW  = 16,
  parameter  int unsigned RAM_LAT = 2,
  parameter  bit          FAST_CLK = 1'b0,   // the bounds test of a row one stage earlier (RAM_LAT >= 2; the setup is stable)
  localparam int unsigned JW      = 1 + SPM_AW + 16 + 16 + TSR_LINE_BITS
) (
  input  logic                         clk,
  input  logic                         rst_n,

  // ---- op setup (stable from start to the last job) ----
  input  logic                         start,        // pulse: counters cleared
  input  logic [3:0]                   tact,         // T_q GS (1..T_MAX)
  input  logic [8:0]                   nseg,         // E / D
  input  logic [7:0]                   prows,        // nseg tact
  input  logic [12:0]                  ng,           // groups of this channel (0: no work)
  input  logic [4:0]                   off0,         // u_first - u0
  input  logic [T_MAX-1:0][2:0]        qt,           // query of slot t
  input  logic [7:0][17:0]             lo,           // per query: signed bounds (AT §5.2)
  input  logic [7:0][17:0]             hi,
  input  logic                         tb_we,        // rowbase table builder
  input  logic [6:0]                   tb_addr,
  input  logic [SPM_AW-1:0]            tb_data,
  input  logic                         tb_done,

  // ---- SEG stream (back end) ----
  input  logic                         seg_valid,
  input  logic [T_MAX-1:0][31:0]       seg_y,

  // ---- jobs (to the gather's job pipe) ----
  output logic                         job_valid,
  input  logic                         job_ready,
  output logic [JW-1:0]                job,

  output logic [12:0]                  groups_done,
  output logic                         busy          // a group buffered or draining
);

  if (T_MAX < 1 || T_MAX > 8 || RAM_LAT < 1) begin : g_chk
    $fatal(1, "tsr_mvm_aqk: T_MAX in [1, 8], RAM_LAT >= 1");
  end

  localparam logic [31:0] NEG_INF = 32'hFF80_0000;
  localparam int unsigned OQ      = RAM_LAT + 4;           // output queue entries (credit)
  localparam int unsigned OQW     = $clog2(OQ + 1);

  // -------------------------------------------------------------------------------------------
  // Write side: SEG -> 16 banks
  // -------------------------------------------------------------------------------------------
  logic [3:0]  wj_q;            // cell j of the next SEG
  logic [8:0]  wg_q;            // segment g
  logic [7:0]  wgt_q;           // g tact
  logic        wh_q;            // half
  logic [12:0] wl_q;            // groups completely written
  logic        full_q [2];      // half holds a complete group not yet drained

  wire w_last = seg_valid & (wj_q == 4'd15) & (wg_q == nseg - 9'd1);

  // stage W: per-bank select (registered), then the RAM write
  logic [15:0]      bw_q;
  logic [15:0][7:0] ba_q;       // {half, e}
  logic [15:0][31:0] bd_q;
  wire  [3:0] rot = 4'(wgt_q) + wj_q;                      // (g tact + j) mod 16
  logic [15:0][3:0] bt;                                     // slot of bank b: (b - rot) mod 16
  always_comb begin
    for (int b = 0; b < 16; b++) bt[b] = 4'(b) - rot;       // 4-bit result (a wider context would not wrap)
  end
  always_ff @(posedge clk) begin
    if (!rst_n) bw_q <= '0;
    else begin
      for (int b = 0; b < 16; b++) begin
        bw_q[b] <= seg_valid & (bt[b] < tact);
      end
    end
  end
  always_ff @(posedge clk) begin
    for (int b = 0; b < 16; b++) begin
      ba_q[b] <= {wh_q, 7'(wgt_q + 8'(bt[b]))};
      bd_q[b] <= '0;                                         // (slot bt[b] < T_MAX; else unused)
      for (int t = 0; t < int'(T_MAX); t++) if (bt[b] == 4'(t)) bd_q[b] <= seg_y[t];
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      wj_q  <= '0;
      wg_q  <= '0;
      wgt_q <= '0;
      wh_q  <= 1'b0;
      wl_q  <= '0;
    end else if (seg_valid) begin
      if (wg_q == nseg - 9'd1) begin
        wg_q  <= '0;
        wgt_q <= '0;
        wj_q  <= wj_q + 4'd1;
        if (wj_q == 4'd15) begin
          wh_q <= ~wh_q;
          wl_q <= wl_q + 13'd1;
        end
      end else begin
        wg_q  <= wg_q + 9'd1;
        wgt_q <= wgt_q + 8'(tact);
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Banks
  // -------------------------------------------------------------------------------------------
  logic        rd_v;           // read of {rh, re} in this cycle
  logic [7:0]  rd_a;
  logic [15:0][31:0] bq;
  for (genvar b = 0; b < 16; b++) begin : g_bank
    tsr_sram_1r1w #(.DEPTH(256), .WIDTH(32), .RD_LAT(RAM_LAT)) u_ram (
      .clk, .we(bw_q[b]), .waddr(ba_q[b]), .wdata(bd_q[b]), .wbe(4'hF),
      .re(rd_v), .raddr(rd_a), .rdata(bq[b])
    );
  end

  // -------------------------------------------------------------------------------------------
  // Drain: rows of a full half, one read per cycle while the output queue has room
  // -------------------------------------------------------------------------------------------
  logic        dact_q;          // draining
  logic        dh_q;            // half being drained
  logic [7:0]  de_q;            // next row
  logic [3:0]  dt_q;            // its slot t (e = g tact + t)
  logic [12:0] dl_q;            // group lambda being drained (= groups_done)
  logic [17:0] juo_q;           // 16 (u - u0) of the drained group: cell offset (relative to 16 u0)
  logic [SPM_AW-1:0] uoff_q;    // u - u0 of the drained group
  logic [OQW-1:0] oq_cnt;       // queue entries + reads in flight
  logic [SPM_AW-1:0] rowbase [128];

  wire d_go = dact_q & (oq_cnt < OQW'(OQ));
  wire d_end = d_go & (de_q == prows - 8'd1);
  assign rd_v = d_go;
  assign rd_a = {dh_q, de_q[6:0]};

  always_ff @(posedge clk) begin
    if (tb_we) rowbase[tb_addr] <= tb_data;
  end

  // tag pipeline of a read: {e, slot's query, last}
  typedef struct packed {
    logic [6:0]        e;
    logic [2:0]        q;
    logic              last;
    logic [SPM_AW-1:0] line;
    logic [17:0]       juo;
  } rtag_t;
  rtag_t             rt_q [RAM_LAT];
  logic [RAM_LAT-1:0] rv_q;
  always_ff @(posedge clk) begin
    if (!rst_n) rv_q <= '0;
    else        rv_q <= RAM_LAT'({rv_q, d_go});
  end
  always_ff @(posedge clk) begin
    rt_q[0] <= '{e: de_q[6:0], q: qt[3'(dt_q)], last: d_end & (dl_q == ng - 13'd1),
                 line: rowbase[de_q[6:0]] + uoff_q, juo: juo_q};
    for (int i = 1; i < int'(RAM_LAT); i++) rt_q[i] <= rt_q[i-1];
  end

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      dact_q    <= 1'b0;
      dh_q      <= 1'b0;
      de_q      <= '0;
      dt_q      <= '0;
      dl_q      <= '0;
      full_q[0] <= 1'b0;
      full_q[1] <= 1'b0;
      juo_q     <= 18'({off0, 4'd0});
      uoff_q    <= SPM_AW'(off0);
    end else begin
      if (w_last) full_q[wh_q] <= 1'b1;
      if (!dact_q && full_q[dh_q] && tb_done) begin
        dact_q <= 1'b1;
        de_q   <= '0;
        dt_q   <= '0;
      end
      if (d_go) begin
        de_q <= de_q + 8'd1;
        dt_q <= (dt_q == tact - 4'd1) ? 4'd0 : dt_q + 4'd1;
      end
      if (d_end) begin
        dact_q       <= 1'b0;
        full_q[dh_q] <= 1'b0;
        dh_q         <= ~dh_q;
        dl_q         <= dl_q + 13'd1;
        juo_q        <= juo_q + 18'(16 * NUM_CH);
        uoff_q       <= uoff_q + SPM_AW'(NUM_CH);
      end
    end
  end

  // read data -> rotate by e mod 16 -> bounds -> output queue
  logic [15:0][31:0] rot_w;
  rtag_t             ht;
  assign ht = rt_q[RAM_LAT-1];
  logic [15:0]       vis;
  // the bounds test of lane j of the row read (AT §5.2): -inf outside [lo_q, hi_q) in the group's cell offsets
  function automatic logic [15:0] vis_of(logic [2:0] q, logic [17:0] juo);
    logic signed [18:0] lo_j, hi_j;
    logic [15:0] v;
    lo_j = $signed({lo[q][17], lo[q]}) - $signed({1'b0, juo});
    hi_j = $signed({hi[q][17], hi[q]}) - $signed({1'b0, juo});
    for (int j = 0; j < 16; j++) v[j] = (19'sd0 + 19'(j) >= lo_j) && (19'sd0 + 19'(j) < hi_j);
    return v;
  endfunction
  always_comb begin
    for (int j = 0; j < 16; j++) rot_w[j] = bq[4'(ht.e[3:0] + 4'(j))];
  end
  if (FAST_CLK && RAM_LAT >= 2) begin : g_visq
    // the tags of the read one stage before the data: lo / hi are the op's setup, stable from start to the last job
    always_ff @(posedge clk) vis <= vis_of(rt_q[RAM_LAT-2].q, rt_q[RAM_LAT-2].juo);
    wire unused_ht = &{1'b0, ht.juo, ht.q};
  end else begin : g_viscomb
    assign vis = vis_of(ht.q, ht.juo);
  end

  logic [JW-1:0] oq_mem [OQ];
  logic [$clog2(OQ)-1:0] oq_wp_q, oq_rp_q;
  logic [OQW-1:0] oq_n_q;                                 // entries
  wire  oq_pop = job_valid & job_ready;
  logic [TSR_LINE_BITS-1:0] jy;
  always_comb begin
    for (int j = 0; j < 16; j++) jy[32*j +: 32] = vis[j] ? rot_w[j] : NEG_INF;
  end
  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      oq_wp_q <= '0;
      oq_rp_q <= '0;
      oq_n_q  <= '0;
      oq_cnt  <= '0;
    end else begin
      if (rv_q[RAM_LAT-1]) oq_wp_q <= (32'(oq_wp_q) == OQ - 1) ? '0 : oq_wp_q + 1'b1;
      if (oq_pop)          oq_rp_q <= (32'(oq_rp_q) == OQ - 1) ? '0 : oq_rp_q + 1'b1;
      oq_n_q <= oq_n_q + OQW'(rv_q[RAM_LAT-1]) - OQW'(oq_pop);
      oq_cnt <= oq_cnt + OQW'(d_go) - OQW'(oq_pop);
    end
  end
  always_ff @(posedge clk) begin
    if (rv_q[RAM_LAT-1]) oq_mem[oq_wp_q] <= {ht.last, ht.line, 16'hFFFF, 9'h100, ht.e, jy};
  end
  assign job_valid   = (oq_n_q != '0);
  assign job         = oq_mem[oq_rp_q];
  assign groups_done = dl_q;
  assign busy        = full_q[0] | full_q[1] | dact_q | (oq_cnt != '0) | (wj_q != 4'd0) | (wg_q != 9'd0)
                     | (|bw_q);

  wire unused_ok = &{1'b0, wl_q, ht.e[6:4]};

`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && a_on_q && !start) begin
      if (seg_valid) assert (!full_q[wh_q]) else $error("%m: SEG into a half that is not drained");
      assert (32'(oq_cnt) <= OQ) else $error("%m: output queue overflow");
      if (start) assert (!busy) else $error("%m: start while busy");
    end
  end
`endif

endmodule

// ================================================================================================
// tsr_mvm_rvec — row-vector emitter (AT §5.2 MAXOUT, §5.3 LSUM): after `start`, walks the op's pass
// rows in increasing r (q = 0 .. tq - 1, g = 0 .. nseg - 1, i = 0 .. gs - 1; r = q n_head + g G + hg0
// + i), reads the row's value from the caller's table (entry e = g tact + q gs + i; val_addr ->
// val_data combinational) and writes word r mod 16 of line base + r / 16: consecutive rows of one
// line merge into one job with their words in `own` (byte enables). Jobs {last, line, own, 0, y};
// `last` on the final job when last_en. done (pulse) after the final job was taken.
// PIPE = 0: one stage, the row address, the table read, the line compare and the line register in one cycle
//   (12 levels with the table read in the caller; the 250 MHz work of docs/platform-u200.md §8.9 found them
//   the slowest paths of the PV unit and the gather).
// PIPE = 1: three stages with the same jobs in the same order: the walker (the loop counters and the
//   row's r and table address, registered in the descriptor slice A), the read stage (the table read, line
//   address and the "same line as the row before" flag, registered in slice B) and the line assembler. The
//   slices are tsr_rslice (registered ready and valid), so no path runs through more than one stage. The
//   table must not change while the emitter runs (the callers: the PV unit's l table after its last slice,
//   the gather's key table after its last job). The first job is offered 3 cycles later and `done` comes 3
//   cycles later than with PIPE = 0; the rate stays one row per cycle.
// ================================================================================================
module tsr_mvm_rvec
  import tsr_pkg::*;
#(
  parameter  int unsigned SPM_AW = 16,
  parameter  bit          PIPE   = 1'b0,
  localparam int unsigned JW     = 1 + SPM_AW + 16 + 16 + TSR_LINE_BITS
) (
  input  logic                  clk,
  input  logic                  rst_n,

  input  logic                  start,
  input  logic [3:0]            tq,          // T_q
  input  logic [3:0]            gs,          // GS
  input  logic [3:0]            tact,        // T_q GS
  input  logic [8:0]            nseg,
  input  logic [6:0]            g_heads,     // G
  input  logic [5:0]            hg0,
  input  logic [14:0]           n_head,
  input  logic [SPM_AW-1:0]     base,
  input  logic                  last_en,

  output logic [6:0]            val_addr,
  input  logic [31:0]           val_data,

  output logic                  job_valid,
  input  logic                  job_ready,
  output logic [JW-1:0]         job,
  output logic                  done
);

  // The line assembler: the current line, the job register and the final-line handshake. Rows come from the
  // caller of this block (the flat walk, PIPE = 0, or the read stage's slice B, PIPE = 1).
  logic              cur_v_q;         // a line is being collected
  logic [SPM_AW-1:0] cur_l_q;
  logic [15:0]       cur_m_q;
  logic [15:0][31:0] cur_d_q;
  logic              ov_q;            // job register
  logic [JW-1:0]     o_q;
  logic              fin_q, done_q;
  wire               o_free = ~ov_q | job_ready;

  if (!PIPE) begin : g_flat
    logic        act_q;
    logic [3:0]  q_q, i_q;
    logic [8:0]  g_q;
    logic [17:0] rq_q;            // q n_head
    logic [17:0] rg_q;            // q n_head + g G
    logic [7:0]  eq_q;            // q gs
    logic [7:0]  eg_q;            // g tact

    wire [17:0] r     = rg_q + 18'(hg0) + 18'(i_q);
    wire [SPM_AW-1:0] rl = base + SPM_AW'(r >> 4);
    wire [3:0]  rw    = r[3:0];
    wire        last_row = (i_q == gs - 4'd1) & (g_q == nseg - 9'd1) & (q_q == tq - 4'd1);
    // a row is consumed when it joins the current line, or when the current line can be emitted
    wire        same  = cur_v_q & (rl == cur_l_q);
    wire        step  = act_q & (same | ~cur_v_q | o_free);
    assign val_addr = 7'(eg_q + eq_q + 8'(i_q));

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        act_q   <= 1'b0;
        cur_v_q <= 1'b0;
        ov_q    <= 1'b0;
        fin_q   <= 1'b0;
        done_q  <= 1'b0;
      end else begin
        done_q <= 1'b0;
        if (ov_q && job_ready) ov_q <= 1'b0;
        if (start) begin
          act_q   <= 1'b1;
          q_q     <= '0;
          g_q     <= '0;
          i_q     <= '0;
          rq_q    <= '0;
          rg_q    <= '0;
          eq_q    <= '0;
          eg_q    <= '0;
          cur_v_q <= 1'b0;
          fin_q   <= 1'b0;
        end else if (step) begin
          if (cur_v_q && !same) begin                     // emit the current line, start a new one
            ov_q <= 1'b1;
            o_q  <= {1'b0, cur_l_q, cur_m_q, 16'd0, cur_d_q};
          end
          cur_v_q <= 1'b1;
          cur_l_q <= rl;
          cur_m_q <= (same ? cur_m_q : 16'd0) | (16'd1 << rw);
          for (int k = 0; k < 16; k++) begin
            if (!same) cur_d_q[k] <= 32'd0;
            if (4'(k) == rw) cur_d_q[k] <= val_data;
          end
          // next row
          if (i_q == gs - 4'd1) begin
            i_q <= '0;
            if (g_q == nseg - 9'd1) begin
              g_q  <= '0;
              rq_q <= rq_q + 18'(n_head);
              rg_q <= rq_q + 18'(n_head);
              eg_q <= '0;
              q_q  <= q_q + 4'd1;
              eq_q <= eq_q + 8'(gs);
              if (q_q == tq - 4'd1) begin
                act_q <= 1'b0;
                fin_q <= 1'b1;
              end
            end else begin
              g_q  <= g_q + 9'd1;
              rg_q <= rg_q + 18'(g_heads);
              eg_q <= eg_q + 8'(tact);
            end
          end else begin
            i_q <= i_q + 4'd1;
          end
        end else if (fin_q && cur_v_q && o_free) begin     // the final line
          ov_q    <= 1'b1;
          o_q     <= {last_en, cur_l_q, cur_m_q, 16'd0, cur_d_q};
          cur_v_q <= 1'b0;
        end else if (fin_q && !cur_v_q && (!ov_q || job_ready)) begin
          fin_q  <= 1'b0;
          done_q <= 1'b1;
        end
      end
    end

    wire unused_ok = &{1'b0, last_row, r[17:4]};
  end else begin : g_pipe
    // ---- stage 1, the walker: loop counters, r and the table address of the next row ----
    logic [3:0]  q_q, i_q;
    logic [8:0]  g_q;
    logic [17:0] rq_q, rg_q;
    logic [7:0]  eq_q, eg_q;
    logic [3:0]  tqm1_q, gsm1_q;                  // tq - 1, gs - 1, nseg - 1 as seen at start
    logic [8:0]  nsm1_q;
    logic [3:0]  gsw_q;                           // gs, tact, n_head, g_heads, hg0 of the op
    logic [3:0]  tactw_q;
    logic [14:0] nhw_q;
    logic [6:0]  ghw_q;
    logic [5:0]  hg0w_q;
    logic        w_act_q;
    wire w_last_i = (i_q == gsm1_q);
    wire w_last_g = (g_q == nsm1_q);
    wire w_last_q = (q_q == tqm1_q);
    wire [17:0] w_r  = rg_q + 18'(hg0w_q) + 18'(i_q);
    wire [6:0]  w_va = 7'(eg_q + eq_q + 8'(i_q));
    localparam int unsigned DW = 1 + 18 + 7;      // {last, r, va}
    logic          a_sv, a_sr, a_mv, a_mr;
    logic [DW-1:0] a_sd, a_md;
    assign a_sv = w_act_q;
    assign a_sd = {w_last_i & w_last_g & w_last_q, w_r, w_va};
    wire   w_adv = a_sv & a_sr;
    tsr_rslice #(.W(DW)) u_a (
      .clk, .rst_n, .s_valid(a_sv), .s_ready(a_sr), .s_data(a_sd), .m_valid(a_mv), .m_ready(a_mr), .m_data(a_md)
    );
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        w_act_q <= 1'b0;
      end else if (start) begin
        w_act_q <= 1'b1;
      end else if (w_adv && w_last_i && w_last_g && w_last_q) begin
        w_act_q <= 1'b0;
      end
    end
    always_ff @(posedge clk) begin
      if (start) begin
        q_q     <= '0;
        g_q     <= '0;
        i_q     <= '0;
        rq_q    <= '0;
        rg_q    <= '0;
        eq_q    <= '0;
        eg_q    <= '0;
        tqm1_q  <= tq - 4'd1;
        gsm1_q  <= gs - 4'd1;
        nsm1_q  <= nseg - 9'd1;
        gsw_q   <= gs;
        tactw_q <= tact;
        nhw_q   <= n_head;
        ghw_q   <= g_heads;
        hg0w_q  <= hg0;
      end else if (w_adv) begin
        if (w_last_i) begin
          i_q <= '0;
          if (w_last_g) begin
            g_q  <= '0;
            rq_q <= rq_q + 18'(nhw_q);
            rg_q <= rq_q + 18'(nhw_q);
            eg_q <= '0;
            q_q  <= q_q + 4'd1;
            eq_q <= eq_q + 8'(gsw_q);
          end else begin
            g_q  <= g_q + 9'd1;
            rg_q <= rg_q + 18'(ghw_q);
            eg_q <= eg_q + 8'(tactw_q);
          end
        end else begin
          i_q <= i_q + 4'd1;
        end
      end
    end

    // ---- stage 2, the read: table value, line address, same line as the row before ----
    wire [17:0]       a_r    = a_md[7 +: 18];
    wire              a_last = a_md[7 + 18];
    assign val_addr = a_md[6:0];
    logic [17:0] pr_q;                            // r of the row before, valid while pv_q
    logic        pv_q;
    wire [SPM_AW-1:0] rd_l   = base + SPM_AW'(a_r >> 4);
    wire              rd_same = pv_q & (SPM_AW'(a_r >> 4) == SPM_AW'(pr_q >> 4));
    localparam int unsigned BW = 1 + 1 + SPM_AW + 4 + 32;     // {last, same, line, word, value}
    logic          b_sv, b_sr, b_mv, b_mr;
    logic [BW-1:0] b_sd, b_md;
    assign b_sv = a_mv;
    assign a_mr = b_sr;
    assign b_sd = {a_last, rd_same, rd_l, a_r[3:0], val_data};
    tsr_rslice #(.W(BW)) u_b (
      .clk, .rst_n, .s_valid(b_sv), .s_ready(b_sr), .s_data(b_sd), .m_valid(b_mv), .m_ready(b_mr), .m_data(b_md)
    );
    always_ff @(posedge clk) begin
      if (!rst_n || start) pv_q <= 1'b0;
      else if (a_mv && b_sr) pv_q <= 1'b1;
      if (a_mv && b_sr) pr_q <= a_r;
    end

    // ---- stage 3, the line assembler ----
    wire [31:0]       h_val  = b_md[31:0];
    wire [3:0]        h_rw   = b_md[35:32];
    wire [SPM_AW-1:0] h_rl   = b_md[36 +: SPM_AW];
    wire              h_same = b_md[36 + SPM_AW];
    wire              h_last = b_md[37 + SPM_AW];
    wire              step   = b_mv & (h_same | ~cur_v_q | o_free);
    assign b_mr = step;

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        cur_v_q <= 1'b0;
        ov_q    <= 1'b0;
        fin_q   <= 1'b0;
        done_q  <= 1'b0;
      end else begin
        done_q <= 1'b0;
        if (ov_q && job_ready) ov_q <= 1'b0;
        if (start) begin
          cur_v_q <= 1'b0;
          fin_q   <= 1'b0;
        end else if (step) begin
          if (cur_v_q && !h_same) begin                   // emit the current line, start a new one
            ov_q <= 1'b1;
            o_q  <= {1'b0, cur_l_q, cur_m_q, 16'd0, cur_d_q};
          end
          cur_v_q <= 1'b1;
          cur_l_q <= h_rl;
          cur_m_q <= (h_same ? cur_m_q : 16'd0) | (16'd1 << h_rw);
          for (int k = 0; k < 16; k++) begin
            if (!h_same) cur_d_q[k] <= 32'd0;
            if (4'(k) == h_rw) cur_d_q[k] <= h_val;
          end
          if (h_last) fin_q <= 1'b1;
        end else if (fin_q && cur_v_q && o_free) begin     // the final line
          ov_q    <= 1'b1;
          o_q     <= {last_en, cur_l_q, cur_m_q, 16'd0, cur_d_q};
          cur_v_q <= 1'b0;
        end else if (fin_q && !cur_v_q && (!ov_q || job_ready)) begin
          fin_q  <= 1'b0;
          done_q <= 1'b1;
        end
      end
    end
  end

  assign job_valid = ov_q;
  assign job       = o_q;
  assign done      = done_q;

endmodule
/* verilator lint_on DECLFILENAME */
