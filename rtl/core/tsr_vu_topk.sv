// tsr_vu_topk.sv — TOPK (0x33) of the VU: per row the k <= 16 best (index, value) pairs under the total order,
// by a running-threshold filter and one insertion per cycle into a sorted 16-entry list.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §5.5 encoding, §6.8 semantics, §7.9 rate, VU-14).
// Contract: the output rows equal tsr_ref_topk (sw/sim/tsr_ref_vu.c) bit for bit: candidates (j, x̂_j) with
// x̂ = DAZ(x), NaN -> 0x7FC00000; sorted by x̂ descending under the total order (NaN above everything), a tie
// to the smaller j; pair i < k at words 2i (j as u32) and 2i + 1 (x̂); missing pairs (m < k) are
// (0xFFFFFFFF, 0xFF800000); the words past 2k of the row's last output line are 0 (VU §4.2).
// Flow: lines arrive from the injector (SK_TOPK slots) into an input FIFO (in_ready is a level the
// injector checks before taking a slot, with room for the two lines in its pipeline). Per line, the
// filter keeps the elements that would enter the current list (all while it holds fewer than k; else
// those before its k-th entry; a stale threshold only lets more through, the insertion decides); the
// survivors are inserted in element order, one per cycle (later elements never beat an equal earlier
// one: the tie rule); a line without survivors costs one cycle. After the row's last line the k pairs leave as ceil(k/8) lines through o_*
// (tsr_vu_wr's TOPK FIFO, valid / ready), then the list is cleared. Worst case one insertion per element
// (ascending rows, VU §7.9).

module tsr_vu_topk
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
(
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         in_v,
  output logic                         in_ready,
  input  logic [TSR_LINE_BITS-1:0]     in_line,
  input  tsr_vu_slot_t                 in_slot,      // tail, d0 (row output line), d1 (element index of word 0)
  input  logic [4:0]                   in_k,
  output logic                         o_v,
  input  logic                         o_ready,
  output logic                         o_tag,
  output logic [TSR_VU_LW-1:0]         o_line,
  output logic [TSR_LINE_BITS-1:0]     o_data,
  output logic                         idle
);
  import tsr_fp_pkg::*;

  localparam int unsigned QD = 8;

  // candidate order: a before b (VU §6.8)
  function automatic logic ahead(input logic [31:0] va, input logic [23:0] ja, input logic [31:0] vb,
                                  input logic [23:0] jb);
    logic na, nb;
    logic [31:0] ka, kb;
    na = (va == TSR_VU_NAN);
    nb = (vb == TSR_VU_NAN);
    ka = va[31] ? ~va : (va | 32'h8000_0000);
    kb = vb[31] ? ~vb : (vb | 32'h8000_0000);
    if (na != nb) return na;
    if (!na && ka != kb) return ka > kb;
    return ja < jb;
  endfunction

  // -------------------------------------------------------------------------------------------
  // Input FIFO
  // -------------------------------------------------------------------------------------------
  // entry: the line, its valid words, the row's output line, the element index of word 0, tag, last, k
  typedef struct packed {
    logic [TSR_LINE_BITS-1:0] line;
    logic [4:0]               tail;
    logic [TSR_VU_LW-1:0]     d0;
    logic [23:0]              d1;
    logic                     tag;
    logic                     r_last;
    logic [4:0]               k;
  } qe_t;
  localparam int unsigned EW = $bits(qe_t);
  logic          q_v, q_pop;
  logic [EW-1:0] q_h;
  logic [3:0]    q_n;
  qe_t           qi, hs;
  always_comb begin
    qi.line   = in_line;
    qi.tail   = in_slot.tail;
    qi.d0     = in_slot.d0;
    qi.d1     = 24'(in_slot.d1);
    qi.tag    = in_slot.tag;
    qi.r_last = in_slot.r_last;
    qi.k      = in_k;
  end
  tsr_vu_fifo #(.W(EW), .D(QD)) u_q (
    .clk, .rst_n, .push(in_v), .din(qi), .pop(q_pop), .v(q_v), .dout(q_h), .n(q_n)
  );
  assign in_ready = (32'(q_n) <= QD - 3);
  assign hs = qe_t'(q_h);
  wire [TSR_LINE_BITS-1:0] hl = hs.line;
  wire [4:0]               hk = hs.k;

  // -------------------------------------------------------------------------------------------
  // List, filter, insertion
  // -------------------------------------------------------------------------------------------
  // F (take): the FIFO head's survivors against the threshold entry k - 1 (thr_q, registered from the
  //   list each cycle: one cycle stale, which only lets more through, as the list only improves)
  // A (select): the lowest survivor of the current line -> the candidate register
  // B (insert): systolic: entry i keeps its value, takes the candidate (the first i it goes before) or
  //   entry i - 1 (the entries after it); entries >= ln are empty, entries >= k are never output
  typedef enum logic [1:0] { T_RUN, T_EMIT0, T_EMIT1 } tst_e;
  tst_e        st_q;
  logic [15:0][31:0] lv_q;                 // values, sorted
  logic [15:0][23:0] lj_q;                 // element indices
  logic [4:0]  ln_q;                       // entries held (<= k)
  logic [4:0]  k_q;
  logic        tag_q, last_q, lc_v_q;      // the current line's row: tag, last line; a line is held
  logic [TSR_VU_LW-1:0] dl_q;
  logic [15:0]       sm_q;                 // survivors of the current line not yet selected
  logic [15:0][31:0] sv_q;
  logic [23:0]       sb_q;                 // element index of word 0
  logic [31:0]       thr_x_q;
  logic [23:0]       thr_j_q;
  logic              c_v_q;                // candidate (stage B)
  logic [31:0]       c_x_q;
  logic [23:0]       c_j_q;

  always_ff @(posedge clk) begin
    thr_x_q <= lv_q[4'(k_q - 5'd1)];
    thr_j_q <= lj_q[4'(k_q - 5'd1)];
  end

  // F: filter (on the FIFO head)
  logic [15:0]       f_m;
  logic [15:0][31:0] f_x;
  always_comb begin
    for (int i = 0; i < 16; i++) begin
      logic [31:0] x;
      x = hl[32*i +: 32];
      f_x[i] = tsr_f32_is_nan(x) ? TSR_VU_NAN : tsr_f32_daz(x);
      f_m[i] = (32'(i) < 32'(hs.tail)) && ((ln_q < hk) || ahead(f_x[i], hs.d1 + 24'(i), thr_x_q, thr_j_q));
    end
  end

  // A: the lowest survivor
  logic [3:0]  s_i;
  logic        s_any;
  always_comb begin
    s_i   = 4'd0;
    s_any = 1'b0;
    for (int i = 15; i >= 0; i--) begin
      if (sm_q[i]) begin
        s_i   = 4'(i);
        s_any = 1'b1;
      end
    end
  end
  wire last_sel = (sm_q & (sm_q - 16'd1)) == 16'd0;   // at most one survivor left (selected now)

  // B: the candidate against every entry
  logic [15:0] bf;                          // the candidate goes before entry i (or entry i is empty)
  always_comb begin
    for (int i = 0; i < 16; i++) bf[i] = (5'(i) >= ln_q) || ahead(c_x_q, c_j_q, lv_q[i], lj_q[i]);
  end

  // a line is taken while none is held, or once the held line's survivors are all selected and it is
  // not its row's last (that row is emitted first): one line per cycle while nothing enters
  wire take    = q_v && (st_q == T_RUN) && (!lc_v_q || (last_sel && !last_q));
  wire row_end = (st_q == T_RUN) && lc_v_q && last_q && !s_any && !c_v_q;
  assign q_pop = take;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q   <= T_RUN;
      ln_q   <= '0;
      sm_q   <= '0;
      lc_v_q <= 1'b0;
      c_v_q  <= 1'b0;
    end else begin
      // A
      c_v_q <= s_any;
      if (s_any) sm_q[s_i] <= 1'b0;
      // F
      if (take) begin
        sm_q   <= f_m;
        lc_v_q <= 1'b1;
      end
      // B
      if (c_v_q && bf[4'(k_q - 5'd1)] && ln_q < k_q) ln_q <= ln_q + 5'd1;
      case (st_q)
        T_RUN:   if (row_end) st_q <= T_EMIT0;
        T_EMIT0: if (o_ready) st_q <= (k_q > 5'd8) ? T_EMIT1 : T_RUN;
        T_EMIT1: if (o_ready) st_q <= T_RUN;
        default: st_q <= T_RUN;
      endcase
      if ((st_q == T_EMIT0 && o_ready && k_q <= 5'd8) || (st_q == T_EMIT1 && o_ready)) begin
        ln_q   <= '0;
        lc_v_q <= 1'b0;
      end
    end
    c_x_q <= sv_q[s_i];
    c_j_q <= sb_q + 24'(s_i);
    if (take) begin
      sv_q   <= f_x;
      sb_q   <= hs.d1;
      k_q    <= hk;
      tag_q  <= hs.tag;
      last_q <= hs.r_last;
      dl_q   <= hs.d0;
    end
    if (c_v_q) begin
      for (int i = 0; i < 16; i++) begin
        if (bf[i]) begin
          if (i == 0 || !bf[(i > 0) ? i - 1 : 0]) begin
            lv_q[i] <= c_x_q;
            lj_q[i] <= c_j_q;
          end else begin
            lv_q[i] <= lv_q[(i > 0) ? i - 1 : 0];
            lj_q[i] <= lj_q[(i > 0) ? i - 1 : 0];
          end
        end
      end
    end
  end

  // output line: pairs 8·h .. 8·h + 7
  always_comb begin
    logic [15:0][31:0] d;
    int unsigned h;
    h = (st_q == T_EMIT1) ? 1 : 0;
    for (int p = 0; p < 8; p++) begin
      int unsigned i;
      i = 8 * h + p;
      d[2*p]     = (5'(i) < ln_q) ? 32'(lj_q[i]) : 32'hFFFF_FFFF;
      d[2*p + 1] = (5'(i) < ln_q) ? lv_q[i] : TSR_VU_NINF;
      if (5'(i) >= k_q) begin
        d[2*p]     = 32'd0;
        d[2*p + 1] = 32'd0;
      end
    end
    o_data = d;
  end
  assign o_v    = (st_q == T_EMIT0) || (st_q == T_EMIT1);
  assign o_tag  = tag_q;
  assign o_line = dl_q + ((st_q == T_EMIT1) ? TSR_VU_LW'(1) : TSR_VU_LW'(0));
  assign idle   = (st_q == T_RUN) && !lc_v_q && !q_v && !c_v_q;
  // the slot's other fields are not needed here
  wire unused_ok = &{1'b0, in_slot};

endmodule
