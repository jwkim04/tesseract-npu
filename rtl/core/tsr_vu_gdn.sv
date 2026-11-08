// tsr_vu_gdn.sv — the fused Gated DeltaNet state update unit (docs/design/02-vector-unit.md §7.12).
//
// For one head, the op reads a state tile S (rows x m f32), the vectors k, q, v and the scalars
// alpha, beta, and produces S' and o with
//     u = S k;  e = beta (v - alpha u);  S' = alpha S + e k^T;  o = S' q
// evaluated as in sw/sim/tsr_ref_gdn.c (tsr_ref_gdn; VU §7.12.3): the dot products are exact sums of
// integer products in a block frame, converted once to f32 (VU-29); the update is FA(FM(alpha, S),
// FM(e, k)) on the VU primitives. The unit is bit-exact with that reference.
//
// Use (a stream interface; not instantiated in tsr_vu, VU §7.12.9):
//   1. cfg_*: m (a multiple of LANES, 4 * LANES .. LANES * NB_MAX), rows, alpha, beta, accepted when
//      cfg_ready. Then the vectors k (m words), q (m words), v (rows words), in this order, one word
//      per cycle on vec_* (vec_ready). The engine (tsr_vu_gdn_coef) builds the head tables in one of two
//      banks while the rows of the previous head are still flowing.
//   2. in_*: the rows, LANES words per beat (m / LANES beats per row), first the vector k as a pseudo
//      row (the unit takes kappa = k . q from it, through the same dot array), then the `rows` state
//      rows. in_ready is low when no head is configured or all row slots are in use.
//   3. out_*: S' rows, LANES words per beat, in row order (out_last on the last beat of a row);
//      o_*: one word per row in row order; done: one pulse when the head's last row and last o are out.
// There is no back pressure on the outputs: the consumer must take every beat and word.
//
// Structure: ingest (writes the two row buffer copies B1 / B2 in URAM, tracks the row's largest
// exponent), the dot array (tsr_vu_gdn_dot: 36 DSP48E2, 16 URAM), the converter (tsr_vu_gdn_fx2f), the
// scalar chain (one shared FM and FA: e = FS(bv, u'), o = FA(FM(alpha, w), FM(e, kappa))), the update
// array (tsr_vu_gdn_upd). Rows are processed in order; a row occupies one of NSLOT slots from its first
// ingest beat until its update has been issued.

module tsr_vu_gdn #(
  parameter  int unsigned LANES  = 8,
  parameter  int unsigned NSLOT  = 16,
  parameter  int unsigned NB_MAX = 32,
  parameter  int unsigned R_MAX  = 1024,
  parameter  int unsigned RD_LAT = 3,
  parameter  int unsigned FM_LAT = 4,
  parameter  int unsigned FA_LAT = 5,
  parameter                STYLE  = "ultra",
  localparam int unsigned SW     = $clog2(NSLOT),
  localparam int unsigned NBW    = $clog2(NB_MAX),
  localparam int unsigned LW     = (LANES > 1) ? $clog2(LANES) : 1,
  localparam int unsigned MW     = $clog2(LANES * NB_MAX) + 1,
  localparam int unsigned RW     = $clog2(R_MAX) + 1,
  localparam int unsigned RB     = $clog2(R_MAX)               // bits of a row index
) (
  input  logic                    clk,
  input  logic                    rst_n,
  // head command
  input  logic                    cfg_valid,
  output logic                    cfg_ready,
  input  logic [MW-1:0]           cfg_m,
  input  logic [RW-1:0]           cfg_rows,
  input  logic [31:0]             cfg_alpha,
  input  logic [31:0]             cfg_beta,
  // vectors k, q, v
  input  logic                    vec_valid,
  input  logic [31:0]             vec_data,
  output logic                    vec_ready,
  // rows
  input  logic                    in_valid,
  output logic                    in_ready,
  input  logic [LANES*32-1:0]     in_data,
  // results
  output logic                    out_valid,
  output logic                    out_last,
  output logic [LANES*32-1:0]     out_data,
  output logic                    o_valid,
  output logic [31:0]             o_data,
  output logic                    done
);

  localparam int unsigned NBB = NBW + 1;           // bits of a beat count

  // A row has at most 256 elements: the 48-bit chain sums are exact only up to that (VU §7.12.5).
  if (LANES < 2 || (LANES & (LANES - 1)) != 0 || NB_MAX < 8 || (NB_MAX & (NB_MAX - 1)) != 0 ||
      LANES * NB_MAX > 256 || NSLOT < 4 || (NSLOT & (NSLOT - 1)) != 0 || R_MAX < 2 || (R_MAX & (R_MAX - 1)) != 0 ||
      RD_LAT < 2) begin : g_chk_cfg
    $fatal(1, "tsr_vu_gdn: LANES and NB_MAX powers of two with LANES * NB_MAX <= 256 and NB_MAX >= 8, NSLOT >= 4 and R_MAX powers of two, RD_LAT >= 2");
  end

  // ============================================================================================
  // Head command, banks, setup engine
  // ============================================================================================
  logic        cfg_bank_q;                         // bank of the next head
  logic [1:0]  bk_busy, bk_ready;
  logic [31:0] bk_alpha [2];
  logic [7:0]  bk_ek [2], bk_eq [2];
  logic [1:0]  bk_badk, bk_badq;
  logic [31:0] bk_kappa [2];
  logic [RW-1:0] bk_o_left [2], bk_s_left [2];

  logic        coef_busy, coef_done, coef_bank_q, coef_badk, coef_badq;
  logic [7:0]  coef_ek, coef_eq;
  logic        cfg_fire;
  logic        hq_full, hq_empty;
  logic        ing_take_head;
  assign cfg_ready = ~coef_busy & ~bk_busy[cfg_bank_q] & ~hq_full;
  assign cfg_fire  = cfg_valid & cfg_ready;

  // head queue for the ingest: {bank, nb, rows + 1}, two entries
  logic        hq_bank [2];
  logic [NBB-1:0] hq_nb [2];
  logic [RW-1:0] hq_rows [2];
  logic        hq_wp, hq_rp;
  logic [1:0]  hq_cnt;
  assign hq_full  = (hq_cnt == 2'd2);
  assign hq_empty = (hq_cnt == 2'd0);

  logic                kf_we, bv_we;
  logic [LW-1:0]       kf_lane, tr_lane;
  logic [NBW:0]        kf_waddr, tr_waddr;
  logic [31:0]         kf_wdata, tr_wdata, bv_wdata;
  logic [1:0]          tr_we;
  logic [RB:0]         bv_waddr;

  tsr_vu_gdn_coef #(.LANES(LANES), .NB_MAX(NB_MAX), .R_MAX(R_MAX), .RD_LAT(RD_LAT), .FM_LAT(FM_LAT), .STYLE(STYLE)) u_coef (
    .clk, .rst_n,
    .start(cfg_fire), .bank(cfg_bank_q), .m(cfg_m), .rows(cfg_rows), .alpha(cfg_alpha), .beta(cfg_beta),
    .busy(coef_busy),
    .vec_valid, .vec_data, .vec_ready,
    .kf_we(kf_we), .kf_lane(kf_lane), .kf_waddr(kf_waddr), .kf_wdata(kf_wdata),
    .tr_we(tr_we), .tr_lane(tr_lane), .tr_waddr(tr_waddr), .tr_wdata(tr_wdata),
    .bv_we(bv_we), .bv_waddr(bv_waddr), .bv_wdata(bv_wdata),
    .done(coef_done), .done_ek(coef_ek), .done_eq(coef_eq), .done_badk(coef_badk), .done_badq(coef_badq));

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cfg_bank_q  <= 1'b0;
      coef_bank_q <= 1'b0;
    end else if (cfg_fire) begin
      cfg_bank_q  <= ~cfg_bank_q;
      coef_bank_q <= cfg_bank_q;
    end
  end

  // done of the head, set below
  logic head_done_v;
  logic head_done_bank;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      bk_busy  <= 2'b00;
      bk_ready <= 2'b00;
      bk_badk  <= 2'b00;
      bk_badq  <= 2'b00;
    end else begin
      if (cfg_fire) begin
        bk_busy[cfg_bank_q] <= 1'b1;
      end
      if (coef_done) begin
        bk_ready[coef_bank_q] <= 1'b1;
        bk_badk[coef_bank_q]  <= coef_badk;
        bk_badq[coef_bank_q]  <= coef_badq;
      end
      if (head_done_v) begin
        bk_busy[head_done_bank]  <= 1'b0;
        bk_ready[head_done_bank] <= 1'b0;
      end
    end
  end
  always_ff @(posedge clk) begin
    if (cfg_fire) bk_alpha[cfg_bank_q] <= cfg_alpha;
    if (coef_done) begin
      bk_ek[coef_bank_q] <= coef_ek;
      bk_eq[coef_bank_q] <= coef_eq;
    end
  end

  // head queue
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      hq_wp <= 1'b0;
      hq_rp <= 1'b0;
      hq_cnt <= 2'd0;
    end else begin
      if (cfg_fire) hq_wp <= ~hq_wp;
      if (ing_take_head) hq_rp <= ~hq_rp;
      hq_cnt <= hq_cnt + {1'b0, cfg_fire} - {1'b0, ing_take_head};
    end
  end
  always_ff @(posedge clk) begin
    if (cfg_fire) begin
      hq_bank[hq_wp] <= cfg_bank_q;
      hq_nb[hq_wp]   <= NBB'(cfg_m >> LW);
      hq_rows[hq_wp] <= cfg_rows + 1'b1;               // the pseudo row first
    end
  end

  // ============================================================================================
  // Ingest
  // ============================================================================================
  logic           ing_act;
  logic           ing_bank;
  logic [NBB-1:0] ing_nb, ing_beat;
  logic [RW-1:0]  ing_left, ing_ridx;
  logic [SW:0]    c_alloc, c_done, c_dot, c_upd, c_free;
  logic           slot_ok;
  logic           ing_row_first;

  assign slot_ok       = ((c_alloc - c_free) != (SW + 1)'(NSLOT));
  assign in_ready      = ing_act & slot_ok;
  assign ing_take_head = ~ing_act & ~hq_empty;
  logic           take_in;
  assign take_in       = in_valid & in_ready;
  logic           ing_last_beat;
  assign ing_last_beat = take_in & (ing_beat == ing_nb - 1'b1);
  assign ing_row_first = (ing_beat == '0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ing_act <= 1'b0;
    end else if (ing_take_head) begin
      ing_act <= 1'b1;
    end else if (ing_last_beat && ing_left == (RW)'(1)) begin
      ing_act <= 1'b0;
    end
  end
  always_ff @(posedge clk) begin
    if (ing_take_head) begin
      ing_bank <= hq_bank[hq_rp];
      ing_nb   <= hq_nb[hq_rp];
      ing_left <= hq_rows[hq_rp];
      ing_ridx <= '0;
      ing_beat <= '0;
    end else if (take_in) begin
      if (ing_last_beat) begin
        ing_beat <= '0;
        ing_left <= ing_left - 1'b1;
        ing_ridx <= ing_ridx + 1'b1;
      end else begin
        ing_beat <= ing_beat + 1'b1;
      end
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) c_alloc <= '0;
    else if (ing_last_beat) c_alloc <= c_alloc + 1'b1;
  end

  // the row buffers (both copies) are written from the ingest port
  logic                b_we;
  logic [SW+NBW-1:0]   b_waddr;
  assign b_we    = take_in;
  assign b_waddr = {c_alloc[SW-1:0], ing_beat[NBW-1:0]};

  // Row exponent: the largest biased exponent of the row (Es). The exponents of an accepted beat are
  // registered, reduced to their maximum in a pipelined tree (one comparator per level), and folded into
  // the running maximum of the row. An exponent of 255 (Inf, NaN) makes Es = 255, which is the row's
  // bad flag; the dot products of a bad row are the canonical NaN whatever Es is.
  localparam int unsigned LV = $clog2(LANES);
  typedef struct packed {
    logic              last;
    logic              first;
    logic              bank;
    logic              pseudo;
    logic [SW-1:0]     slot;
    logic [NBB-1:0]    nb;
    logic [RB-1:0]     ridx;
  } ctl_t;
  logic                   st_v [LV+1];
  ctl_t                   st_c [LV+1];
  logic [LANES-1:0][7:0]  st_e0;
  always_ff @(posedge clk) begin
    if (!rst_n) st_v[0] <= 1'b0; else st_v[0] <= take_in;
  end
  always_ff @(posedge clk) begin
    st_c[0] <= '{last: ing_last_beat, first: ing_row_first, bank: ing_bank, pseudo: (ing_ridx == '0),
                 slot: c_alloc[SW-1:0], nb: ing_nb, ridx: RB'(ing_ridx - 1'b1)};
    for (int l = 0; l < LANES; l++) st_e0[l] <= in_data[32*l + 23 +: 8];
  end
  for (genvar lv = 1; lv <= LV; lv++) begin : g_emax
    logic [7:0] lvl [LANES >> lv];
    for (genvar i = 0; i < (LANES >> lv); i++) begin : g_i
      logic [7:0] a_e, b_e;
      if (lv == 1) begin : g_first
        assign a_e = st_e0[2*i];
        assign b_e = st_e0[2*i+1];
      end else begin : g_next
        assign a_e = g_emax[lv-1].lvl[2*i];
        assign b_e = g_emax[lv-1].lvl[2*i+1];
      end
      always_ff @(posedge clk) lvl[i] <= (a_e > b_e) ? a_e : b_e;
    end
    always_ff @(posedge clk) begin
      if (!rst_n) st_v[lv] <= 1'b0; else st_v[lv] <= st_v[lv-1];
    end
    always_ff @(posedge clk) st_c[lv] <= st_c[lv-1];
  end

  logic        ex_v, ex_last, ex_first, ex_bank, ex_pseudo;
  logic [SW-1:0]  ex_slot;
  logic [NBB-1:0] ex_nb;
  logic [RB-1:0]  ex_ridx;
  logic [7:0]  beat_max;
  assign ex_v      = st_v[LV];
  assign ex_last   = st_c[LV].last;
  assign ex_first  = st_c[LV].first;
  assign ex_bank   = st_c[LV].bank;
  assign ex_pseudo = st_c[LV].pseudo;
  assign ex_slot   = st_c[LV].slot;
  assign ex_nb     = st_c[LV].nb;
  assign ex_ridx   = st_c[LV].ridx;
  assign beat_max  = g_emax[LV].lvl[0];

  logic [7:0]  row_max_q;
  logic [7:0]  row_max_n;
  logic        row_bad_n;
  assign row_max_n = (ex_first || beat_max > row_max_q) ? beat_max : row_max_q;
  assign row_bad_n = (row_max_n == 8'hFF);
  always_ff @(posedge clk) begin
    if (ex_v) row_max_q <= row_max_n;
  end

  // slot metadata, written when a row is complete; two copies for the two readers
  //   a (dot launch): {ridx, pseudo, bad, bank, nb, es}    b (update launch): {pseudo, bank, nb}
  localparam int unsigned MA_W = RB + 1 + 1 + 1 + NBB + 8;
  localparam int unsigned MB_W = 1 + 1 + NBB;
  (* ram_style = "distributed" *) logic [MA_W-1:0] meta_a [NSLOT];
  (* ram_style = "distributed" *) logic [MB_W-1:0] meta_b [NSLOT];
  logic meta_we;
  assign meta_we = ex_v & ex_last;
  always_ff @(posedge clk) begin
    if (meta_we) begin
      meta_a[ex_slot] <= {ex_ridx, ex_pseudo, row_bad_n, ex_bank, ex_nb, row_max_n};
      meta_b[ex_slot] <= {ex_pseudo, ex_bank, ex_nb};
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) c_done <= '0;
    else if (meta_we) c_done <= c_done + 1'b1;
  end

  // ============================================================================================
  // Dot launch
  // ============================================================================================
  logic [MA_W-1:0] ma;
  logic [RB-1:0]   ma_ridx;
  logic            ma_pseudo, ma_bad, ma_bank;
  logic [NBB-1:0]  ma_nb;
  logic [7:0]      ma_es;
  logic [SW-1:0]   dot_slot;
  assign dot_slot = c_dot[SW-1:0];
  assign ma = meta_a[dot_slot];
  assign {ma_ridx, ma_pseudo, ma_bad, ma_bank, ma_nb, ma_es} = ma;
  logic [NBB-1:0]  dot_cool;
  logic            dot_go;
  assign dot_go = (c_done != c_dot) & (dot_cool == '0) & bk_ready[ma_bank];
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      c_dot    <= '0;
      dot_cool <= '0;
    end else if (dot_go) begin
      c_dot    <= c_dot + 1'b1;
      dot_cool <= ma_nb - 1'b1;
    end else if (dot_cool != '0) begin
      dot_cool <= dot_cool - 1'b1;
    end
  end

  logic                res_valid;
  logic signed [47:0]  res_kh, res_kl, res_qh, res_ql;

  // launch record: popped at the result
  typedef struct packed {
    logic [SW-1:0] slot;
    logic [RB-1:0] ridx;
    logic          pseudo;
    logic          bad;
    logic          bank;
    logic [7:0]    es;
  } lrec_t;
  (* ram_style = "distributed" *) lrec_t lrec [NSLOT];
  logic [SW:0] lwp_q, lrp_q;
  always_ff @(posedge clk) begin
    if (dot_go) lrec[lwp_q[SW-1:0]] <= '{slot: dot_slot, ridx: ma_ridx, pseudo: ma_pseudo, bad: ma_bad, bank: ma_bank, es: ma_es};
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      lwp_q <= '0;
      lrp_q <= '0;
    end else begin
      if (dot_go) lwp_q <= lwp_q + 1'b1;
      if (res_valid) lrp_q <= lrp_q + 1'b1;
    end
  end

  tsr_vu_gdn_dot #(.LANES(LANES), .NSLOT(NSLOT), .NB_MAX(NB_MAX), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_dot (
    .clk, .rst_n,
    .b_we, .b_waddr, .b_wdata(in_data),
    .t_we(tr_we), .t_lane(tr_lane), .t_waddr(tr_waddr), .t_wdata(tr_wdata),
    .go(dot_go), .go_slot(dot_slot), .go_bank(ma_bank), .go_es(ma_es), .go_nb(ma_nb),
    .res_valid, .res_kh, .res_kl, .res_qh, .res_ql);

  // ============================================================================================
  // Conversion of the two sums of a row
  // ============================================================================================
  lrec_t           rrec;
  assign rrec = lrec[lrp_q[SW-1:0]];

  localparam int unsigned TGW = 1 + SW + 1 + 1 + RB;     // kind, slot, bank, pseudo, ridx
  logic                cv_v;
  logic [TGW-1:0]      cv_tag;
  logic signed [47:0]  cv_hi, cv_lo;
  logic signed [9:0]   cv_x;
  logic                cv_bad;

  // the Q sums wait one cycle
  logic                q_pend;
  logic signed [47:0]  q_hi, q_lo;
  lrec_t               q_rec;
  always_ff @(posedge clk) begin
    if (!rst_n) q_pend <= 1'b0; else q_pend <= res_valid;
  end
  always_ff @(posedge clk) begin
    if (res_valid) begin
      q_hi  <= res_qh;
      q_lo  <= res_ql;
      q_rec <= rrec;
    end
  end
  // x = Es + Et - 302 (the scale of the sum, tsr_ref_gdn.h), in [-302, 206]
  function automatic logic signed [9:0] xscale(input logic [7:0] es, input logic [7:0] et);
    return signed'(10'(es) + 10'(et) - 10'd302);
  endfunction
  always_comb begin
    if (res_valid) begin
      cv_v   = 1'b1;
      cv_tag = {1'b0, rrec.slot, rrec.bank, rrec.pseudo, rrec.ridx};
      cv_hi  = res_kh;
      cv_lo  = res_kl;
      cv_x   = xscale(rrec.es, bk_ek[rrec.bank]);
      cv_bad = rrec.bad | bk_badk[rrec.bank];
    end else begin
      cv_v   = q_pend;
      cv_tag = {1'b1, q_rec.slot, q_rec.bank, q_rec.pseudo, q_rec.ridx};
      cv_hi  = q_hi;
      cv_lo  = q_lo;
      cv_x   = xscale(q_rec.es, bk_eq[q_rec.bank]);
      cv_bad = q_rec.bad | bk_badq[q_rec.bank];
    end
  end

  logic                cvo_v;
  logic [TGW-1:0]      cvo_tag;
  logic [31:0]         cvo_y;
  tsr_vu_gdn_fx2f #(.TW(TGW)) u_cvt (
    .clk, .rst_n, .in_valid(cv_v), .in_tag(cv_tag), .in_hi(cv_hi), .in_lo(cv_lo), .in_x(cv_x), .in_bad(cv_bad),
    .out_valid(cvo_v), .out_tag(cvo_tag), .out_y(cvo_y));

  logic            cvo_kind, cvo_bank, cvo_pseudo;
  logic [SW-1:0]   cvo_slot;
  logic [RB-1:0]   cvo_ridx;
  assign {cvo_kind, cvo_slot, cvo_bank, cvo_pseudo, cvo_ridx} = cvo_tag;

  // ============================================================================================
  // Scalar chain: e = FS(bv, u'); o = FA(FM(alpha, w), FM(e, kappa))
  // ============================================================================================
  // bv read for u' of a real row; u' and the tag wait for the data
  logic            u_v;
  assign u_v = cvo_v & ~cvo_kind & ~cvo_pseudo;
  logic [31:0]     bv_rdata;
  tsr_vu_gdn_ram #(.DEPTH(2 * R_MAX), .WIDTH(32), .NWE(1), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_bv (
    .clk, .we(bv_we), .waddr(bv_waddr), .wdata(bv_wdata),
    .re(u_v), .raddr({cvo_bank, cvo_ridx}), .rdata(bv_rdata));

  logic [RD_LAT-1:0] ud_v;
  logic [31:0]       ud_u [RD_LAT];
  logic [SW-1:0]     ud_slot [RD_LAT];
  logic              ud_bank [RD_LAT];
  always_ff @(posedge clk) begin
    if (!rst_n) ud_v <= '0; else ud_v <= RD_LAT'({ud_v, u_v});
  end
  always_ff @(posedge clk) begin
    ud_u[0] <= cvo_y;
    ud_slot[0] <= cvo_slot;
    ud_bank[0] <= cvo_bank;
    for (int i = 1; i < RD_LAT; i++) begin
      ud_u[i] <= ud_u[i-1];
      ud_slot[i] <= ud_slot[i-1];
      ud_bank[i] <= ud_bank[i-1];
    end
  end

  // FS request: bv data valid with the delayed u'
  logic            fs_req;
  assign fs_req = ud_v[RD_LAT-1];

  // w request for the multiplier (A): real row; kappa from the pseudo row
  logic            wa_v;
  assign wa_v = cvo_v & cvo_kind & ~cvo_pseudo;
  logic            wa_pend;
  logic [31:0]     wa_y;
  logic [SW-1:0]   wa_slot;
  logic            wa_bank;
  logic            eb_pend;
  logic [31:0]     eb_e;
  logic [SW-1:0]   eb_slot;
  logic            eb_bank;
  logic            fm_issueB, fm_issueA;
  assign fm_issueB = eb_pend;
  assign fm_issueA = wa_pend & ~eb_pend;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wa_pend <= 1'b0;
    end else if (wa_v) begin
      wa_pend <= 1'b1;
    end else if (fm_issueA) begin
      wa_pend <= 1'b0;
    end
    if (wa_v) begin
      wa_y    <= cvo_y;
      wa_slot <= cvo_slot;
      wa_bank <= cvo_bank;
    end
  end
  always_ff @(posedge clk) begin
    if (cvo_v & cvo_kind & cvo_pseudo) bk_kappa[cvo_bank] <= cvo_y;
  end

  // shared FM
  logic            fm_v, fm_ov;
  logic [31:0]     fm_a, fm_b, fm_y;
  assign fm_v = fm_issueA | fm_issueB;
  assign fm_a = fm_issueB ? eb_e : bk_alpha[wa_bank];
  assign fm_b = fm_issueB ? bk_kappa[eb_bank] : wa_y;
  tsr_fp_mul #(.LAT(FM_LAT)) u_sfm (
    .clk, .rst_n, .ce(1'b1), .in_valid(fm_v), .a(fm_a), .b(fm_b), .out_valid(fm_ov), .y(fm_y));
  logic [FM_LAT-1:0] fmt_v, fmt_b;
  logic [SW-1:0]     fmt_slot [FM_LAT];
  logic              fmt_bank [FM_LAT];
  always_ff @(posedge clk) begin
    if (!rst_n) fmt_v <= '0; else fmt_v <= FM_LAT'({fmt_v, fm_v});
    fmt_b <= FM_LAT'({fmt_b, fm_issueB});
  end
  always_ff @(posedge clk) begin
    fmt_slot[0] <= fm_issueB ? eb_slot : wa_slot;
    fmt_bank[0] <= fm_issueB ? eb_bank : wa_bank;
    for (int i = 1; i < FM_LAT; i++) begin
      fmt_slot[i] <= fmt_slot[i-1];
      fmt_bank[i] <= fmt_bank[i-1];
    end
  end

  // results of the multiplier: A -> x2 fifo, B -> x4 fifo
  logic [31:0]     x2_q [8];
  logic [31:0]     x4_q [8];
  logic [SW-1:0]   x4_slot [8];
  logic            x4_bank [8];
  logic [3:0]      x2_wp, x2_rp, x4_wp, x4_rp;
  wire             fm_ra = fm_ov & ~fmt_b[FM_LAT-1];
  wire             fm_rb = fm_ov & fmt_b[FM_LAT-1];
  logic            pair_ok, fa_issueO, fs_issue;
  assign pair_ok   = (x2_wp != x2_rp) & (x4_wp != x4_rp);
  assign fs_issue  = fs_req;
  assign fa_issueO = pair_ok & ~fs_issue;
  always_ff @(posedge clk) begin
    if (fm_ra) x2_q[x2_wp[2:0]] <= fm_y;
    if (fm_rb) begin
      x4_q[x4_wp[2:0]]    <= fm_y;
      x4_slot[x4_wp[2:0]] <= fmt_slot[FM_LAT-1];
      x4_bank[x4_wp[2:0]] <= fmt_bank[FM_LAT-1];
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      x2_wp <= '0; x2_rp <= '0; x4_wp <= '0; x4_rp <= '0;
    end else begin
      if (fm_ra)     x2_wp <= x2_wp + 1'b1;
      if (fm_rb)     x4_wp <= x4_wp + 1'b1;
      if (fa_issueO) begin
        x2_rp <= x2_rp + 1'b1;
        x4_rp <= x4_rp + 1'b1;
      end
    end
  end

  // shared FA
  logic            fa_v, fa_ov;
  logic [31:0]     fa_a, fa_b, fa_y;
  assign fa_v = fs_issue | fa_issueO;
  assign fa_a = fs_issue ? bv_rdata : x2_q[x2_rp[2:0]];
  assign fa_b = fs_issue ? ud_u[RD_LAT-1] : x4_q[x4_rp[2:0]];
  tsr_fp_add #(.LAT(FA_LAT)) u_sfa (
    .clk, .rst_n, .ce(1'b1), .in_valid(fa_v), .a(fa_a), .b(fa_b), .sub(fs_issue), .out_valid(fa_ov), .y(fa_y));
  logic [FA_LAT-1:0] fat_fs;
  logic [SW-1:0]     fat_slot [FA_LAT];
  logic              fat_bank [FA_LAT];
  always_ff @(posedge clk) begin
    fat_fs <= FA_LAT'({fat_fs, fs_issue});
    fat_slot[0] <= fs_issue ? ud_slot[RD_LAT-1] : x4_slot[x4_rp[2:0]];
    fat_bank[0] <= fs_issue ? ud_bank[RD_LAT-1] : x4_bank[x4_rp[2:0]];
    for (int i = 1; i < FA_LAT; i++) begin
      fat_slot[i] <= fat_slot[i-1];
      fat_bank[i] <= fat_bank[i-1];
    end
  end

  // FA results: e (FS) -> E store, FM request B; o (FA) -> output
  logic            e_v;
  assign e_v = fa_ov & fat_fs[FA_LAT-1];
  (* ram_style = "distributed" *) logic [31:0] e_store [NSLOT];
  logic [NSLOT-1:0] e_valid;
  logic            upd_launch, upd_skip, upd_launch_d;
  logic [SW-1:0]   upd_slot;
  always_ff @(posedge clk) begin
    if (e_v) e_store[fat_slot[FA_LAT-1]] <= fa_y;
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      e_valid <= '0;
    end else begin
      if (e_v)        e_valid[fat_slot[FA_LAT-1]] <= 1'b1;
      if (cvo_v & cvo_kind & cvo_pseudo) e_valid[cvo_slot] <= 1'b1;     // the pseudo row is over: skip it
      if (upd_launch | upd_skip) e_valid[upd_slot] <= 1'b0;
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      eb_pend <= 1'b0;
    end else if (e_v) begin
      eb_pend <= 1'b1;
    end else if (fm_issueB) begin
      eb_pend <= 1'b0;
    end
    if (e_v) begin
      eb_e    <= fa_y;
      eb_slot <= fat_slot[FA_LAT-1];
      eb_bank <= fat_bank[FA_LAT-1];
    end
  end

  assign o_valid = fa_ov & ~fat_fs[FA_LAT-1];
  assign o_data  = fa_y;
  logic            o_bank;
  assign o_bank  = fat_bank[FA_LAT-1];

  // ============================================================================================
  // Update launch
  // ============================================================================================
  logic [MB_W-1:0] mb;
  logic            mb_pseudo, mb_bank;
  logic [NBB-1:0]  mb_nb;
  assign upd_slot = c_upd[SW-1:0];
  assign mb = meta_b[upd_slot];
  assign {mb_pseudo, mb_bank, mb_nb} = mb;
  logic [NBB-1:0]  upd_cool;
  assign upd_launch = e_valid[upd_slot] & ~mb_pseudo & (upd_cool == '0);
  assign upd_skip   = e_valid[upd_slot] & mb_pseudo;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      c_upd        <= '0;
      upd_cool     <= '0;
      c_free       <= '0;
      upd_launch_d <= 1'b0;
    end else begin
      upd_launch_d <= upd_launch;
      if (upd_launch) begin
        c_upd    <= c_upd + 1'b1;
        upd_cool <= mb_nb - 1'b1;
      end else if (upd_skip) begin
        c_upd    <= c_upd + 1'b1;
      end else if (upd_cool != '0) begin
        upd_cool <= upd_cool - 1'b1;
      end
      // a slot is free for the ingest in the cycle after its update (or skip) was launched; the reads of
      // the update run in the following NB cycles, but the ingest writes beat g no earlier than that
      // cycle + g, one cycle after the read of beat g
      c_free <= c_free + (SW + 1)'(upd_launch_d | upd_skip);
    end
  end

  tsr_vu_gdn_upd #(.LANES(LANES), .NSLOT(NSLOT), .NB_MAX(NB_MAX), .RD_LAT(RD_LAT), .FM_LAT(FM_LAT), .FA_LAT(FA_LAT), .STYLE(STYLE)) u_upd (
    .clk, .rst_n,
    .b_we, .b_waddr, .b_wdata(in_data),
    .kf_we, .kf_lane, .kf_waddr, .kf_wdata,
    .go(upd_launch), .go_slot(upd_slot), .go_bank(mb_bank), .go_nb(mb_nb),
    .go_alpha(bk_alpha[mb_bank]), .go_e(e_store[upd_slot]),
    .out_valid, .out_last, .out_data);

  // ============================================================================================
  // Completion of a head
  // ============================================================================================
  logic ob_q;                                           // bank whose S' rows are coming out
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ob_q <= 1'b0;
      for (int b = 0; b < 2; b++) begin
        bk_o_left[b] <= '0;
        bk_s_left[b] <= '0;
      end
    end else begin
      if (cfg_fire) begin
        bk_o_left[cfg_bank_q] <= cfg_rows;
        bk_s_left[cfg_bank_q] <= cfg_rows;
      end
      if (o_valid) bk_o_left[o_bank] <= bk_o_left[o_bank] - 1'b1;
      if (out_valid && out_last) begin
        bk_s_left[ob_q] <= bk_s_left[ob_q] - 1'b1;
        if (bk_s_left[ob_q] == (RW)'(1)) ob_q <= ~ob_q;
      end
    end
  end
  // a head is done in the cycle after both its counters reached zero
  logic [1:0] bk_fin;
  always_comb begin
    for (int b = 0; b < 2; b++) bk_fin[b] = bk_busy[b] & bk_ready[b] & (bk_o_left[b] == '0) & (bk_s_left[b] == '0);
  end
  assign head_done_v    = |bk_fin;
  assign head_done_bank = bk_fin[1];
  assign done           = head_done_v;


endmodule
