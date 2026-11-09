// tsr_vu_red.sv — the VU's reduction unit and scalar path: row drain from the lanes' partials, the
// 8-input subtree, the pairing top combiner, the post unit (FM, FA, SQ, RC), the row-result packer, and
// the SOP scalar functions SQRT, RSQRT (post unit) and DIV (divider).
//
// Spec: docs/design/02-vector-unit.md "VU" §6.2 reductions, §7.5 reduction unit and row post; §5.8 SOP;
// AT §5.5 AMASK row maxima.
// Row flow: the injector's row token (at the row's last line) is delayed past the lanes' last partial
// write; the drain reads group q = 0 .. ng (8 partials: entry ring + q/2, lanes 8·(q mod 2) ..) one per
// cycle into the subtree, then frees the row's np + 1 ring entries. An empty row (RED TO_S with m = 0)
// drains one group of init values. Each top level pairs consecutive results of a row: a pair -> FA (or
// MAXT), an unpaired last result -> FA(x, +0) = z(x) for SUM / SUMSQ (VU §6.2: a skipped subtree passes
// z(q)), x for MAX. With ng + 1 present groups this equals the complete 128-partial tree bit for bit.
// MAX and AMASK use MAXT at every node (any order gives the same total-order maximum); their partials
// start at -inf / M_FLOOR (tsr_vu_lane).
// Post (VU §6.2): NONE y; RCP RC(y); RMS RC(SQ(FA(FM(y, sk), se))) — a fixed pipeline FM -> FA -> SQ -> RC
// whose stages are bypassed by the row's mode; SOP SQRT = SQ, RSQRT = RC(SQ) (VU §5.8).
// Results: TO_S and SOP -> the s register; vectors -> 16 words per line at rd + row/16: RED writes the
// whole line (tail words 0), AMASK writes only the words of its rows (the words past R keep their
// bits, AT tsr_ref_at.c). Lines go to tsr_vu_wr's RED FIFO, whose places the generator reserved.

module tsr_vu_red
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
#(
  parameter int unsigned TQD = 128,         // row tokens between the drain and the post output
  parameter bit LOCAL_L1 = 1'b0,           // first subtree level resides in the half-width lane banks
  parameter int unsigned DRL = 0            // cycles from dr_addr to its dr_data (0: combinational; tsr_vu_lanes' pipes)
) (
  input  logic                           clk,
  input  logic                           rst_n,
  input  tsr_vu_opk_t [1:0]              opk,
  // row tokens (tsr_vu_inj)
  input  logic                           row_v,
  input  tsr_vu_row_t                    row_tok,
  // partial drain (all lanes, a combinational read DRL cycles behind the address)
  output logic [3:0]                     dr_addr,
  output logic                           dr_sel, dr_empty, dr_mx,
  input  logic [15:0][31:0]              dr_data,
  input  logic [3:0][31:0]               dr_l1,
  output logic [4:0]                     ring_rel,   // entries freed this cycle
  // scalar path (SOP SQRT / RSQRT / DIV; operands in opk[tag].k1, k2, mode in opk[tag].post)
  input  logic                           sopx_v,
  input  logic                           sopx_tag,
  output logic                           sopx_idle,
  // results
  output logic                           r_v,
  output logic                           r_tag,
  output logic [TSR_VU_LW-1:0]           r_line,
  output logic [TSR_LINE_BITS-1:0]       r_data,
  output logic [TSR_SPM_BE_W-1:0]        r_be,
  output logic                           s_we,
  output logic [3:0]                     s_wi,
  output logic [31:0]                    s_wd,
  output logic                           s_wtag,
  output logic                           idle
);
  import tsr_fp_pkg::*;

  localparam int unsigned TOKD = 15;        // token delay: the drain reads after the partial writes (end of lane cycle 13)
  localparam int unsigned FAL  = TSR_VU_FADD_LAT;
  localparam int unsigned TQW  = $clog2(TQD);

  function automatic logic [31:0] key_of(input logic [31:0] d);
    return d[31] ? ~d : (d | 32'h8000_0000);
  endfunction
  function automatic logic [31:0] maxt(input logic [31:0] a, input logic [31:0] b);
    logic [31:0] ad, bd;
    if (tsr_f32_is_nan(a) || tsr_f32_is_nan(b)) return TSR_VU_NAN;
    ad = tsr_f32_daz(a);
    bd = tsr_f32_daz(b);
    return (key_of(ad) >= key_of(bd)) ? ad : bd;
  endfunction

  // -------------------------------------------------------------------------------------------
  // Row tokens: delay, then the drain queue
  // -------------------------------------------------------------------------------------------
  localparam int unsigned RTW = $bits(tsr_vu_row_t);
  logic        rd_v;
  tsr_vu_row_t rd_t;
  logic [TOKD-1:0] rdv_sr;
  always_ff @(posedge clk) begin
    if (!rst_n) rdv_sr <= '0;
    else        rdv_sr <= {rdv_sr[TOKD-2:0], row_v};
  end
  assign rd_v = rdv_sr[TOKD-1];
  tsr_vu_dly #(.W(RTW), .D(TOKD)) u_rtd (.clk, .d(row_tok), .q(rd_t));

  logic          dq_v, dq_pop;
  logic [RTW-1:0] dq_h;
  logic [4:0]    dq_n;
  tsr_vu_fifo #(.W(RTW), .D(16)) u_dq (
    .clk, .rst_n, .push(rd_v), .din(rd_t), .pop(dq_pop), .v(dq_v), .dout(dq_h), .n(dq_n)
  );
  tsr_vu_row_t dh;
  assign dh = tsr_vu_row_t'(dq_h);

  // -------------------------------------------------------------------------------------------
  // Drain: one group per cycle
  // -------------------------------------------------------------------------------------------
  logic [3:0] q_q;                          // group index of the head row
  wire        mx   = opk[dh.tag].rkind[1];  // MAX or AMASK
  wire        dlast = dq_v && (dh.empty || q_q == dh.ng);
  assign dq_pop  = dlast;
  assign dr_sel = q_q[0];
  assign dr_empty = dh.empty;
  assign dr_mx = mx;
  assign dr_addr = dh.ring + {1'b0, q_q[3:1]};
  assign ring_rel = (dlast && !dh.empty) ? 5'(dh.np) + 5'd1 : 5'd0;

  // the drain step whose partials dr_data carries: the step's controls, DRL cycles after dr_addr
  logic dv_c, dl_c, mx_c, sel_c, em_c, dr_busy;
  if (DRL == 0) begin : g_dr0
    assign dr_busy = 1'b0;
    assign dv_c  = dq_v;
    assign dl_c  = dlast;
    assign mx_c  = mx;
    assign sel_c = q_q[0];
    assign em_c  = dh.empty;
  end else begin : g_drn
    logic [DRL-1:0] drv_sr;
    always_ff @(posedge clk) begin
      if (!rst_n) drv_sr <= '0;
      else        drv_sr <= DRL'({drv_sr, dq_v});
    end
    assign dv_c    = drv_sr[DRL-1];
    assign dr_busy = |drv_sr;
    tsr_vu_dly #(.W(4), .D(DRL)) u_drd (.clk, .d({dlast, mx, q_q[0], dh.empty}), .q({dl_c, mx_c, sel_c, em_c}));
  end

  logic        g_v_q, g_last_q, g_mx_q;
  logic [7:0][31:0] g_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      q_q   <= '0;
      g_v_q <= 1'b0;
    end else begin
      g_v_q <= dv_c;
      if (dq_v) q_q <= dlast ? 4'd0 : q_q + 4'd1;
    end
    g_last_q <= dl_c;
    g_mx_q   <= mx_c;
    if (!LOCAL_L1) for (int i = 0; i < 8; i++) begin
      if (em_c) g_q[i] <= mx_c ? TSR_VU_NINF : 32'd0;   // empty row: SUM +0, MAX -inf (VU §6.2)
      else      g_q[i] <= dr_data[(sel_c ? 8 : 0) + i];
    end
  end

  // drained row tokens, in order, until the row's result enters the post unit
  logic          tq_pop;
  logic          tq_v;
  logic [RTW-1:0] tq_h;
  logic [TQW:0]  tq_n;
  tsr_vu_fifo #(.W(RTW), .D(TQD)) u_tq (
    .clk, .rst_n, .push(dlast), .din(dh), .pop(tq_pop), .v(tq_v), .dout(tq_h), .n(tq_n)
  );

  // -------------------------------------------------------------------------------------------
  // Node: FA (SUM) or MAXT (MAX), fixed latency FAL
  // -------------------------------------------------------------------------------------------
  // (instantiated below as tsr_vu_rnode)

  // subtree: 8 -> 4 -> 2 -> 1
  logic [3:0][31:0] l1;
  logic [1:0][31:0] l2;
  logic [31:0]      l3;
  logic [3*FAL-1:0] sv_sr, sl_sr, sm_sr;
  for (genvar i = 0; i < 4; i++) begin : g_l1
    if (LOCAL_L1) begin : g_remote
      assign l1[i] = dr_l1[i];
    end else begin : g_local
      tsr_vu_rnode u_n (.clk, .rst_n, .mx(g_mx_q), .a(g_q[2*i]), .b(g_q[2*i+1]), .y(l1[i]));
    end
  end
  logic mx1, mx2;
  tsr_vu_dly #(.W(1), .D(FAL)) d_mx1 (.clk, .d(g_mx_q), .q(mx1));
  tsr_vu_dly #(.W(1), .D(FAL)) d_mx2 (.clk, .d(mx1), .q(mx2));
  for (genvar i = 0; i < 2; i++) begin : g_l2
    tsr_vu_rnode u_n (.clk, .rst_n, .mx(mx1), .a(l1[2*i]), .b(l1[2*i+1]), .y(l2[i]));
  end
  tsr_vu_rnode u_l3 (.clk, .rst_n, .mx(mx2), .a(l2[0]), .b(l2[1]), .y(l3));
  always_ff @(posedge clk) begin
    if (!rst_n) sv_sr <= '0;
    else        sv_sr <= {sv_sr[3*FAL-2:0], g_v_q};
    sl_sr <= {sl_sr[3*FAL-2:0], g_last_q};
    sm_sr <= {sm_sr[3*FAL-2:0], g_mx_q};
  end

  // top combiner: 4 pairing levels (16 -> 8 -> 4 -> 2 -> 1)
  logic [4:0]        tv;                     // level inputs: valid, value, last-of-row, mode
  logic [4:0][31:0]  tx;
  logic [4:0]        tl, tm;
  assign tv[0] = sv_sr[3*FAL-1];
  assign tx[0] = l3;
  assign tl[0] = sl_sr[3*FAL-1];
  assign tm[0] = sm_sr[3*FAL-1];
  for (genvar k = 0; k < 4; k++) begin : g_top
    logic        h_v;
    logic [31:0] h_x;
    logic        iv, il, im;
    logic [31:0] ia, ib;
    always_ff @(posedge clk) begin
      if (!rst_n) h_v <= 1'b0;
      else if (tv[k]) h_v <= !h_v && !tl[k];
      if (tv[k] && !h_v) h_x <= tx[k];
    end
    // issue: a pair, or the unpaired last result with the identity (+0 for FA: z(x); x for MAXT)
    assign iv = tv[k] && (h_v || tl[k]);
    assign ia = h_v ? h_x : tx[k];
    assign ib = h_v ? tx[k] : (tm[k] ? tx[k] : 32'd0);
    assign il = tl[k];
    assign im = tm[k];
    tsr_vu_rnode u_n (.clk, .rst_n, .mx(im), .a(ia), .b(ib), .y(tx[k+1]));
    logic [FAL-1:0] v_sr, l_sr, m_sr;
    always_ff @(posedge clk) begin
      if (!rst_n) v_sr <= '0;
      else        v_sr <= {v_sr[FAL-2:0], iv};
      l_sr <= {l_sr[FAL-2:0], il};
      m_sr <= {m_sr[FAL-2:0], im};
    end
    assign tv[k+1] = v_sr[FAL-1];
    assign tl[k+1] = l_sr[FAL-1];
    assign tm[k+1] = m_sr[FAL-1];
  end

  // -------------------------------------------------------------------------------------------
  // Post unit: FM -> FA -> SQ -> RC, stages enabled by the mode {fm, fa, sq, rc}
  // -------------------------------------------------------------------------------------------
  // inputs: a row result (tv[4]) or a scalar-path SQRT / RSQRT (sopx_v with mode 1 / 2); DIV uses the
  // divider. The injector admits a scalar op only while this unit is idle.
  wire        sx_post = sopx_v && (opk[sopx_tag].post != 2'd3);
  wire        sx_div  = sopx_v && (opk[sopx_tag].post == 2'd3);

  logic        p_v;
  logic [31:0] p_x, p_k, p_e;
  logic [3:0]  p_m;
  tsr_vu_row_t p_t;                          // the entering row's token ({sopx} rides along apart)
  assign tq_pop = tv[4];
  always_comb begin
    tsr_vu_row_t t;
    t   = tsr_vu_row_t'(tq_h);
    p_t = t;
    if (sx_post) begin
      p_t     = '0;
      p_t.tag = sopx_tag;
    end
    p_v = tv[4] || sx_post;
    p_x = sx_post ? opk[sopx_tag].k1 : tx[4];
    p_k = opk[t.tag].sk;
    p_e = opk[t.tag].se;
    if (sx_post) p_m = (opk[sopx_tag].post == 2'd1) ? 4'b0010 : 4'b0011;      // SQRT, RSQRT
    else case (opk[t.tag].post)
      2'd1:    p_m = 4'b0001;               // RCP
      2'd2:    p_m = 4'b1111;               // RMS
      default: p_m = 4'b0000;
    endcase
  end

  localparam int unsigned FML = TSR_VU_FMUL_LAT, SQL = TSR_VU_SQRT_LAT, RCL = TSR_VU_RCP_LAT;
  logic [31:0] pfm, pfa, psq, prc;
  logic [1:0]  ov;
  logic [31:0] x1, x2, x3, x4, k1d, e2;
  logic [3:0]  m1, m2, m3, m4;
  tsr_fp_mul #(.LAT(FML)) u_pfm (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(p_x), .b(p_k), .out_valid(ov[0]), .y(pfm));
  tsr_vu_dly #(.W(36), .D(FML)) d_p1 (.clk, .d({p_m, p_x}), .q({m1, k1d}));
  tsr_vu_dly #(.W(32), .D(FML)) d_pe (.clk, .d(p_e), .q(e2));
  assign x1 = m1[3] ? pfm : k1d;
  tsr_fp_add #(.LAT(FAL)) u_pfa (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(x1), .b(e2), .sub(1'b0), .out_valid(ov[1]), .y(pfa));
  logic [31:0] x1d;
  tsr_vu_dly #(.W(36), .D(FAL)) d_p2 (.clk, .d({m1, x1}), .q({m2, x1d}));
  assign x2 = m2[2] ? pfa : x1d;
  tsr_vu_sqrt u_psq (.clk, .x(x2), .y(psq));
  logic [31:0] x2d;
  tsr_vu_dly #(.W(36), .D(SQL)) d_p3 (.clk, .d({m2, x2}), .q({m3, x2d}));
  assign x3 = m3[1] ? psq : x2d;
  tsr_vu_rcp u_prc (.clk, .x(x3), .y(prc));
  logic [31:0] x3d;
  tsr_vu_dly #(.W(36), .D(RCL)) d_p4 (.clk, .d({m3, x3}), .q({m4, x3d}));
  assign x4 = m4[0] ? prc : x3d;
  localparam int unsigned PL = FML + FAL + SQL + RCL;
  logic [PL-1:0] pv_sr;
  always_ff @(posedge clk) begin
    if (!rst_n) pv_sr <= '0;
    else        pv_sr <= {pv_sr[PL-2:0], p_v};
  end
  wire pout = pv_sr[PL-1];
  tsr_vu_row_t pt;
  logic        psx;
  tsr_vu_dly #(.W(RTW + 1), .D(PL)) d_pt (.clk, .d({sx_post, p_t}), .q({psx, pt}));

  // divider (SOP DIV)
  logic        dv_in_ready, dv_out_v;
  logic [31:0] dv_y;
  logic        dv_busy_q, dv_tag_q;
  tsr_fp_div u_div (
    .clk, .rst_n, .in_valid(sx_div), .in_ready(dv_in_ready), .a(opk[sopx_tag].k1), .b(opk[sopx_tag].k2),
    .out_valid(dv_out_v), .out_ready(1'b1), .y(dv_y)
  );
  always_ff @(posedge clk) begin
    if (!rst_n) dv_busy_q <= 1'b0;
    else if (sx_div) dv_busy_q <= 1'b1;
    else if (dv_out_v) dv_busy_q <= 1'b0;
    if (sx_div) dv_tag_q <= sopx_tag;
  end

  // -------------------------------------------------------------------------------------------
  // Results: s register, or the packer
  // -------------------------------------------------------------------------------------------
  logic [15:0][31:0] pk_q;                   // the result line being packed
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s_we <= 1'b0;
      r_v  <= 1'b0;
    end else begin
      s_we <= (pout && (psx || opk[pt.tag].to_s)) || dv_out_v;
      r_v  <= pout && !psx && !opk[pt.tag].to_s && (pt.row[3:0] == 4'd15 || pt.lrow);
    end
    s_wi   <= dv_out_v ? opk[dv_tag_q].sd : opk[pt.tag].sd;
    s_wd   <= dv_out_v ? dv_y : x4;
    s_wtag <= dv_out_v ? dv_tag_q : pt.tag;
    if (pout && !psx && !opk[pt.tag].to_s) begin
      pk_q[pt.row[3:0]] <= x4;
      if (pt.row[3:0] == 4'd0) begin
        for (int i = 1; i < 16; i++) pk_q[i] <= 32'd0;
      end
    end
    r_tag  <= pt.tag;
    r_line <= opk[pt.tag].rd + TSR_VU_LW'(pt.row[15:4]);
    begin
      logic [15:0][31:0] ln;
      logic [15:0]       wm;                 // words written: those of rows 16·(row/16) .. row
      ln = pk_q;
      ln[pt.row[3:0]] = x4;
      for (int i = 0; i < 16; i++) begin
        if (pt.row[3:0] == 4'd0 && i != 0) ln[i] = 32'd0;
        wm[i] = (4'(i) <= pt.row[3:0]);
      end
      r_data <= ln;
      for (int i = 0; i < 16; i++) r_be[4*i +: 4] <= (opk[pt.tag].keep && !wm[i]) ? 4'h0 : 4'hF;
    end
  end

  assign sopx_idle = !dv_busy_q && !sx_div;
  assign idle = !dq_v && (rdv_sr == '0) && !tq_v && (tv == '0) && (pv_sr == '0) && !g_v_q && !dr_busy &&
                (sv_sr == '0) && !dv_busy_q;

  wire unused_ok = &{1'b0, dr_l1, g_q, ov, dv_in_ready, dq_n, tq_n, tl[4], tm[4], m4[3:1], pt.ring, pt.np, pt.ng, pt.empty};

endmodule

// One reduction node: FA(a, b) (mx = 0) or MAXT(a, b) (mx = 1), fixed latency TSR_VU_FADD_LAT.
/* verilator lint_off DECLFILENAME */
module tsr_vu_rnode
  import tsr_vu_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        mx,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] y
);
  import tsr_fp_pkg::*;
  localparam int unsigned FAL = TSR_VU_FADD_LAT;
  function automatic logic [31:0] key_of(input logic [31:0] d);
    return d[31] ? ~d : (d | 32'h8000_0000);
  endfunction
  logic [31:0] fa_y, mt_y, mt;
  logic        mxd, ov;
  always_comb begin
    logic [31:0] ad, bd;
    ad = tsr_f32_daz(a);
    bd = tsr_f32_daz(b);
    if (tsr_f32_is_nan(a) || tsr_f32_is_nan(b)) mt = TSR_VU_NAN;
    else mt = (key_of(ad) >= key_of(bd)) ? ad : bd;
  end
  tsr_fp_add #(.LAT(FAL)) u_fa (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(a), .b(b), .sub(1'b0), .out_valid(ov), .y(fa_y));
  tsr_vu_dly #(.W(32), .D(FAL)) d_mt (.clk, .d(mt), .q(mt_y));
  tsr_vu_dly #(.W(1),  .D(FAL)) d_mx (.clk, .d(mx), .q(mxd));
  assign y = mxd ? mt_y : fa_y;
  wire unused_ok = &{1'b0, ov};
endmodule
/* verilator lint_on DECLFILENAME */
