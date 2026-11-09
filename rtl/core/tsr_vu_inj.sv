// tsr_vu_inj.sv — the VU injector: takes slots in order once their operands are in, forms the physical lanes'
// operands for each sub-slot, and hands the exit descriptors, the reduction row tokens and the
// pseudo-slot actions (c-line loads, ROPE table / positions, TOPK lines, scalar path) on.
//
// Spec: docs/design/02-vector-unit.md "VU" §7.1 input reorder, §7.3 half-rate presentation, §7.5
// partials, §7.6 rotation, §7.8 sequencer.
// Stages: D (decide: the head slot and its operands are ready, the lanes' exit order allows it; pop the
// FIFOs, read the local memory) -> I (one cycle per sub-slot: operand selection) -> registered outputs
// to the lanes (their input register is the next stage).
// GDN (HAS_GDN, VU §7.12.5): a slot of kind SK_GDN goes to the GDN engine, not to the lanes. It takes nsub injection
//   cycles like a lane slot (4 for the head command and the vector lines, 1 for a row line), and fires only when the
//   engine says it can take it (gdn_cfg_ok, gdn_vec_ok, gdn_row_ok for the first line of a row). The engine sees
//   gdn_v once per slot, with the registered slot (topk_slot) and the line (pa_data) of the cycle.
// Rules:
//   - Exit order: every sub-slot leaves the lanes after every earlier one. A slot with exit latency L
//     (TSR_VU_LF front, TSR_VU_LB back) and n sub-slots is admitted only if L >= ex_rem, the cycles to
//     the last scheduled exit; it sets ex_rem = L + n - 1 (a back slot followed by a front slot waits
//     LB - LF cycles). Reduction slots have no exit.
//   - Split rate (VU §7.3): sub-slot k of an n-way slot carries elements [16k/n, 16(k+1)/n) in lanes
//     0 .. 16/n - 1 (the back lanes, or the EXPK lanes with VU_EXP_LANES 8).
//   - FRONT8 presents each architectural line in two halves. NEOX uses four sub-slots:
//     line = k / 2, part = k % 2. Other operations retain their logical element index.
//   - ROPE (VU §6.7, §7.6): NEOX pair slot: its first line is output line j = FS(FM(x0, c), FM(x1, s)), sub-slot 1
//     the second line j + n_dims/32 = FA(FM(x0, s), FM(x1, c)), x0 = A[l], x1 = B[l], angle i = 16j + l. NORM: lane l
//     pairs elements l & ~1 and l | 1 with angle 8o + l/2: even lanes FS, odd lanes FA. Both lines of
//     a pair are read before either is written (in-place safe, VU §4.5).
//   - Partials: a row's first line allocates P = min(L, 8) entries of the 16-entry ring (L = the row's
//     lines); line li accumulates into entry base + li mod 8; the row token carries base and P, and
//     tsr_vu_red frees the entries after the row's drain.
//   - DQROW (VU §6.8, tsr_vu_dq): the chunk's meta pseudo-slot starts the constant table at its fire;
//     a DQROW line slot waits until its table entry is written, and its operands are formed from the
//     quant line (Q8_0: a = 0x4B000000 | (q ^ 0x80), b = -(2^23 + 128), c1 = d; Q4_K: a = F32(q), b =
//     F32(m), c1 = d1, c2 = dmin, neg).

module tsr_vu_inj
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
#(
  parameter int unsigned FRONT_LANES = 16,
  parameter int unsigned LMAW      = 9,
  parameter bit          HAS_DQROW = 1'b0,
  parameter bit          HAS_GDN   = 1'b0,
  parameter int unsigned LANE_X    = 0       // cycles the lane pipes add to every exit (tsr_vu_lanes: 2 x stages)
) (
  input  logic                           clk,
  input  logic                           rst_n,
  // slots and SPM operands (tsr_vu_rd)
  input  logic                           q_valid,
  input  tsr_vu_slot_t                   q_slot,
  output logic                           q_pop,
  input  logic [1:0][1:0]                f_valid,
  input  logic [1:0][1:0][TSR_LINE_BITS-1:0] f_data,
  output logic [1:0][1:0]                f_pop,
  input  tsr_vu_opk_t [1:0]              opk,
  // local memory read ports (1-cycle read; the output holds while not reading)
  output logic                           lm_re_a,
  output logic [LMAW-1:0]                lm_ra_a,
  input  logic [TSR_LINE_BITS-1:0]       lm_rd_a,
  output logic                           lm_re_b,
  output logic [LMAW-1:0]                lm_ra_b,
  input  logic [TSR_LINE_BITS-1:0]       lm_rd_b,
  // ROPE angle buffer (tsr_vu_angle): [bank] = {c, s} at (half, entry), combinational
  input  logic [1:0]                     ang_ready,
  output logic                           ang_rel,
  output logic                           ang_rel_h,
  output logic                           ang_rh,
  output logic [4:0]                     ang_re,
  input  logic [15:0][63:0]              ang_cs,
  output logic                           tbl_we,
  output logic [4:0]                     tbl_line,
  output logic                           pos_we,
  output logic [15:0]                    pos_idx,
  output logic [TSR_LINE_BITS-1:0]       pa_data,     // TBL / POS / TOPK line
  // reductions (tsr_vu_red)
  input  logic [4:0]                     ring_free,
  output logic [3:0]                     ring_take,   // entries allocated this cycle
  output logic                           row_v,
  output tsr_vu_row_t                    row_tok,
  input  logic                           red_idle,
  // scalar path (SOP SQRT / RSQRT / DIV)
  output logic                           sopx_v,
  output logic                           sopx_tag,
  input  logic                           sopx_idle,
  // TOPK
  input  logic                           topk_ready,
  output logic                           topk_v,
  output tsr_vu_slot_t                   topk_slot,
  // GDN engine (registered slot: topk_slot)
  input  logic                           gdn_cfg_ok,
  input  logic                           gdn_vec_ok,
  input  logic                           gdn_row_ok,
  output logic                           gdn_v,
  // lanes (registered)
  output logic [15:0]                    l_v,
  output tsr_vu_uop_e                    l_uop,
  output logic [15:0][31:0]              l_a,
  output logic [15:0][31:0]              l_b,
  output logic [(FRONT_LANES == 8 ? 1 : 16)-1:0][31:0] l_c1,
  output logic [(FRONT_LANES == 8 ? 1 : 16)-1:0][31:0] l_c2,
  output tsr_vu_bank_coeff_t            l_coeff,
  output logic [15:0]                    l_neg,
  output logic [15:0]                    l_lv,
  output logic                           l_fresh,
  output logic [3:0]                     l_pa,
  output logic                           l_ctx,
  // exit descriptor of the sub-slot on the lane outputs
  output logic                           d_v,
  output logic                           d_back,
  output tsr_vu_desc_t                   d_desc,
  output logic                           busy
);

  // -------------------------------------------------------------------------------------------
  // D: decide
  // -------------------------------------------------------------------------------------------
  localparam int unsigned RF = 16 / FRONT_LANES;
  tsr_vu_slot_t s;
  assign s = q_slot;
  wire na = (s.asrc == SRC_SPM), nb = (s.bsrc == SRC_SPM);
  wire pa = s.aline[0], pb = s.bline[0];
  wire lane_slot = (s.kind == SK_LANE);
  wire dq_line   = lane_slot && s.dq[5];
  wire dq_meta   = (s.kind == SK_CL2) && s.dq[5];
  logic [5:0] dq_prod;
  wire has_exit  = lane_slot && !s.red || (lane_slot && s.uop == U_AMASK);
  wire [7:0] lat = s.exitb ? 8'(TSR_VU_LB + LANE_X) : 8'(TSR_VU_LF + LANE_X);
  wire [3:0] np  = 4'((5'(s.r_ng) + 5'd2) >> 1);         // min(L, 8) partial entries

  logic [7:0] ex_rem_q;
  logic       i_v;                                       // I holds a slot
  logic [2:0] k_q;                                       // its current sub-slot
  tsr_vu_slot_t i_s;
  wire        i_gdn  = HAS_GDN && i_s.kind == SK_GDN;
  wire        i_last = i_v && ((i_s.kind != SK_LANE && !i_gdn) || k_q == i_s.nsub - 3'd1);
  wire        i_free = !i_v || i_last;

  // tsr_vu_red's idle, registered: a scalar-path op starts only after every earlier op is done (the
  // hazard barrier of tsr_vu), so this check is a guard and a cycle's lag is harmless
  logic red_idle_q;
  always_ff @(posedge clk) red_idle_q <= red_idle;

  logic ops_in;
  always_comb begin
    ops_in = (!na || f_valid[pa][0]) && (!nb || f_valid[pb][1]);
    case (s.kind)
      SK_LANE: begin
        if (has_exit && lat < ex_rem_q) ops_in = 1'b0;
        if (s.red && s.r_first && 5'(np) > ring_free) ops_in = 1'b0;
        if (s.rope != 2'd0 && !ang_ready[s.ang_h]) ops_in = 1'b0;
        if (dq_line && dq_prod <= 6'(s.dqb)) ops_in = 1'b0;
      end
      SK_TOPK: if (!topk_ready) ops_in = 1'b0;
      SK_GDN: if (HAS_GDN) begin
        case (s.gsub)
          GS_CFG:  if (!gdn_cfg_ok) ops_in = 1'b0;
          GS_ROW:  if (s.r_first && !gdn_row_ok) ops_in = 1'b0;
          default: if (!gdn_vec_ok) ops_in = 1'b0;
        endcase
      end
      SK_SOPX: if (!red_idle_q || !sopx_idle || ex_rem_q != 8'd0) ops_in = 1'b0;
      default: ;
    endcase
  end
  wire fire = q_valid && i_free && ops_in;

  assign q_pop = fire;
  always_comb begin
    f_pop = '0;
    if (fire && na) f_pop[pa][0] = 1'b1;
    if (fire && nb) f_pop[pb][1] = 1'b1;
  end
  assign lm_re_a = fire && (s.asrc == SRC_LM);
  assign lm_ra_a = LMAW'(s.aline);
  assign lm_re_b = fire && (s.bsrc == SRC_LM);
  assign lm_ra_b = LMAW'(s.bline);

  // partial ring allocation
  logic [3:0] ring_wp_q, cur_base_q, i_base_q;
  assign ring_take = (fire && lane_slot && s.red && s.r_first) ? np : 4'd0;

  logic [TSR_LINE_BITS-1:0] a_q, b_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      i_v        <= 1'b0;
      k_q        <= '0;
      ex_rem_q   <= '0;
      ring_wp_q  <= '0;
      cur_base_q <= '0;
    end else begin
      if (fire) begin
        i_v <= 1'b1;
        k_q <= '0;
      end else if (i_last) begin
        i_v <= 1'b0;
      end else if (i_v) begin
        k_q <= k_q + 3'd1;
      end
      if (fire && has_exit) ex_rem_q <= lat + 8'(s.nsub) - 8'd1;
      else if (ex_rem_q != 8'd0) ex_rem_q <= ex_rem_q - 8'd1;
      if (fire && lane_slot && s.red && s.r_first) begin
        ring_wp_q  <= ring_wp_q + np;
        cur_base_q <= ring_wp_q;
      end
    end
  end
  always_ff @(posedge clk) begin
    if (fire) begin
      i_s      <= s;
      a_q      <= f_data[pa][0];
      b_q      <= f_data[pb][1];
      i_base_q <= (s.red && s.r_first) ? ring_wp_q : cur_base_q;
    end
  end

  // -------------------------------------------------------------------------------------------
  // I: operands of sub-slot k_q
  // -------------------------------------------------------------------------------------------
  wire [TSR_LINE_BITS-1:0] A = (i_s.asrc == SRC_LM) ? lm_rd_a : a_q;
  wire [TSR_LINE_BITS-1:0] B = (i_s.bsrc == SRC_LM) ? lm_rd_b : b_q;
  logic [TSR_LINE_BITS-1:0] cl1_q, cl2_q;               // c-line registers (VU §4.3)

  function automatic logic [31:0] wd(input logic [TSR_LINE_BITS-1:0] x, input int unsigned i);
    return x[32*i +: 32];
  endfunction

  assign ang_rh = i_s.ang_h;
  assign ang_re = i_s.ang_e;

  // DQROW constant table (tsr_vu_dq)
  logic [31:0] dq_c1, dq_b, dq_c2;
  if (HAS_DQROW) begin : g_dq
    tsr_vu_dq u_dq (
      .clk, .rst_n, .start(fire && dq_meta), .start_fmt(s.dq[4]), .start_meta(f_data[pa][0]),
      .prod(dq_prod), .rd_e(i_s.dqb), .rd_c1(dq_c1), .rd_b(dq_b), .rd_c2(dq_c2)
    );
  end else begin : g_nodq
    assign dq_prod = 6'd32;
    assign dq_c1   = '0;
    assign dq_b    = '0;
    assign dq_c2   = '0;
    wire unused_dq = &{1'b0, dq_meta};
  end

  logic [15:0][31:0] o_a, o_b, o_c1, o_c2;
  logic [15:0]       o_neg, o_v, o_lv;
  logic o_ctx;
  tsr_vu_bank_coeff_t coeff;
  wire load_c1 = i_v && i_s.kind == SK_CL1;
  wire load_c2 = i_v && i_s.kind == SK_CL2 && !i_s.dq[5];
  always_comb begin
    coeff = '0;
    if (FRONT_LANES == 8) begin
      coeff.load = {load_c2, load_c1};
      coeff.elem = {i_s.c2sel == C_CLE, i_s.c1sel == C_CLE};
      coeff.rope = i_s.rope;
      coeff.upper_quad = i_s.rope != 2'd1 && i_s.nsub == 3'd4 && k_q[0];
      coeff.angle_addr = {i_s.ang_h, i_s.ang_e};
      coeff.angle_hi = i_s.ang_hi;
      if (HAS_DQROW && i_s.dq[5]) begin
        coeff.elem = '0;
        coeff.rope = '0;
      end
    end
  end
  always_comb begin
    int unsigned w, e;
    logic [31:0] k1, k2, ac, as, c1v, c2v;
    logic [7:0]  qb;
    qb = '0;
    k1 = opk[i_s.tag].k1;
    k2 = opk[i_s.tag].k2;
    w  = (i_s.rope == 2'd1) ? FRONT_LANES : (i_s.nsub == 3'd4) ? 4 : (i_s.nsub == 3'd2) ? 8 : 16;
    o_ctx = (FRONT_LANES == 8) && (((i_s.rope == 2'd1 ? 32'(k_q) % RF : 32'(k_q)) * w) >= 8);
    for (int l = 0; l < 16; l++) begin
      e = ((i_s.rope == 2'd1 ? 32'(k_q) % RF : 32'(k_q)) * w + l) % 16;
      o_v[l]  = i_v && (i_s.kind == SK_LANE) && (l < w);
      o_lv[l] = (32'(e) < 32'(i_s.tail));
      // a, b
      case (i_s.asrc)
        SRC_K:   o_a[l] = k1;
        default: o_a[l] = wd(A, e);
      endcase
      case (i_s.bsrc)
        SRC_K:   o_b[l] = k2;
        default: o_b[l] = wd(B, e);
      endcase
      // c
      case (i_s.c1sel)
        C_K:     c1v = k1;
        C_CLW:   c1v = wd(cl1_q, 32'(i_s.cword));
        C_CLE:   c1v = (FRONT_LANES == 8) ? 32'd0 : wd(cl1_q, e);
        default: c1v = 32'd0;
      endcase
      case (i_s.c2sel)
        C_K:     c2v = k2;
        C_CLW:   c2v = wd(cl2_q, 32'(i_s.cword));
        C_CLE:   c2v = (FRONT_LANES == 8) ? 32'd0 : wd(cl2_q, e);
        default: c2v = 32'd0;
      endcase
      o_c1[l]  = c1v;
      o_c2[l]  = c2v;
      o_neg[l] = 1'b0;
      // ROPE rotation operands
      if (i_s.rope == 2'd1) begin                       // NEOX pair: x0 = A, x1 = B, angle bank l
        ac = (FRONT_LANES == 8) ? 32'd0 : ang_cs[e][63:32];
        as = (FRONT_LANES == 8) ? 32'd0 : ang_cs[e][31:0];
        o_c1[l]  = (32'(k_q) / RF == 0) ? ac : as;
        o_c2[l]  = (32'(k_q) / RF == 0) ? as : ac;
        o_neg[l] = (32'(k_q) / RF == 0);
      end else if (i_s.rope == 2'd2) begin              // NORM: elements 2p, 2p + 1, angle 8·hi + p
        ac = (FRONT_LANES == 8) ? 32'd0 : ang_cs[(i_s.ang_hi ? 8 : 0) + e / 2][63:32];
        as = (FRONT_LANES == 8) ? 32'd0 : ang_cs[(i_s.ang_hi ? 8 : 0) + e / 2][31:0];
        o_a[l]   = wd(A, e & 14);
        o_b[l]   = wd(A, e | 1);
        o_c1[l]  = (e % 2 == 0) ? ac : as;
        o_c2[l]  = (e % 2 == 0) ? as : ac;
        o_neg[l] = (e % 2 == 0);
      end
      // DQROW: byte 16p + l of the quant line
      if (HAS_DQROW && i_s.dq[5]) begin
        qb = A[8 * (16 * 32'(i_s.dq[3:2]) + e) +: 8];
        if (!i_s.dq[4]) begin                           // Q8_0: FM(FA(2^23 + q + 128, -(2^23 + 128)), d)
          o_a[l]   = {16'h4B00, 8'h00, qb ^ 8'h80};
          o_b[l]   = 32'hCB00_0080;
          o_c1[l]  = dq_c1;
          o_c2[l]  = 32'd0;
          o_neg[l] = 1'b0;
        end else begin                                  // Q4_K: FS(FM(q, d1), FM(m, dmin))
          o_a[l]   = tsr_vu_u4_to_f32(i_s.dq[1] ? qb[7:4] : qb[3:0]);
          o_b[l]   = dq_b;
          o_c1[l]  = dq_c1;
          o_c2[l]  = dq_c2;
          o_neg[l] = 1'b1;
        end
      end
      if (FRONT_LANES == 8 && (load_c1 || load_c2)) begin
        // A/B carry the two logical contexts in the existing non-lane slot.
        o_a[l] = wd(A, l % 8);
        o_b[l] = wd(A, (l % 8) + 8);
      end
    end
  end

  if (FRONT_LANES == 8) begin : g_scalar_coeff
    wire unused_coeff = &{1'b0, o_c1[15:1], o_c2[15:1]};
  end

  // registered outputs
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      l_v  <= '0;
      l_coeff <= '0;
      d_v  <= 1'b0;
    end else begin
      l_v <= o_v;
      l_coeff <= coeff;
      d_v <= i_v && (i_s.kind == SK_LANE) && (!i_s.red || i_s.uop == U_AMASK);
    end
    l_uop   <= i_s.uop;
    l_a     <= o_a;
    l_b     <= o_b;
    l_c1    <= o_c1[$bits(l_c1)/32-1:0];
    l_c2    <= o_c2[$bits(l_c2)/32-1:0];
    l_neg   <= o_neg;
    l_lv    <= o_lv;
    l_fresh <= i_s.r_fresh;
    l_ctx   <= o_ctx;
    l_pa    <= i_base_q + 4'(i_s.r_li);
    d_back  <= i_s.exitb;
    d_desc.tag    <= i_s.tag;
    d_desc.dst    <= i_s.dst;
    d_desc.line   <= (i_s.rope == 2'd1 && 32'(k_q) / RF != 0) ? i_s.d1 : i_s.d0;
    d_desc.part   <= (i_s.rope == 2'd1) ? 2'(32'(k_q) % RF) : k_q[1:0];
    d_desc.nparts <= (i_s.rope == 2'd1) ? 3'(RF) : i_s.nsub;
    d_desc.tail   <= i_s.tail;
    d_desc.dword  <= i_s.dword;
    d_desc.sts    <= i_s.sts;
    d_desc.keep   <= i_s.keep;
  end

  // pseudo-slot actions and row tokens (I stage, registered)
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      tbl_we  <= 1'b0;
      pos_we  <= 1'b0;
      row_v   <= 1'b0;
      ang_rel <= 1'b0;
      sopx_v  <= 1'b0;
      topk_v  <= 1'b0;
      gdn_v   <= 1'b0;
    end else begin
      tbl_we  <= i_v && i_s.kind == SK_TBL;
      pos_we  <= i_v && i_s.kind == SK_POS;
      row_v   <= i_v && ((i_s.kind == SK_LANE && i_s.red && i_s.r_last && i_last) || i_s.kind == SK_REDE);
      ang_rel <= i_v && i_s.kind == SK_LANE && i_s.tok_last && i_last;   // the token's angles are read
      sopx_v  <= i_v && i_s.kind == SK_SOPX;
      topk_v  <= i_v && i_s.kind == SK_TOPK;
      gdn_v   <= i_v && i_gdn && k_q == 3'd0;                             // once per slot
    end
    tbl_line  <= i_s.d0[4:0];
    pos_idx   <= 16'(i_s.d0);
    pa_data   <= A;
    ang_rel_h <= i_s.ang_h;
    sopx_tag  <= i_s.tag;
    topk_slot <= i_s;
    row_tok.tag   <= i_s.tag;
    row_tok.ring  <= i_base_q;
    row_tok.np    <= 4'((5'(i_s.r_ng) + 5'd2) >> 1) - 4'd1;
    row_tok.ng    <= i_s.r_ng;
    row_tok.empty <= (i_s.kind == SK_REDE);
    row_tok.row   <= i_s.r_row;
    row_tok.lrow  <= i_s.r_lrow;
    if (load_c1) cl1_q <= A;
    if (load_c2) cl2_q <= A;
  end

  assign busy = i_v || q_valid || (ex_rem_q != 8'd0);

  if (!HAS_GDN) begin : g_nogdn
    wire unused_gdn = &{1'b0, gdn_cfg_ok, gdn_vec_ok, gdn_row_ok, s.gsub, s.r_first};
  end
  wire unused_ok = &{1'b0, i_s.d1[0], s.d0, s.cword, i_s.dq[0]};

endmodule
