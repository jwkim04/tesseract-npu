// tsr_vu_lanes.sv — 8 or 16 physical VU lanes in two banks behind registered pipes.
// The banks can sit in opposite outer SLRs, away from the SPM-facing part of the unit.
//
// Spec: docs/design/02-vector-unit.md "VU" §7.10 placement, remedy 3 of CC §14.1; floorplan C'' of
// hw/u200/k4_floorplan.tcl.
// Contract: with PIPE = 0 the inputs and selected exit are wired straight to the ports.
// With PIPE = p >= 1, the injector's lane inputs (l_*), the partial drain address (dr_addr) and the reset
// pass p register stages on the way in, and the selected lane exit and drain read data dr_data p
// stages on the way out: every lane result leaves 2p cycles later than the lane's own latency
// (TSR_VU_LF / TSR_VU_LB). FRONT16 returns raw partials after 2p cycles. FRONT8
// returns L1 results after 2p + 1 + FAL cycles, replacing the original central
// staging/L1 work without changing any architectural result cycle.
// The consumers allow for it (tsr_vu_inj and tsr_vu_wr: LANE_X = 2p, tsr_vu_red: DRL = 2p); the lanes
// have no back-pressure, so no handshake needs to cross.
// Each bank carries its own reset, controls and semantic back_valid token. Its u_share selects f_y/b_y before
// the return pipe. The writer keeps its descriptor timing, and the reduction drain stays independent.
// Why the reset is piped like the inputs: a lane leaves reset p cycles after the injector does, which is
// exactly when the first element of a later op can arrive, so the reset stays in step with the data.
// Placement (hw/u200/k4_floorplan.tcl, PIPE >= 2): the first stage of the input pipes and the last stage
// of the output pipes are pinned beside the injector / writer. Each bank keeps its other stages
// beside its physical lanes (four per bank for FRONT8, eight for FRONT16).
// Each bank has equal back/front-only capacity. FRONT8 retains two independent
// partial contexts per physical lane. Its first reduction level stays beside the
// lanes: select/init plus one staging register and two unchanged FP nodes per bank.
// Only their two 32-bit results enter the return pipe. The reduction unit consumes
// them at the original L1 result cycle, so row timing and ordinary exits do not change.
// Each crossing is one register-to-register hop. Kept stage registers prevent
// reset/control merging across banks and shreg_extract = "no" prevents SRLs.

module tsr_vu_lanes
  import tsr_vu_pkg::*;
#(
  parameter int unsigned FRONT_LANES = 16,
  parameter int unsigned EXP_LANES  = 16,
  parameter int unsigned BACK_LANES = 8,
  parameter int unsigned PIPE       = 0,         // register stages on each side of the lanes
  parameter bit          HAS_LOCAL  = 1'b0
) (
  input  logic              clk,
  input  logic              rst_n,
  // element and reduction controls of the injector (registered there)
  input  logic [15:0]       l_v,
  input  tsr_vu_uop_e       l_uop,
  input  logic [15:0][31:0] l_a,
  input  logic [15:0][31:0] l_b,
  input  logic [(FRONT_LANES == 8 ? 1 : 16)-1:0][31:0] l_c1,
  input  logic [(FRONT_LANES == 8 ? 1 : 16)-1:0][31:0] l_c2,
  input  tsr_vu_bank_coeff_t l_coeff,
  input  tsr_vu_bank_angle_t [1:0] bank_angle,
  input  logic [15:0]       l_neg,
  input  logic [15:0]       l_lv,
  input  logic              l_fresh,
  input  logic [3:0]        l_pa,
  input  logic              l_ctx,
  input  logic              back_valid,   // injector d_v && d_back, aligned with l_*
  // partial drain (tsr_vu_red): address in, data PIPE + PIPE cycles later
  input  logic [3:0]        dr_addr,
  input  logic              dr_sel, dr_empty, dr_mx,
  output logic [15:0][31:0] dr_data,
  output logic [3:0][31:0] dr_l1,
  // exactly one scheduled front/back exit; the reduction drain stays separate
  output logic [15:0][31:0] lane_y,
  input  tsr_vu_local_req_t [1:0] local_req,
  output tsr_vu_local_rsp_t [1:0] local_rsp
);

  localparam int unsigned NL = FRONT_LANES/2;
  localparam int unsigned Q = FRONT_LANES/4;
  localparam int unsigned CTX = 16/FRONT_LANES;
  if (FRONT_LANES == 8) begin : g_half_unused
    assign lane_y[15:8] = '0;
  end

  // Independent bank modules retain their transport and selector hierarchy.
  for (genvar bank = 0; bank < 2; bank++) begin : g_bank
    logic [7:0][31:0] bank_dr;
    logic [NL-1:0][31:0] bank_y;
    logic [1:0][31:0] bank_l1;
    localparam int CW = FRONT_LANES == 8 ? 1 : NL;
    logic [NL-1:0][31:0] a, b;
    logic [CW-1:0][31:0] c1, c2;
    logic [NL-1:0] v, neg, lv;
    if (FRONT_LANES == 8) begin : g_scalar
      assign c1[0] = l_c1[0];
      assign c2[0] = l_c2[0];
    end
    (* keep_hierarchy = "yes" *) tsr_vu_lane_bank #(
      .BANK(bank), .FRONT_LANES(FRONT_LANES), .EXP_LANES(EXP_LANES), .BACK_LANES(BACK_LANES), .PIPE(PIPE), .HAS_LOCAL(HAS_LOCAL)
    ) u_bank (
      .clk, .rst_n, .l_v(v), .l_uop, .l_a(a), .l_b(b), .l_c1(c1), .l_c2(c2),
      .l_coeff, .angle(bank_angle[bank]), .l_neg(neg), .l_lv(lv), .l_fresh, .l_pa, .l_ctx, .back_valid, .dr_addr, .dr_sel, .dr_empty, .dr_mx,
      .dr_data(bank_dr), .dr_l1(bank_l1), .lane_y(bank_y),
      .local_req(local_req[bank]), .local_rsp(local_rsp[bank])
    );
    for (genvar pair = 0; pair < 2; pair++) begin : g_pair_map
      assign dr_l1[bank+2*pair] = bank_l1[pair];
    end
    for (genvar j = 0; j < NL; j++) begin : g_map
      localparam int L = (j < Q) ? Q*bank+j : FRONT_LANES/2+Q*bank+j-Q;
      assign v[j] = l_v[L];
      assign a[j] = l_a[L];
      assign b[j] = l_b[L];
      assign neg[j] = l_neg[L];
      assign lv[j] = l_lv[L];
      if (FRONT_LANES != 8) begin : g_vector
        assign c1[j] = l_c1[L];
        assign c2[j] = l_c2[L];
      end
      assign dr_data[L] = bank_dr[j];
      assign lane_y[L] = bank_y[j];
      if (CTX == 2) begin : g_context_hi
        assign dr_data[L+8] = bank_dr[j+NL];
      end
    end
  end
endmodule

// Kept implementation bank shares the wrapper's source and build lists.
/* verilator lint_off DECLFILENAME */
module tsr_vu_lane_bank
  import tsr_vu_pkg::*;
#(
  parameter int unsigned BANK = 0,
  parameter int unsigned FRONT_LANES = 16,
  parameter int unsigned EXP_LANES  = 16,
  parameter int unsigned BACK_LANES = 8,
  parameter int unsigned PIPE       = 0,
  parameter bit          HAS_LOCAL  = 1'b0,         // register stages on each side of the lanes, >= 0
  localparam int unsigned NL = FRONT_LANES/2,
  localparam int unsigned CW = FRONT_LANES == 8 ? 1 : NL
) (
  input  logic              clk,
  input  logic              rst_n,
  // element and reduction controls of the injector (registered there)
  input  logic [NL-1:0]     l_v,
  input  tsr_vu_uop_e       l_uop,
  input  logic [NL-1:0][31:0] l_a,
  input  logic [NL-1:0][31:0] l_b,
  input  logic [CW-1:0][31:0] l_c1,
  input  logic [CW-1:0][31:0] l_c2,
  input  tsr_vu_bank_coeff_t l_coeff,
  input  tsr_vu_bank_angle_t angle,
  input  logic [NL-1:0]     l_neg,
  input  logic [NL-1:0]     l_lv,
  input  logic              l_fresh,
  input  logic [3:0]        l_pa,
  input  logic              l_ctx,
  input  logic              back_valid,   // injector d_v && d_back, aligned with l_*
  // partial drain (tsr_vu_red): address in, data PIPE + PIPE cycles later
  input  logic [3:0]        dr_addr,
  input  logic              dr_sel, dr_empty, dr_mx,
  output logic [7:0][31:0] dr_data,
  output logic [1:0][31:0] dr_l1,
  // exactly one scheduled front/back exit; the reduction drain stays separate
  output logic [FRONT_LANES/2-1:0][31:0] lane_y,
  input  tsr_vu_local_req_t local_req,
  output tsr_vu_local_rsp_t local_rsp
);

  localparam int unsigned Q = FRONT_LANES/4;
  localparam int unsigned CTX = 16/FRONT_LANES;
  localparam int unsigned PAW = 4+$clog2(CTX);
  localparam bit LOCAL_L1 = (FRONT_LANES == 8);
  localparam int unsigned DR_WORDS = LOCAL_L1 ? 2 : 8;

  // everything that goes into the lane array, and everything that comes out of it
  typedef struct packed {
    logic [NL-1:0]       v;
    logic [5:0]        uop;
    logic [NL-1:0][31:0] a, b;
    logic [CW-1:0][31:0] c1, c2;
    logic [NL-1:0]       neg, lv;
    logic              fresh;
    logic [PAW-1:0]    pa;
    logic [3:0]        dr_addr;
    logic              back_valid;
  } lin_base_t;
  typedef struct packed {
    logic [DR_WORDS-1:0][31:0] dr_data;
    logic [NL-1:0][31:0] lane_y;
  } lout_t;

  localparam int unsigned IW = $bits(lin_base_t) + (LOCAL_L1 ? 3 + $bits(tsr_vu_bank_coeff_t) : 0);
  localparam int unsigned OW = $bits(lout_t);

  logic [IW-1:0] li, lo_in;
  lin_base_t li_base, lo_base;
  logic [2:0] dr_ctrl;
  logic [NL-1:0][31:0] partial_lo, partial_hi;
  // Input tokens are inserted above back_valid only for the reduced transport.
  lout_t lq, lp;                     // lane array output: the lanes' / after the output pipe
  logic  lane_rst_n;

  for (genvar j = 0; j < NL; j++) begin : g_map
    if (!LOCAL_L1) begin : g_raw_drain
      assign dr_data[j] = lp.dr_data[j];
    end
    assign lane_y[j] = lp.lane_y[j];

  end
  assign li_base.v = l_v;
  assign li_base.a = l_a;
  assign li_base.b = l_b;
  assign li_base.c1 = l_c1;
  assign li_base.c2 = l_c2;
  assign li_base.neg = l_neg;
  assign li_base.lv = l_lv;
  logic [NL-1:0][31:0] lane_c1, lane_c2;
  assign li_base.uop = 6'(l_uop);
  assign li_base.fresh = l_fresh;
  assign li_base.pa = PAW'({l_ctx, l_pa});
  assign li_base.dr_addr = dr_addr;
  assign li_base.back_valid = back_valid;

  if (LOCAL_L1) begin : g_reduced_transport
    tsr_vu_bank_coeff_t coeff;
    assign coeff = tsr_vu_bank_coeff_t'(lo_in[IW-1 -: $bits(coeff)]);
    assign li = {l_coeff, li_base[$bits(lin_base_t)-1:1], dr_sel, dr_empty, dr_mx, li_base[0]};
    assign lo_base = lin_base_t'({lo_in[$bits(lin_base_t)+2:4], lo_in[0]});
    assign dr_ctrl = lo_in[3:1];
    assign dr_data = '0;
    assign dr_l1 = lp.dr_data;

    // A coefficient-load slot carries both contexts alongside the ordinary lane stream.
    logic [1:0][NL-1:0][31:0] cl1_q, cl2_q;
    always_ff @(posedge clk) begin
      if (lane_rst_n && coeff.load[0]) cl1_q <= {lo_base.b, lo_base.a};
      if (lane_rst_n && coeff.load[1]) cl2_q <= {lo_base.b, lo_base.a};
    end

    // The producer sends each angle only to the bank that consumes it.
    logic [7:0][63:0] cs;
    for (genvar i = 0; i < 8; i++) begin : g_angle
      logic [63:0] mem [64];
      always_ff @(posedge clk) begin
        if (angle.valid && angle.addr[2:0] == 3'(i)) mem[angle.addr[8:3]] <= angle.data;
      end
      assign cs[i] = mem[coeff.angle_addr];
    end
    for (genvar j = 0; j < NL; j++) begin : g_coeff
      wire quad = 1'(j >> 1) ^ coeff.upper_quad;
      wire [2:0] elem_idx = {lo_base.pa[4], quad, 1'(j)};
      // Drop the bank-owner bit from the global angle index. NORM shares
      // each angle between the two adjacent lanes of a pair.
      wire [2:0] angle_idx = coeff.rope == 2'd1 ? elem_idx :
                                           {coeff.angle_hi, lo_base.pa[4], quad};
      wire [63:0] pair_cs = cs[angle_idx];
      assign lane_c1[j] = coeff.rope != 0 ? (lo_base.neg[j] ? pair_cs[63:32] : pair_cs[31:0]) :
                         coeff.elem[0] ? cl1_q[elem_idx[2]][elem_idx[1:0]] : lo_base.c1[0];
      assign lane_c2[j] = coeff.rope != 0 ? (lo_base.neg[j] ? pair_cs[31:0] : pair_cs[63:32]) :
                         coeff.elem[1] ? cl2_q[elem_idx[2]][elem_idx[1:0]] : lo_base.c2[0];
    end

  end else begin : g_raw_transport
    assign li = li_base;
    assign lo_base = lin_base_t'(lo_in);
    assign dr_ctrl = '0;
    assign dr_l1 = '0;
    assign lane_c1 = lo_base.c1;
    assign lane_c2 = lo_base.c2;
    wire unused_control = &{1'b0, dr_sel, dr_empty, dr_mx, partial_hi, dr_ctrl, l_coeff, angle};
  end

  if (LOCAL_L1) begin : g_reduce
    (* keep_hierarchy = "yes" *) tsr_vu_reduce_bank u_reduce (
      .clk, .rst_n(lane_rst_n), .sel(dr_ctrl[2]), .empty(dr_ctrl[1]), .mx(dr_ctrl[0]),
      .partial_lo, .partial_hi, .y(lq.dr_data)
    );
  end

  if (PIPE == 0) begin : g_wire
    assign lo_in      = li;
    assign lane_rst_n = rst_n;
    assign lp         = lq;
  end else begin : g_pipe
    // input side, p stages; the reset rides along (its flops are not reset themselves: during a reset
    // pulse of >= p cycles the chain fills with zeros, and a flop starts at 0 on the device)
    (* shreg_extract = "no", keep = "true", dont_touch = "true" *) logic [PIPE-1:0][IW-1:0] ip_q;
    (* shreg_extract = "no", keep = "true", dont_touch = "true" *) logic [PIPE-1:0]         rp_q;
    always_ff @(posedge clk) begin
      ip_q[0] <= li;
      rp_q[0] <= rst_n;
      for (int unsigned i = 1; i < PIPE; i++) begin
        ip_q[i] <= ip_q[i-1];
        rp_q[i] <= rp_q[i-1];
      end
    end
    assign lo_in      = ip_q[PIPE-1];
    assign lane_rst_n = rp_q[PIPE-1];
    // output side, p stages
    (* shreg_extract = "no", keep = "true", dont_touch = "true" *) logic [PIPE-1:0][OW-1:0] op_q;
    always_ff @(posedge clk) begin
      op_q[0] <= lq;
      for (int unsigned i = 1; i < PIPE; i++) op_q[i] <= op_q[i-1];
    end
    assign lp = lout_t'(op_q[PIPE-1]);
  end

  logic [NL-1:0][31:0] front_y, back_y;
  // Selection stays beside the lanes, before the return stages. The delayed
  // semantic token includes S0, just as the writer's descriptor delay does.
  (* keep_hierarchy = "yes" *) tsr_vu_lane_share #(.LANES(NL)) u_share (
    .clk, .rst_n(lane_rst_n), .back_valid(lo_base.back_valid),
    .front_y, .back_y, .y(lq.lane_y)
  );

  logic local_in;
  logic [NL-1:0][31:0] in_a, in_b, in_c1, in_c2;
  logic [NL-1:0] in_v;
  tsr_vu_uop_e in_uop;
  if (HAS_LOCAL) begin : g_local
    if (NL != 4 || EXP_LANES != 8 || BACK_LANES != 4) begin : g_chk
      $fatal(1, "tsr_vu_lane_bank: local mode needs four EXP lanes");
    end
    // Local requests bypass the crossing stages and reserve the front exits.
    // An S0 register precedes LF, so track validity through the same delay.
    logic [TSR_VU_LF:0][3:0] valid_q;
    logic [TSR_VU_LB:0][1:0] back_q;
    always_ff @(posedge clk) begin
      if (!lane_rst_n) begin
        valid_q <= '0;
        back_q <= '0;
      end else begin
        valid_q <= {valid_q[TSR_VU_LF-1:0], local_req.tanh_op ? 4'd0 : local_req.valid};
        back_q <= {back_q[TSR_VU_LB-1:0], local_req.tanh_op ? local_req.valid[1:0] : 2'd0};
      end
    end
    assign local_rsp.valid = valid_q[TSR_VU_LF];
    assign local_rsp.y = front_y;
    assign local_rsp.back_valid = back_q[TSR_VU_LB];
    assign local_rsp.back_y = back_y[1:0];
    assign local_in = |local_req.valid;
    assign in_a = local_in ? local_req.a : lo_base.a;
    assign in_b = local_in ? '0 : lo_base.b;
    assign in_c1 = local_in ? {4{local_req.scale_only ? local_req.scale : local_req.max_value}} : lane_c1;
    assign in_c2 = local_in ? {4{local_req.scale}} : lane_c2;
    assign in_v = local_in ? local_req.valid : lo_base.v;
    assign in_uop = local_in ? (local_req.tanh_op ? U_TANH : local_req.scale_only ? U_MULC : U_ASMAX) :
                              tsr_vu_uop_e'(lo_base.uop);
`ifndef SYNTHESIS
    logic checks_on_q;
    initial checks_on_q = 1'b0;
    always_ff @(posedge clk) begin
      if (!lane_rst_n) checks_on_q <= 1'b1;
      else if (checks_on_q && local_in) assert (lo_base.v == '0);
    end
`endif
  end else begin : g_no_local
    assign local_in = 1'b0;
    assign local_rsp = '0;
    assign in_a = lo_base.a;
    assign in_b = lo_base.b;
    assign in_c1 = lane_c1;
    assign in_c2 = lane_c2;
    assign in_v = lo_base.v;
    assign in_uop = tsr_vu_uop_e'(lo_base.uop);
    wire unused_local = &{1'b0, local_req};
  end

  for (genvar l = 0; l < NL; l++) begin : g_lane
    localparam int L = (l < Q) ? Q*BANK+l : FRONT_LANES/2+Q*BANK+l-Q;
    if (!LOCAL_L1) begin : g_raw_drain
      assign lq.dr_data[l] = partial_lo[l];
    end
    tsr_vu_lane #(.HAS_EXP(L < EXP_LANES), .HAS_BACK(L < BACK_LANES), .PART_CTX(CTX)) u_lane (
      .clk, .rst_n(lane_rst_n), .in_v(in_v[l]), .in_uop(in_uop), .in_a(in_a[l]),
      .in_b(in_b[l]), .in_c1(in_c1[l]), .in_c2(in_c2[l]), .in_neg(lo_base.neg[l] && !local_in), .acc_lv(lo_base.lv[l] && !local_in),
      .acc_fresh(lo_base.fresh), .acc_pa(lo_base.pa), .dr_addr(lo_base.dr_addr), .dr_data(partial_lo[l]), .dr_data_hi(partial_hi[l]),
      .f_y(front_y[l]), .b_y(back_y[l])
    );
  end

endmodule
/* verilator lint_on DECLFILENAME */

// Two unchanged first-level subtree nodes, selected/staged beside their lanes.
/* verilator lint_off DECLFILENAME */
module tsr_vu_reduce_bank
  import tsr_vu_pkg::*;
(
  input logic clk, rst_n,
  input logic sel, empty, mx,
  input logic [3:0][31:0] partial_lo, partial_hi,
  output logic [1:0][31:0] y
);
  logic [3:0][31:0] g_q;
  logic g_mx_q;
  always_ff @(posedge clk) begin
    g_mx_q <= mx;
    for (int i = 0; i < 4; i++) begin
      if (empty) g_q[i] <= mx ? TSR_VU_NINF : 32'd0;
      else g_q[i] <= sel ? partial_hi[i] : partial_lo[i];
    end
  end
  for (genvar pair = 0; pair < 2; pair++) begin : g_pair
    (* keep_hierarchy = "yes" *) tsr_vu_rnode u_n (
      .clk, .rst_n, .mx(g_mx_q), .a(g_q[2*pair]), .b(g_q[2*pair+1]), .y(y[pair])
    );
  end
endmodule
/* verilator lint_on DECLFILENAME */
