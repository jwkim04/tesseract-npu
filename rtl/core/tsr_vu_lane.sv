// tsr_vu_lane.sv — one VU lane: the front lane (FM_A ∥ FM_B → FA → FM_C → EXPK → select) with its RED
// partials and, when HAS_BACK, the back lane (ONE → RC → FM_R → fp16 rounding → FM_S → select).
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §7.2 front lane, §7.3 back lane, §7.4 function mapping,
// §6 numerics; §7.5 the per-lane partials of the reduction).
// Contract: for every element injected with micro-op u (tsr_vu_pkg::tsr_vu_uop_e) and operands a, b, c1,
// c2, the front exit f_y carries the result TSR_VU_LF cycles after the input register (cycle 0) and the
// back exit b_y TSR_VU_LB cycles after it; the result is the simulator's (tsr_ref_vop_elem, tsr_ref_at.c,
// tsr_ref_vu.c) bit for bit:
//   VOP (u = func + 1)  as VU §5.2 with c = c1 (front exit; the H functions SILU, SWIGLU, SIGMOID, GELU,
//                       GEGLU, TANH, RECIP, GELU16, GEGLU16 at the back exit)
//   U_ROT    FA(FM(a, c1), FM(b, c2)), or FS when neg (ROPE §6.7 rotation, AMERGE combine)
//   U_ASMAX  EXPK(FM(FS(a, c1), c2), P1);  U_ESCB  EXPK(FM(FS(a, b), c2), P1)
//   U_AMASK  (bits(b) = 0xFF800000) ? -inf : FA(a, b)
//   U_RCPMAX RC(MAXT(a, 2^-126)) (back exit)
//   U_RSUM / U_RSUMSQ / U_RMAX / U_AMASK accumulate into partial {buf, phase} = acc_pa (VU §6.2): SUM-type
//            p <- FA(p', x) with x = a or FM(a, a), MAX-type p <- MAXT(p', v) with v = a or the AMASK value;
//            p' = the init value (+0, -inf, M_FLOOR for AMASK) when acc_fresh (the first line of that phase
//            in the row), else the partial. A lane without an element (acc_lv = 0) writes the init value
//            when fresh and nothing otherwise (VU §6.2 skip rule: unwritten partials are +0).
// The datapath has no clock enable and never stalls (VU-9); in_v only qualifies the partial writes.
// HAS_EXP = 0 removes the EXPK core (VU §7.10 knob VU_EXP_LANES: these lanes never get an EXPK micro-op),
// HAS_BACK = 0 the back lane (VU_BACK_LANES).
//
// Pipeline (cycle of each value relative to the input register S0 = cycle 0):
//   S1  1   DAZ(a), F16(DAZ(a)) (GELU16 input rounding, VU §7.3), the flags (NaN, <= -10, >= 10, -inf,
//           mask -inf, TANH small path |a| < 2^-6, sign), MAXT / MINT / MAXT(a, 2^-126)
//   S2  2   H(a) = F32(F16(DAZ(a))); FM_A / FM_B operands; the chains x (the "x operand": a, H(a), |a|
//           or MAXT(a, 2^-126)) and r (the result so far: MAX, MIN, COPY, SETC, DAZ(a) for the GELU16 clamp)
//   FM  6   FM_A, FM_B (tsr_fp_mul LAT 4);  r <- FM_A for MUL, MULC, MULCB
//   FA  11  FA (tsr_fp_add LAT 5, operands muxed into its input register);  r <- FA (or the AMASK value)
//           for the add-type functions; SUM partial read at cycle 6, MAX read at 10, the partial's new
//           value at 11 - 12, written at the end of cycle 13 (a partial is touched at most every 8
//           cycles; tsr_vu_red's drain reads it after cycle 14)
//   FC  15  FM_C;  r <- FM_C for MULCB, ADDMULC, TANH (small-path value t1)
//   E   41  EXPK (tsr_vu_exp, LAT 26)
//   S5  42  front result: e, the EXPSUB -inf rule, or r                                  = TSR_VU_LF
//   back:   ONE 46 (FA(ONE, e) ∥ FS(ONE, e)), RC 67 (tsr_vu_rcp, LAT 21), FM_R 71, B3a 72 F16(FM_R),
//           B3b 73 g = the GELU16 value, FM_S 77, B5 78 final select                     = TSR_VU_LB
// Operands of every unit come from the previous unit's output register and delay lines (tsr_vu_dly).

module tsr_vu_lane
  import tsr_vu_pkg::*;
#(
  parameter bit HAS_EXP  = 1'b1,
  parameter bit HAS_BACK = 1'b1
) (
  input  logic        clk,
  input  logic        rst_n,
  // element (registered here: S0)
  input  logic        in_v,
  input  tsr_vu_uop_e in_uop,
  input  logic [31:0] in_a,
  input  logic [31:0] in_b,
  input  logic [31:0] in_c1,
  input  logic [31:0] in_c2,
  input  logic        in_neg,
  // reduction controls
  input  logic        acc_lv,             // the lane holds an element of the row
  input  logic        acc_fresh,          // first touch of this partial in the row
  input  logic [3:0]  acc_pa,             // {buffer, phase}
  // partial drain read (combinational)
  input  logic [3:0]  dr_addr,
  output logic [31:0] dr_data,
  // exits
  output logic [31:0] f_y,
  output logic [31:0] b_y
);
  import tsr_fp_pkg::*;

  localparam int unsigned T_FM = 6, T_FA = 11, T_FC = 15, T_E = 41;
  localparam int unsigned T_ONE = 46, T_RC = 67, T_FR = 71, T_B3B = 73, T_FS = 77;

  if (TSR_VU_FMUL_LAT != 4 || TSR_VU_FADD_LAT != 5 || TSR_VU_EXP_LAT != 26 || TSR_VU_RCP_LAT != 21 ||
      TSR_VU_LF != T_E + 1 || TSR_VU_LB != T_FS + 1) begin : g_chk_lat
    $fatal(1, "tsr_vu_lane: the latency constants of tsr_vu_pkg and the lane schedule disagree");
  end

  // -------------------------------------------------------------------------------------------
  // MAXT / MINT (VU §6.8)
  // -------------------------------------------------------------------------------------------
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
  function automatic logic [31:0] mint(input logic [31:0] a, input logic [31:0] b);
    logic [31:0] ad, bd;
    logic        na, nb;
    na = tsr_f32_is_nan(a);
    nb = tsr_f32_is_nan(b);
    ad = tsr_f32_daz(a);
    bd = tsr_f32_daz(b);
    if (na && nb) return TSR_VU_NAN;
    if (na) return bd;
    if (nb) return ad;
    return (key_of(ad) <= key_of(bd)) ? ad : bd;
  endfunction

  // -------------------------------------------------------------------------------------------
  // S0: input registers
  // -------------------------------------------------------------------------------------------
  logic        v0_q, neg0_q, lv0_q, fr0_q;
  tsr_vu_uop_e u0_q;
  logic [31:0] a0_q, b0_q, c10_q, c20_q;
  logic [3:0]  pa0_q;
  always_ff @(posedge clk) begin
    if (!rst_n) v0_q <= 1'b0;
    else        v0_q <= in_v;
    u0_q   <= in_uop;
    a0_q   <= in_a;
    b0_q   <= in_b;
    c10_q  <= in_c1;
    c20_q  <= in_c2;
    neg0_q <= in_neg;
    lv0_q  <= acc_lv;
    fr0_q  <= acc_fresh;
    pa0_q  <= acc_pa;
  end

  // -------------------------------------------------------------------------------------------
  // S1: DAZ, F16, flags, compares
  // -------------------------------------------------------------------------------------------
  typedef struct packed {
    logic nan;       // a is NaN
    logic le10;      // DAZ(a) <= -10 (non-NaN)
    logic ge10;      // DAZ(a) >= 10 (non-NaN)
    logic aninf;     // DAZ(a) = -inf
    logic bninf;     // bits(b) = 0xFF800000
    logic tsmall;    // |DAZ(a)| < 2^-6
    logic sign;      // sign bit of a (raw)
  } flags_t;

  logic        v1_q, neg1_q, lv1_q, fr1_q;
  tsr_vu_uop_e u1_q;
  logic [31:0] a1_q, ad1_q, b1_q, c11_q, c21_q, maxab1_q, minab1_q, maxlf1_q;
  logic [15:0] h16_1_q;
  flags_t      fl1_q;
  logic [3:0]  pa1_q;
  always_ff @(posedge clk) begin
    logic [31:0] ad;
    ad = tsr_f32_daz(a0_q);
    if (!rst_n) v1_q <= 1'b0;
    else        v1_q <= v0_q;
    u1_q     <= u0_q;
    a1_q     <= a0_q;
    ad1_q    <= ad;
    b1_q     <= b0_q;
    c11_q    <= c10_q;
    c21_q    <= c20_q;
    neg1_q   <= neg0_q;
    lv1_q    <= lv0_q;
    fr1_q    <= fr0_q;
    pa1_q    <= pa0_q;
    h16_1_q  <= tsr_f32_to_fp16(ad);
    maxab1_q <= maxt(a0_q, b0_q);
    minab1_q <= mint(a0_q, b0_q);
    maxlf1_q <= maxt(a0_q, TSR_VU_LFLOOR);
    fl1_q.nan   <= tsr_f32_is_nan(a0_q);
    fl1_q.le10  <= !tsr_f32_is_nan(a0_q) && ad[31] && (ad[30:0] >= TSR_VU_TEN[30:0]);
    fl1_q.ge10  <= !tsr_f32_is_nan(a0_q) && !ad[31] && (ad[30:0] >= TSR_VU_TEN[30:0]);
    fl1_q.aninf <= (ad == TSR_VU_NINF);
    fl1_q.bninf <= (b0_q == TSR_VU_NINF);
    fl1_q.tsmall <= (ad[30:0] < 31'h3C80_0000);
    fl1_q.sign  <= a0_q[31];
  end

  // -------------------------------------------------------------------------------------------
  // S2: H(a), FM operands, chains x and r
  // -------------------------------------------------------------------------------------------
  logic        v2_q, neg2_q, lv2_q, fr2_q;
  tsr_vu_uop_e u2_q;
  logic [31:0] a2_q, b2_q, c12_q, c22_q, x2_q, r2_q;
  logic [31:0] fma_x_q, fma_y_q, fmb_x_q, fmb_y_q;
  flags_t      fl2_q;
  logic [3:0]  pa2_q;
  always_ff @(posedge clk) begin
    logic [31:0] h, ax;
    h  = tsr_fp16_to_f32(h16_1_q);
    ax = {1'b0, ad1_q[30:0]};
    if (!rst_n) v2_q <= 1'b0;
    else        v2_q <= v1_q;
    u2_q   <= u1_q;
    a2_q   <= a1_q;
    b2_q   <= b1_q;
    c12_q  <= c11_q;
    c22_q  <= c21_q;
    neg2_q <= neg1_q;
    lv2_q  <= lv1_q;
    fr2_q  <= fr1_q;
    pa2_q  <= pa1_q;
    fl2_q  <= fl1_q;
    // x operand
    case (u1_q)
      U_GELU16, U_GEGLU16: x2_q <= h;
      U_TANH:              x2_q <= ax;
      U_RCPMAX:            x2_q <= maxlf1_q;
      default:             x2_q <= a1_q;
    endcase
    // result so far
    case (u1_q)
      U_MAX:               r2_q <= maxab1_q;
      U_MIN:               r2_q <= minab1_q;
      U_SETC:              r2_q <= c11_q;
      U_GELU16, U_GEGLU16: r2_q <= ad1_q;
      default:             r2_q <= a1_q;      // COPY, RMAX value, and the raw a elsewhere
    endcase
    // FM_A / FM_B operands
    fma_x_q <= a1_q;
    fma_y_q <= b1_q;
    fmb_x_q <= 32'd0;
    fmb_y_q <= 32'd0;
    case (u1_q)
      U_MULC, U_MULCB, U_MULCADD: fma_y_q <= c11_q;
      U_RSUMSQ, U_GELU, U_GEGLU:  fma_y_q <= a1_q;
      U_GELU16, U_GEGLU16: begin
        fma_x_q <= h;
        fma_y_q <= h;
      end
      U_TANH: begin
        fma_x_q <= ax;
        fma_y_q <= ax;
      end
      U_ROT: begin
        fma_y_q <= c11_q;
        fmb_x_q <= b1_q;
        fmb_y_q <= c21_q;
      end
      default: ;
    endcase
  end

  // -------------------------------------------------------------------------------------------
  // FM_A, FM_B (cycle 2 -> 6)
  // -------------------------------------------------------------------------------------------
  logic [3:0] ov;                          // the fp blocks' valid outputs (in_valid is tied to 1)
  logic [31:0] fma_y, fmb_y;
  tsr_fp_mul #(.LAT(TSR_VU_FMUL_LAT)) u_fma (
    .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(fma_x_q), .b(fma_y_q), .out_valid(ov[0]), .y(fma_y)
  );
  tsr_fp_mul #(.LAT(TSR_VU_FMUL_LAT)) u_fmb (
    .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(fmb_x_q), .b(fmb_y_q), .out_valid(ov[1]), .y(fmb_y)
  );

  // chains to cycle 6
  tsr_vu_uop_e u6;
  logic [5:0] u6_d;                // tsr_vu_dly's output is plain logic (XSim: enum assignment rules)
  assign u6 = tsr_vu_uop_e'(u6_d);
  logic [31:0] a6, b6, c16, r6d;
  logic        neg6, lv6, fr6, v6;
  logic [3:0]  pa6;
  tsr_vu_dly #(.W(6),  .D(T_FM - 2)) d_u6  (.clk, .d(u2_q), .q(u6_d));
  tsr_vu_dly #(.W(32), .D(T_FM - 2)) d_a6  (.clk, .d(a2_q), .q(a6));
  tsr_vu_dly #(.W(32), .D(T_FM - 2)) d_b6  (.clk, .d(b2_q), .q(b6));
  tsr_vu_dly #(.W(32), .D(T_FM - 2)) d_c16 (.clk, .d(c12_q), .q(c16));
  tsr_vu_dly #(.W(32), .D(T_FM - 2)) d_r6  (.clk, .d(r2_q), .q(r6d));
  tsr_vu_dly #(.W(8),  .D(T_FM - 2)) d_k6  (.clk, .d({neg2_q, lv2_q, fr2_q, v2_q, pa2_q}),
                                            .q({neg6, lv6, fr6, v6, pa6}));

  logic [31:0] r6;
  always_comb begin
    case (u6)
      U_MUL, U_MULC, U_MULCB: r6 = fma_y;
      default:                       r6 = r6d;
    endcase
  end

  // -------------------------------------------------------------------------------------------
  // Partials (VU §6.2, §7.5): 2 buffers x 8 phases; SUM read at cycle 6, MAX read at 10, writes at the end of 13
  // -------------------------------------------------------------------------------------------
  logic [31:0] prf [16];
  logic [31:0] p_sum;                      // SUM-type p' at cycle 6
  assign p_sum = fr6 ? 32'd0 : prf[pa6];
  assign dr_data = prf[dr_addr];

  // -------------------------------------------------------------------------------------------
  // FA (cycle 6 -> 11)
  // -------------------------------------------------------------------------------------------
  logic [31:0] fa_x, fa_y_in;
  logic        fa_sub;
  always_comb begin
    fa_x    = a6;
    fa_y_in = b6;
    fa_sub  = 1'b0;
    case (u6)
      U_SUB:              fa_sub = 1'b1;
      U_ADDC:             fa_y_in = c16;
      U_SUBC:             begin fa_y_in = c16; fa_sub = 1'b1; end
      U_MULCADD:          fa_x = fma_y;
      U_EXPSUB, U_ESCB:   fa_sub = 1'b1;
      U_EXPSUBC, U_ASMAX: begin fa_y_in = c16; fa_sub = 1'b1; end
      U_GELU, U_GEGLU, U_GELU16, U_GEGLU16: begin fa_x = fma_y; fa_y_in = TSR_VU_KG; end
      U_TANH:             begin fa_x = TSR_VU_THREE; fa_y_in = fma_y; fa_sub = 1'b1; end
      U_RSUM:             begin fa_x = p_sum; fa_y_in = lv6 ? a6 : 32'd0; end
      U_RSUMSQ:           begin fa_x = p_sum; fa_y_in = lv6 ? fma_y : 32'd0; end
      U_ROT:              begin fa_x = fma_y; fa_y_in = fmb_y; fa_sub = neg6; end
      default: ;                                      // ADD, ADDMULC, AMASK: FA(a, b)
    endcase
  end
  logic [31:0] fa_y;
  tsr_fp_add #(.LAT(TSR_VU_FADD_LAT)) u_fa (
    .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(fa_x), .b(fa_y_in), .sub(fa_sub), .out_valid(ov[2]), .y(fa_y)
  );

  // chains to cycle 11
  tsr_vu_uop_e u11;
  logic [5:0] u11_d;                // tsr_vu_dly's output is plain logic (XSim: enum assignment rules)
  assign u11 = tsr_vu_uop_e'(u11_d);
  logic [31:0] b11, c111, c211, x11, r11d;
  logic        lv11, fr11, v11;
  logic [3:0]  pa11;
  flags_t      fl11;
  tsr_vu_dly #(.W(6),  .D(T_FA - T_FM)) d_u11  (.clk, .d(u6), .q(u11_d));
  tsr_vu_dly #(.W(32), .D(T_FA - T_FM)) d_b11  (.clk, .d(b6), .q(b11));
  tsr_vu_dly #(.W(32), .D(T_FA - T_FM)) d_c111 (.clk, .d(c16), .q(c111));
  tsr_vu_dly #(.W(32), .D(T_FA - 2))    d_c211 (.clk, .d(c22_q), .q(c211));
  tsr_vu_dly #(.W(32), .D(T_FA - 2))    d_x11  (.clk, .d(x2_q), .q(x11));
  tsr_vu_dly #(.W(32), .D(T_FA - T_FM)) d_r11  (.clk, .d(r6), .q(r11d));
  logic [3:0]  pa10;
  logic [6:0]  k10;
  tsr_vu_dly #(.W(7),  .D(T_FA - T_FM - 1)) d_k10 (.clk, .d({lv6, fr6, v6, pa6}), .q(k10));
  tsr_vu_dly #(.W(7),  .D(1))           d_k11  (.clk, .d(k10), .q({lv11, fr11, v11, pa11}));
  assign pa10 = k10[3:0];
  // MAX-type partial read one cycle ahead of its write (the same entry is touched at most every 8
  // cycles, VU §7.5), so the compare starts from a register
  logic [31:0] prd_q;
  always_ff @(posedge clk) prd_q <= prf[pa10];
  tsr_vu_dly #(.W($bits(flags_t)), .D(T_FA - 2)) d_f11 (.clk, .d(fl2_q), .q(fl11));

  logic [31:0] amask_v, r11;
  assign amask_v = fl11.bninf ? TSR_VU_NINF : fa_y;
  always_comb begin
    case (u11)
      U_ADD, U_SUB, U_ADDC, U_SUBC, U_MULCADD, U_EXPSUB, U_EXPSUBC, U_ROT: r11 = fa_y;
      U_AMASK: r11 = amask_v;
      default: r11 = r11d;
    endcase
  end

  // partial writes: the operands at cycle 11 (registered: NaN flags, DAZ), MAXT and the new value at 12
  // (registered), written at the end of cycle 13
  logic        p1_v_q, p1_sum_q, p1_lv_q, p1_na_q, p1_nb_q;
  logic [3:0]  p1_a_q;
  logic [31:0] p1_init_q, p1_fa_q, p1_pm_q, p1_mv_q;
  always_ff @(posedge clk) begin
    logic        sumk, maxk;
    logic [31:0] init, pm, mv;
    sumk = (u11 == U_RSUM) || (u11 == U_RSUMSQ);
    maxk = (u11 == U_RMAX) || (u11 == U_AMASK);
    init = (u11 == U_AMASK) ? TSR_VU_MFLOOR : (u11 == U_RMAX) ? TSR_VU_NINF : 32'd0;
    pm   = fr11 ? init : prd_q;
    mv   = (u11 == U_AMASK) ? amask_v : r11d;          // RMAX: the raw a on the r chain
    if (!rst_n) p1_v_q <= 1'b0;
    else        p1_v_q <= v11 && (lv11 || fr11) && (sumk || maxk);
    p1_sum_q  <= sumk;
    p1_lv_q   <= lv11;
    p1_a_q    <= pa11;
    p1_init_q <= init;
    p1_fa_q   <= fa_y;
    p1_na_q   <= tsr_f32_is_nan(pm);
    p1_nb_q   <= tsr_f32_is_nan(mv);
    p1_pm_q   <= tsr_f32_daz(pm);
    p1_mv_q   <= tsr_f32_daz(mv);
  end
  logic        pw_v_q;
  logic [3:0]  pw_a_q;
  logic [31:0] pw_d_q;
  always_ff @(posedge clk) begin
    logic [31:0] mx;                                   // MAXT (VU §6.8) of the registered operands
    mx = (p1_na_q || p1_nb_q) ? TSR_VU_NAN : (key_of(p1_pm_q) >= key_of(p1_mv_q)) ? p1_pm_q : p1_mv_q;
    if (!rst_n) pw_v_q <= 1'b0;
    else        pw_v_q <= p1_v_q;
    pw_a_q <= p1_a_q;
    pw_d_q <= p1_sum_q ? p1_fa_q : (p1_lv_q ? mx : p1_init_q);
  end
  always_ff @(posedge clk) if (pw_v_q) prf[pw_a_q] <= pw_d_q;

  // -------------------------------------------------------------------------------------------
  // FM_C (cycle 11 -> 15)
  // -------------------------------------------------------------------------------------------
  logic [31:0] fc_x, fc_y_in;
  always_comb begin
    fc_x    = fa_y;
    fc_y_in = x11;                                     // GELU*, TANH: FM(w, x)
    case (u11)
      U_MULCB:          begin fc_x = r11d; fc_y_in = b11; end   // FM(FM(a, c), b): FM_A on the r chain
      U_ADDMULC:        fc_y_in = c111;
      U_ASMAX, U_ESCB:  fc_y_in = c211;
      default: ;
    endcase
  end
  logic [31:0] fc_y;
  tsr_fp_mul #(.LAT(TSR_VU_FMUL_LAT)) u_fmc (
    .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(fc_x), .b(fc_y_in), .out_valid(ov[3]), .y(fc_y)
  );

  // chains to cycle 15
  tsr_vu_uop_e u15;
  logic [5:0] u15_d;                // tsr_vu_dly's output is plain logic (XSim: enum assignment rules)
  assign u15 = tsr_vu_uop_e'(u15_d);
  logic [31:0] x15, r15d;
  tsr_vu_dly #(.W(6),  .D(T_FC - T_FA)) d_u15 (.clk, .d(u11), .q(u15_d));
  tsr_vu_dly #(.W(32), .D(T_FC - T_FA)) d_x15 (.clk, .d(x11), .q(x15));
  tsr_vu_dly #(.W(32), .D(T_FC - T_FA)) d_r15 (.clk, .d(r11), .q(r15d));

  logic [31:0] r15;
  always_comb begin
    case (u15)
      U_MULCB, U_ADDMULC, U_TANH: r15 = fc_y;
      default:                    r15 = r15d;
    endcase
  end

  // -------------------------------------------------------------------------------------------
  // EXPK (cycle 15 -> 41)
  // -------------------------------------------------------------------------------------------
  logic [31:0] e_in, e_y;
  logic [1:0]  e_k;
  always_comb begin
    e_in = x15;
    e_k  = TSR_VU_K_P1;
    case (u15)
      U_EXPSUB, U_EXPSUBC:            e_in = r15d;     // FS result on the r chain
      U_SIGMOID, U_SILU, U_SWIGLU:    e_k  = TSR_VU_K_N1;
      U_GELU, U_GEGLU, U_GELU16, U_GEGLU16: begin e_in = fc_y; e_k = TSR_VU_K_NG; end
      U_TANH:                         e_k  = TSR_VU_K_N2;
      U_ASMAX, U_ESCB:                e_in = fc_y;
      default: ;                                       // EXP: EXPK(a, P1)
    endcase
  end
  if (HAS_EXP) begin : g_exp
    tsr_vu_exp u_exp (.clk, .v(e_in), .kappa(e_k), .y(e_y));
  end else begin : g_noexp
    assign e_y = 32'd0;
    wire unused_e = &{1'b0, e_in, e_k};
  end

  // chains to cycle 41
  tsr_vu_uop_e u41;
  logic [5:0] u41_d;                // tsr_vu_dly's output is plain logic (XSim: enum assignment rules)
  assign u41 = tsr_vu_uop_e'(u41_d);
  logic [31:0] r41;
  flags_t      fl41;
  tsr_vu_dly #(.W(6),  .D(T_E - T_FC)) d_u41 (.clk, .d(u15), .q(u41_d));
  tsr_vu_dly #(.W(32), .D(T_E - T_FC)) d_r41 (.clk, .d(r15), .q(r41));
  tsr_vu_dly #(.W($bits(flags_t)), .D(T_E - T_FA)) d_f41 (.clk, .d(fl11), .q(fl41));

  // S5: front result (cycle 42)
  always_ff @(posedge clk) begin
    case (u41)
      U_EXP, U_ASMAX, U_ESCB: f_y <= e_y;
      U_EXPSUB, U_EXPSUBC:    f_y <= fl41.aninf ? 32'd0 : e_y;     // VU §6.5 EXPSUB -inf rule
      default:                f_y <= r41;
    endcase
  end

  // -------------------------------------------------------------------------------------------
  // Back lane
  // -------------------------------------------------------------------------------------------
  if (HAS_BACK) begin : g_back
    // ONE: FA(ONE, e), FS(ONE, e) (cycle 41 -> 46)
    logic [31:0] ep, em;
    logic [3:0]  ovb;
    tsr_fp_add #(.LAT(TSR_VU_FADD_LAT)) u_onep (
      .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(TSR_VU_ONE), .b(e_y), .sub(1'b0), .out_valid(ovb[0]), .y(ep)
    );
    tsr_fp_add #(.LAT(TSR_VU_FADD_LAT)) u_onem (
      .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(TSR_VU_ONE), .b(e_y), .sub(1'b1), .out_valid(ovb[1]), .y(em)
    );

    tsr_vu_uop_e u46;

    logic [5:0] u46_d;                // tsr_vu_dly's output is plain logic (XSim: enum assignment rules)

    assign u46 = tsr_vu_uop_e'(u46_d);
    logic [31:0] x46;
    tsr_vu_dly #(.W(6),  .D(T_ONE - T_E))  d_u46 (.clk, .d(u41), .q(u46_d));
    tsr_vu_dly #(.W(32), .D(T_ONE - T_FC)) d_x46 (.clk, .d(x15), .q(x46));

    // RC (cycle 46 -> 67)
    logic [31:0] rc_in, rc_y;
    assign rc_in = (u46 == U_RECIP || u46 == U_RCPMAX) ? x46 : ep;
    tsr_vu_rcp u_rcp (.clk, .x(rc_in), .y(rc_y));

    tsr_vu_uop_e u67;

    logic [5:0] u67_d;                // tsr_vu_dly's output is plain logic (XSim: enum assignment rules)

    assign u67 = tsr_vu_uop_e'(u67_d);
    logic [31:0] x67, em67;
    tsr_vu_dly #(.W(6),  .D(T_RC - T_ONE)) d_u67  (.clk, .d(u46), .q(u67_d));
    tsr_vu_dly #(.W(32), .D(T_RC - T_ONE)) d_x67  (.clk, .d(x46), .q(x67));
    tsr_vu_dly #(.W(32), .D(T_RC - T_ONE)) d_em67 (.clk, .d(em), .q(em67));

    // FM_R (cycle 67 -> 71)
    logic [31:0] fr_y;
    tsr_fp_mul #(.LAT(TSR_VU_FMUL_LAT)) u_fmr (
      .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a((u67 == U_TANH) ? em67 : x67), .b(rc_y), .out_valid(ovb[2]), .y(fr_y)
    );

    // B3a (72): F16 of FM_R;  B3b (73): g = the GELU16 value (or FM_R)
    tsr_vu_uop_e u71, u73;
    logic [5:0] u71_d, u73_d;                // tsr_vu_dly's outputs are plain logic (XSim: enum assignment rules)
    assign u71 = tsr_vu_uop_e'(u71_d);
    assign u73 = tsr_vu_uop_e'(u73_d);
    logic [31:0] r73, b73;
    flags_t      fl73;
    tsr_vu_dly #(.W(6),  .D(T_FR - T_RC)) d_u71 (.clk, .d(u67), .q(u71_d));
    tsr_vu_dly #(.W(6),  .D(2))           d_u73 (.clk, .d(u71), .q(u73_d));
    tsr_vu_dly #(.W(32), .D(T_B3B - T_E)) d_r73 (.clk, .d(r41), .q(r73));
    tsr_vu_dly #(.W(32), .D(T_B3B - T_FA)) d_b73 (.clk, .d(b11), .q(b73));
    tsr_vu_dly #(.W($bits(flags_t)), .D(T_B3B - T_E)) d_f73 (.clk, .d(fl41), .q(fl73));

    logic [15:0] fr16_q;
    logic [31:0] fr72_q, g_q;
    always_ff @(posedge clk) begin
      fr16_q <= tsr_f32_to_fp16(tsr_f32_daz(fr_y));
      fr72_q <= fr_y;
    end
    tsr_vu_uop_e u72;
    logic [5:0] u72_d;                // tsr_vu_dly's output is plain logic (XSim: enum assignment rules)
    assign u72 = tsr_vu_uop_e'(u72_d);
    flags_t      fl72;
    logic [31:0] r72;
    tsr_vu_dly #(.W(6),  .D(1)) d_u72 (.clk, .d(u71), .q(u72_d));
    tsr_vu_dly #(.W($bits(flags_t)), .D(T_B3B - T_E - 1)) d_f72 (.clk, .d(fl41), .q(fl72));
    tsr_vu_dly #(.W(32), .D(T_B3B - T_E - 1)) d_r72 (.clk, .d(r41), .q(r72));
    always_ff @(posedge clk) begin
      if (u72 == U_GELU16 || u72 == U_GEGLU16) begin
        // GELU16 (VU §6.5): NaN -> NaN; x' >= 10 -> x' (r = DAZ(a)); else H(GELU(H(x')))
        g_q <= fl72.nan ? TSR_VU_NAN : fl72.ge10 ? r72 : tsr_fp16_to_f32(fr16_q);
      end else begin
        g_q <= fr72_q;
      end
    end

    // FM_S (cycle 73 -> 77)
    logic [31:0] fs_x, fs_yin, fs_y;
    always_comb begin
      fs_x   = g_q;
      fs_yin = b73;
      if (u73 == U_TANH) begin
        fs_x   = r73;                                  // t1 = FM(FM... small path value
        fs_yin = TSR_VU_THIRD;
      end
    end
    tsr_fp_mul #(.LAT(TSR_VU_FMUL_LAT)) u_fms (
      .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(fs_x), .b(fs_yin), .out_valid(ovb[3]), .y(fs_y)
    );

    // B5 (78): final select
    tsr_vu_uop_e u77;
    logic [5:0] u77_d;                // tsr_vu_dly's output is plain logic (XSim: enum assignment rules)
    assign u77 = tsr_vu_uop_e'(u77_d);
    logic [31:0] rc77, fr77, g77;
    flags_t      fl77;
    tsr_vu_dly #(.W(6),  .D(T_FS - T_B3B)) d_u77  (.clk, .d(u73), .q(u77_d));
    tsr_vu_dly #(.W(32), .D(T_FS - T_RC))  d_rc77 (.clk, .d(rc_y), .q(rc77));
    tsr_vu_dly #(.W(32), .D(T_FS - T_FR))  d_fr77 (.clk, .d(fr_y), .q(fr77));
    tsr_vu_dly #(.W(32), .D(T_FS - T_B3B)) d_g77  (.clk, .d(g_q), .q(g77));
    tsr_vu_dly #(.W($bits(flags_t)), .D(T_FS - T_B3B)) d_f77 (.clk, .d(fl73), .q(fl77));
    always_ff @(posedge clk) begin
      logic [31:0] t;
      t = fl77.tsmall ? fs_y : fr77;
      case (u77)
        U_SIGMOID, U_RECIP, U_RCPMAX: b_y <= rc77;
        U_SILU, U_GELU:               b_y <= fr77;
        U_GELU16:                     b_y <= fl77.le10 ? 32'd0 : g77;
        U_GEGLU16:                    b_y <= fl77.le10 ? 32'd0 : fs_y;
        U_TANH:                       b_y <= fl77.nan ? TSR_VU_NAN : {t[31] ^ fl77.sign, t[30:0]};
        default:                      b_y <= fs_y;     // SWIGLU, GEGLU
      endcase
    end
    wire unused_bk = &{1'b0, fl73, fl72, fl77, ovb};
  end else begin : g_noback
    assign b_y = 32'd0;
  end

  wire unused_ok = &{1'b0, fl41, ov};

endmodule
