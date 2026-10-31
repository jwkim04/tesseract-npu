// tsr_vu_sincos.sv — SIN and COS of one f32 angle per cycle, fully pipelined (the ROPE angle unit's core).
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §6.6, Appendix A; §7.6 angle unit).
// Contract: (s, c) = tsr_m_sincos(t) of sw/sim/tsr_ref_math.c bit for bit; fixed latency TSR_VU_SC_LAT, no
// clock enable. Verified in tb/vu/test_vu_arith.py (Verilator C++ against libtsr_ref: |θ| <= 2^18 densely,
// the whole exponent range, specials).
//
// Steps (VU §6.6 numbering):
//   C1   1-2  θ <- DAZ(θ); NaN / ±inf -> (NaN, NaN); E < 115 (|θ| < 2^-12, zeros included) -> (θ, 1)
//   C2-C4 3   P = M × C64 (24 × 64 = 88 bits exact), sh = 214 - E in [-40, 99]
//   C5-C6 3   P as a fixed-point number with its binary point at bit sh: k = the two bits above it, F = the
//             62 bits below it ({P, 62 zeros} >> sh, a staged shifter); sh < 0: F = 0, k = P << -sh
//   C7   4    F >= 2^61: k + 1, F <- 2^62 - F (|F| with the sign rneg)
//   C8-C11 5  R = |F| × PIO2 (62 × 63 = 125 bits exact)
//   C12-C14 5 r = RNE(R·2^-124) with sign rneg (+0 if F = 0): msb, shift + sticky, round
//   C15- 6    z = FM(r, r); sn = FA(FM(FM(FA(FM(FA(FM(SN3, z), SN2), z), SN1), z), r), r);
//             cs = FA(FS(FM(FM(FA(FM(FA(FM(CS3, z), CS2), z), CS1), z), z), FM(HALF, z)), ONE)
//             (tsr_fp_mul / tsr_fp_add, operands from the previous unit's output register)
//   end  7-8  (S, C) by the quadrant q = k mod 4; SIN = sign(θ) ? -S : S; the special results win
// Cost: 10 FM, 7 FA, about 15 DSP48E2 for the two integer products.

module tsr_vu_sincos (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] t,
  output logic [31:0] s,
  output logic [31:0] c
);
  import tsr_vu_pkg::*;

  localparam logic [63:0] C64  = 64'hA2F9836E4E44152A;
  localparam logic [62:0] PIO2 = 63'h6487ED5110B4611A;
  localparam logic [31:0] SN1 = 32'hBE2AAAA3, SN2 = 32'h3C08839E, SN3 = 32'hB94CA1F9;
  localparam logic [31:0] CS1 = 32'h3D2AAAA5, CS2 = 32'hBAB6061A, CS3 = 32'h37CCF5CE;

  localparam int unsigned LAT = TSR_VU_SC_LAT;

  // C0: input register
  logic [31:0] t0_q;
  always_ff @(posedge clk) t0_q <= t;

  // C1: classification
  logic        c1_spec_q, c1_sgn_q;
  logic [31:0] c1_ss_q, c1_cs_q;           // special results
  logic [23:0] c1_m_q;
  logic signed [8:0] c1_sh_q;              // 214 - E
  always_ff @(posedge clk) begin
    logic [31:0] u;
    logic [7:0]  e;
    u = (t0_q[30:23] == 8'd0) ? {t0_q[31], 31'd0} : t0_q;
    e = u[30:23];
    c1_sgn_q <= u[31];
    c1_m_q   <= {1'b1, u[22:0]};
    c1_sh_q  <= 9'sd214 - $signed({1'b0, e});
    if (e == 8'hFF) begin
      c1_spec_q <= 1'b1;
      c1_ss_q   <= TSR_VU_NAN;
      c1_cs_q   <= TSR_VU_NAN;
    end else if (e < 8'd115) begin
      c1_spec_q <= 1'b1;
      c1_ss_q   <= u;
      c1_cs_q   <= TSR_VU_ONE;
    end else begin
      c1_spec_q <= 1'b0;
      c1_ss_q   <= 32'd0;
      c1_cs_q   <= 32'd0;
    end
  end

  // C2-C4: P = M × C64
  logic [87:0] p2_q, p3_q, p4_q;
  logic signed [8:0] sh_d [3];
  logic        sgn_d [3];
  always_ff @(posedge clk) begin
    p2_q <= 88'(c1_m_q) * 88'(C64);
    p3_q <= p2_q;
    p4_q <= p3_q;
    sh_d[0]  <= c1_sh_q;
    sgn_d[0] <= c1_sgn_q;
    for (int i = 1; i < 3; i++) begin
      sh_d[i]  <= sh_d[i-1];
      sgn_d[i] <= sgn_d[i-1];
    end
  end

  // C5: coarse shift ({P, 62 zeros} >> sh) by sh[6:4] (multiples of 16), C6: fine shift by sh[3:0].
  // sh in [0, 99]: the window [sh, sh + 64) of a 150-bit value; sh < 0 handled apart.
  logic [149:0] q5_q;
  logic [3:0]   sh5_q;
  logic         neg5_q, sgn5_q;
  logic [1:0]   kneg5_q;                   // k for sh < 0: (P << -sh) mod 4
  always_ff @(posedge clk) begin
    logic [149:0] q;
    q = {p4_q, 62'd0};
    neg5_q  <= sh_d[2] < 0;
    kneg5_q <= (sh_d[2] == -9'sd1) ? {p4_q[0], 1'b0} : 2'd0;
    q5_q    <= (sh_d[2] < 0) ? 150'd0 : (q >> {sh_d[2][6:4], 4'd0});
    sh5_q   <= sh_d[2][3:0];
    sgn5_q  <= sgn_d[2];
  end
  logic [1:0]  k6_q;
  logic [61:0] f6_q;
  logic        sgn6_q;
  wire [149:0] q6 = q5_q >> sh5_q;
  always_ff @(posedge clk) begin
    k6_q   <= neg5_q ? kneg5_q : q6[63:62];
    f6_q   <= neg5_q ? 62'd0 : q6[61:0];
    sgn6_q <= sgn5_q;
  end

  // C7: nearest quadrant
  logic [1:0]  k7_q;
  logic [61:0] fm7_q;                      // |F| <= 2^61
  logic        rneg7_q, sgn7_q;
  always_ff @(posedge clk) begin
    if (f6_q[61]) begin
      k7_q    <= k6_q + 2'd1;
      fm7_q   <= 62'((63'd1 << 62) - 63'(f6_q));            // 2^62 - F
      rneg7_q <= 1'b1;
    end else begin
      k7_q    <= k6_q;
      fm7_q   <= f6_q;
      rneg7_q <= 1'b0;
    end
    sgn7_q <= sgn6_q;
  end

  // C8-C11: R = |F| × PIO2 (125 bits)
  logic [124:0] r8_q, r9_q, r10_q, r11_q;
  always_ff @(posedge clk) begin
    r8_q  <= 125'(fm7_q) * 125'(PIO2);
    r9_q  <= r8_q;
    r10_q <= r9_q;
    r11_q <= r10_q;
  end

  // C12: msb index b of R (R = 0, or R >= PIO2 > 2^62: b in [62, 124])
  logic [124:0] r12_q;
  logic [6:0]   b12_q;
  logic         z12_q;
  always_ff @(posedge clk) begin
    logic [6:0] b;
    b = 7'd62;
    for (int i = 62; i <= 124; i++) if (r11_q[i]) b = 7'(i);
    b12_q <= b;
    z12_q <= (r11_q == '0);
    r12_q <= r11_q;
  end
  // C13: sig = R >> (b - 23) (24 bits), guard, sticky
  logic [23:0] sig13_q;
  logic        g13_q, st13_q, z13_q;
  logic [6:0]  b13_q;
  wire [6:0]   n13   = b12_q - 7'd24;                  // (b - 23) - 1, >= 38
  wire [124:0] sh13  = r12_q >> n13;                   // [0] guard, [24:1] the significand
  wire [124:0] msk13 = (125'd1 << n13) - 125'd1;       // the sticky bits below the guard
  always_ff @(posedge clk) begin
    sig13_q <= sh13[24:1];
    g13_q   <= sh13[0];
    st13_q  <= (r12_q & msk13) != '0;
    z13_q   <= z12_q;
    b13_q   <= b12_q;
  end
  // C14: round to nearest even, pack r (normal: exponent b + 3)
  logic rq_neg_d;                            // rneg aligned to C13 (C7 + 6)
  logic [1:0] kq_d;                          // quadrant aligned to the polynomial outputs
  tsr_vu_dly #(.W(1), .D(6)) d_rneg (.clk, .d(rneg7_q), .q(rq_neg_d));
  logic [31:0] r14_q;
  always_ff @(posedge clk) begin
    logic [24:0] sg;
    logic [8:0]  be;
    sg = {1'b0, sig13_q} + 25'(g13_q && (st13_q || sig13_q[0]));
    be = 9'(b13_q) + 9'd3;
    if (sg[24]) begin
      sg = sg >> 1;
      be = be + 9'd1;
    end
    r14_q <= z13_q ? 32'd0 : {rq_neg_d, be[7:0], sg[22:0]};
  end

  // C15-: polynomials (each unit: operands from the previous output register; times relative to r14)
  localparam int unsigned FM = TSR_VU_FMUL_LAT, FAL = TSR_VU_FADD_LAT;
  logic [2:0] ov;
  logic [10:0] ovm;
  logic [31:0] z, t1, t2, t3, t4, t5, t6, sn, u1, u2, u3, u4, u5, u6, hz, u7, cs;
  logic [31:0] r_z, r_t6;                    // r delayed to z (0) and to t6's input
  logic [31:0] z_t3, z_t5, z_u3, z_u5, z_u6, z_h;

  // z = FM(r, r): ready at FM
  tsr_fp_mul #(.LAT(FM)) u_z  (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(r14_q), .b(r14_q), .out_valid(ovm[0]), .y(z));
  // sin chain
  tsr_fp_mul #(.LAT(FM))  u_t1 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(SN3), .b(z), .out_valid(ovm[1]), .y(t1));
  tsr_fp_add #(.LAT(FAL)) u_t2 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(t1), .b(SN2), .sub(1'b0), .out_valid(ov[0]), .y(t2));
  tsr_vu_dly #(.W(32), .D(FM + FAL)) d_zt3 (.clk, .d(z), .q(z_t3));
  tsr_fp_mul #(.LAT(FM))  u_t3 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(t2), .b(z_t3), .out_valid(ovm[2]), .y(t3));
  tsr_fp_add #(.LAT(FAL)) u_t4 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(t3), .b(SN1), .sub(1'b0), .out_valid(ov[1]), .y(t4));
  tsr_vu_dly #(.W(32), .D(2 * FM + 2 * FAL)) d_zt5 (.clk, .d(z), .q(z_t5));
  tsr_fp_mul #(.LAT(FM))  u_t5 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(t4), .b(z_t5), .out_valid(ovm[3]), .y(t5));
  tsr_vu_dly #(.W(32), .D(4 * FM + 2 * FAL)) d_rt6 (.clk, .d(r14_q), .q(r_t6));
  tsr_fp_mul #(.LAT(FM))  u_t6 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(t5), .b(r_t6), .out_valid(ovm[4]), .y(t6));
  tsr_vu_dly #(.W(32), .D(5 * FM + 2 * FAL)) d_rz (.clk, .d(r14_q), .q(r_z));
  tsr_fp_add #(.LAT(FAL)) u_sn (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(t6), .b(r_z), .sub(1'b0), .out_valid(ov[2]), .y(sn));
  // cos chain
  tsr_fp_mul #(.LAT(FM))  u_u1 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(CS3), .b(z), .out_valid(ovm[5]), .y(u1));
  logic [3:0] ovc;
  tsr_fp_add #(.LAT(FAL)) u_u2 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(u1), .b(CS2), .sub(1'b0), .out_valid(ovc[0]), .y(u2));
  tsr_vu_dly #(.W(32), .D(FM + FAL)) d_zu3 (.clk, .d(z), .q(z_u3));
  tsr_fp_mul #(.LAT(FM))  u_u3 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(u2), .b(z_u3), .out_valid(ovm[6]), .y(u3));
  tsr_fp_add #(.LAT(FAL)) u_u4 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(u3), .b(CS1), .sub(1'b0), .out_valid(ovc[1]), .y(u4));
  tsr_vu_dly #(.W(32), .D(2 * FM + 2 * FAL)) d_zu5 (.clk, .d(z), .q(z_u5));
  tsr_fp_mul #(.LAT(FM))  u_u5 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(u4), .b(z_u5), .out_valid(ovm[7]), .y(u5));
  tsr_vu_dly #(.W(32), .D(3 * FM + 2 * FAL)) d_zu6 (.clk, .d(z), .q(z_u6));
  tsr_fp_mul #(.LAT(FM))  u_u6 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(u5), .b(z_u6), .out_valid(ovm[8]), .y(u6));
  tsr_vu_dly #(.W(32), .D(3 * FM + 2 * FAL)) d_zh (.clk, .d(z), .q(z_h));
  tsr_fp_mul #(.LAT(FM))  u_h  (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(TSR_VU_HALF), .b(z_h), .out_valid(ovm[9]), .y(hz));
  tsr_fp_add #(.LAT(FAL)) u_u7 (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(u6), .b(hz), .sub(1'b1), .out_valid(ovc[2]), .y(u7));
  tsr_fp_add #(.LAT(FAL)) u_cs (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(u7), .b(TSR_VU_ONE), .sub(1'b0), .out_valid(ovc[3]), .y(cs));
  assign ovm[10] = 1'b0;

  // ready times relative to r14: sn after z, t1, t2, t3, t4, t5, t6, sn (5 FM + 3 FA); cs after z, u1,
  // u2, u3, u4, u5, u6 (with h), u7, cs (5 FM + 4 FA). sn waits for cs.
  localparam int unsigned T_SN = 5 * FM + 3 * FAL;
  localparam int unsigned T_CS = 5 * FM + 4 * FAL;
  localparam int unsigned T_END = (T_SN > T_CS) ? T_SN : T_CS;
  logic [31:0] sn_a, cs_a;
  tsr_vu_dly #(.W(32), .D(T_END - T_SN)) d_sna (.clk, .d(sn), .q(sn_a));
  tsr_vu_dly #(.W(32), .D(T_END - T_CS)) d_csa (.clk, .d(cs), .q(cs_a));

  // side band: specials, sign, quadrant (C1 / C7 -> the final register)
  localparam int unsigned T_R14 = 14;                   // r14_q is valid at cycle 14 (t0_q at cycle 0)
  localparam int unsigned T_FIN = T_R14 + T_END;
  logic        spec_f, sgn_f;
  logic [31:0] ss_f, cs_f;
  tsr_vu_dly #(.W(65), .D(T_FIN - 1)) d_sp (.clk, .d({c1_spec_q, c1_ss_q, c1_cs_q}), .q({spec_f, ss_f, cs_f}));
  tsr_vu_dly #(.W(1),  .D(T_FIN - 7)) d_sg (.clk, .d(sgn7_q), .q(sgn_f));
  tsr_vu_dly #(.W(2),  .D(T_FIN - 7)) d_kq (.clk, .d(k7_q), .q(kq_d));

  always_ff @(posedge clk) begin
    logic [31:0] ss, cc;
    case (kq_d)
      2'd0:    begin ss = sn_a;                    cc = cs_a; end
      2'd1:    begin ss = cs_a;                    cc = {~sn_a[31], sn_a[30:0]}; end
      2'd2:    begin ss = {~sn_a[31], sn_a[30:0]}; cc = {~cs_a[31], cs_a[30:0]}; end
      default: begin ss = {~cs_a[31], cs_a[30:0]}; cc = sn_a; end
    endcase
    s <= spec_f ? ss_f : (sgn_f ? {~ss[31], ss[30:0]} : ss);
    c <= spec_f ? cs_f : cc;
  end

  // t (input) -> t0_q -> ... -> s / c: T_FIN + 2 cycles
  if (LAT != T_FIN + 2) begin : g_chk_lat
    $fatal(1, "tsr_vu_sincos: TSR_VU_SC_LAT must be %0d", T_FIN + 2);
  end

  wire unused_ok = &{1'b0, ov, ovc, ovm, q6[149:64], sh13[124:25]};

endmodule
