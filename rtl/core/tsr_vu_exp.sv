// tsr_vu_exp.sv — EXPK(v, κ): the VU's integer exponential with a folded constant, fully pipelined.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §6.3, Appendix A; §7.2 "EXPK core").
// Contract: y = tsr_m_expk(v, kappa) of sw/sim/tsr_ref_math.c bit for bit, for every v and κ in
// {P1 0, N1 1, N2 2, NG 3}; fixed latency LAT (= TSR_VU_EXP_LAT) cycles, one operation per cycle, no
// clock enable (the VU's lanes never stall, VU §7.7 / VU-9). EXPK never returns -0 or a subnormal.
// Verified: tb/vu/test_vu_arith.py (Verilator C++ sweep against libtsr_ref: 2^24 strided points per κ
// plus specials by default, all 2^32 inputs per κ opt-in).
//
// Algorithm (VU §6.3 step numbers):
//   E1  1-5   DAZ, NaN -> 0x7FC00000, ±inf -> sgn ? +0 : +inf, ±0 -> 1.0, Ev > 156 saturates; sgn =
//             sign(v) ^ sK; Mv = 2^23 | fraction; sh = 156 - Ev
//   E2-E4 6   P = Mv × LK (24 × 40 bits, DSP)
//   E5-E6 6   T = P >> sh (two shift stages: sh[5:3], sh[2:0]; sh >= 64 gives 0), T >= 2^40 saturates
//   E7    7-8 t = ±T; n = t >> 32 (floor); F = t mod 2^32; j = F >> 26; G = F mod 2^26
//   E8-E18 9  Horner in units of 2^-32: X3 = (G·P3F) >> 32, S2 = P2F + X3, X2 = (G·S2) >> 32,
//             S1 = P1F + X2, PF = (G·S1) >> 32 (three 26 × ≤32-bit products, 3 stages each + add)
//   E19-E22 10 M = T64[j] + ((T64[j]·PF) >> 32)
//   E23   11-12 normalise (M >= 2^31: drop 8, n + 1) and round to 24 bits (RNE on the dropped bits;
//             sig = 2^24 -> 2^23, n + 1)
//   E24   13  be = n + 127: >= 255 -> +inf, <= 0 -> +0, else {be, sig - 2^23}; special results win
// Registers: every product is fed from registers and followed by two registers (DSP M / P); the
// shift and the adds are one stage each (<= 6 LUT levels, docs/rtl-m2.md §1).

module tsr_vu_exp (
  input  logic        clk,
  input  logic [31:0] v,
  input  logic [1:0]  kappa,
  output logic [31:0] y
);
  import tsr_vu_pkg::*;

  // Appendix A constants
  localparam logic [39:0] LK_P1 = 40'h5C551D94AE;
  localparam logic [39:0] LK_N2 = 40'hB8AA3B295C;
  localparam logic [39:0] LK_NG = 40'h06969F3DD6;

  localparam logic [31:0] C_P3F = 32'h0E35846C;
  localparam logic [31:0] C_P2F = 32'h3D7F7BFF;
  localparam logic [31:0] C_P1F = 32'hB17217F8;

  function automatic logic [30:0] t64(input logic [5:0] j);
    case (j)
      6'd0:  return 31'h40000000; 6'd1:  return 31'h40B268FA; 6'd2:  return 31'h4166C34C; 6'd3:  return 31'h421D1462;
      6'd4:  return 31'h42D561B4; 6'd5:  return 31'h438FB0CB; 6'd6:  return 31'h444C0740; 6'd7:  return 31'h450A6ABB;
      6'd8:  return 31'h45CAE0F2; 6'd9:  return 31'h468D6FAE; 6'd10: return 31'h47521CC6; 6'd11: return 31'h4818EE22;
      6'd12: return 31'h48E1E9BA; 6'd13: return 31'h49AD1598; 6'd14: return 31'h4A7A77D4; 6'd15: return 31'h4B4A169C;
      6'd16: return 31'h4C1BF829; 6'd17: return 31'h4CF022CA; 6'd18: return 31'h4DC69CDD; 6'd19: return 31'h4E9F6CD4;
      6'd20: return 31'h4F7A9930; 6'd21: return 31'h50582888; 6'd22: return 31'h51382182; 6'd23: return 31'h521A8AD7;
      6'd24: return 31'h52FF6B55; 6'd25: return 31'h53E6C9DA; 6'd26: return 31'h54D0AD5A; 6'd27: return 31'h55BD1CDB;
      6'd28: return 31'h56AC1F75; 6'd29: return 31'h579DBC57; 6'd30: return 31'h5891FAC1; 6'd31: return 31'h5988E209;
      6'd32: return 31'h5A82799A; 6'd33: return 31'h5B7EC8F2; 6'd34: return 31'h5C7DD7A4; 6'd35: return 31'h5D7FAD59;
      6'd36: return 31'h5E8451D0; 6'd37: return 31'h5F8BCCDB; 6'd38: return 31'h60962665; 6'd39: return 31'h61A3666D;
      6'd40: return 31'h62B39509; 6'd41: return 31'h63C6BA64; 6'd42: return 31'h64DCDEC3; 6'd43: return 31'h65F60A7F;
      6'd44: return 31'h6712460B; 6'd45: return 31'h683199ED; 6'd46: return 31'h69540EC9; 6'd47: return 31'h6A79AD56;
      6'd48: return 31'h6BA27E65; 6'd49: return 31'h6CCE8AE1; 6'd50: return 31'h6DFDDBCC; 6'd51: return 31'h6F307A41;
      6'd52: return 31'h70666F76; 6'd53: return 31'h719FC4B9; 6'd54: return 31'h72DC8374; 6'd55: return 31'h741CB528;
      6'd56: return 31'h75606374; 6'd57: return 31'h76A7980F; 6'd58: return 31'h77F25CCE; 6'd59: return 31'h7940BB9E;
      6'd60: return 31'h7A92BE8B; 6'd61: return 31'h7BE86FBA; 6'd62: return 31'h7D41D96E; default: return 31'h7E9F0606;
    endcase
  endfunction

  // ---------------------------------------------------------------------------------------------
  // Special-result side band: valid flag and value, delayed to the output
  // ---------------------------------------------------------------------------------------------
  localparam int unsigned LAT = TSR_VU_EXP_LAT;

  // E0: input registers
  logic [31:0] v0_q;
  logic [1:0]  k0_q;
  always_ff @(posedge clk) begin
    v0_q <= v;
    k0_q <= kappa;
  end

  // E1: classification (steps 1-5)
  logic        e1_spec, e1_sgn;
  logic [31:0] e1_val;
  logic [23:0] e1_mv;
  logic [7:0]  e1_sh;
  logic [39:0] e1_lk;
  always_comb begin
    logic [31:0] u;
    logic [7:0]  ev;
    u      = (v0_q[30:23] == 8'd0) ? {v0_q[31], 31'd0} : v0_q;      // DAZ
    ev     = u[30:23];
    e1_sgn = u[31] ^ (k0_q != 2'd0);                                // sK = 1 for N1, N2, NG
    e1_mv  = {1'b1, u[22:0]};
    e1_sh  = 8'd156 - ev;
    case (k0_q)
      2'd2:    e1_lk = LK_N2;
      2'd3:    e1_lk = LK_NG;
      default: e1_lk = LK_P1;
    endcase
    e1_spec = 1'b1;
    if (ev == 8'hFF && u[22:0] != 23'd0) e1_val = TSR_VU_NAN;       // 1. NaN
    else if (ev == 8'hFF)                e1_val = e1_sgn ? 32'd0 : TSR_VU_INF;   // 3. ±inf
    else if (ev == 8'd0)                 e1_val = TSR_VU_ONE;       // 3. ±0
    else if (ev > 8'd156)                e1_val = e1_sgn ? 32'd0 : TSR_VU_INF;   // 5. saturate
    else begin
      e1_spec = 1'b0;
      e1_val  = 32'd0;
    end
  end

  logic        s1_spec_q, s1_sgn_q;
  logic [31:0] s1_val_q;
  logic [23:0] s1_mv_q;
  logic [7:0]  s1_sh_q;
  logic [39:0] s1_lk_q;
  always_ff @(posedge clk) begin
    s1_spec_q <= e1_spec;
    s1_val_q  <= e1_val;
    s1_sgn_q  <= e1_sgn;
    s1_mv_q   <= e1_mv;
    s1_sh_q   <= e1_sh;
    s1_lk_q   <= e1_lk;
  end

  // E2-E4: P = Mv × LK (64 bits)
  logic [63:0] p1_q, p2_q, p3_q;
  logic [7:0]  sh_d [3];
  logic        sgn_d [3];
  always_ff @(posedge clk) begin
    p1_q <= 64'(s1_mv_q) * 64'(s1_lk_q);
    p2_q <= p1_q;
    p3_q <= p2_q;
    sh_d[0]  <= s1_sh_q;
    sgn_d[0] <= s1_sgn_q;
    for (int i = 1; i < 3; i++) begin
      sh_d[i]  <= sh_d[i-1];
      sgn_d[i] <= sgn_d[i-1];
    end
  end

  // E5: coarse shift by sh[5:3] (sh >= 64: 0); E6: fine shift, saturation
  logic [63:0] t5_q;
  logic [2:0]  sh5_q;
  logic        sgn5_q;
  always_ff @(posedge clk) begin
    t5_q   <= (sh_d[2] >= 8'd64) ? 64'd0 : (p3_q >> {sh_d[2][5:3], 3'b000});
    sh5_q  <= sh_d[2][2:0];
    sgn5_q <= sgn_d[2];
  end
  logic [63:0] t6;
  assign t6 = t5_q >> sh5_q;
  logic        t6_sat_q, sgn6_q;
  logic [39:0] t6_q;
  always_ff @(posedge clk) begin
    t6_sat_q <= (t6[63:40] != 24'd0);
    t6_q     <= t6[39:0];
    sgn6_q   <= sgn5_q;
  end

  // E7: t = ±T, n, F
  logic signed [9:0] n7_q;
  logic [31:0]       f7_q;
  logic              sat7_q, sgn7_q;
  always_ff @(posedge clk) begin
    logic [41:0] tt;
    tt     = sgn6_q ? (42'd0 - {2'b00, t6_q}) : {2'b00, t6_q};
    n7_q   <= $signed(tt[41:32]);
    f7_q   <= tt[31:0];
    sat7_q <= t6_sat_q;
    sgn7_q <= sgn6_q;
  end

  // E8-E18: Horner. Stage groups: product (3 registers), add (1 register).
  logic [25:0] g8;
  assign g8 = f7_q[25:0];
  // step X3 = (G·P3F) >> 32
  logic [57:0] h1a_q, h1b_q, h1c_q;
  logic [25:0] g_d1 [4];
  always_ff @(posedge clk) begin
    h1a_q <= 58'(g8) * 58'(C_P3F);
    h1b_q <= h1a_q;
    h1c_q <= h1b_q;
    g_d1[0] <= g8;
    for (int i = 1; i < 4; i++) g_d1[i] <= g_d1[i-1];
  end
  logic [31:0] s2_q;
  always_ff @(posedge clk) s2_q <= C_P2F + 32'(h1c_q >> 32);
  // step X2 = (G·S2) >> 32
  logic [57:0] h2a_q, h2b_q, h2c_q;
  logic [25:0] g_d2 [4];
  always_ff @(posedge clk) begin
    h2a_q <= 58'(g_d1[3]) * 58'(s2_q);
    h2b_q <= h2a_q;
    h2c_q <= h2b_q;
    g_d2[0] <= g_d1[3];
    for (int i = 1; i < 4; i++) g_d2[i] <= g_d2[i-1];
  end
  logic [31:0] s1_q;
  always_ff @(posedge clk) s1_q <= C_P1F + 32'(h2c_q >> 32);
  // step PF = (G·S1) >> 32
  logic [57:0] h3a_q, h3b_q, h3c_q;
  always_ff @(posedge clk) begin
    h3a_q <= 58'(g_d2[3]) * 58'(s1_q);
    h3b_q <= h3a_q;
    h3c_q <= h3b_q;
  end
  logic [25:0] pf_q;
  always_ff @(posedge clk) pf_q <= 26'(h3c_q >> 32);

  // side band through the Horner stages: n, j (T64), saturation, sign  (E7 -> E18: 12 stages)
  localparam int unsigned HD = 12;
  logic signed [9:0] n_d [HD];
  logic [30:0]       tj_d [HD];
  logic              sat_d [HD], sgnh_d [HD];
  always_ff @(posedge clk) begin
    n_d[0]    <= n7_q;
    tj_d[0]   <= t64(f7_q[31:26]);
    sat_d[0]  <= sat7_q;
    sgnh_d[0] <= sgn7_q;
    for (int i = 1; i < HD; i++) begin
      n_d[i]    <= n_d[i-1];
      tj_d[i]   <= tj_d[i-1];
      sat_d[i]  <= sat_d[i-1];
      sgnh_d[i] <= sgnh_d[i-1];
    end
  end

  // E19-E22: M = T64 + ((T64·PF) >> 32)
  logic [56:0] m1_q, m2_q, m3_q;
  logic [30:0] tj_m [3];
  logic signed [9:0] n_m [3];
  logic        sat_m [3], sgn_m [3];
  always_ff @(posedge clk) begin
    m1_q <= 57'(tj_d[HD-1]) * 57'(pf_q);
    m2_q <= m1_q;
    m3_q <= m2_q;
    tj_m[0]  <= tj_d[HD-1];
    n_m[0]   <= n_d[HD-1];
    sat_m[0] <= sat_d[HD-1];
    sgn_m[0] <= sgnh_d[HD-1];
    for (int i = 1; i < 3; i++) begin
      tj_m[i]  <= tj_m[i-1];
      n_m[i]   <= n_m[i-1];
      sat_m[i] <= sat_m[i-1];
      sgn_m[i] <= sgn_m[i-1];
    end
  end
  logic [31:0]       mm_q;
  logic signed [9:0] nn_q;
  logic              sat_q, sgnf_q;
  always_ff @(posedge clk) begin
    mm_q   <= 32'(tj_m[2]) + 32'(m3_q >> 32);
    nn_q   <= n_m[2];
    sat_q  <= sat_m[2];
    sgnf_q <= sgn_m[2];
  end

  // E23: normalise and round (steps 11-12)
  logic [23:0]       sig_q;
  logic signed [10:0] n23_q;
  logic              sat23_q, sgn23_q;
  always_ff @(posedge clk) begin
    logic [24:0] sg;
    logic        up;
    logic signed [10:0] n1;
    n1 = 11'(nn_q);
    if (mm_q[31]) begin                       // drop 8
      sg = {1'b0, mm_q[31:8]};
      up = mm_q[7] && (mm_q[6:0] != 7'd0 || mm_q[8]);
      n1 = n1 + 11'sd1;
    end else begin                            // drop 7
      sg = {1'b0, mm_q[30:7]};
      up = mm_q[6] && (mm_q[5:0] != 6'd0 || mm_q[7]);
    end
    sg = sg + 25'(up);
    if (sg[24]) begin
      sig_q <= 24'h800000;
      n1 = n1 + 11'sd1;
    end else begin
      sig_q <= sg[23:0];
    end
    n23_q   <= n1;
    sat23_q <= sat_q;
    sgn23_q <= sgnf_q;
  end

  // E24: pack; specials (delayed) win
  localparam int unsigned SD = LAT - 3;      // spec_d[i] is the E1 result delayed by i + 1
  logic        spec_d [SD];
  logic [31:0] sval_d [SD];
  always_ff @(posedge clk) begin
    spec_d[0] <= s1_spec_q;
    sval_d[0] <= s1_val_q;
    for (int i = 1; i < SD; i++) begin
      spec_d[i] <= spec_d[i-1];
      sval_d[i] <= sval_d[i-1];
    end
  end
  always_ff @(posedge clk) begin
    logic signed [10:0] be;
    be = n23_q + 11'sd127;
    if (spec_d[SD-1])        y <= sval_d[SD-1];
    else if (sat23_q)        y <= sgn23_q ? 32'd0 : TSR_VU_INF;
    else if (be >= 11'sd255) y <= TSR_VU_INF;
    else if (be <= 11'sd0)   y <= 32'd0;
    else                     y <= {1'b0, be[7:0], sig_q[22:0]};
  end

  wire unused_ok = &{1'b0, h1c_q[31:0], h2c_q[31:0], h3c_q[31:0], m3_q[31:0], sig_q[23]};

endmodule
