// tsr_qact_scale.sv — the A32 block scales of QACT without dividers: one block per cycle, fixed latency.
//
// Spec: docs/rtl-core.md §2, block (e); numerics §8.1 (A32: d = fdiv(amax, 127), id = fdiv(1, d)).
// Contract, bit for bit with tsr_fdiv / tsr_frecip / tsr_f32_to_fp16 of sw/sim/tsr_ref.c, for every amag that
// tsr_qact can produce (|DAZ(x)| bits of the amax of a block: 0, a normal value or Inf, never NaN):
//   d     = tsr_fdiv(amag, 127)
//   scale = (d != +-0) ? tsr_frecip(d) : +0       (tsr_fdiv(1, d) = tsr_frecip(d), tsr_fp_div.sv)
//   h16   = tsr_f32_to_fp16(d)
//   zero  = (amag == 0)
// Verified exhaustively over every amag from 0 to 0x7F800000 (tb/qact/test_qact_scale.py, opt-in full run;
// the default run covers every mantissa at fourteen exponents).
// Timing: out_valid is LAT cycles after in_valid, one block per cycle, no stall, no enable. rst_n clears the
// valid pipeline only; data registers have no reset. Every output is a register.
//
// d = amag / 127 is a multiply by M = ceil(2^39 / 127) (33 bits) and the normal result rounding. With A the
// 24-bit significand of amag, P = A * M has 57 bits; Q = P[56:31] has 26 bits and Q[25] or Q[24] is set. The
// product is cut into two DSP48E2 products (A * M[16:0], A * M[32:17]) and added in the next stage. The
// remainder of A * 2^8 / 127 is Q mod 128, and the guard bit g (Q[0], or Q[1] when Q[25] is set) is one of
// its bits: when g is set the sticky bit |Q[6:0] is set too, so the rounding increments exactly when g is set
// and round_pack gets st = 1 (a division by 127 has no tie). The significand of amag == 0 gives a result
// exponent below zero, so d is +0 there without a rule of its own. 1 / d is the vector unit's reciprocal
// core tsr_vu_rcp (correctly rounded, TSR_VU_RCP_LAT stages); the other results wait in a delay line of the
// same length.
//
//   K0  amag register
//   K1  two partial products pl, ph (one DSP48E2 each)
//   K2  P = (ph << 17) + pl, Q
//   K3  normalization: significand s, guard g, exponent e, amag == Inf
//   K4  d = round_pack(...) (+Inf for amag == Inf)
//   RC  scale = frecip(d) (tsr_vu_rcp); beside it the delay line of {h16(d), d == 0, zero, valid}
//   Out scale (+0 when d == 0), h16, zero, out_valid

module tsr_qact_scale
  import tsr_fp_pkg::*;
  import tsr_vu_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,      // clears the valid pipeline only
  input  logic        in_valid,
  input  logic [30:0] amag,       // amax of a block: |DAZ(x)| bits, never NaN (0, a normal value or Inf)
  output logic        out_valid,  // LAT cycles after in_valid
  output logic [31:0] scale,      // id = frecip(d), +0 when d == +-0
  output logic [15:0] h16,        // tsr_f32_to_fp16(d)
  output logic        zero        // amag == 0
);
  // The latency from in_valid to out_valid (documents the fixed latency; tsr_qact does not depend on it).
  /* verilator lint_off UNUSEDPARAM */
  localparam int unsigned LAT = 5 + TSR_VU_RCP_LAT + 1;
  /* verilator lint_on UNUSEDPARAM */
  localparam int unsigned RL  = TSR_VU_RCP_LAT;
  localparam logic [32:0] M   = 33'h1_0204_0811;   // ceil(2^39 / 127)

  // K0: input register
  logic        k0_v, k0_zero;
  logic [7:0]  k0_e;
  logic [23:0] k0_a;
  always_ff @(posedge clk) begin
    if (!rst_n) k0_v <= 1'b0;
    else        k0_v <= in_valid;
  end
  always_ff @(posedge clk) begin
    k0_e    <= amag[30:23];
    k0_a    <= {1'b1, amag[22:0]};
    k0_zero <= (amag == 31'd0);
  end

  // K1: partial products
  logic        k1_v, k1_zero;
  logic [7:0]  k1_e;
  (* use_dsp = "yes" *) logic [40:0] k1_pl;
  (* use_dsp = "yes" *) logic [39:0] k1_ph;
  always_ff @(posedge clk) begin
    if (!rst_n) k1_v <= 1'b0;
    else        k1_v <= k0_v;
  end
  always_ff @(posedge clk) begin
    k1_pl   <= 41'(k0_a) * 41'(M[16:0]);
    k1_ph   <= 40'(k0_a) * 40'(M[32:17]);
    k1_e    <= k0_e;
    k1_zero <= k0_zero;
  end

  // K2: P = (ph << 17) + pl, Q = P[56:31]
  logic        k2_v, k2_zero;
  logic [7:0]  k2_e;
  logic [25:0] k2_q;
  wire  [56:0] k2_p = {k1_ph, 17'd0} + 57'(k1_pl);
  always_ff @(posedge clk) begin
    if (!rst_n) k2_v <= 1'b0;
    else        k2_v <= k1_v;
  end
  always_ff @(posedge clk) begin
    k2_q    <= 26'(k2_p >> 31);
    k2_e    <= k1_e;
    k2_zero <= k1_zero;
  end

  // K3: normalize
  logic                k3_v, k3_zero, k3_g, k3_inf;
  logic [23:0]         k3_s;
  logic signed [9:0]   k3_e;
  wire                 k2_hi = k2_q[25];
  always_ff @(posedge clk) begin
    if (!rst_n) k3_v <= 1'b0;
    else        k3_v <= k2_v;
  end
  always_ff @(posedge clk) begin
    k3_s    <= k2_hi ? k2_q[25:2] : k2_q[24:1];
    k3_g    <= k2_hi ? k2_q[1] : k2_q[0];
    k3_e    <= 10'($signed({2'b00, k2_e}) - 10'sd7 + 10'(k2_hi));
    k3_inf  <= (k2_e == 8'hFF);
    k3_zero <= k2_zero;
  end

  // K4: d
  logic        k4_v, k4_zero;
  logic [31:0] k4_d;
  always_ff @(posedge clk) begin
    if (!rst_n) k4_v <= 1'b0;
    else        k4_v <= k3_v;
  end
  always_ff @(posedge clk) begin
    if (k3_inf) k4_d <= 32'h7F80_0000;
    else        k4_d <= tsr_f32_round_pack(1'b0, k3_e, k3_s, k3_g, 1'b1);
    k4_zero <= k3_zero;
  end

  // RC: 1 / d, and the delay line of what does not go through it
  logic [31:0] rc_y;
  tsr_vu_rcp u_rcp (.clk, .x(k4_d), .y(rc_y));

  logic [15:0] dl_h16  [RL];
  logic        dl_dz   [RL];
  logic        dl_zero [RL];
  logic        dl_v    [RL];
  always_ff @(posedge clk) begin
    dl_h16[0]  <= tsr_f32_to_fp16(k4_d);
    dl_dz[0]   <= (k4_d[30:0] == 31'd0);
    dl_zero[0] <= k4_zero;
    for (int i = 1; i < int'(RL); i++) begin
      dl_h16[i]  <= dl_h16[i-1];
      dl_dz[i]   <= dl_dz[i-1];
      dl_zero[i] <= dl_zero[i-1];
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int i = 0; i < int'(RL); i++) dl_v[i] <= 1'b0;
    end else begin
      dl_v[0] <= k4_v;
      for (int i = 1; i < int'(RL); i++) dl_v[i] <= dl_v[i-1];
    end
  end

  // Out
  always_ff @(posedge clk) begin
    if (!rst_n) out_valid <= 1'b0;
    else        out_valid <= dl_v[RL-1];
  end
  always_ff @(posedge clk) begin
    scale <= dl_dz[RL-1] ? 32'd0 : rc_y;
    h16   <= dl_h16[RL-1];
    zero  <= dl_zero[RL-1];
  end

endmodule
