// tsr_vu_rcp.sv — RC(x): correctly rounded f32 reciprocal with the spec's fp rules, fully pipelined.
//
// Spec: docs/design/02-vector-unit.md "VU" §6.4, §7.3 "RC"; VU-5.
// Contract: y = tsr_frecip(x) of sw/sim/tsr_ref.c bit for bit (= fin(1 / DAZ(x))): ±0 -> ±inf,
// ±inf -> ±0, NaN -> 0x7FC00000, a result below 2^-126 after gradual-underflow rounding -> ±0 (for
// x = 2^126·m, m > 1, and every x >= 2^127). Fixed latency TSR_VU_RCP_LAT, one operation per cycle,
// no clock enable. Verified exhaustively over all 2^23 mantissas at several exponents plus the
// exponent edges (tb/vu/test_vu_arith.py, Verilator C++ against libtsr_ref).
//
// Algorithm (the design is free, VU §7.3; this one uses no table RAM): for x = 2^(E-127)·M/2^23 with
// M = 2^23 | f, f != 0, the significand of 1/x is S = round(2^47 / M) in (2^23, 2^24) with exponent
// 253 - E (f = 0: 1/x = 2^(127-E) exactly, exponent 254 - E). There are no ties (M·(2S+1) = 2^48 has
// no solution for M < 2^24 other than a power of two).
//   seed  Y0 = round(2^19 / (513 + 2k)), k = f[22:15]: 1/m at the segment center, Q0.10
//   NR 1  R0 = 2^33 - M·Y0 (Q1.33, signed), Y1 = Y0·2^12 + (Y0·R0) >> 21 (Q0.22, floor shift)
//   NR 2  R1 = 2^45 - M·Y1 (Q1.45, > 0), Y2 = Y1·2^8 + (Y1·R1) >> 37 (Q0.30)
//   cand  Qc = (Y2 + 32) >> 6: |Qc - 2^47/M| < 1 for every f (checked exhaustively)
//   fix   P = M·Qc; D+ = 2^48 - 2P - M, D- = 2^48 - 2P + M: S = Qc + 1 if D+ > 0, Qc - 1 if D- < 0,
//         else Qc (one exact 24 × 25-bit correction product, VU §7.3).
// Products: 24×10, 10×27, 24×23, 23×28, 24×25 (≈ 8 DSP48E2), each from registers and followed by two
// or three registers; every add is one stage.

module tsr_vu_rcp (
  input  logic        clk,
  input  logic [31:0] x,
  output logic [31:0] y
);
  import tsr_vu_pkg::*;

  localparam int unsigned LAT = TSR_VU_RCP_LAT;

  // seed ROM (256 x 10 bits), evaluated at elaboration: a run-time division here would be a 60-level
  // combinational divider
  function automatic logic [255:0][9:0] seed_rom();
    logic [255:0][9:0] t;
    for (int k = 0; k < 256; k++) t[k] = 10'((((32'd1 << 20) / (32'd513 + 32'(k) * 32'd2)) + 32'd1) >> 1);
    return t;
  endfunction
  localparam logic [255:0][9:0] SEED = seed_rom();

  // R0: input register
  logic [31:0] x0_q;
  always_ff @(posedge clk) x0_q <= x;

  // R1: classification, seed, result exponent, specials
  logic        r1_spec;
  logic [31:0] r1_val;
  logic [7:0]  r1_exp;
  always_comb begin
    logic [7:0] e;
    logic       fz;
    e  = x0_q[30:23];
    fz = (x0_q[22:0] == 23'd0);
    r1_exp  = fz ? (8'd254 - e) : (8'd253 - e);
    r1_spec = 1'b1;
    if (e == 8'hFF && !fz)                 r1_val = TSR_VU_NAN;                    // NaN
    else if (e == 8'hFF)                   r1_val = {x0_q[31], 31'd0};             // ±inf -> ±0
    else if (e == 8'd0)                    r1_val = {x0_q[31], 8'hFF, 23'd0};      // DAZ ±0 -> ±inf
    else if (fz ? (e >= 8'd254) : (e >= 8'd253)) r1_val = {x0_q[31], 31'd0};       // FTZ
    else begin
      r1_spec = 1'b0;
      r1_val  = 32'd0;
    end
  end

  logic        s1_spec_q, s1_fz_q, s1_sign_q;
  logic [31:0] s1_val_q;
  logic [7:0]  s1_exp_q;
  logic [23:0] s1_m_q;
  logic [9:0]  s1_y0_q;
  always_ff @(posedge clk) begin
    s1_spec_q <= r1_spec;
    s1_val_q  <= r1_val;
    s1_fz_q   <= (x0_q[22:0] == 23'd0);
    s1_sign_q <= x0_q[31];
    s1_exp_q  <= r1_exp;
    s1_m_q    <= {1'b1, x0_q[22:0]};
    s1_y0_q   <= SEED[x0_q[22:15]];
  end

  // M delayed to every product that needs it
  localparam int unsigned MD = 17;
  logic [23:0] m_d [MD];
  always_ff @(posedge clk) begin
    m_d[0] <= s1_m_q;
    for (int i = 1; i < MD; i++) m_d[i] <= m_d[i-1];
  end

  // NR 1: P0 = M·Y0 (2 stages), R0 (1), T0 = Y0·R0 (2), Y1 (1)
  logic [33:0] p0a_q, p0b_q;
  logic [9:0]  y0_d [5];
  always_ff @(posedge clk) begin
    p0a_q <= 34'(s1_m_q) * 34'(s1_y0_q);
    p0b_q <= p0a_q;
    y0_d[0] <= s1_y0_q;
    for (int i = 1; i < 5; i++) y0_d[i] <= y0_d[i-1];
  end
  logic signed [26:0] r0_q;
  always_ff @(posedge clk) r0_q <= 27'(35'sd8589934592 - $signed({1'b0, p0b_q}));   // 2^33 - P0
  logic signed [37:0] t0a_q, t0b_q;
  always_ff @(posedge clk) begin
    t0a_q <= $signed({28'd0, y0_d[2]}) * 38'(r0_q);
    t0b_q <= t0a_q;
  end
  logic [22:0] y1_q;
  always_ff @(posedge clk) y1_q <= 23'($signed({1'b0, y0_d[4], 12'd0}) + (t0b_q >>> 21));

  // NR 2: P1 = M·Y1 (3), R1 (1), T1 = Y1·R1 (3), Qc (1)
  logic [46:0] p1a_q, p1b_q, p1c_q;
  logic [22:0] y1_d [7];
  always_ff @(posedge clk) begin
    p1a_q <= 47'(m_d[5]) * 47'(y1_q);
    p1b_q <= p1a_q;
    p1c_q <= p1b_q;
    y1_d[0] <= y1_q;
    for (int i = 1; i < 7; i++) y1_d[i] <= y1_d[i-1];
  end
  logic [27:0] r1_q;
  always_ff @(posedge clk) r1_q <= 28'(47'h2000_0000_0000 - p1c_q);                     // 2^45 - P1
  logic [50:0] t1a_q, t1b_q, t1c_q;
  always_ff @(posedge clk) begin
    t1a_q <= 51'(y1_d[3]) * 51'(r1_q);
    t1b_q <= t1a_q;
    t1c_q <= t1b_q;
  end
  logic [24:0] qc_q;
  always_ff @(posedge clk) qc_q <= 25'(({8'd0, y1_d[6], 8'd0} + 39'(t1c_q >> 37) + 39'd32) >> 6);

  // Correction: P = M·Qc (3), decision (1)
  logic [48:0] pa_q, pb_q, pc_q;
  logic [24:0] qc_d [3];
  always_ff @(posedge clk) begin
    pa_q <= 49'(m_d[13]) * 49'(qc_q);
    pb_q <= pa_q;
    pc_q <= pb_q;
    qc_d[0] <= qc_q;
    for (int i = 1; i < 3; i++) qc_d[i] <= qc_d[i-1];
  end
  logic        up_q, dn_q;
  logic [24:0] qcs_q;
  always_ff @(posedge clk) begin
    logic signed [50:0] dp, dm;
    dp = 51'sh1_0000_0000_0000 - $signed({1'b0, pc_q, 1'b0}) - $signed({27'd0, m_d[MD-1]});
    dm = 51'sh1_0000_0000_0000 - $signed({1'b0, pc_q, 1'b0}) + $signed({27'd0, m_d[MD-1]});
    up_q  <= (dp > 51'sd0);
    dn_q  <= (dm < 51'sd0);
    qcs_q <= qc_d[2];
  end

  // side band: specials, f = 0, sign, exponent (R1 register -> the pack stage)
  localparam int unsigned SD = LAT - 3;
  logic        spec_d [SD], fz_d [SD], sign_d [SD];
  logic [31:0] sval_d [SD];
  logic [7:0]  exp_d [SD];
  always_ff @(posedge clk) begin
    spec_d[0] <= s1_spec_q;
    sval_d[0] <= s1_val_q;
    fz_d[0]   <= s1_fz_q;
    sign_d[0] <= s1_sign_q;
    exp_d[0]  <= s1_exp_q;
    for (int i = 1; i < SD; i++) begin
      spec_d[i] <= spec_d[i-1];
      sval_d[i] <= sval_d[i-1];
      fz_d[i]   <= fz_d[i-1];
      sign_d[i] <= sign_d[i-1];
      exp_d[i]  <= exp_d[i-1];
    end
  end

  // pack (S is in (2^23, 2^24): its bit 23 is the hidden one)
  wire [24:0] sig = up_q ? qcs_q + 25'd1 : dn_q ? qcs_q - 25'd1 : qcs_q;
  always_ff @(posedge clk) begin
    if (spec_d[SD-1])    y <= sval_d[SD-1];
    else if (fz_d[SD-1]) y <= {sign_d[SD-1], exp_d[SD-1], 23'd0};
    else                 y <= {sign_d[SD-1], exp_d[SD-1], sig[22:0]};
  end

  wire unused_ok = &{1'b0, t1c_q[36:0], t1c_q[50:50], p0b_q[33:33], sig[24:23]};

endmodule
