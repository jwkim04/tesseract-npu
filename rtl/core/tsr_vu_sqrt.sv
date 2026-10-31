// tsr_vu_sqrt.sv — SQ(x): correctly rounded f32 square root with the spec's fp rules, fully pipelined.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §6.4; RED post RMS §6.2, SOP SQRT / RSQRT §5.8).
// Contract: y = tsr_fsqrt(x) of sw/sim/tsr_ref.c bit for bit (= fin(sqrtf(DAZ(x)))): ±0 -> ±0 (a
// subnormal input is a signed zero), +inf -> +inf, NaN or any x < 0 -> 0x7FC00000. A square root of a
// normal f32 is never subnormal and never overflows. Fixed latency TSR_VU_SQRT_LAT, one operation per
// cycle, no clock enable. Verified over every significand at both exponent parities plus the edges
// (tb/vu/test_vu_arith.py).
//
// Algorithm: x = 2^(E-127)·M/2^23, M = 2^23 | f. X = M << (E even) puts an odd exponent into the
// radicand; s = floor(sqrt(X·2^25)) in [2^24, 2^25) by restoring digit recurrence (25 iterations,
// one per stage: rem = 4·rem + 2 bits of the radicand; trial = 4q + 1; rem >= trial -> subtract, q bit
// 1). The significand is (s >> 1) + s[0]: sqrt never lies on a rounding midpoint (an odd square cannot
// equal a 24-bit significand times a power of two), and s <= 2^25 - 2 so the increment never carries
// out. Result exponent (E + 127) >> 1.

module tsr_vu_sqrt (
  input  logic        clk,
  input  logic [31:0] x,
  output logic [31:0] y
);
  import tsr_vu_pkg::*;

  localparam int unsigned IT  = 25;
  localparam int unsigned LAT = TSR_VU_SQRT_LAT;
  if (LAT != IT + 3) begin : g_chk_lat
    $fatal(1, "tsr_vu_sqrt: TSR_VU_SQRT_LAT must be IT + 3");
  end

  // S0: input register; S1: classification and radicand
  logic [31:0] x0_q;
  always_ff @(posedge clk) x0_q <= x;

  logic        c_spec;
  logic [31:0] c_val;
  always_comb begin
    logic [7:0] e;
    e = x0_q[30:23];
    c_spec = 1'b1;
    if (e == 8'hFF && x0_q[22:0] != 23'd0)  c_val = TSR_VU_NAN;          // NaN
    else if (e == 8'd0)                     c_val = {x0_q[31], 31'd0};    // ±0 (DAZ)
    else if (x0_q[31])                      c_val = TSR_VU_NAN;          // x < 0, -inf
    else if (e == 8'hFF)                    c_val = TSR_VU_INF;          // +inf
    else begin
      c_spec = 1'b0;
      c_val  = 32'd0;
    end
  end

  logic [49:0] n_q   [IT + 1];        // radicand bits not consumed yet (left-aligned)
  logic [26:0] rem_q [IT + 1];
  logic [24:0] q_q   [IT + 1];
  logic        spec_q [IT + 1];
  logic [31:0] sval_q [IT + 1];
  logic [7:0]  ex_q   [IT + 1];

  // all stages in one block (one procedural driver per array, as synthesis needs)
  always_ff @(posedge clk) begin
    logic [24:0] xr;
    logic [26:0] r, t;
    xr = x0_q[23] ? {1'b0, 1'b1, x0_q[22:0]} : {1'b1, x0_q[22:0], 1'b0};   // E even (exponent odd): 2M
    n_q[0]    <= {xr, 25'd0};
    rem_q[0]  <= '0;
    q_q[0]    <= '0;
    spec_q[0] <= c_spec;
    sval_q[0] <= c_val;
    ex_q[0]   <= 8'((9'(x0_q[30:23]) + 9'd127) >> 1);
    for (int i = 0; i < IT; i++) begin
      r = {rem_q[i][24:0], n_q[i][49:48]};
      t = {q_q[i], 2'b01};
      if (r >= t) begin
        rem_q[i+1] <= r - t;
        q_q[i+1]   <= {q_q[i][23:0], 1'b1};
      end else begin
        rem_q[i+1] <= r;
        q_q[i+1]   <= {q_q[i][23:0], 1'b0};
      end
      n_q[i+1]    <= {n_q[i][47:0], 2'b00};
      spec_q[i+1] <= spec_q[i];
      sval_q[i+1] <= sval_q[i];
      ex_q[i+1]   <= ex_q[i];
    end
  end

  // pack: significand (s >> 1) + s[0]
  wire [24:0] sig = {1'b0, q_q[IT][24:1]} + 25'(q_q[IT][0]);
  always_ff @(posedge clk) begin
    y <= spec_q[IT] ? sval_q[IT] : {1'b0, ex_q[IT], sig[22:0]};
  end

  wire unused_ok = &{1'b0, rem_q[IT], n_q[IT], sig[24:23]};

endmodule
