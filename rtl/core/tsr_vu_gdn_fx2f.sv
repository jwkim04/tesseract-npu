// tsr_vu_gdn_fx2f.sv — fixed-point sum to f32 conversion of the GDN dot products
// (docs/design/02-vector-unit.md §7.12.3; bit-exact with tsr_ref_gdn_fx2f in sw/sim/tsr_ref_gdn.c).
//
// Input: I = in_hi * 2^13 + in_lo (two's complement, |I| < 2^59) and the scale x: the value is
// I * 2^x. Output: the f32 bits of I * 2^x, rounded to nearest even, overflow to Inf, a result below
// 2^-126 - 2^-150 flushed to a signed zero (the spec's FTZ rule), I = 0 gives +0, and in_bad gives
// the canonical NaN 0x7FC00000. in_hi / in_lo are the sums of the DSP chains for the high and the
// low table pieces (tsr_vu_gdn_dot). Fully pipelined, one conversion per cycle, fixed latency
// LAT_CVT = 8 (the tag travels along). No reset of data registers; the valid pipeline is reset.
//
// Stages: S0 input registers; S1 I = hi * 2^13 + lo; S2 sign and magnitude; S3 leading zeros and
// the biased exponent of the leading bit E0 = 186 - lz + x; S4 / S5 left shift by lz (16s, then
// 1s to 15); S6 rounding (guard, sticky, increment) and classification; S7 packing. The subnormal
// range needs no shifter: below 2^-126 only a value that reaches 2^-126 - 2^-150 survives, and
// that is E0 = 0 with all 24 kept bits set (the rule of tsr_fp_pkg, e = 0 case).

module tsr_vu_gdn_fx2f #(
  parameter int unsigned TW = 4
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                in_valid,
  input  logic [TW-1:0]       in_tag,
  input  logic signed [47:0]  in_hi,
  input  logic signed [47:0]  in_lo,
  input  logic signed [9:0]   in_x,
  input  logic                in_bad,
  output logic                out_valid,
  output logic [TW-1:0]       out_tag,
  output logic [31:0]         out_y
);

  localparam int unsigned MW = 60;       // magnitude width

  // S0 --------------------------------------------------------------------------------------
  logic                v0, bad0;
  logic [TW-1:0]       tag0;
  logic signed [47:0]  hi0, lo0;
  logic signed [9:0]   x0;
  always_ff @(posedge clk) begin
    if (!rst_n) v0 <= 1'b0; else v0 <= in_valid;
  end
  always_ff @(posedge clk) begin
    tag0 <= in_tag; hi0 <= in_hi; lo0 <= in_lo; x0 <= in_x; bad0 <= in_bad;
  end

  // S1 --------------------------------------------------------------------------------------
  logic                v1, bad1;
  logic [TW-1:0]       tag1;
  logic signed [60:0]  sum1;
  logic signed [9:0]   x1;
  always_ff @(posedge clk) begin
    if (!rst_n) v1 <= 1'b0; else v1 <= v0;
  end
  always_ff @(posedge clk) begin
    tag1 <= tag0; x1 <= x0; bad1 <= bad0;
    sum1 <= (61'(hi0) <<< 13) + 61'(lo0);
  end

  // S2 --------------------------------------------------------------------------------------
  logic                v2, bad2, sgn2, z2;
  logic [TW-1:0]       tag2;
  logic [MW-1:0]       mag2;
  logic signed [9:0]   x2;
  always_ff @(posedge clk) begin
    if (!rst_n) v2 <= 1'b0; else v2 <= v1;
  end
  always_ff @(posedge clk) begin
    tag2 <= tag1; x2 <= x1; bad2 <= bad1;
    sgn2 <= sum1[60];
    z2   <= (sum1 == 61'sd0);
    mag2 <= sum1[60] ? (~sum1[MW-1:0] + MW'(1)) : sum1[MW-1:0];
  end

  // S3 --------------------------------------------------------------------------------------
  logic                v3, bad3, sgn3, z3;
  logic [TW-1:0]       tag3;
  logic [MW-1:0]       mag3;
  logic [5:0]          lz3;
  logic signed [11:0]  e0_3;
  logic [5:0]          lz_c;
  always_comb begin
    lz_c = 6'(MW);
    for (int i = 0; i < MW; i++) begin
      if (mag2[i]) lz_c = 6'(MW - 1 - i);          // the last (highest) set bit wins
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) v3 <= 1'b0; else v3 <= v2;
  end
  always_ff @(posedge clk) begin
    tag3 <= tag2; bad3 <= bad2; sgn3 <= sgn2; z3 <= z2;
    mag3 <= mag2;
    lz3  <= lz_c;
    e0_3 <= 12'sd186 - 12'(signed'({1'b0, lz_c})) + 12'(x2);
  end

  // S4 --------------------------------------------------------------------------------------
  logic                v4, bad4, sgn4, z4;
  logic [TW-1:0]       tag4;
  logic [MW-1:0]       n4;
  logic [3:0]          lz4;
  logic signed [11:0]  e0_4;
  always_ff @(posedge clk) begin
    if (!rst_n) v4 <= 1'b0; else v4 <= v3;
  end
  always_ff @(posedge clk) begin
    tag4 <= tag3; bad4 <= bad3; sgn4 <= sgn3; z4 <= z3; e0_4 <= e0_3; lz4 <= lz3[3:0];
    n4 <= mag3 << {lz3[5:4], 4'b0000};
  end

  // S5 --------------------------------------------------------------------------------------
  logic                v5, bad5, sgn5, z5;
  logic [TW-1:0]       tag5;
  logic [MW-1:0]       nm5;
  logic signed [11:0]  e0_5;
  always_ff @(posedge clk) begin
    if (!rst_n) v5 <= 1'b0; else v5 <= v4;
  end
  always_ff @(posedge clk) begin
    tag5 <= tag4; bad5 <= bad4; sgn5 <= sgn4; z5 <= z4; e0_5 <= e0_4;
    nm5 <= n4 << lz4;
  end

  // S6 --------------------------------------------------------------------------------------
  logic                v6, bad6, sgn6, z6, carry6, allones6;
  logic [TW-1:0]       tag6;
  logic [22:0]         man6;
  logic signed [11:0]  e0_6;
  logic [23:0]         q_c;
  logic                inc_c;
  logic [23:0]         f1_c;          // the 23 fraction bits plus the increment; bit 23 is the carry
  assign q_c   = nm5[MW-1 -: 24];
  assign inc_c = nm5[MW-25] & ((|nm5[MW-26:0]) | q_c[0]);
  assign f1_c  = {1'b0, q_c[22:0]} + 24'(inc_c);
  always_ff @(posedge clk) begin
    if (!rst_n) v6 <= 1'b0; else v6 <= v5;
  end
  always_ff @(posedge clk) begin
    tag6 <= tag5; bad6 <= bad5; sgn6 <= sgn5; z6 <= z5; e0_6 <= e0_5;
    man6     <= f1_c[22:0];
    carry6   <= f1_c[23];
    allones6 <= &q_c;
  end

  // S7 --------------------------------------------------------------------------------------
  logic signed [11:0] e1_c;
  logic [31:0]        y_c;
  assign e1_c = e0_6 + 12'(signed'({11'b0, carry6}));
  always_comb begin
    if (bad6) begin
      y_c = 32'h7FC0_0000;
    end else if (z6) begin
      y_c = 32'h0;
    end else if (e0_6 >= 12'sd1) begin
      if (e1_c >= 12'sd255) y_c = {sgn6, 8'hFF, 23'h0};
      else                  y_c = {sgn6, e1_c[7:0], man6};
    end else if (e0_6 == 12'sd0 && allones6) begin
      y_c = {sgn6, 8'h01, 23'h0};
    end else begin
      y_c = {sgn6, 31'h0};
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) out_valid <= 1'b0; else out_valid <= v6;
  end
  always_ff @(posedge clk) begin
    out_tag <= tag6;
    out_y   <= y_c;
  end

endmodule
