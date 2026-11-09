// tsr_fp_mul.sv — pipelined f32 multiplier with the spec's fp rules.
//
// Spec: docs/rtl-core.md §2, block (c).
//
// Contract (bit-exact with tsr_ref.c tsr_fmul):
//   y = fin(DAZ(a) * DAZ(b)): inputs with exponent 0 are signed zeros; the product is rounded
//   RNE in the binary32 exponent range (gradual underflow) and a subnormal rounded result becomes
//   a signed zero (|exact| < 2^-126 - 2^-150 -> +-0; e.g. 0x3F7FFFFF * 0x00800000 = 0x00800000);
//   any NaN result is 0x7FC00000; Inf * 0 = NaN; Inf propagates with the XOR sign; zero results
//   carry the XOR sign.
// Timing: fixed latency LAT cycles (LAT >= 1). The pipeline advances only when ce = 1: every
//   register (valid and data) updates only at edges with ce = 1, in_valid/a/b are sampled only
//   at such edges, and out_valid/y are registers. A result taken at advancing edge n is on
//   out_valid/y from advancing edge n + LAT - 1 until the next advancing edge (a consumer that
//   is stalled by the same ce takes it at that edge). No backpressure beyond ce. Reset clears
//   the valid pipeline only (data registers have no reset). No combinational input->output path.
//   Consumers take the latency from LAT (pass it explicitly); do not assume the default.
// Pipeline (stage registers in order; LAT selects which exist):
//   [IN]    raw operands a, b, in_valid                              LAT >= 2
//   stage 1 24x24 mantissa product (`*`, inferred: 2 x DSP48E2 on UltraScale+, no primitives),
//           in parallel the operand classification and the result exponent for both product
//           binades (side band, fabric)
//   [MUL]   product + side band: 1 register at LAT 3 (DSP P register), 2 at LAT 4 and 5 (DSP
//           M, P), LAT - 3 above
//   stage 2 normalise (binade select by p[47]), RNE increment bit, and the special / FTZ /
//           overflow result folded into {sign, magnitude, inc} (inc = 0 for special results)
//   [NORM]  {sign, magnitude, inc}                                   LAT >= 5
//   stage 3 y = {sign, magnitude + inc}: a single 31-bit increment
//   [OUT]   y, out_valid                                             always
//   No logic in front of the input flops ([IN] holds the raw operands). Stage 1 is <= 4 LUT
//   levels, stages 2 + 3 together <= 4 LUT levels + the 31-bit carry chain (4 CARRY8), so every
//   LAT >= 3 is within the ~6-level budget of docs/rtl-core.md §1. Vivado 2024.2 out of context,
//   xcu200-fsgd2104-2-e, 3.333 ns, I/O delay 1.6 ns (tb/fp/viv/fp_ooc.tcl; tb/fp/test_fp.py
//   test_fp_ooc_timing):
//     LAT 3: WNS +0.79 ns, set by the DSP cascade without M register ([IN] flop -> 2 x DSP48E2
//            -> P; synthesis keeps the operand register in fabric since stage 1 reads it too);
//            P -> y (stages 2 + 3) +1.49 ns
//     LAT 4: WNS +1.16 ns (the default: the 300 MHz U200 configuration), worst path P -> y
//            (3 LUT + 4 CARRY8); max 4 LUTs on any path
//     LAT 5: WNS +1.33 ns ([NORM] splits stages 2 and 3)
//   LAT 1 and 2 are for low clock rates only.
// Rounding (equivalent to tsr_fp_pkg::tsr_f32_round_pack, which tsr_fp_div uses): for the
//   truncated significand 1.s (s = 23 fraction bits), guard g, sticky st and biased exponent e,
//   1 <= e <= 254: y = {sign, {e[7:0], s} + inc} with inc = g & (st | s[0]) (RNE) — a carry out
//                  of the fraction is the binade carry into the exponent, and 254 + carry gives
//                  0x7F800000 = Inf, the RNE overflow;
//   e > 254       : Inf;   e == 0: 2^-126 iff s is all ones, else 0 (FTZ rule, see
//                  tsr_fp_pkg);   e < 0: 0   (all with the XOR sign; inc = 0).
//   e is ea + eb - 127 for a product in [1, 2) and one more for [2, 4); both candidates are
//   decided in stage 1, stage 2 only selects by the product's top bit.

module tsr_fp_mul #(
  parameter int unsigned LAT = 4
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        ce,
  input  logic        in_valid,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        out_valid,
  output logic [31:0] y
);
  import tsr_fp_pkg::*;

  localparam int unsigned N_IN   = (LAT >= 2) ? 1 : 0;
  localparam int unsigned N_NORM = (LAT >= 5) ? 1 : 0;
  localparam int unsigned N_MUL  = LAT - 1 - N_IN - N_NORM;   // LAT 1.. : 0, 0, 1, 2, 2, 3, ...

  // Side band of stage 1, indexed [c] by the product binade: c = 0 for a product in [1, 2)
  // (biased exponent ea + eb - 127), c = 1 for [2, 4) (one more).
  typedef struct packed {
    logic            sign;   // XOR of the operand signs
    logic            nan;    // result is the canonical NaN
    logic [1:0]      spc;    // [c]: special result (NaN, Inf, zero, FTZ, 2^-126, overflow)
    logic [1:0]      ff;     // [c]: special result with exponent 0xFF (NaN, Inf, overflow)
    logic [1:0]      ez;     // [c]: rounded product with biased exponent 0 (2^-126 or zero)
    logic [1:0][7:0] e8;     // [c]: biased exponent of a rounded product (valid if !spc[c])
  } side_t;

  // Stage 2 result: y = {sign, mag + inc}.
  typedef struct packed {
    logic        sign;
    logic [30:0] mag;        // exponent and fraction before the RNE increment
    logic        inc;        // RNE increment (0 for special results)
  } norm_t;

  // ---------------------------------------------------------------------------------------------
  // [IN] register: raw operands.
  // ---------------------------------------------------------------------------------------------
  logic        v1;
  logic [31:0] a1, b1;

  if (N_IN != 0) begin : g_in
    logic        v_q;
    logic [31:0] a_q, b_q;
    always_ff @(posedge clk) begin
      if (!rst_n)  v_q <= 1'b0;
      else if (ce) v_q <= in_valid;
    end
    always_ff @(posedge clk) begin
      if (ce) begin
        a_q <= a;
        b_q <= b;
      end
    end
    assign v1 = v_q;
    assign a1 = a_q;
    assign b1 = b_q;
  end else begin : g_no_in
    assign v1 = in_valid;
    assign a1 = a;
    assign b1 = b;
  end

  // ---------------------------------------------------------------------------------------------
  // Stage 1: mantissa product; classification and exponent candidates (side band).
  // The hidden bit is a constant 1: zero / subnormal / Inf / NaN operands give a special
  // result, their mantissa product is never used.
  // ---------------------------------------------------------------------------------------------
  side_t       sd1;
  logic [47:0] p1;

  assign p1 = {1'b1, a1[22:0]} * {1'b1, b1[22:0]};   // in [2^46, 2^48)

  always_comb begin
    logic              za, zb, ia, ib, cnan, cinf, czero, fix;
    logic signed [9:0] ec;
    za    = (a1[30:23] == 8'd0);
    zb    = (b1[30:23] == 8'd0);
    ia    = (a1[30:23] == 8'hFF) && (a1[22:0] == 23'd0);
    ib    = (b1[30:23] == 8'hFF) && (b1[22:0] == 23'd0);
    cnan  = tsr_f32_is_nan(a1) || tsr_f32_is_nan(b1) || (ia && zb) || (ib && za);
    cinf  = !cnan && (ia || ib);
    czero = !cnan && (za || zb);   // (an Inf with a zero is NaN)
    fix   = cnan || cinf || czero;
    sd1.sign = a1[31] ^ b1[31];
    sd1.nan  = cnan;
    for (int c = 0; c < 2; c++) begin
      ec = $signed({2'b00, a1[30:23]}) + $signed({2'b00, b1[30:23]})
         - ((c == 0) ? 10'sd127 : 10'sd126);            // in [-125, 382] for finite operands
      sd1.spc[c] = fix || (ec > 10'sd254) || (ec <= 10'sd0);
      sd1.ff[c]  = cnan || cinf || (!fix && (ec > 10'sd254));
      sd1.ez[c]  = !fix && (ec == 10'sd0);
      sd1.e8[c]  = ec[7:0];
    end
  end

  // ---------------------------------------------------------------------------------------------
  // [MUL] registers: a plain register chain after `*` (synthesis absorbs the product registers
  // into the DSP M/P registers); the side band travels alongside.
  // ---------------------------------------------------------------------------------------------
  logic        v2;
  side_t       sd2;
  logic [47:0] p2;

  if (N_MUL != 0) begin : g_mul
    logic        v_q  [N_MUL];
    side_t       sd_q [N_MUL];
    logic [47:0] p_q  [N_MUL];
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        for (int i = 0; i < int'(N_MUL); i++) v_q[i] <= 1'b0;
      end else if (ce) begin
        v_q[0] <= v1;
        for (int i = 1; i < int'(N_MUL); i++) v_q[i] <= v_q[i-1];
      end
    end
    always_ff @(posedge clk) begin
      if (ce) begin
        sd_q[0] <= sd1;
        p_q[0]  <= p1;
        for (int i = 1; i < int'(N_MUL); i++) begin
          sd_q[i] <= sd_q[i-1];
          p_q[i]  <= p_q[i-1];
        end
      end
    end
    assign v2  = v_q[N_MUL-1];
    assign sd2 = sd_q[N_MUL-1];
    assign p2  = p_q[N_MUL-1];
  end else begin : g_no_mul
    assign v2  = v1;
    assign sd2 = sd1;
    assign p2  = p1;
  end

  // ---------------------------------------------------------------------------------------------
  // Stage 2: normalise, RNE increment bit, special results.
  // ---------------------------------------------------------------------------------------------
  norm_t n2;

  always_comb begin
    logic        sel, g, st, ones;
    logic [22:0] s23;
    sel  = p2[47];                                   // product in [2, 4)
    s23  = sel ? p2[46:24] : p2[45:23];
    g    = sel ? p2[23] : p2[22];
    st   = (p2[21:0] != 22'd0) || (sel && p2[22]);
    ones = &s23;
    n2.sign = sd2.sign && !sd2.nan;
    if (sd2.spc[sel]) begin
      n2.mag = {sd2.ff[sel] ? 8'hFF : {7'd0, sd2.ez[sel] && ones}, sd2.nan, 22'd0};
      n2.inc = 1'b0;
    end else begin
      n2.mag = {sd2.e8[sel], s23};
      n2.inc = g && (st || s23[0]);
    end
  end

  // [NORM] register (LAT >= 5)
  logic  v3;
  norm_t n3;

  if (N_NORM != 0) begin : g_norm
    logic  v_q;
    norm_t n_q;
    always_ff @(posedge clk) begin
      if (!rst_n)  v_q <= 1'b0;
      else if (ce) v_q <= v2;
    end
    always_ff @(posedge clk) begin
      if (ce) n_q <= n2;
    end
    assign v3 = v_q;
    assign n3 = n_q;
  end else begin : g_no_norm
    assign v3 = v2;
    assign n3 = n2;
  end

  // ---------------------------------------------------------------------------------------------
  // Stage 3: RNE increment, and the [OUT] register.
  // ---------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n)  out_valid <= 1'b0;
    else if (ce) out_valid <= v3;
  end
  always_ff @(posedge clk) begin
    if (ce) y <= {n3.sign, n3.mag + 31'(n3.inc)};
  end

  // Elaboration check.
  if (LAT < 1) begin : g_bad_lat
    $fatal(1, "tsr_fp_mul: LAT must be >= 1 (got %0d)", LAT);
  end

endmodule
