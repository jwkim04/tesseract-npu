// tsr_fp_pkg.sv — combinational f32 / fp16 / int8 helpers with the spec's fp rules.
//
// Owner: fp agent (docs/rtl-core.md §2, block (c); contract §8.2).
//
// Normative definition: sw/sim/tsr_ref.c (bit-exact) — the RTL matches it for every input
// (tb/fp: exhaustive over all 2^32 / 2^16 inputs for the conversions, directed + random sweeps
// for the arithmetic helpers).
//   tsr_f32_daz        = tsr_f32_daz / daz_bits()      (subnormal or zero -> signed zero)
//   tsr_f32_is_nan     = exponent all ones, mantissa != 0
//   tsr_f32_amag       = DAZ magnitude bits used by the QACT amax / max search (amax_candidate())
//   tsr_f32_amax_*     = the amax / max search itself (amax_candidate() + "a > amax" scan order)
//   tsr_fp16_to_f32    = tsr_fp16_to_f32 (exact; NaN keeps sign/payload, quiet bit set)
//   tsr_f32_to_fp16    = tsr_f32_to_fp16 (RNE, overflow -> Inf, NaN -> sign | 0x7E00,
//                        |f| < 2^-25 or f32 subnormal -> signed 0)
//   tsr_f32_to_q8      = tsr_q8_round_away (half_even = 0) / tsr_q8_round_even (half_even = 1):
//                        NaN -> 0, rounded value saturated to [-128, 127]
//   tsr_i32_to_f32     = tsr_i32_to_f32 (RNE; needed from M3)
//   tsr_f32_round_pack = the result rounding of tsr_fmul / tsr_fdiv (RNE + the spec's FTZ rule
//                        + overflow to Inf); used by tsr_fp_div. tsr_fp_mul implements the same
//                        rule split over two pipeline stages (see its header, "Rounding").
//
// These are pure functions; callers register their inputs/outputs as timing requires. Logic
// depth (UltraScale+, 6-LUT, estimates): daz / is_nan / amag / amax_cand / amax_merge /
// fp16_to_f32 are 1-3 levels; f32_to_q8 and f32_to_fp16 are a 9..12-way shift + sticky + small
// increment (~5-7 levels: meant as one 300 MHz stage with registered input and output);
// i32_to_f32 (32-bit CLZ + shift + 24-bit increment) and round_pack are ~7-9 levels (round_pack
// is the last stage of tsr_fp_div).
//
// FTZ rule (spec, Implementation clarifications, "FTZ decision point"): the result is RNE in the binary32 exponent range
// (gradual underflow) and becomes a signed zero iff that rounded result is subnormal, i.e. iff
// |exact| < 2^-126 - 2^-150. No subnormal is ever produced, so no denormalisation shifter is
// needed: for a truncated significand s (1.xxx, 24 bits) with biased exponent e,
//   e >= 1 : normal RNE (may carry into the next binade / overflow to Inf);
//   e == 0 : exact = 1.s.. * 2^-127 >= 2^-126 - 2^-150  <=>  s == 0xFFFFFF  -> +-2^-126,
//            otherwise +-0 (the gradual-underflow result would be subnormal);
//   e <= -1: exact < 2^-127 -> +-0.

package tsr_fp_pkg;

  // Shared constants: not every user needs every constant (UNUSEDPARAM disabled for them only).
  /* verilator lint_off UNUSEDPARAM */
  localparam logic [31:0] TSR_F32_CANON_NAN = 32'h7FC0_0000;
  localparam logic [31:0] TSR_F32_ONE       = 32'h3F80_0000;
  localparam logic [31:0] TSR_F32_127       = 32'h42FE_0000;  // 127.0f
  localparam logic [31:0] TSR_F32_M127      = 32'hC2FE_0000;  // -127.0f
  localparam logic [30:0] TSR_F32_INF_MAG   = 31'h7F80_0000;  // |Inf|: largest non-NaN magnitude
  /* verilator lint_on UNUSEDPARAM */

  // -------------------------------------------------------------------------------------------
  // Classification / DAZ
  // -------------------------------------------------------------------------------------------

  // Subnormal (or zero) -> signed zero; everything else unchanged.
  function automatic logic [31:0] tsr_f32_daz(logic [31:0] x);
    return (x[30:23] == 8'd0) ? {x[31], 31'd0} : x;
  endfunction

  function automatic logic tsr_f32_is_nan(logic [31:0] x);
    return (x & 32'h7FFF_FFFF) > 32'h7F80_0000;
  endfunction

  // Magnitude bits of DAZ(x). For non-NaN x an unsigned compare of these equals the float
  // compare of |x| (Inf = 0x7F800000 is the largest). Callers exclude NaN with tsr_f32_is_nan.
  function automatic logic [30:0] tsr_f32_amag(logic [31:0] x);
    return 31'(tsr_f32_daz(x) & 32'h7FFF_FFFF);
  endfunction

  // -------------------------------------------------------------------------------------------
  // QACT amax / max search (tsr_ref.c tsr_qact: amax_candidate() and the scan
  //   "if (candidate && a > amax) { amax = a; max = DAZ(x); }" starting from amax = 0, max = +0).
  //
  // A candidate is {mag, neg}: mag = |DAZ(x)| bits, neg = sign of DAZ(x). NaN does not take part:
  // it becomes {0, 0}, which can never win (only a strictly larger magnitude replaces the
  // current one, and the scan starts at magnitude 0). A zero magnitude also carries neg = 0, so
  // a block of zeros / NaNs yields amax = 0 and max = +0.0 exactly like the reference.
  // tsr_f32_amax_merge(first, second) keeps `first` unless `second` is strictly larger: it is
  // associative, so any reduction tree that keeps the element order (earlier operand first)
  // gives the reference's sequential result, including the first-occurrence rule for ties.
  //   A32 : amax = result.mag            (as f32 bits: {1'b0, mag})
  //   A256: max  = tsr_f32_amax_val(result) = {neg, mag} (signed DAZ value, +0 if amax == 0)
  // -------------------------------------------------------------------------------------------
  typedef struct packed {
    logic [30:0] mag;   // |DAZ(x)| bits; 0 for NaN, zero and subnormal x
    logic        neg;   // sign of DAZ(x); 0 whenever mag == 0
  } tsr_f32_amax_t;     // 32 bits

  function automatic tsr_f32_amax_t tsr_f32_amax_cand(logic [31:0] x);
    tsr_f32_amax_t c;
    if (tsr_f32_is_nan(x) || x[30:23] == 8'd0) begin
      c.mag = 31'd0;
      c.neg = 1'b0;
    end else begin
      c.mag = x[30:0];
      c.neg = x[31];
    end
    return c;
  endfunction

  // `first` precedes `second` in element order.
  function automatic tsr_f32_amax_t tsr_f32_amax_merge(tsr_f32_amax_t first, tsr_f32_amax_t second);
    return (second.mag > first.mag) ? second : first;
  endfunction

  // Signed DAZ value of the winner (A256 `max`).
  function automatic logic [31:0] tsr_f32_amax_val(tsr_f32_amax_t c);
    return {c.neg, c.mag};
  endfunction

  // -------------------------------------------------------------------------------------------
  // fp16 <-> f32 (conversions, not f32 operations: no DAZ/FTZ/canonical-NaN rule; spec
  // Implementation clarifications, "NaN in fp16 conversion")
  // -------------------------------------------------------------------------------------------

  // Exact fp16 -> f32 (tsr_fp16_to_f32). Subnormal halves m * 2^-24 become normal f32;
  // NaN keeps sign and payload with the quiet bit set; Inf -> Inf.
  function automatic logic [31:0] tsr_fp16_to_f32(logic [15:0] h);
    logic       sign;
    logic [4:0] e;
    logic [9:0] m;
    logic [3:0] lz;      // leading zeros of the 10-bit subnormal mantissa (0..9)
    logic [9:0] mn;      // m << (lz + 1): the bits below the leading one, left-aligned
    sign = h[15];
    e    = h[14:10];
    m    = h[9:0];
    lz   = 4'd9;
    for (int i = 0; i < 10; i++) begin
      if (m[i]) lz = 4'(9 - i);   // the last (highest) set bit wins
    end
    mn = m << (lz + 4'd1);
    if (e == 5'd0) begin
      if (m == 10'd0) return {sign, 31'd0};
      return {sign, 8'(8'd112 - {4'd0, lz}), mn, 13'd0};   // exponent 103 + msb index
    end
    if (e == 5'h1F) return {sign, 8'hFF, (m != 10'd0), m[8:0], 13'd0};
    return {sign, 8'({3'd0, e} + 8'd112), m, 13'd0};
  endfunction

  // f32 -> fp16, round-to-nearest-even (tsr_f32_to_fp16; = ggml GGML_FP32_TO_FP16).
  //   NaN -> sign | 0x7E00, Inf and |f| >= 65520 -> sign | Inf, |f| < 2^-25 (incl. f32
  //   subnormals) -> signed 0, fp16 subnormals produced exactly (RNE in units of 2^-24).
  // Both the normal and the subnormal range are "(ex << 10) + (sig >> sh)" plus an RNE
  // increment: normal: sh = 13, ex = E - 113 (the hidden bit of sig >> 13 adds the final 1);
  // subnormal: sh = 126 - E in [14, 24], ex = 0. A carry out of the mantissa runs into the
  // exponent (round up to 2^-14 or to Inf) naturally.
  function automatic logic [15:0] tsr_f32_to_fp16(logic [31:0] f);
    logic        sign;
    logic [7:0]  ee;
    logic [23:0] sig;
    logic [4:0]  sh;
    logic [4:0]  ex;
    logic [38:0] sft;    // {sig, 24'b0} >> sh: [38:24] = sig >> sh (< 2^11), [23:0] = bits out
    logic [14:0] base;
    logic        inc;
    sign = f[31];
    ee   = f[30:23];
    sig  = {1'b1, f[22:0]};
    if (ee == 8'hFF) return (f[22:0] != 23'd0) ? {sign, 15'h7E00} : {sign, 15'h7C00};
    if (ee < 8'd102) return {sign, 15'd0};        // E == 0 (zero / subnormal) or |f| < 2^-25
    if (ee > 8'd142) return {sign, 15'h7C00};     // |f| >= 2^16
    if (ee >= 8'd113) begin                        // fp16 normal range (e >= -14)
      sh = 5'd13;
      ex = 5'(ee - 8'd113);
    end else begin                                 // fp16 subnormal range, E in [102, 112]
      sh = 5'(8'd126 - ee);
      ex = 5'd0;
    end
    sft  = 39'({sig, 24'd0} >> sh);
    base = {ex, 10'd0} + sft[38:24];               // sig >> sh < 2^11
    inc  = sft[23] & ((sft[22:0] != 23'd0) | sft[24]);
    return {sign, 15'(base + {14'd0, inc})};
  endfunction

  // -------------------------------------------------------------------------------------------
  // f32 -> int8 (QACT): tsr_q8_round_away (half_even = 0, A32) / tsr_q8_round_even (half_even =
  // 1, A256). NaN -> 0; |x| >= 256 (incl. Inf) saturates; |x| < 0.5 (incl. zero and
  // subnormals) -> 0; otherwise round, negate, saturate to [-128, 127].
  // -------------------------------------------------------------------------------------------
  function automatic logic [7:0] tsr_f32_to_q8(logic [31:0] x, logic half_even);
    logic        neg;
    logic [7:0]  ee;
    logic [3:0]  k;      // ee - 126 in [0, 8]: |x| in [2^(k-1), 2^k)
    logic [31:0] sft;    // sig << k = |x| * 2^24: [31:24] = integer part, [23:0] = fraction
    logic [8:0]  ip;     // rounded magnitude 0..256
    logic        g, st, inc;
    neg = x[31];
    ee  = x[30:23];
    if (tsr_f32_is_nan(x)) return 8'd0;
    if (ee >= 8'd135) return neg ? 8'h80 : 8'h7F;  // |x| >= 256 (incl. Inf)
    if (ee < 8'd126) return 8'd0;                   // |x| < 0.5
    k   = 4'(ee - 8'd126);
    sft = {8'd0, 1'b1, x[22:0]} << k;
    g   = sft[23];
    st  = sft[22:0] != 23'd0;
    inc = half_even ? (g & (st | sft[24])) : g;
    ip  = {1'b0, sft[31:24]} + {8'd0, inc};
    if (!neg) return (ip > 9'd127) ? 8'h7F : ip[7:0];
    return (ip > 9'd127) ? 8'h80 : 8'(-ip[7:0]);
  endfunction

  // -------------------------------------------------------------------------------------------
  // int32 -> f32, round-to-nearest-even (tsr_i32_to_f32; M3 term computation).
  // -------------------------------------------------------------------------------------------
  function automatic logic [31:0] tsr_i32_to_f32(logic [31:0] v);
    logic        neg;
    logic [31:0] mag;    // |v|; INT32_MIN -> 2^31
    logic [4:0]  lz;
    logic [30:0] nrm;    // (mag << lz) without its leading one
    logic [23:0] fr;     // rounded fraction; fr[23] = carry into the next binade
    logic        inc;
    neg = v[31];
    mag = neg ? (32'd0 - v) : v;
    if (mag == 32'd0) return 32'd0;
    lz = 5'd31;
    for (int i = 0; i < 32; i++) begin
      if (mag[i]) lz = 5'(31 - i);  // the last (highest) set bit wins
    end
    nrm = 31'(mag << lz);
    inc = nrm[7] & ((nrm[6:0] != 7'd0) | nrm[8]);
    fr  = {1'b0, nrm[30:8]} + {23'd0, inc};
    // biased exponent 127 + (31 - lz) (+1 when the rounding carries to 2^24; fr[22:0] is 0 then)
    return {neg, 8'(8'd158 - {3'd0, lz} + {7'd0, fr[23]}), fr[22:0]};
  endfunction

  // -------------------------------------------------------------------------------------------
  // Result rounding of tsr_fp_mul / tsr_fp_div (finite nonzero exact result, not NaN/Inf/0).
  //   value = 1.s[22:0] g st.. * 2^(e - 127): s = truncated significand (s[23] must be 1), g =
  //   the next bit, st = OR of every bit below g (and of the remainder); e = biased exponent of
  //   the truncated significand, any value in [-512, 511].
  // Result: RNE; Inf on overflow; the spec's FTZ rule (see the header) below 2^-126.
  // -------------------------------------------------------------------------------------------
  function automatic logic [31:0] tsr_f32_round_pack(logic sign, logic signed [9:0] e,
                                                     logic [23:0] s, logic g, logic st);
    logic        inc;
    logic [23:0] fr;     // rounded fraction; fr[23] = carry into the next binade (fr[22:0] = 0)
    logic        e_pos, e_zero, ovf;
    inc    = g & (st | s[0]);
    fr     = {1'b0, s[22:0]} + {23'd0, inc};
    e_pos  = (e > 10'sd0);
    e_zero = (e == 10'sd0);
    ovf    = (e > 10'sd254) || ((e == 10'sd254) && fr[23]);
    if (e_pos) begin
      if (ovf) return {sign, 8'hFF, 23'd0};
      return {sign, 8'(e[7:0] + {7'd0, fr[23]}), fr[22:0]};
    end
    if (e_zero && (&s)) return {sign, 8'd1, 23'd0};          // [2^-126 - 2^-150, 2^-126) -> 2^-126
    return {sign, 31'd0};
  endfunction

endpackage
