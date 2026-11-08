// tsr_fp_add.sv — pipelined f32 adder / subtractor with the spec's fp rules.
//
// Owner: fp agent (docs/rtl-m3.md §2 block (fp), contract §8.1).
//
// Contract (bit-exact with sw/sim/tsr_ref.c tsr_fadd / tsr_fsub):
//   y = fin(DAZ(a) + DAZ(b)) when sub = 0, fin(DAZ(a) - DAZ(b)) when sub = 1:
//   - inputs with exponent 0 (zero, subnormal) are signed zeros (DAZ);
//   - the exact sum is rounded RNE in the binary32 exponent range; a subnormal rounded result
//     becomes a signed zero with the sign of that result (FTZ; an add/sub result is subnormal
//     only when it is exact, e.g. 0x00C00000 - 0x00800000 -> +0);
//   - exact zero results: x - x = +0, (-0) + (-0) = -0, (+0) + (-0) = +0 (IEEE RNE rules);
//   - any NaN result is the canonical NaN 0x7FC00000 (NaN input, Inf - Inf, -Inf + Inf);
//     Inf +- finite = Inf; overflow rounds to Inf.
//   - commutative for sub = 0 (bit-exact), so operand order never matters for additions.
// Timing: fixed latency LAT cycles, 1 <= LAT <= 8 (the MVM accumulators need LAT <= 8: two terms
//   for the same Q8_0 partial are >= 8 cycles apart, docs/rtl-m3.md §7.3). The pipeline advances
//   only when ce = 1, exactly like tsr_fp_mul: every register (valid and data) updates only at
//   edges with ce = 1; in_valid/a/b/sub are sampled only at such edges; out_valid/y are
//   registers; a result taken at advancing edge n is on out_valid/y from advancing edge
//   n + LAT - 1 until the next advancing edge. Reset clears the valid pipeline only. No
//   combinational input -> output path. Fully pipelined: one operation per cycle.
//   The default LAT (5) is the smallest latency that meets 300 MHz on the U200 within the ~6 LUT
//   level budget (docs/rtl-m2.md §1; measured below). tsr_pkg::TSR_DEF_FADD_LAT, the LAT the MVM
//   blocks pass, must equal it (checked by tb/fp test_fp_default_lats). Users pass LAT explicitly
//   (TSR_DEF_FADD_LAT); every LAT in 1..8 gives the same results (tb/fp: LAT 1..8), LAT 6..8 are
//   LAT 5 plus output registers.
//
// Algorithm (no subnormal is ever produced, so there is no denormalisation and no rounding of
// tiny results: below 2^-126 the exact sum of two DAZ'd operands is a multiple of 2^-149, hence
// exact, and the spec's FTZ rule flushes it to a signed zero):
//   Operands are ordered by exponent only (big: the larger exponent, a on a tie); d = e_big -
//   e_small; eff_sub = sign(a) ^ sign(b) ^ sub. m = {e != 0, fraction} (0 for DAZ'd operands).
//   far path  (effective addition, or effective subtraction with d >= 2, or d = 1 without
//             cancellation): the small significand is aligned into a 27-bit window (24 bits +
//             guard + round + sticky; every bit shifted out is OR-ed into bit 0), then
//             R = {m_big, 000} +- S (28 bits). R is normalised by at most one position (sum
//             >= 2^27: right; difference < 2^26: left — d >= 2 bounds the cancellation to one
//             bit) and rounded RNE: y = {sign, {e, s} + inc}, inc = g & (st | lsb), where the
//             increment's carry out of the fraction is the binade carry into the exponent and
//             e = 254 + carry is the RNE overflow to Inf. The jammed sticky bit makes R odd
//             whenever the alignment was inexact and keeps R within one unit of the exact value,
//             so R and the exact result agree in every bit at and above the guard position
//             (always >= bit 1) and in "anything below the guard bit".
//             A left normalisation never reaches exponent 0 (it needs a nonzero small operand,
//             so e_big >= e_small + 2 >= 3), and a far zero result occurs only for two zero
//             operands with an effective addition: y = {sign of both, 0}.
//   near path (effective subtraction with d <= 1 and a result below 2^24 units, i.e. at least
//             one bit of cancellation): N = 2 m_big - (m_small << (1 - d)) is computed exactly
//             (on a tie of exponents |m_a - m_b| by a second subtractor; the result takes the
//             sign of the larger magnitude), N < 2^24 fits the significand, so there is no
//             rounding: y = {sign, e_big - 1 - lzc(N), N << lzc(N)}; e_big - 1 - lzc <= 0 is an
//             exact subnormal -> signed zero (FTZ), N = 0 -> +0 (x - x).
//   specials  NaN operand or Inf - Inf -> 0x7FC00000; otherwise an Inf operand -> that Inf.
//
// Pipeline (logic segments S1..S4; LAT selects which cut registers exist):
//   [IN]  raw operands a, b, sub, in_valid                                   LAT >= 2
//   S1    classification (DAZ, Inf/NaN, eff_sub), exponent differences e_a - e_b and e_b - e_a,
//         swap by exponent, d -> shift amount min(d, 31), near flag
//   [C1]                                                                     LAT >= 4
//   S2    far: 27-bit alignment shift + sticky; near: the exact differences N (26-bit
//         subtractors) and |m_a - m_b|; e_big +- 1
//   [C2]                                                                     LAT >= 3
//   S3    far: 28-bit add/sub, 1-bit normalisation select, RNE increment bit, overflow;
//         near: 24-bit leading-zero count, path select
//   [C3]                                                                     LAT >= 5
//   S4    far: the 31-bit {e, s} + inc increment; near: left shift by lzc, exponent e_big - 1 -
//         lzc and the FTZ compare; final select (NaN / Inf / near / far)
//   [OUT] y, out_valid (always); LAT 6..8 add LAT - 5 plain output registers after [OUT].
//   LAT 4 merges S3 + S4, LAT 3 S1 + S2 and S3 + S4, LAT 2 and 1 everything.
// Logic depth / timing, measured: Vivado 2024.2 out-of-context place and route,
// xcu200-fsgd2104-2-e, 3.333 ns, I/O delay 1.6 ns (tb/fp/viv/fp_ooc.tcl; tb/fp/test_fp.py
// test_fp_ooc_timing[add]). "LUT" = LUT cells on the worst register-to-register path of a segment.
//   LAT 5 (default): WNS +1.10 ns (set by the ce input port under the 1.6 ns I/O budget);
//          segments S1 +1.33 ns / 5 LUT, S2 +1.53 ns / 5 LUT, S3 +1.37 ns / 4 LUT + 4 CARRY8,
//          S4 +1.59 ns / 5 LUT; 421 LUT, 350 FF, 16 CARRY8, no DSP / RAM.
//   LAT 4: WNS +1.02 ns but 8 LUT levels (near path: lzc -> left shift -> select, S3 + S4); the
//          other LAT 4 split (S1 + S2 merged) measured WNS +0.51 ns, also 8 LUT levels (exponent
//          difference -> swap -> alignment shift). Both exceed the ~6-level budget of rtl-m2 §1.
//   LAT 3: WNS +0.41 ns, 8 LUT levels.  LAT 1, 2: for low clock rates only.
//   LAT 6..8: the LAT 5 segments plus LAT - 5 output registers (no added logic).

`ifdef TSR_FP_ADD_DSP
// Area experiment only (docs/design/02-vector-unit.md §7.10, run E2): with +define+TSR_FP_ADD_DSP=1 every
// tsr_fp_add is the DSP-aligned adder tsr_fp_add_dsp (same ports and results, LAT 5..8); =2 also does the
// near-path left shift in a DSP (NORM_DSP). The define is not set by any filelist, script or build in the
// tree, so the default RTL is the module below.
module tsr_fp_add #(
  parameter int unsigned LAT = 5
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        ce,
  input  logic        in_valid,
  input  logic [31:0] a,
  input  logic [31:0] b,
  input  logic        sub,
  output logic        out_valid,
  output logic [31:0] y
);
  tsr_fp_add_dsp #(.LAT(LAT), .NORM_DSP(`TSR_FP_ADD_DSP == 2)) u_add (
    .clk, .rst_n, .ce, .in_valid, .a, .b, .sub, .out_valid, .y
  );
endmodule
`else
module tsr_fp_add #(
  parameter int unsigned LAT = 5
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        ce,
  input  logic        in_valid,
  input  logic [31:0] a,
  input  logic [31:0] b,
  input  logic        sub,        // 1: a - b, 0: a + b
  output logic        out_valid,
  output logic [31:0] y
);
  import tsr_fp_pkg::*;

  if (LAT < 1 || LAT > 8) begin : g_chk_lat
    $fatal(1, "tsr_fp_add: LAT must be in [1, 8]");
  end

  // Cut registers present at this LAT (see the header).
  localparam bit          CUT_IN = (LAT >= 2);
  localparam bit          CUT_1  = (LAT >= 4);
  localparam bit          CUT_2  = (LAT >= 3);
  localparam bit          CUT_3  = (LAT >= 5);
  localparam int unsigned N_XOUT = (LAT > 5) ? LAT - 5 : 0;

  // ---------------------------------------------------------------------------------------------
  // Stage records
  // ---------------------------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0] a;
    logic [31:0] b;
    logic        sub;
  } in_t;

  typedef struct packed {
    logic        nan;        // result is the canonical NaN
    logic        inf;        // result is an Inf of sign inf_s (operand Inf; S3 adds overflow)
    logic        inf_s;
  } spc_t;

  typedef struct packed {
    spc_t        spc;
    logic        eff_sub;    // effective subtraction
    logic        s_big;      // effective sign of the big operand (sign of a nonzero far result)
    logic [7:0]  e_big;      // exponent of the big operand (the larger one; a on a tie)
    logic [23:0] m_big;      // significands with the hidden bit; 0 for DAZ'd operands
    logic [23:0] m_small;
    logic [4:0]  shamt;      // min(d, 31): far alignment shift (>= 27 shifts everything out)
    logic        d0;         // d[0] (near path: d is 0 or 1)
    logic        nr;         // near path candidate: eff_sub & d <= 1
  } s1_t;

  typedef struct packed {
    spc_t        spc;
    logic        eff_sub;
    logic        s_big;
    logic [7:0]  e_big;
    logic [7:0]  e_p1;       // e_big + 1
    logic [7:0]  e_m1;       // e_big - 1
    logic        e254;       // e_big == 254 (a far carry overflows)
    logic [23:0] m_big;
    logic [26:0] s_al;       // aligned small significand, bit 0 = OR of every bit shifted out
    logic        nr;
    logic [24:0] n;          // near: exact |difference| in units of 2^(e_big - 151)
    logic        n_neg;      // near: the small operand had the larger magnitude
  } s2_t;

  typedef struct packed {
    logic        nan;
    logic        inf;        // operand Inf or far overflow
    logic        inf_s;
    logic        use_near;
    logic        s_big;      // sign of a far result
    logic [30:0] f_mag;      // far: {e, s[22:0]} before the RNE increment
    logic        f_inc;      // far: RNE increment
    logic        n_s;        // near: sign of a nonzero result
    logic        n_zero;     // near: exact zero (x - x) -> +0
    logic [4:0]  n_lz;       // near: leading zeros of n[23:0]
    logic [23:0] n_m;        // near: n[23:0]
    logic [7:0]  e_m1;       // near: e_big - 1
  } s3_t;

  // Leading zeros of a 24-bit value (24 for zero).
  function automatic logic [4:0] lzc24(logic [23:0] v);
    logic [4:0] n;
    n = 5'd24;
    for (int i = 0; i < 24; i++) begin
      if (v[i]) n = 5'(23 - i);   // the last (highest) set bit wins
    end
    return n;
  endfunction

  // ---------------------------------------------------------------------------------------------
  // [IN] raw operands
  // ---------------------------------------------------------------------------------------------
  logic v0;
  in_t  i0;

  if (CUT_IN) begin : g_in
    logic v_q;
    in_t  d_q;
    always_ff @(posedge clk) begin
      if (!rst_n)  v_q <= 1'b0;
      else if (ce) v_q <= in_valid;
    end
    always_ff @(posedge clk) begin
      if (ce) d_q <= '{a: a, b: b, sub: sub};
    end
    assign v0 = v_q;
    assign i0 = d_q;
  end else begin : g_no_in
    assign v0 = in_valid;
    assign i0 = '{a: a, b: b, sub: sub};
  end

  // ---------------------------------------------------------------------------------------------
  // S1: classification, exponent order, shift amount.
  // ---------------------------------------------------------------------------------------------
  s1_t s1d;

  always_comb begin
    logic [7:0]  ea, eb, d;
    logic        sa, sb, ha, hb, xa, xb, fa, fb, a_ge;
    logic [8:0]  dab;
    logic [7:0]  dba;
    logic [23:0] ma, mb;
    ea  = i0.a[30:23];
    eb  = i0.b[30:23];
    sa  = i0.a[31];
    sb  = i0.b[31] ^ i0.sub;                 // effective sign of b
    ha  = (ea != 8'd0);                      // normal (else DAZ: signed zero)
    hb  = (eb != 8'd0);
    xa  = (ea == 8'hFF);                     // Inf or NaN
    xb  = (eb == 8'hFF);
    fa  = (i0.a[22:0] != 23'd0);
    fb  = (i0.b[22:0] != 23'd0);
    ma  = {ha, i0.a[22:0] & {23{ha}}};
    mb  = {hb, i0.b[22:0] & {23{hb}}};
    dab = {1'b0, ea} - {1'b0, eb};
    dba = eb - ea;                           // used only when e_b > e_a
    a_ge = !dab[8];
    d   = a_ge ? dab[7:0] : dba;
    s1d.eff_sub = sa ^ sb;
    s1d.spc.nan   = (xa && fa) || (xb && fb) || (xa && xb && (sa ^ sb));
    s1d.spc.inf   = xa || xb;                // (NaN overrides it)
    s1d.spc.inf_s = xa ? sa : sb;
    s1d.s_big   = a_ge ? sa : sb;
    s1d.e_big   = a_ge ? ea : eb;
    s1d.m_big   = a_ge ? ma : mb;
    s1d.m_small = a_ge ? mb : ma;
    s1d.shamt   = (d[7:5] != 3'd0) ? 5'd31 : d[4:0];
    s1d.d0      = d[0];
    s1d.nr      = (sa ^ sb) && (d[7:1] == 7'd0);
  end

  // [C1]
  logic v1;
  s1_t  s1q;

  if (CUT_1) begin : g_c1
    logic v_q;
    s1_t  d_q;
    always_ff @(posedge clk) begin
      if (!rst_n)  v_q <= 1'b0;
      else if (ce) v_q <= v0;
    end
    always_ff @(posedge clk) begin
      if (ce) d_q <= s1d;
    end
    assign v1  = v_q;
    assign s1q = d_q;
  end else begin : g_no_c1
    assign v1  = v0;
    assign s1q = s1d;
  end

  // ---------------------------------------------------------------------------------------------
  // S2: far alignment + sticky; near exact differences; exponent neighbours.
  // ---------------------------------------------------------------------------------------------
  s2_t s2d;

  always_comb begin
    logic [26:0] shr;
    logic        st;
    logic [25:0] x;          // 2 m_big - (m_small << (1 - d0)), two's complement
    logic [23:0] yd;         // m_small - m_big (used when x < 0, then d0 = 0 and it is > 0)
    shr = {s1q.m_small, 3'b000} >> s1q.shamt;
    st  = 1'b0;
    for (int k = 0; k < 24; k++) begin
      // m_small[k] sits at window bit k + 3 and is shifted out iff k + 3 < shamt
      if (5'(k + 3) < s1q.shamt) st = st | s1q.m_small[k];
    end
    x  = {1'b0, s1q.m_big, 1'b0} - (s1q.d0 ? {2'b00, s1q.m_small} : {1'b0, s1q.m_small, 1'b0});
    yd = s1q.m_small - s1q.m_big;
    s2d.spc     = s1q.spc;
    s2d.eff_sub = s1q.eff_sub;
    s2d.s_big   = s1q.s_big;
    s2d.e_big   = s1q.e_big;
    s2d.e_p1    = s1q.e_big + 8'd1;
    s2d.e_m1    = s1q.e_big - 8'd1;
    s2d.e254    = (s1q.e_big == 8'd254);
    s2d.m_big   = s1q.m_big;
    s2d.s_al    = {shr[26:1], shr[0] | st};
    s2d.nr      = s1q.nr;
    s2d.n       = x[25] ? {yd, 1'b0} : x[24:0];
    s2d.n_neg   = x[25];
  end

  // [C2]
  logic v2;
  s2_t  s2q;

  if (CUT_2) begin : g_c2
    logic v_q;
    s2_t  d_q;
    always_ff @(posedge clk) begin
      if (!rst_n)  v_q <= 1'b0;
      else if (ce) v_q <= v1;
    end
    always_ff @(posedge clk) begin
      if (ce) d_q <= s2d;
    end
    assign v2  = v_q;
    assign s2q = d_q;
  end else begin : g_no_c2
    assign v2  = v1;
    assign s2q = s2d;
  end

  // ---------------------------------------------------------------------------------------------
  // S3: far add / normalise / RNE bit; near leading zeros; path select.
  // ---------------------------------------------------------------------------------------------
  s3_t s3d;

  always_comb begin
    logic [27:0] bb, ss, r;
    logic        hi, lo, g, st;
    logic [22:0] s;          // fraction (the hidden bit is implied)
    logic [7:0]  e;
    bb = {1'b0, s2q.m_big, 3'b000};
    ss = {1'b0, s2q.s_al};
    r  = s2q.eff_sub ? (bb - ss) : (bb + ss);
    hi = !s2q.eff_sub && r[27];             // sum in [2^27, 2^28): one position right
    lo = s2q.eff_sub && !r[26];             // difference in [2^25, 2^26): one position left
    if (hi) begin
      s  = r[26:4];
      g  = r[3];
      st = (r[2:0] != 3'd0);
      e  = s2q.e_p1;
    end else if (lo) begin
      s  = r[24:2];
      g  = r[1];
      st = r[0];
      e  = s2q.e_m1;
    end else begin
      s  = r[25:3];
      g  = r[2];
      st = (r[1:0] != 2'd0);
      e  = s2q.e_big;
    end
    s3d.nan      = s2q.spc.nan;
    s3d.inf      = s2q.spc.inf || (hi && s2q.e254);    // far overflow: e_big + 1 = 255
    s3d.inf_s    = s2q.spc.inf ? s2q.spc.inf_s : s2q.s_big;
    s3d.use_near = s2q.nr && !s2q.n[24];
    s3d.s_big    = s2q.s_big;
    s3d.f_mag    = {e, s};
    s3d.f_inc    = g && (st || s[0]);
    s3d.n_s      = s2q.s_big ^ s2q.n_neg;
    s3d.n_zero   = (s2q.n[23:0] == 24'd0);
    s3d.n_lz     = lzc24(s2q.n[23:0]);
    s3d.n_m      = s2q.n[23:0];
    s3d.e_m1     = s2q.e_m1;
  end

  // [C3]
  logic v3;
  s3_t  s3q;

  if (CUT_3) begin : g_c3
    logic v_q;
    s3_t  d_q;
    always_ff @(posedge clk) begin
      if (!rst_n)  v_q <= 1'b0;
      else if (ce) v_q <= v2;
    end
    always_ff @(posedge clk) begin
      if (ce) d_q <= s3d;
    end
    assign v3  = v_q;
    assign s3q = d_q;
  end else begin : g_no_c3
    assign v3  = v2;
    assign s3q = s3d;
  end

  // ---------------------------------------------------------------------------------------------
  // S4: far increment, near normalisation / FTZ, final select; [OUT].
  // ---------------------------------------------------------------------------------------------
  logic [31:0] y4;

  always_comb begin
    logic [22:0] nm;         // fraction of n << lzc (its bit 23 is the hidden 1)
    logic [7:0]  ne;
    logic        ftz;
    nm  = 23'(s3q.n_m << s3q.n_lz);
    ne  = s3q.e_m1 - {3'd0, s3q.n_lz};
    ftz = (s3q.e_m1 <= {3'd0, s3q.n_lz});   // e_big - 1 - lzc <= 0: exact subnormal
    if (s3q.nan) begin
      y4 = TSR_F32_CANON_NAN;
    end else if (s3q.inf) begin
      y4 = {s3q.inf_s, TSR_F32_INF_MAG};
    end else if (s3q.use_near) begin
      if (s3q.n_zero)  y4 = 32'd0;
      else if (ftz)    y4 = {s3q.n_s, 31'd0};
      else             y4 = {s3q.n_s, ne, nm};
    end else begin
      y4 = {s3q.s_big, s3q.f_mag + 31'(s3q.f_inc)};
    end
  end

  logic        v4;
  logic [31:0] y4q;

  always_ff @(posedge clk) begin
    if (!rst_n)  v4 <= 1'b0;
    else if (ce) v4 <= v3;
  end
  always_ff @(posedge clk) begin
    if (ce) y4q <= y4;
  end

  // LAT 6..8: plain output registers (kept as flip-flops, not shift-register LUTs).
  if (N_XOUT != 0) begin : g_xout
    logic                         v_q [N_XOUT];
    (* shreg_extract = "no" *) logic [31:0] y_q [N_XOUT];
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        for (int i = 0; i < int'(N_XOUT); i++) v_q[i] <= 1'b0;
      end else if (ce) begin
        v_q[0] <= v4;
        for (int i = 1; i < int'(N_XOUT); i++) v_q[i] <= v_q[i-1];
      end
    end
    always_ff @(posedge clk) begin
      if (ce) begin
        y_q[0] <= y4q;
        for (int i = 1; i < int'(N_XOUT); i++) y_q[i] <= y_q[i-1];
      end
    end
    assign out_valid = v_q[N_XOUT-1];
    assign y         = y_q[N_XOUT-1];
  end else begin : g_no_xout
    assign out_valid = v4;
    assign y         = y4q;
  end

endmodule
`endif
