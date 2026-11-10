// tsr_fp_add_dsp.sv — tsr_fp_add with the far-path alignment shifter done by one DSP48E2.
//
// Measurement prototype (docs/design/02-vector-unit.md §7.10, area run E1c). Not instantiated by default:
// nothing in rtl/ names it. Same function, ports and cut-register scopes as tsr_fp_add, bit-exact with
// sw/sim/tsr_ref.c tsr_fadd / tsr_fsub (tb/fp/test_fp.py test_fp_add_dsp, the sweeps of fp_sweep.cpp).
// LAT must be 5..8 (the DSP sits between [C1] and [C2]; LAT 6..8 add output registers as tsr_fp_add does).
//
// What changes against tsr_fp_add (read its header first): the right shift of the small significand by
// shamt = min(d, 31) is a multiplication by a one-hot power of two. With oh = 2^(16 - j):
//   P = m_small * oh          24 x 17 unsigned, one DSP48E2 (A = m_small, B = oh; no fabric multiplier)
//   shamt <= 16   j = shamt         window = P[39:13], shifted-out bits P[12:0]
//   17..25        j = shamt - 17    window = P[39:30], shifted-out bits P[29:0] (a second right shift by 17,
//                                   a 2:1 select in front of the far adder)
//   >= 26         oh = 0            window = 0, the sticky bit is m_small[23] (m_small != 0): the baseline
//                                   gives the same s_al = 1 for every shamt >= 26
// The sticky bit is the OR of the shifted-out bits of P, so the comparator chain of the baseline goes away.
// The decode of shamt into oh sits in S1 (the DSP input registers are [C1]); [C2] is the DSP P register.
// NORM_DSP = 1 does the near path's left shift by the leading-zero count the same way, in S4:
//   q = min(lzc, 16), P = n_m * 2^q (one DSP48E2 between [C3] and the final select); lzc >= 17 leaves n_m below
//   2^7, and the last lzc - 16 positions are a 7-bit shift in fabric. The DSP inputs are the [C3] registers.
// Everything else (S1 classification, near path, S3, S4) is tsr_fp_add unchanged.

module tsr_fp_add_dsp #(
  parameter int unsigned LAT      = 5,
  parameter bit          NORM_DSP = 1'b0     // 1: the near-path left shift is a second DSP48E2 as well
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

  if (LAT < 5 || LAT > 8) begin : g_chk_lat
    $fatal(1, "tsr_fp_add_dsp: LAT must be in [5, 8]");
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
    logic [16:0] oh;         // far alignment: one-hot 2^(16 - j), 0 when shamt >= 26 (j: see the header)
    logic        t;          // 17 <= shamt <= 25: second right shift by 17
    logic        gone;       // shamt >= 26: everything shifted out
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
    logic [40:0] p;          // m_small * oh (the DSP P register)
    logic        t;
    logic        gone_m;     // shamt >= 26 and m_small != 0: the aligned significand is 1
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
    logic [16:0] n_oh;       // NORM_DSP: one-hot 2^min(lzc, 16)
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
    logic [4:0]  sh5;
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
    sh5         = (d[7:5] != 3'd0) ? 5'd31 : d[4:0];       // min(d, 31)
    s1d.gone    = (sh5 >= 5'd26);
    s1d.t       = (sh5 >= 5'd17) && (sh5 <= 5'd25);
    for (int i = 0; i < 17; i++) begin
      s1d.oh[i] = (sh5 == 5'(16 - i)) || (i >= 8 && sh5 == 5'(33 - i));
    end
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
  // S2: far alignment product (the DSP, P register = [C2]); near exact differences; exponent neighbors.
  // ---------------------------------------------------------------------------------------------
  s2_t s2d;

  always_comb begin
    logic [25:0] x;          // 2 m_big - (m_small << (1 - d0)), two's complement
    logic [23:0] yd;         // m_small - m_big (used when x < 0, then d0 = 0 and it is > 0)
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
    s2d.p       = s1q.m_small * s1q.oh;                // inferred DSP48E2 (A = m_small, B = oh)
    s2d.t       = s1q.t;
    s2d.gone_m  = s1q.gone && s1q.m_small[23];
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
  // S3: far window select and sticky from P, add / normalize / RNE bit; near leading zeros; path select.
  // ---------------------------------------------------------------------------------------------
  s3_t s3d;

  always_comb begin
    logic [27:0] bb, ss, r;
    logic        hi, lo, g, st;
    logic [22:0] s;          // fraction (the hidden bit is implied)
    logic [7:0]  e;
    logic [26:0] win, s_al;
    logic        pst;
    win  = s2q.t ? {17'd0, s2q.p[39:30]} : s2q.p[39:13];
    pst  = s2q.t ? (s2q.p[29:0] != 30'd0) : (s2q.p[12:0] != 13'd0);
    s_al = {win[26:1], win[0] | pst | s2q.gone_m};
    bb = {1'b0, s2q.m_big, 3'b000};
    ss = {1'b0, s_al};
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
    for (int i = 0; i < 17; i++) begin
      s3d.n_oh[i] = NORM_DSP && ((i == 16) ? (s3d.n_lz >= 5'd16) : (s3d.n_lz == 5'(i)));   // 2^min(lzc, 16)
    end
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
  // S4: far increment, near normalization / FTZ, final select; [OUT].
  // ---------------------------------------------------------------------------------------------
  logic [31:0] y4;

  logic [22:0] n_p;          // NORM_DSP: n_m * 2^min(lzc, 16), inferred DSP48E2 (combinational from [C3]); bits 22:0
  assign n_p = NORM_DSP ? 23'(s3q.n_m * s3q.n_oh) : 23'd0;

  always_comb begin
    logic [22:0] nm;         // fraction of n << lzc (its bit 23 is the hidden 1)
    logic [7:0]  ne;
    logic        ftz;
    if (NORM_DSP) begin
      logic [6:0] u7;
      u7 = 7'(s3q.n_m[6:0] << (s3q.n_lz - 5'd16));       // used for lzc >= 17 only (n_m < 2^7)
      nm = (s3q.n_lz >= 5'd17) ? {u7, 16'd0} : n_p[22:0];
    end else begin
      nm = 23'(s3q.n_m << s3q.n_lz);
    end
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
