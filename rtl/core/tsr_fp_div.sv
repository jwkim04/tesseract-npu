// tsr_fp_div.sv — iterative, correctly rounded f32 divider with the spec's fp rules.
//
// Owner: fp agent (docs/rtl-core.md §2, block (c)).
//
// Contract (bit-exact with tsr_ref.c tsr_fdiv; tsr_frecip(x) == tsr_fdiv(1.0f, x)):
//   y = fin(DAZ(a) / DAZ(b)), correctly rounded RNE in the binary32 exponent range, subnormal
//   rounded result -> signed zero (|exact| < 2^-126 - 2^-150 -> +-0; 0x00FFFFFF / 2 =
//   0x00800000), x/0 = +-Inf (x finite nonzero), 0/0 = Inf/Inf = NaN -> 0x7FC00000,
//   any NaN operand -> 0x7FC00000, Inf/x = +-Inf, x/Inf = +-0. Sign = XOR of the input signs
//   (for zero and Inf results too).
// Handshake: in_valid/in_ready accepts (a, b) (latched at the transfer); out_valid/out_ready
//   delivers y; out_valid and y hold until out_ready (docs/rtl-core.md §1 R1-R3). in_ready is a
//   register (1 while the iteration engine is idle); out_valid/y are registers. No
//   combinational input->output path.
// Latency (fixed, every input incl. special values): an operation accepted at edge n has
//   out_valid = 1 after edge n + 14 when the output register is free by then (else it waits in
//   the engine until out_ready frees it); in_ready is 1 again after that same edge, so
//   back-to-back operations are accepted every 15 cycles. Up to two operations are held (one
//   in the output register, one in the engine).
// Algorithm: mantissas A, B in [2^23, 2^24); A' = A < B ? 2A : A so that Q = A'/B is in
//   [1, 2); restoring radix-4 division (digit by comparing 4r against B, 2B, 3B in parallel:
//   three 27-bit subtractors per cycle) gives 1 + 12 x 2 = 25 quotient bits (24-bit significand
//   + guard) and the exact remainder (sticky); tsr_fp_pkg::tsr_f32_round_pack rounds (RNE, FTZ
//   rule, overflow). Cycles: accept, setup, 12 iterations, round.

module tsr_fp_div (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        in_valid,
  output logic        in_ready,
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic        out_valid,
  input  logic        out_ready,
  output logic [31:0] y
);
  import tsr_fp_pkg::*;

  localparam int unsigned ITERS = 12;   // 2 quotient bits per iteration

  typedef enum logic [1:0] {
    CLS_NORM = 2'd0,
    CLS_ZERO = 2'd1,
    CLS_INF  = 2'd2,
    CLS_NAN  = 2'd3
  } cls_e;

  typedef enum logic [1:0] {
    S_IDLE  = 2'd0,
    S_SETUP = 2'd1,
    S_ITER  = 2'd2,
    S_ROUND = 2'd3
  } state_e;

  // Control (reset).
  state_e     state;
  logic [3:0] cnt;

  // Datapath (no reset).
  logic [31:0]       a_q, b_q;    // latched operands
  logic              sign_q;
  cls_e              cls_q;
  logic signed [9:0] e_q;         // biased exponent of Q in [1, 2)
  logic [23:0]       dv_q;        // divisor mantissa B
  logic [25:0]       dv3_q;       // 3B
  logic [23:0]       r_q;         // partial remainder, always < B
  logic [24:0]       q_q;         // quotient bits (MSB = the integer bit 1)

  // ---------------------------------------------------------------------------------------------
  // Setup (from the latched operands).
  // ---------------------------------------------------------------------------------------------
  logic              su_sign;
  cls_e              su_cls;
  logic signed [9:0] su_e;
  logic [23:0]       su_dv;
  logic [25:0]       su_dv3;
  logic [23:0]       su_r;

  always_comb begin
    logic        za, zb, ia, ib, na, nb, lt;
    logic [23:0] ma, mb;
    logic [24:0] d1;
    logic [23:0] d2;
    za = (a_q[30:23] == 8'd0);
    zb = (b_q[30:23] == 8'd0);
    ia = (a_q[30:23] == 8'hFF) && (a_q[22:0] == 23'd0);
    ib = (b_q[30:23] == 8'hFF) && (b_q[22:0] == 23'd0);
    na = tsr_f32_is_nan(a_q);
    nb = tsr_f32_is_nan(b_q);
    su_sign = a_q[31] ^ b_q[31];
    if (na || nb || (ia && ib) || (za && zb)) su_cls = CLS_NAN;
    else if (ia || zb)                        su_cls = CLS_INF;
    else if (za || ib)                        su_cls = CLS_ZERO;
    else                                      su_cls = CLS_NORM;
    ma = {1'b1, a_q[22:0]};
    mb = {1'b1, b_q[22:0]};
    d1 = {1'b0, ma} - {1'b0, mb};            // A - B  (negative <=> A < B)
    d2 = 24'({ma, 1'b0} - {1'b0, mb});       // 2A - B (in [0, B) whenever A < B)
    lt = d1[24];
    su_r   = lt ? d2 : d1[23:0];             // A' - B, in [0, B)
    su_e   = $signed({2'b00, a_q[30:23]}) - $signed({2'b00, b_q[30:23]}) + 10'sd127
             - (lt ? 10'sd1 : 10'sd0);
    su_dv  = mb;
    su_dv3 = {2'b00, mb} + {1'b0, mb, 1'b0};
  end

  // ---------------------------------------------------------------------------------------------
  // One radix-4 iteration: x = 4r in [0, 4B); digit = floor(x / B); r' = x - digit * B.
  // ---------------------------------------------------------------------------------------------
  logic [23:0] it_r;
  logic [1:0]  it_d;
  logic [26:0] x, t1, t2, t3;   // [26] = sign; [25:24] only feed the sign (chosen r' < B)

  always_comb begin
    x  = {1'b0, r_q, 2'b00};
    t1 = x - {3'd0, dv_q};
    t2 = x - {2'd0, dv_q, 1'b0};
    t3 = x - {1'd0, dv3_q};
    if (!t3[26]) begin
      it_d = 2'd3;
      it_r = t3[23:0];
    end else if (!t2[26]) begin
      it_d = 2'd2;
      it_r = t2[23:0];
    end else if (!t1[26]) begin
      it_d = 2'd1;
      it_r = t1[23:0];
    end else begin
      it_d = 2'd0;
      it_r = x[23:0];
    end
  end

  wire unused_ok = &{1'b0, t1[25:24], t2[25:24], t3[25:24]};

  // ---------------------------------------------------------------------------------------------
  // Result.
  // ---------------------------------------------------------------------------------------------
  logic [31:0] res;
  always_comb begin
    unique case (cls_q)
      CLS_NAN:  res = TSR_F32_CANON_NAN;
      CLS_INF:  res = {sign_q, 8'hFF, 23'd0};
      CLS_ZERO: res = {sign_q, 31'd0};
      default:  res = tsr_f32_round_pack(sign_q, e_q, q_q[24:1], q_q[0], (r_q != 24'd0));
    endcase
  end

  wire accept  = in_valid && in_ready;
  wire deliver = (state == S_ROUND) && (!out_valid || out_ready);

  // ---------------------------------------------------------------------------------------------
  // Control.
  // ---------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      in_ready  <= 1'b0;
      cnt       <= '0;
      out_valid <= 1'b0;
    end else begin
      unique case (state)
        S_IDLE: begin
          in_ready <= 1'b1;
          if (accept) begin
            state    <= S_SETUP;
            in_ready <= 1'b0;
          end
        end
        S_SETUP: begin
          state <= S_ITER;
          cnt   <= '0;
        end
        S_ITER: begin
          cnt <= cnt + 4'd1;
          if (cnt == 4'(ITERS - 1)) state <= S_ROUND;
        end
        default: begin   // S_ROUND
          if (deliver) begin
            state    <= S_IDLE;
            in_ready <= 1'b1;
          end
        end
      endcase
      if (deliver)                 out_valid <= 1'b1;
      else if (out_valid && out_ready) out_valid <= 1'b0;
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Datapath.
  // ---------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (accept) begin
      a_q <= a;
      b_q <= b;
    end
    if (state == S_SETUP) begin
      sign_q <= su_sign;
      cls_q  <= su_cls;
      e_q    <= su_e;
      dv_q   <= su_dv;
      dv3_q  <= su_dv3;
      r_q    <= su_r;
      q_q    <= 25'd1;
    end
    if (state == S_ITER) begin
      r_q <= it_r;
      q_q <= {q_q[22:0], it_d};
    end
    if (deliver) y <= res;
  end

`ifndef SYNTHESIS
  // The engine never runs with in_ready high, and accepts only while idle. Armed once a reset
  // has been seen (same scheme as tsr_spm / tsr_mem_arb / tsr_core): Verilator's first evaluation
  // may clock this instance with rst_n high and random state before any reset.
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      assert (!(in_ready && state != S_IDLE)) else $error("tsr_fp_div: in_ready while busy");
    end
  end
`endif

endmodule
