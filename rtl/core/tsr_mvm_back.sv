// tsr_mvm_back.sv — MVM cluster back end (spec MVM 클러스터 stages 5-9): activation reads, MAC
// lanes (tsr_mac_lane), integer scale stage, fp term stage, 16-partial accumulators with a
// background streaming tree; produces one row result (T_MAX f32) per row.
//
// Owner: back agent (docs/rtl-m3.md §2 block (back); contract §3.3-§3.5, numerics §7).
//
// Contract (normative text: docs/rtl-m3.md §7; golden model tb/common/mvm_model.py):
//   - cfg_fmt / cfg_d_lines are stable while beats of one MVM flow (the cluster wrapper registers
//     them from the cluster command); every accepted beat belongs to that MVM. Beats are offered
//     only when the activation buffer holds the MVM's QA images (the wrapper gates beat_valid).
//     They may change only while idle = 1 (a new MVM's beats arrive only after every result of
//     the previous one was taken, docs/rtl-m3.md §6.2); asserted in simulation.
//   - per beat (tsr_mvm_beat_t, R1/R2), for every token t < T_MAX, bit-exact with tsr_ref.c:
//       Q8_0: isum_g = sum_{L<32} w[32g+L] * qs_t[32b + 32g + L] (g = 0, and g = 1 if two);
//             term(b+g) = fmul(fmul(d_g, Dx_t[b+g]), i32_to_f32(isum_g))   (d_0 = d_a, d_1 = d_b)
//       Q4_K: S_j = 32-lane group sums (j = 4*half + g); isum = sum_j sc_j*S_j over both halves;
//             msum = sum_{j<8} m_j * (BS_t[16s+2j] + BS_t[16s+2j+1]);
//             term(s) = fsub(fmul(fmul(d_a, Dx_t[s]), i32_to_f32(isum)),
//                            fmul(fmul(d_b, Dx_t[s]), i32_to_f32(msum)))   (at half = 1)
//       term i of a row -> partial p[i % 16] = fadd(p[i % 16], term), partials start at +0.0 per
//       row (a partial that gets no term stays +0.0; the first add is a real fadd(+0, t));
//       at row_last: y_t = tsr_tree16(p) = ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)) + ... (4 levels).
//     Integer sums are exact (any order); every fp op is a separately rounded tsr_fp_* op in
//     exactly this order (spec 수치 규칙: no FMA, the parenthesisation above).
//   - activation buffer: qs_re/qs_addr and dbs_re/dbs_addr, data exactly ACT_RD_LAT cycles later
//     (tsr_mvm_actbuf). QS word for a beat: Q8_0 line b/2 = bank (b/2)%2 of word b/4 (64 lanes);
//     Q4_K word 2s + half (128 lanes). D line: (Q8_0 b, Q4_K s) >> 4, element & 15; BS line
//     cfg_d_lines + s/2, int16 elements 16*(s%2) .. +15.
//   - res_valid/res_ready/res_y: one result per row, in row order (R1/R2); res_y[t] for every
//     t < T_MAX (tokens >= T of the MVM are don't-care). The back end never drops a result:
//     internal pipelines run with ce = 1; beat_ready falls only to reserve result space and
//     partial-set / tree capacity (docs/rtl-m3.md §7.5: no stall for rows of >= RB_MIN beats).
//   - idle: no beat or row in flight, no result pending.
// Datapath per token replicated T_MAX times (docs/rtl-m3.md §9.3). Q8_0 delivers 2 terms per quant
// beat: two accumulate lanes (even / odd partials); Q4_K 1 term per 2 quant beats (lane s % 2).
//
// ================================================================================================
// Microarchitecture (all registers advance every cycle: no stage ever stalls; beat_ready, a flop,
// is the only flow control). Cycle numbers below are relative to the cycle in which a beat is
// accepted (cycle 0); "in cycle c" = on a register output during cycle c. Defaults in brackets
// (ACT_RD_LAT 2, MAC_LAT 3, MUL_LAT 4, ADD_LAT 5).
//
//   cycle 1        input stage: qs_re/qs_addr, dbs_re/dbs_addr registers (Q8_0 word b/4, Q4_K
//                  2s+half; D line blk>>4, Q4_K half 1: BS line cfg_d_lines + s/2); the control
//                  tags (row_last, set, first, two, half, blk[3:0]) enter the tag pipeline; the
//                  weights (lanes 0..63 bytes + lanes 64..127 low nibbles, 768 bits), d_a/d_b, sc
//                  and m enter delay lines (SRL-friendly, no reset).
//   M0 = 1+ACT [3] activation data. MAC operands, both straight from flops (tsr_mac_lane has no
//                  input register): x = the QS word (all 1024 bits, every token) from the actbuf
//                  RAM output register, w = the weight placement register wm_q (placed per format
//                  in cycle M0 - 1: Q8_0 lanes 0..63 for an even QS line, lanes 64..127 for an odd
//                  one, so the Q8_0 half select is on the shared weights, not on T_MAX x 512
//                  activation bits; Q4_K nibbles, upper nibbles forced 0).
//                  d*Dx operands: Q8_0 Dx elements (b & 15) | 0 / | 1 of the D line; Q4_K: the
//                  half-0 beat reads the D line and stores Dx_t[s] (dxh), the half-1 beat reads
//                  the BS line and uses dxh. BS pair sums (msum stage 1).
//   G = M0+MAC [6] group sums S_g (tsr_mac_lane, 4 instances x T_MAX tokens, 21 bits; generic
//                  MAC_LAT 3: products | 2 adder levels | 3 adder levels, docs/rtl-m3.md §9.2).
//   G..G+2         integer scale: (1) Q4_K sc_g * S_g (6u x 17s), Q8_0 block sums select by the
//                  QS-line parity; (2) Q4_K part = sum of 4; (3) Q4_K half 0: part0 := part,
//                  half 1: isum = part + part0 (26 bits). msum: pair sums, m_k * pair_k, two
//                  adder levels (26 bits) from M0, delayed MAC_LAT - 1 to meet isum.
//   G+3..G+5       2 x tsr_mvm_i2f (3 stages: magnitude | lzc + normalise | RNE round), bit-exact
//                  tsr_i32_to_f32 (tb/mvmback exhaustive sweep over all 2^26 inputs).
//   F = max(G+6, M0+MUL) [12]   second products fmul(d*Dx, f32(sum)) (the shorter of the f32 /
//                  d*Dx paths is delayed to F).
//   P = F+MUL [16] products: Q8_0 terms b, b+1 -> accumulate issue (A8 = P);
//                  Q4_K fsub(t1, t2) (tsr_fp_add sub = 1) -> accumulate issue A4 = P+ADD [21].
//   accumulate     two tsr_fp_add lanes per token; lane l holds partials p[2k+l] in slot k, two
//                  partial sets (row parity). Operand = +0.0 for the first term of a partial in
//                  the row (term index < 16: no reset-before-use race), else the slot register,
//                  forwarded from the adder output when that result is being written in the issue
//                  cycle (ADD_LAT 8: two terms of one partial are exactly 8 cycles apart).
//   tree           when a row's last add has been written, its set is pending; the level-1 adder
//                  reads it pair by pair (pair k = slot k of lanes 0 and 1 = (p[2k], p[2k+1])) on
//                  8 consecutive cycles and clears each slot to +0.0 as it reads it (so partials
//                  without terms read +0.0 next time; reset clears every slot). Levels 2..4 are
//                  streaming: one tsr_fp_add per level and token; a level issues on every second
//                  result of the level below (hold register + toggle), which is exactly the
//                  tsr_tree16 pair order. Level-4 results go to the result FIFO.
//   control        the slot write enables / clears, accumulate operand selects, level-1 operand
//                  selects and hold enables reach each token from that token's own registered
//                  copies of the shared control (cycle-identical, "Per-token control copies").
//   result FIFO    RES_DEPTH = ceil(HOLD/8) + 1 entries of T_MAX x 32 bits (HOLD = row_last accept ->
//                  result pop with res_ready = 1, Q4_K: A4 + 5*ADD_LAT + 10 [56] cycles -> 8
//                  entries); res_valid is a flop, res_y the head entry (registered read pointer).
//
// Flow control (beat_ready is a flop computed one cycle ahead, so it cannot look at the beat):
//   (a) result space: a row's FIFO entry is reserved when its row_last beat is accepted;
//       beat_ready = (reserved entries after this edge) < RES_DEPTH (any beat may be a row_last).
//   (b) partial sets / tree: the first beat of a row (the beat after a row_last) is accepted only
//       >= 8 cycles after the previous row's first beat. Proof sketch (F_r / E_r = accept cycles
//       of row r's first / last beat, D = accept -> issue, L = ADD_LAT): row r's set is complete
//       at C_r = E_r + D + L and read by level 1 in [S_r, S_r + 7] with S_r = max(C_r + 2,
//       S_{r-1} + 8) <= max_j(E_{r-j} + D + L + 2 + 8j); row r+2 reuses the set and writes slot k
//       first at >= F_{r+2} + D + L + k, and F_{r+2} >= F_{r-j+1} + 8(j+1) >= E_{r-j} + 9 + 8j, so
//       every slot is read (and cleared) >= 7 cycles before row r+2 writes it; a completed set
//       waits for level 1 at most until S_r + 8 (at most one waiting set). A row of >= 8 quant
//       beats takes >= 8 cycles, so (b) never throttles such rows.
//   RB_MIN (docs/rtl-m3.md §7.5): with tagged beats offered back to back and res_ready = 1, no
//   beat_ready drop for rows of >= 8 quant beats = ROW_BEATS >= 9 for both formats (Q8_0 K 480 /
//   512, Q4_K K 1024: 9 beats incl. META), any T and every latency parameter value: RB_MIN = 9.
//   With the front end's META cycles (one stream cycle per chunk, tsr_mvm_front) Q8_0 rows of
//   ROW_BEATS 8 (K 416 / 448, 7 quant beats + META) also run without a drop (row starts 8 cycles
//   apart; beat_ready is then low only in the META cycle, where no beat is offered). Shorter rows
//   run at one row per 8 cycles (exactly 8.00 measured for Q8_0 K 32..224, Q4_K K 256..768).
//
// Measured (tb/mvmback, every configuration: T_MAX 1 / 2 / 3 / 5 / 8, latencies 1 .. 8):
//   - 1.000 cycles per quant beat for rows of ROW_BEATS >= 9 (both formats, T = 1 and T = T_MAX,
//     K up to K_MAX), 0 beat_ready drops; ROW_BEATS 8 back to back: exactly 1 stall per row;
//   - row_last accept -> res_valid (res_ready = 1): Q8_0 HOLD - ADD_LAT, Q4_K HOLD cycles
//     (defaults 51 / 56; all latencies 1: 25 / 26; all 8: 81 / 89); first QS / DBS read 1 cycle
//     after a beat is accepted;
//   - RES_DEPTH 8 at the defaults (5 / 13 at all latencies 1 / 8); with res_ready = 0 exactly
//     RES_DEPTH rows are taken, then beat_ready stays 0.
// Logic depth (by construction, not yet timed: Vivado is part C): every stage register-bounded;
//   the deepest segments are the 16:1 partial read + first / forward mux into the accumulate adder
//   input (~4 LUT), the 6u x 17s products (LUT multipliers unless mapped to DSP), the i2f
//   normalise stage (26-bit lzc + barrel shift, ~6 LUT) and the generic tsr_mac_lane at MAC_LAT 3
//   (tb/mvmback mac_lane_structure): the 8 x 8 products from the flops above into the product
//   register (LUT multipliers, or DSP48E2 with the product register as MREG), then 2 and 3 adder
//   levels of 17..21 bits (DSP cascades in M4). The RAM-to-MAC route of x has no logic; if it
//   needs a fabric register, ACT_RD_LAT + 1 provides it. beat_ready: front-end beat.row_last
//   flop -> reservation count compare -> flop (~3 LUT).
//
// Cost per token (generic, counted from the RTL, not synthesised): 4 tsr_fp_mul, 7 tsr_fp_add
// (2 accumulate, 1 Q4_K subtract, 4 tree levels; the streaming tree adds 3 adders, ~1.3 K LUT /
// 1 K FF, to the docs/rtl-m3.md §9.3 estimate), 2 tsr_mvm_i2f, 12 small multipliers (4 + 8 x
// 6u * 17s), the MAC products / trees of 128 lanes (tsr_mac_lane; at MAC_LAT 3 2708 flops: 2048
// product register, 576 level-2 sums, 84 group sums), 2 x 2 x 8 x 32 = 1024 partial flops (+ 3 x
// 32 tree hold, 32 dxh, 25 part0), ~1.4 K integer-stage / msum / i2f flops, the d*Dx or f32
// alignment delay (64 x |FI - FD| bits: 320 at the defaults), RES_DEPTH x 32 result FIFO bits,
// 83 control-copy flops; about 8.5 K FF per token at the defaults. Shared per cluster: the weight
// delay line 768 x ACT_RD_LAT bits (1536 at the defaults, SRL-mappable), the placed-weight
// register wm_q (1024), d / sc / m delay lines (~470 bits), ~400 control / tag flops.
// Possible saving (not done): merge tree levels 2..4 into one adder with a fixed 8-cycle frame
// schedule (level-1 starts aligned to a frame; 7 ops at distinct residues) - 2 adders per token
// less, at the cost of a schedule that depends on ADD_LAT and up to 7 cycles more latency.
// ================================================================================================

// The two helpers below live in this file (one filelist entry, like tsr_rslice in tsr_mem_arb.sv);
// targeted waiver for the file-name rule only.
/* verilator lint_off DECLFILENAME */
// ------------------------------------------------------------------------------------------------
// tsr_mvm_dly — fixed delay line of N cycles (N = 0: wire). RST = 1: the registers reset to 0
// (valid / tag bits); RST = 0: no reset (datapath; maps to SRLs). Helper of tsr_mvm_back.
// ------------------------------------------------------------------------------------------------
module tsr_mvm_dly #(
  parameter int unsigned W   = 1,
  parameter int unsigned N   = 1,
  parameter bit          RST = 1'b0
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  if (N == 0) begin : g_wire
    assign q = d;
    wire unused_ok = &{1'b0, clk, rst_n};
  end else begin : g_reg
    logic [W-1:0] r [N];
    if (RST) begin : g_rst
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          for (int dly_i = 0; dly_i < int'(N); dly_i++) r[dly_i] <= '0;
        end else begin
          r[0] <= d;
          for (int dly_i = 1; dly_i < int'(N); dly_i++) r[dly_i] <= r[dly_i-1];
        end
      end
    end else begin : g_nrst
      always_ff @(posedge clk) begin
        r[0] <= d;
        for (int dly_j = 1; dly_j < int'(N); dly_j++) r[dly_j] <= r[dly_j-1];
      end
      wire unused_ok = &{1'b0, rst_n};
    end
    assign q = r[N-1];
  end
endmodule

// ------------------------------------------------------------------------------------------------
// tsr_mvm_i2f — IW-bit two's complement integer -> f32, round to nearest even, 3 register stages,
// free running (y = conversion of the v presented 3 cycles earlier). Bit-exact with
// tsr_fp_pkg::tsr_i32_to_f32(sext(v)) for every IW-bit v, 25 <= IW <= 32 (tb/mvmback
// test_mvm_back_i2f_sweep: all 2^26 inputs at IW = 26). docs/rtl-m3.md §8.2 split:
//   stage 1  sign, magnitude |v| (IW-bit carry chain)
//   stage 2  leading-zero count and left normalisation (priority encoder + barrel shift)
//   stage 3  RNE increment of the 23-bit fraction (guard = the bit below it, sticky = the rest),
//            exponent 126 + IW - lz + carry; 0 -> +0.0
// Helper of tsr_mvm_back (isum / msum, 26 bits).
// ------------------------------------------------------------------------------------------------
module tsr_mvm_i2f #(
  parameter int unsigned IW = 26
) (
  input  logic          clk,
  input  logic [IW-1:0] v,
  output logic [31:0]   y
);
  if (IW < 25 || IW > 32) begin : g_chk
    $fatal(1, "tsr_mvm_i2f: IW must be in [25, 32]");
  end
  localparam int unsigned LZW = $clog2(IW);

  // stage 1
  logic          neg1;
  logic [IW-1:0] mag1;
  always_ff @(posedge clk) begin
    neg1 <= v[IW-1];
    mag1 <= v[IW-1] ? (~v + IW'(1)) : v;      // -2^(IW-1) -> 2^(IW-1) (fits IW bits unsigned)
  end

  // stage 2
  logic [LZW-1:0] lz;
  always_comb begin
    lz = LZW'(IW - 1);
    for (int lzc_i = 0; lzc_i < int'(IW); lzc_i++) begin
      if (mag1[lzc_i]) lz = LZW'(int'(IW) - 1 - lzc_i);   // the highest set bit wins
    end
  end
  logic           neg2, zero2;
  logic [LZW-1:0] lz2;
  logic [IW-1:0]  nrm2;
  always_ff @(posedge clk) begin
    neg2  <= neg1;
    zero2 <= (mag1 == '0);
    lz2   <= lz;
    nrm2  <= mag1 << lz;                       // nrm2[IW-1] = 1 unless zero
  end

  // stage 3
  logic [22:0] m23;
  logic        g, st;
  assign m23 = nrm2[IW-2 -: 23];
  assign g   = nrm2[IW-25];
  if (IW >= 26) begin : g_st
    assign st = |nrm2[IW-26:0];
  end else begin : g_no_st
    assign st = 1'b0;
  end
  wire        inc = g & (st | m23[0]);
  wire [23:0] fr  = {1'b0, m23} + {23'd0, inc};      // fr[23]: carry into the next binade
  wire [7:0]  ex  = 8'(126 + IW) - 8'(lz2) + {7'd0, fr[23]};
  always_ff @(posedge clk) begin
    y <= zero2 ? 32'd0 : {neg2, ex, fr[22:0]};
  end
endmodule

/* verilator lint_on DECLFILENAME */

// ------------------------------------------------------------------------------------------------
// tsr_mvm_back
// ------------------------------------------------------------------------------------------------
module tsr_mvm_back
  import tsr_pkg::*;
#(
  parameter  int unsigned K_MAX      = TSR_DEF_K_MAX,
  parameter  int unsigned T_MAX      = TSR_DEF_T_MAX,
  parameter  int unsigned ACT_RD_LAT = TSR_DEF_ACT_RD_LAT,
  parameter  int unsigned MAC_LAT    = TSR_DEF_MAC_LAT,
  parameter  int unsigned MUL_LAT    = TSR_DEF_FMUL_LAT,
  parameter  int unsigned ADD_LAT    = TSR_DEF_FADD_LAT,
  localparam int unsigned QS_WORDS   = tsr_actbuf_qs_words(K_MAX),
  localparam int unsigned DBS_LINES  = tsr_actbuf_dbs_lines(K_MAX),
  localparam int unsigned QS_AW      = (QS_WORDS > 1) ? $clog2(QS_WORDS) : 1,
  localparam int unsigned DBS_AW     = (DBS_LINES > 1) ? $clog2(DBS_LINES) : 1
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // ---- per-MVM configuration (stable while beats flow) ----
  input  logic                          cfg_fmt,        // 0: Q8_0, 1: Q4_K
  input  logic [7:0]                    cfg_d_lines,    // QA D lines (BS lines follow them)

  // ---- tagged beats from the front end ----
  input  logic                          beat_valid,
  output logic                          beat_ready,
  input  tsr_mvm_beat_t                 beat,

  // ---- activation buffer reads ----
  output logic                          qs_re,
  output logic [QS_AW-1:0]              qs_addr,
  input  logic [T_MAX-1:0][1023:0]      qs_data,
  output logic                          dbs_re,
  output logic [DBS_AW-1:0]             dbs_addr,
  input  logic [T_MAX-1:0][511:0]       dbs_data,

  // ---- row results ----
  output logic                          res_valid,
  input  logic                          res_ready,
  output logic [T_MAX-1:0][31:0]        res_y,

  output logic                          idle
);

  if (T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH) begin : g_chk_t
    $fatal(1, "tsr_mvm_back: T_MAX must be in [1, 8]");
  end
  if (ACT_RD_LAT < 1 || MAC_LAT < 1 || MUL_LAT < 1 || ADD_LAT < 1 || ADD_LAT > 8) begin : g_chk_lat
    $fatal(1, "tsr_mvm_back: latencies >= 1 and ADD_LAT <= 8 required");
  end

  // ---------------------------------------------------------------------------------------------
  // Schedule (see the header)
  // ---------------------------------------------------------------------------------------------
  localparam int unsigned INT_LAT   = 3;                         // integer scale stages
  localparam int unsigned CVT_LAT   = 3;                         // tsr_mvm_i2f
  localparam int unsigned C_M0      = 1 + ACT_RD_LAT;            // activation data, MAC / d*Dx operands
  localparam int unsigned C_G       = C_M0 + MAC_LAT;            // group sums
  localparam int unsigned C_CV      = C_G + INT_LAT;             // i2f inputs (isum / msum / Q8_0 sums)
  localparam int unsigned C_FI      = C_CV + CVT_LAT;            // f32 sums
  localparam int unsigned C_FD      = C_M0 + MUL_LAT;            // d*Dx
  localparam int unsigned C_F       = (C_FI > C_FD) ? C_FI : C_FD;   // second product operands
  localparam int unsigned C_P       = C_F + MUL_LAT;             // products
  localparam int unsigned C_A8      = C_P;                       // Q8_0 accumulate issue
  localparam int unsigned C_A4      = C_P + ADD_LAT;             // Q4_K accumulate issue (after fsub)
  localparam int unsigned NCTL      = C_A4;                      // tag pipeline cycles 1..C_A4
  localparam int unsigned MS_LAT    = 4;                         // msum stages from M0
  // row_last accept edge -> result pop edge with res_ready = 1 (Q4_K, the longer format):
  // issue A4, last add written A4 + L, level 1 starts 2 cycles later, level 4 issues 3L + 7 later,
  // its result L later, FIFO write, res_valid one cycle after it (measured: tb/mvmback
  // back_throughput "latency row_last->result" = HOLD for Q4_K, HOLD - ADD_LAT for Q8_0).
  localparam int unsigned HOLD      = C_A4 + 5 * ADD_LAT + 10;
  // Rows ending every 8 cycles hold ceil(HOLD / 8) reservations right after a row_last is taken,
  // and beat_ready (a flop) needs a free entry for whatever beat comes next: this is the smallest
  // depth that never drops beat_ready at RB_MIN (tb/mvmback back_throughput, every config; one
  // entry less drops it, e.g. HOLD 26 / 89 at all latencies 1 / 8).
  localparam int unsigned RES_DEPTH = (HOLD + 7) / 8 + 1;
  localparam int unsigned RES_AW    = $clog2(RES_DEPTH);
  localparam int unsigned ROW_GAP   = 8;                         // row starts >= 8 cycles apart

  // ---------------------------------------------------------------------------------------------
  // Input stage and flow control
  // ---------------------------------------------------------------------------------------------
  logic                ready_q;      // beat_ready
  logic                at_start_q;   // the next accepted beat is the first beat of a row
  logic [3:0]          since_q;      // cycles since the last row start was accepted (saturates at 8)
  logic                rset_q;       // partial set of the current / next row (row parity)
  logic [RES_AW:0]     resv_q;       // result entries reserved (row_last accepted, not yet popped)
  logic                idle_q;
  logic                res_valid_q;

  wire acc      = beat_valid & ready_q;
  wire acc_last = acc & beat.row_last;
  wire pop      = res_valid_q & res_ready;

  logic [RES_AW:0] resv_n;
  logic            at_start_n;
  logic [3:0]      since_n;
  always_comb begin
    resv_n     = resv_q + (RES_AW+1)'(acc_last) - (RES_AW+1)'(pop);
    at_start_n = acc ? beat.row_last : at_start_q;
    since_n    = (acc & at_start_q) ? 4'd1 : ((since_q >= 4'(ROW_GAP)) ? 4'(ROW_GAP) : since_q + 4'd1);
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ready_q    <= 1'b0;
      at_start_q <= 1'b1;
      since_q    <= 4'(ROW_GAP);
      rset_q     <= 1'b0;
      resv_q     <= '0;
      idle_q     <= 1'b1;
    end else begin
      ready_q    <= (resv_n < (RES_AW+1)'(RES_DEPTH)) & ~(at_start_n & (since_n < 4'(ROW_GAP)));
      at_start_q <= at_start_n;
      since_q    <= since_n;
      rset_q     <= rset_q ^ acc_last;
      resv_q     <= resv_n;
      idle_q     <= at_start_n & (resv_n == '0);
    end
  end

  assign beat_ready = ready_q;
  assign idle       = idle_q;

  // ---- activation buffer read addresses (registered outputs) ----
  logic              qs_re_q, dbs_re_q;
  logic [QS_AW-1:0]  qs_addr_q;
  logic [DBS_AW-1:0] dbs_addr_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      qs_re_q  <= 1'b0;
      dbs_re_q <= 1'b0;
    end else begin
      qs_re_q  <= acc;
      dbs_re_q <= acc;
    end
  end
  always_ff @(posedge clk) begin
    if (acc) begin
      // Q4_K: QS word 2s + half; Q8_0: QS word b/4 (lines b/2 = {2w, 2w+1})
      qs_addr_q  <= cfg_fmt ? QS_AW'({beat.blk, beat.half}) : QS_AW'(beat.blk[TSR_MVM_BLK_W-1:2]);
      // Q4_K half 1: BS line d_lines + s/2; otherwise the D line of block / superblock blk >> 4
      dbs_addr_q <= (cfg_fmt & beat.half)
                    ? DBS_AW'(11'(cfg_d_lines) + 11'(beat.blk[TSR_MVM_BLK_W-1:1]))
                    : DBS_AW'(beat.blk[TSR_MVM_BLK_W-1:4]);
    end
  end
  assign qs_re    = qs_re_q;
  assign qs_addr  = qs_addr_q;
  assign dbs_re   = dbs_re_q;
  assign dbs_addr = dbs_addr_q;

  // ---- tag pipeline (cycles 1 .. NCTL) ----
  typedef struct packed {
    logic       rl;      // row_last
    logic       set;     // partial set of the row
    logic       first;   // term index < 16: the first term of its partial in the row
    logic       two;     // Q8_0: block b+1 present
    logic       half;    // Q4_K half
    logic [3:0] b4;      // blk[3:0]
  } ctl_t;

  logic [NCTL:1] cv;     // cv[c]: a beat is in cycle c of its pipeline
  ctl_t          cd [1:NCTL];

  always_ff @(posedge clk) begin
    if (!rst_n) cv <= '0;
    else        cv <= {cv[NCTL-1:1], acc};
  end
  always_ff @(posedge clk) begin
    cd[1] <= '{rl: beat.row_last, set: rset_q, first: (beat.blk[TSR_MVM_BLK_W-1:4] == '0),
               two: beat.two, half: beat.half, b4: beat.blk[3:0]};
    for (int ctl_c = 2; ctl_c <= int'(NCTL); ctl_c++) cd[ctl_c] <= cd[ctl_c-1];
  end

  // ---- wide per-beat fields: delay lines from cycle 0 (the input beat) ----
  logic [127:0][7:0] w_in;
  assign w_in = beat.w;
  logic [63:0][7:0]  wlo_in, wlo_p;     // lanes 0..63
  logic [63:0][3:0]  whi_in, whi_p;     // lanes 64..127, low nibble (Q8_0: 0; Q4_K: 0..15)
  for (genvar l = 0; l < 64; l++) begin : g_wpack
    assign wlo_in[l] = w_in[l];
    assign whi_in[l] = w_in[64 + l][3:0];
  end

  // cycle M0 - 1 = ACT_RD_LAT >= 1; the placement register below takes them into cycle M0
  tsr_mvm_dly #(.W(768), .N(C_M0 - 1)) u_dly_w (.clk, .rst_n, .d({whi_in, wlo_in}), .q({whi_p, wlo_p}));

  logic [31:0] da_m0, db_m0;
  tsr_mvm_dly #(.W(64), .N(C_M0)) u_dly_d (.clk, .rst_n, .d({beat.d_b, beat.d_a}), .q({db_m0, da_m0}));

  logic [TSR_MVM_GROUPS-1:0][5:0] sc_g;
  tsr_mvm_dly #(.W(24), .N(C_G)) u_dly_sc (.clk, .rst_n, .d(beat.sc), .q(sc_g));

  logic [7:0][5:0] m_m1;
  tsr_mvm_dly #(.W(48), .N(C_M0 + 1)) u_dly_m (.clk, .rst_n, .d(beat.m), .q(m_m1));

  // ---------------------------------------------------------------------------------------------
  // MAC lanes (cycle M0 -> G)
  // ---------------------------------------------------------------------------------------------
  // Weight placement (shared by all tokens), computed in cycle M0 - 1 into the register wm_q, so
  // both MAC operands come straight from flops (tsr_mac_lane has no input register; x is the
  // activation buffer's RAM output register). hs: Q8_0 beat on an odd QS line (b/2 odd): its 64
  // weights go to lanes 64..127, which see bytes 0..63 of that line in the 1024-bit QS word.
  // Q4_K: nibbles, upper nibbles forced 0.
  wire hs_p = ~cfg_fmt & cd[C_M0 - 1].b4[1];
  logic [127:0][7:0] wm_q;    // MAC weights in cycle M0 (no reset: datapath)
  always_ff @(posedge clk) begin
    for (int wp_l = 0; wp_l < 64; wp_l++) begin
      wm_q[wp_l]      <= hs_p ? 8'd0 : {wlo_p[wp_l][7:4] & {4{~cfg_fmt}}, wlo_p[wp_l][3:0]};
      wm_q[64 + wp_l] <= hs_p ? wlo_p[wp_l] : (cfg_fmt ? {4'd0, whi_p[wp_l]} : 8'd0);
    end
  end

  logic [TSR_MVM_GROUPS-1:0][T_MAX-1:0][TSR_MVM_SUM_W-1:0] gs;    // [group][token] at G
  for (genvar g = 0; g < TSR_MVM_GROUPS; g++) begin : g_mac
    logic [T_MAX-1:0][31:0][7:0] xg;
    for (genvar t = 0; t < T_MAX; t++) begin : g_x
      assign xg[t] = qs_data[t][256*g +: 256];
    end
    logic mac_ov;
    tsr_mac_lane #(.N(TSR_MVM_GROUP_LANES), .T(T_MAX), .LAT(MAC_LAT)) u_mac (
      .clk, .rst_n, .ce(1'b1), .in_valid(cv[C_M0]),
      .w(wm_q[32*g +: 32]), .x(xg), .out_valid(mac_ov), .y(gs[g])
    );
    wire unused_mac = &{1'b0, mac_ov};
  end

  // ---------------------------------------------------------------------------------------------
  // Accumulate issue control (shared): Q8_0 at A8 (both lanes), Q4_K at A4 (lane s % 2)
  // ---------------------------------------------------------------------------------------------
  logic [1:0] iv;          // lane l issues an add
  logic       iset, ifirst, irl;
  logic [2:0] islot;
  always_comb begin
    if (cfg_fmt) begin
      iv[0]  = cv[C_A4] & cd[C_A4].half & ~cd[C_A4].b4[0];
      iv[1]  = cv[C_A4] & cd[C_A4].half &  cd[C_A4].b4[0];
      iset   = cd[C_A4].set;
      islot  = cd[C_A4].b4[3:1];
      ifirst = cd[C_A4].first;
      irl    = cv[C_A4] & cd[C_A4].half & cd[C_A4].rl;
    end else begin
      iv[0]  = cv[C_A8];
      iv[1]  = cv[C_A8] & cd[C_A8].two;
      iset   = cd[C_A8].set;
      islot  = cd[C_A8].b4[3:1];
      ifirst = cd[C_A8].first;
      irl    = cv[C_A8] & cd[C_A8].rl;
    end
  end

  // Tags of the accumulate adders' outputs (ADD_LAT later): lane valids, the row's last add,
  // set and slot (both lanes issue to the same slot in a cycle). The last of the ADD_LAT stages is
  // explicit: ovp / osp (stage ADD_LAT - 1) are the next-cycle values that the per-token control
  // copies below are computed from.
  logic [1:0] ov;
  logic       ocmp, oset;
  logic [2:0] oslot;
  logic [2:0] ovp;         // {ov, ocmp} of the next cycle
  logic [3:0] osp;         // {oset, oslot} of the next cycle
  tsr_mvm_dly #(.W(3), .N(ADD_LAT - 1), .RST(1'b1)) u_dly_ov (.clk, .rst_n, .d({iv, irl}), .q(ovp));
  tsr_mvm_dly #(.W(4), .N(ADD_LAT - 1)) u_dly_os (.clk, .rst_n, .d({iset, islot}), .q(osp));
  always_ff @(posedge clk) begin
    if (!rst_n) {ov, ocmp} <= '0;
    else        {ov, ocmp} <= ovp;
  end
  always_ff @(posedge clk) {oset, oslot} <= osp;

  // ---------------------------------------------------------------------------------------------
  // Tree control (shared): level-1 reader, streaming levels 2..4
  // ---------------------------------------------------------------------------------------------
  logic [1:0] pend_q;      // set complete, waiting for level 1
  logic       l1_act_q;    // level 1 issues pair l1_k_q of set l1_set_q in this cycle
  logic       l1_set_q, l1_nxt_q;
  logic [2:0] l1_k_q;
  wire        l1_free = ~l1_act_q | (l1_k_q == 3'd7);
  wire        l1_go   = l1_free & pend_q[l1_nxt_q];
  // next values of l1_act_q / l1_set_q / l1_k_q (also loaded by the per-token copies)
  wire        l1_act_n = l1_go | (l1_act_q & (l1_k_q != 3'd7));
  wire        l1_set_n = l1_go ? l1_nxt_q : l1_set_q;
  wire  [2:0] l1_k_n   = l1_go ? 3'd0 : (l1_act_q ? l1_k_q + 3'd1 : l1_k_q);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      pend_q   <= '0;
      l1_act_q <= 1'b0;
      l1_set_q <= 1'b0;
      l1_nxt_q <= 1'b0;
      l1_k_q   <= '0;
    end else begin
      l1_act_q <= l1_act_n;
      l1_set_q <= l1_set_n;
      l1_k_q   <= l1_k_n;
      if (l1_go) l1_nxt_q <= ~l1_nxt_q;
      for (int pend_s = 0; pend_s < 2; pend_s++) begin
        pend_q[pend_s] <= (pend_q[pend_s] & ~(l1_go & (l1_nxt_q == 1'(pend_s)))) | (ocmp & (oset == 1'(pend_s)));
      end
    end
  end

  // level n results valid (n = 1..4) and the issue of level n+1 on every second result; t1p..t3p
  // (stage ADD_LAT - 1 of the level-1..3 valid lines) are the next-cycle values of t1v..t3v
  logic t1v, t2v, t3v, t4v;
  logic t1p, t2p, t3p;
  logic tg1_q, tg2_q, tg3_q;
  wire  i2v = t1v & tg1_q;
  wire  i3v = t2v & tg2_q;
  wire  i4v = t3v & tg3_q;
  tsr_mvm_dly #(.W(1), .N(ADD_LAT - 1), .RST(1'b1)) u_dly_t1 (.clk, .rst_n, .d(l1_act_q), .q(t1p));
  tsr_mvm_dly #(.W(1), .N(ADD_LAT - 1), .RST(1'b1)) u_dly_t2 (.clk, .rst_n, .d(i2v), .q(t2p));
  tsr_mvm_dly #(.W(1), .N(ADD_LAT - 1), .RST(1'b1)) u_dly_t3 (.clk, .rst_n, .d(i3v), .q(t3p));
  tsr_mvm_dly #(.W(1), .N(ADD_LAT), .RST(1'b1)) u_dly_t4 (.clk, .rst_n, .d(i4v), .q(t4v));

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      t1v   <= 1'b0;
      t2v   <= 1'b0;
      t3v   <= 1'b0;
      tg1_q <= 1'b0;
      tg2_q <= 1'b0;
      tg3_q <= 1'b0;
    end else begin
      t1v   <= t1p;
      t2v   <= t2p;
      t3v   <= t3p;
      tg1_q <= tg1_q ^ t1v;
      tg2_q <= tg2_q ^ t2v;
      tg3_q <= tg3_q ^ t3v;
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Per-token control copies (implementation only, no functional change): the accumulate / tree
  // control above fans out to all T_MAX token datapaths (k4: +0.05..0.10 ns paths from l1_k_q,
  // l1_act_q, the ov / t1v / t3v delay lines and cd[C_A4] into the slot registers, the adder
  // operand registers and the hold registers, docs/rtl-m3.md §14.3). Every token therefore
  // registers its own copies, loaded from the next-cycle values computed here once, so each copy
  // equals the shared signal it replaces in every cycle (reset copies reset with it, the others
  // load every cycle like the tag stages they copy; checked by the simulation assertions at the
  // end): no latency change. Format-dependent selections stay after the copies (cfg_fmt is not
  // registered here), so the A4 and A8 tags are copied separately.
  //   wen_n[l][s][k] = ov[l] & oset == s & oslot == k        slot write enable (next cycle)
  //   clr_n[s][k]    = l1_act_q & l1_set_q == s & l1_k_q == k level-1 clear-on-read (next cycle)
  //   byp4_n / byp8_n[l] = ov[l] & {oset, oslot} == {iset, islot} for the Q4_K (A4) / Q8_0 (A8)
  //                        issue tags (next cycle)
  //   hen_n[n-1]     = tnv & ~tgn_q                           hold register h<n> enable (next cycle)
  // ---------------------------------------------------------------------------------------------
  logic [1:0][1:0][7:0] wen_n;
  logic      [1:0][7:0] clr_n;
  logic [1:0]           byp4_n, byp8_n;
  logic [2:0]           hen_n;
  wire  [3:0]           a4_n = {cd[C_A4 - 1].set, cd[C_A4 - 1].b4[3:1]};   // {iset, islot} next cycle
  wire  [3:0]           a8_n = {cd[C_A8 - 1].set, cd[C_A8 - 1].b4[3:1]};
  always_comb begin
    for (int wn_l = 0; wn_l < 2; wn_l++) begin
      for (int wn_s = 0; wn_s < 2; wn_s++) begin
        for (int wn_k = 0; wn_k < 8; wn_k++) begin
          wen_n[wn_l][wn_s][wn_k] = ovp[1 + wn_l] & (osp == {1'(wn_s), 3'(wn_k)});
        end
      end
      byp4_n[wn_l] = ovp[1 + wn_l] & (osp == a4_n);
      byp8_n[wn_l] = ovp[1 + wn_l] & (osp == a8_n);
    end
    for (int cn_s = 0; cn_s < 2; cn_s++) begin
      for (int cn_k = 0; cn_k < 8; cn_k++) begin
        clr_n[cn_s][cn_k] = l1_act_n & ({l1_set_n, l1_k_n} == {1'(cn_s), 3'(cn_k)});
      end
    end
    hen_n[0] = t1p & ~(tg1_q ^ t1v);
    hen_n[1] = t2p & ~(tg2_q ^ t2v);
    hen_n[2] = t3p & ~(tg3_q ^ t3v);
  end
  // issue tags copied per token and lane: {set, slot, first} of stages A4 / A8
  wire  [4:0]           ta4_n = {a4_n, cd[C_A4 - 1].first};
  wire  [4:0]           ta8_n = {a8_n, cd[C_A8 - 1].first};

  // ---------------------------------------------------------------------------------------------
  // Per-token datapath
  // ---------------------------------------------------------------------------------------------
  logic [T_MAX-1:0][31:0] y4;      // level-4 results (valid with t4v)

  for (genvar t = 0; t < T_MAX; t++) begin : g_tok
    // ---------------- integer scale (G .. G+2) ----------------
    logic signed [20:0] s0, s1, s2, s3;
    assign s0 = gs[0][t];
    assign s1 = gs[1][t];
    assign s2 = gs[2][t];
    assign s3 = gs[3][t];

    // (1) Q4_K products sc_g * S_g (S_g fits 17 bits: |S| <= 32 * 15 * 128); Q8_0 block sums
    logic signed [22:0] is1_q [4];
    wire hs_g = ~cfg_fmt & cd[C_G].b4[1];
    always_ff @(posedge clk) begin
      if (cfg_fmt) begin
        is1_q[0] <= 23'($signed({1'b0, sc_g[0]}) * $signed(s0[16:0]));
        is1_q[1] <= 23'($signed({1'b0, sc_g[1]}) * $signed(s1[16:0]));
        is1_q[2] <= 23'($signed({1'b0, sc_g[2]}) * $signed(s2[16:0]));
        is1_q[3] <= 23'($signed({1'b0, sc_g[3]}) * $signed(s3[16:0]));
      end else begin
        is1_q[0] <= 23'(hs_g ? s2 : s0);       // block b
        is1_q[1] <= 23'(hs_g ? s3 : s1);       // block b + 1
        is1_q[2] <= '0;
        is1_q[3] <= '0;
      end
    end

    // (2) Q4_K part = sum of the 4 products (|part| < 2^24); Q8_0 pass
    logic signed [24:0] is2_q [2];
    always_ff @(posedge clk) begin
      is2_q[0] <= cfg_fmt ? (25'(is1_q[0]) + 25'(is1_q[1])) + (25'(is1_q[2]) + 25'(is1_q[3]))
                          : 25'(is1_q[0]);
      is2_q[1] <= 25'(is1_q[1]);
    end

    // (3) Q4_K: half 0 stores part0, half 1 forms isum = part0 + part1 (|isum| < 2^25); Q8_0 pass
    logic signed [24:0] part0_q;
    logic signed [25:0] is3_q [2];
    always_ff @(posedge clk) begin
      if (cv[C_G + 2] && cfg_fmt && !cd[C_G + 2].half) part0_q <= is2_q[0];
    end
    always_ff @(posedge clk) begin
      is3_q[0] <= cfg_fmt ? (26'(part0_q) + 26'(is2_q[0])) : 26'(is2_q[0]);
      is3_q[1] <= 26'(is2_q[1]);
    end

    // ---------------- msum (M0 .. M0+3, Q4_K half-1 beats) ----------------
    // BS elements 16*(s%2) + 2k, +1 of the BS line; pair sums 17 bits, m_k * pair 23 bits,
    // msum 26 bits (|msum| <= 8 * 63 * 65536 < 2^25 for any int16 BS)
    wire bsel = cd[C_M0].b4[0];
    logic signed [16:0] pair_q [8];
    logic signed [22:0] mp_q   [8];
    logic signed [23:0] m4_q   [4];
    logic signed [25:0] msum_q;
    always_ff @(posedge clk) begin
      for (int ms_k = 0; ms_k < 8; ms_k++) begin
        pair_q[ms_k] <= 17'($signed(dbs_data[t][256*bsel + 32*ms_k +: 16]))
                      + 17'($signed(dbs_data[t][256*bsel + 32*ms_k + 16 +: 16]));
      end
      for (int mp_k = 0; mp_k < 8; mp_k++) mp_q[mp_k] <= 23'($signed({1'b0, m_m1[mp_k]}) * pair_q[mp_k]);
      for (int m4_j = 0; m4_j < 4; m4_j++) m4_q[m4_j] <= 24'(mp_q[2*m4_j]) + 24'(mp_q[2*m4_j+1]);
      msum_q <= (26'(m4_q[0]) + 26'(m4_q[1])) + (26'(m4_q[2]) + 26'(m4_q[3]));
    end
    logic [25:0] msum_cv;      // msum in cycle C_CV
    tsr_mvm_dly #(.W(26), .N(C_CV - C_M0 - MS_LAT)) u_dly_ms (.clk, .rst_n, .d(msum_q), .q(msum_cv));

    // ---------------- int -> f32 (C_CV .. C_FI) ----------------
    logic [31:0] f0, f1, f0_f, f1_f;
    tsr_mvm_i2f #(.IW(26)) u_i2f0 (.clk, .v(is3_q[0]), .y(f0));
    tsr_mvm_i2f #(.IW(26)) u_i2f1 (.clk, .v(cfg_fmt ? msum_cv : is3_q[1]), .y(f1));
    tsr_mvm_dly #(.W(64), .N(C_F - C_FI)) u_dly_f (.clk, .rst_n, .d({f1, f0}), .q({f1_f, f0_f}));

    // ---------------- d * Dx (M0 .. FD) ----------------
    logic [511:0] dline;
    assign dline = dbs_data[t];
    wire [31:0] dx_even = dline[64*cd[C_M0].b4[3:1] +: 32];        // element (b & 15) & ~1
    wire [31:0] dx_odd  = dline[64*cd[C_M0].b4[3:1] + 32 +: 32];   // element (b & 15) | 1
    logic [31:0] dxh_q;    // Q4_K: Dx_t[s] from the half-0 beat's D read
    always_ff @(posedge clk) begin
      if (cv[C_M0] && cfg_fmt && !cd[C_M0].half) dxh_q <= cd[C_M0].b4[0] ? dx_odd : dx_even;
    end

    logic [31:0] dd0, dd1, dd0_f, dd1_f;
    logic        dd0_v, dd1_v;
    tsr_fp_mul #(.LAT(MUL_LAT)) u_mul_a0 (
      .clk, .rst_n, .ce(1'b1), .in_valid(cv[C_M0]),
      .a(da_m0), .b(cfg_fmt ? dxh_q : dx_even), .out_valid(dd0_v), .y(dd0)
    );
    tsr_fp_mul #(.LAT(MUL_LAT)) u_mul_a1 (
      .clk, .rst_n, .ce(1'b1), .in_valid(cv[C_M0]),
      .a(db_m0), .b(cfg_fmt ? dxh_q : dx_odd), .out_valid(dd1_v), .y(dd1)
    );
    tsr_mvm_dly #(.W(64), .N(C_F - C_FD)) u_dly_dd (.clk, .rst_n, .d({dd1, dd0}), .q({dd1_f, dd0_f}));

    // ---------------- second products (F .. P) and the Q4_K subtraction (P .. A4) ----------------
    logic [31:0] pb0, pb1, qd;
    logic        pb0_v, pb1_v, qd_v;
    tsr_fp_mul #(.LAT(MUL_LAT)) u_mul_b0 (
      .clk, .rst_n, .ce(1'b1), .in_valid(cv[C_F]), .a(dd0_f), .b(f0_f), .out_valid(pb0_v), .y(pb0)
    );
    tsr_fp_mul #(.LAT(MUL_LAT)) u_mul_b1 (
      .clk, .rst_n, .ce(1'b1), .in_valid(cv[C_F]), .a(dd1_f), .b(f1_f), .out_valid(pb1_v), .y(pb1)
    );
    tsr_fp_add #(.LAT(ADD_LAT)) u_sub (
      .clk, .rst_n, .ce(1'b1), .in_valid(cv[C_P]), .a(pb0), .b(pb1), .sub(1'b1),
      .out_valid(qd_v), .y(qd)
    );

    // ---------------- this token's control copies (see "Per-token control copies") ----------------
    // tc_wen_q == {ov[l] & oset == s & oslot == k}, tc_clr_q == {l1_act_q & l1_set_q == s &
    // l1_k_q == k}, tc_byp4_q / tc_byp8_q == byp4 / byp8, tc_a4_q / tc_a8_q[l] == {set, slot, first}
    // of cd[C_A4] / cd[C_A8], tc_l1_q[l] == {l1_set_q, l1_k_q}, tc_hen_q == hold enables; `keep`:
    // synthesis must not merge the copies of different tokens back into one register.
    (* keep = "true" *) logic [1:0][1:0][7:0] tc_wen_q;
    (* keep = "true" *) logic      [1:0][7:0] tc_clr_q;
    (* keep = "true" *) logic [1:0]           tc_byp4_q, tc_byp8_q;
    (* keep = "true" *) logic [2:0]           tc_hen_q;
    (* keep = "true" *) logic [1:0][3:0]      tc_l1_q;
    (* keep = "true" *) logic [1:0][4:0]      tc_a4_q, tc_a8_q;
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        tc_wen_q  <= '0;
        tc_clr_q  <= '0;
        tc_byp4_q <= '0;
        tc_byp8_q <= '0;
        tc_hen_q  <= '0;
        tc_l1_q   <= '0;
      end else begin
        tc_wen_q  <= wen_n;
        tc_clr_q  <= clr_n;
        tc_byp4_q <= byp4_n;
        tc_byp8_q <= byp8_n;
        tc_hen_q  <= hen_n;
        tc_l1_q   <= {2{l1_set_n, l1_k_n}};
      end
    end
    always_ff @(posedge clk) begin
      tc_a4_q <= {2{ta4_n}};
      tc_a8_q <= {2{ta8_n}};
    end

    // ---------------- accumulators: 2 lanes x 2 sets x 8 slots ----------------
    logic [31:0] pr_q [2][2][8];     // [lane][set][slot]
    logic [31:0] acc_y [2];
    logic [1:0]  acc_v;
    for (genvar l = 0; l < 2; l++) begin : g_lane
      wire [31:0] term = cfg_fmt ? qd : ((l == 0) ? pb0 : pb1);
      // == {iset, islot, ifirst} and ov[l] & {oset, oslot} == {iset, islot}
      wire [4:0]  ltag = cfg_fmt ? tc_a4_q[l] : tc_a8_q[l];
      wire        lbyp = cfg_fmt ? tc_byp4_q[l] : tc_byp8_q[l];
      wire        lset = ltag[4];
      wire [2:0]  lslot = ltag[3:1];
      wire        lfirst = ltag[0];
      wire [31:0] opnd = lfirst ? 32'd0 : (lbyp ? acc_y[l] : pr_q[l][lset][lslot]);
      tsr_fp_add #(.LAT(ADD_LAT)) u_acc (
        .clk, .rst_n, .ce(1'b1), .in_valid(iv[l]), .a(opnd), .b(term), .sub(1'b0),
        .out_valid(acc_v[l]), .y(acc_y[l])
      );
      always_ff @(posedge clk) begin
        for (int pr_s = 0; pr_s < 2; pr_s++) begin
          for (int pr_k = 0; pr_k < 8; pr_k++) begin
            if (!rst_n || tc_clr_q[pr_s][pr_k]) begin
              pr_q[l][pr_s][pr_k] <= 32'd0;                 // reset / clear-on-read: +0.0
            end else if (tc_wen_q[l][pr_s][pr_k]) begin
              pr_q[l][pr_s][pr_k] <= acc_y[l];
            end
          end
        end
      end
    end

    // ---------------- tree: level 1 (pairs of the pending set), levels 2..4 streaming ----------------
    logic [31:0] y1, y2, y3, h1_q, h2_q, h3_q;
    logic [3:0]  tv;
    tsr_fp_add #(.LAT(ADD_LAT)) u_l1 (
      .clk, .rst_n, .ce(1'b1), .in_valid(l1_act_q),
      .a(pr_q[0][tc_l1_q[0][3]][tc_l1_q[0][2:0]]), .b(pr_q[1][tc_l1_q[1][3]][tc_l1_q[1][2:0]]),
      .sub(1'b0),
      .out_valid(tv[0]), .y(y1)
    );
    tsr_fp_add #(.LAT(ADD_LAT)) u_l2 (
      .clk, .rst_n, .ce(1'b1), .in_valid(i2v), .a(h1_q), .b(y1), .sub(1'b0), .out_valid(tv[1]), .y(y2)
    );
    tsr_fp_add #(.LAT(ADD_LAT)) u_l3 (
      .clk, .rst_n, .ce(1'b1), .in_valid(i3v), .a(h2_q), .b(y2), .sub(1'b0), .out_valid(tv[2]), .y(y3)
    );
    tsr_fp_add #(.LAT(ADD_LAT)) u_l4 (
      .clk, .rst_n, .ce(1'b1), .in_valid(i4v), .a(h3_q), .b(y3), .sub(1'b0), .out_valid(tv[3]), .y(y4[t])
    );
    always_ff @(posedge clk) begin
      if (tc_hen_q[0]) h1_q <= y1;       // left operand of the next pair (t1v & ~tg1_q)
      if (tc_hen_q[1]) h2_q <= y2;
      if (tc_hen_q[2]) h3_q <= y3;
    end

    // fp units' own valid outputs: the shared tag pipelines above carry the control
    wire unused_tok = &{1'b0, dd0_v, dd1_v, pb0_v, pb1_v, qd_v, acc_v, tv};
  end

  // ---------------------------------------------------------------------------------------------
  // Result FIFO
  // ---------------------------------------------------------------------------------------------
  logic [T_MAX-1:0][31:0] rf_q [RES_DEPTH];
  logic [RES_AW-1:0]      rf_wp_q, rf_rp_q;
  logic [RES_AW:0]        rf_cnt_q;
  wire  [RES_AW:0]        rf_cnt_n = rf_cnt_q + (RES_AW+1)'(t4v) - (RES_AW+1)'(pop);

  function automatic logic [RES_AW-1:0] rf_inc(logic [RES_AW-1:0] p);
    return (32'(p) == RES_DEPTH - 1) ? '0 : p + 1'b1;
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rf_wp_q     <= '0;
      rf_rp_q     <= '0;
      rf_cnt_q    <= '0;
      res_valid_q <= 1'b0;
    end else begin
      if (t4v) rf_wp_q <= rf_inc(rf_wp_q);
      if (pop) rf_rp_q <= rf_inc(rf_rp_q);
      rf_cnt_q    <= rf_cnt_n;
      res_valid_q <= (rf_cnt_n != '0);
    end
  end
  always_ff @(posedge clk) begin
    if (t4v) rf_q[rf_wp_q] <= y4;
  end

  assign res_valid = res_valid_q;
  assign res_y     = rf_q[rf_rp_q];

  // beat.last: the cluster counts rows itself; lanes 64..127 above the nibble are 0 by contract
  logic [63:0][3:0] w_hi_top;
  for (genvar l = 0; l < 64; l++) begin : g_unused_w
    assign w_hi_top[l] = w_in[64 + l][7:4];
  end
  wire unused_ok = &{1'b0, beat.last, w_hi_top};

  // ---------------------------------------------------------------------------------------------
  // Simulation checks (armed by the first reset edge)
  // ---------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic       a_on_q;
  logic       cfg_fmt_q;
  logic [7:0] cfg_dl_q;
  logic       half_exp_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    cfg_fmt_q <= cfg_fmt;
    cfg_dl_q  <= cfg_d_lines;
    if (!rst_n) half_exp_q <= 1'b0;
    else if (acc && cfg_fmt) half_exp_q <= ~beat.half;
  end

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (!idle_q) begin
        assert (cfg_fmt == cfg_fmt_q && cfg_d_lines == cfg_dl_q)
          else $error("%m: cfg_fmt / cfg_d_lines changed while the back end is busy");
      end
      if (acc) begin
        assert (cfg_fmt ? (beat.half == half_exp_q && !beat.two) : !beat.half)
          else $error("%m: beat half / two tags inconsistent with the format");
        assert (!(cfg_fmt && beat.row_last && !beat.half)) else $error("%m: Q4_K row ends on a half-0 beat");
      end
      if (ocmp) assert (!pend_q[oset]) else $error("%m: partial set %0d completed while still pending", oset);
      if (t4v) assert (rf_cnt_q < (RES_AW+1)'(RES_DEPTH) || pop) else $error("%m: result FIFO overflow");
      if (pop) assert (resv_q != '0) else $error("%m: result popped without a reservation");
      assert (rf_cnt_q <= resv_q) else $error("%m: more results than reservations");
      // a row start is accepted >= ROW_GAP cycles after the previous one (flow-control rule (b))
      if (acc && at_start_q) assert (since_q >= 4'(ROW_GAP)) else $error("%m: row start spacing violated");
      // no clear and accumulate write of the same slot in one cycle (proof in the header)
      if (l1_act_q && (ov != 2'b00)) begin
        assert (!(oset == l1_set_q && oslot == l1_k_q)) else $error("%m: accumulate write into a slot being cleared");
      end
      // the level-1 reader only reads complete sets, and nothing of the read set is in flight
      if (l1_act_q && (iv != 2'b00) && !ifirst) begin
        assert (iset != l1_set_q) else $error("%m: accumulate read of the set level 1 is reading");
      end
    end
  end

  // Per-token control copies == the shared control they replace, in every cycle after the first
  // reset edge (also during reset and for inputs outside the contract): no latency change.
  logic [1:0][1:0][7:0] wen_ref;
  logic      [1:0][7:0] clr_ref;
  logic [1:0]           byp_ref;
  for (genvar s = 0; s < 2; s++) begin : g_ref_s
    for (genvar k = 0; k < 8; k++) begin : g_ref_k
      assign wen_ref[0][s][k] = ov[0] && oset == 1'(s) && oslot == 3'(k);
      assign wen_ref[1][s][k] = ov[1] && oset == 1'(s) && oslot == 3'(k);
      assign clr_ref[s][k]    = l1_act_q && l1_set_q == 1'(s) && l1_k_q == 3'(k);
    end
  end
  assign byp_ref[0] = ov[0] & (oset == iset) & (oslot == islot);
  assign byp_ref[1] = ov[1] & (oset == iset) & (oslot == islot);
  for (genvar t = 0; t < T_MAX; t++) begin : g_chk_tok
    always_ff @(posedge clk) begin
      if (a_on_q) begin
        assert (g_tok[t].tc_wen_q == wen_ref && g_tok[t].tc_clr_q == clr_ref
                && g_tok[t].tc_hen_q == {t3v & ~tg3_q, t2v & ~tg2_q, t1v & ~tg1_q}
                && g_tok[t].tc_l1_q == {2{l1_set_q, l1_k_q}})
          else $error("%m: token %0d slot / tree control copy differs from the shared control", t);
      end
    end
    for (genvar l = 0; l < 2; l++) begin : g_chk_lane
      always_ff @(posedge clk) begin
        if (a_on_q) begin
          assert ({g_tok[t].g_lane[l].lset, g_tok[t].g_lane[l].lslot, g_tok[t].g_lane[l].lfirst}
                  == {iset, islot, ifirst} && g_tok[t].g_lane[l].lbyp == byp_ref[l])
            else $error("%m: token %0d lane %0d issue control copy differs from the shared control", t, l);
        end
      end
    end
  end
`endif

endmodule
