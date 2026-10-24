// tsr_mac_lane.sv — Xilinx UltraScale+ (U200) variant of the MAC lane group: N signed 8-bit
// products summed, for T tokens that share the weights, with the products on DSP48E2 slices and
// two tokens packed per DSP (spec MVM 클러스터 마이크로아키텍처 "DSP 패킹").
//
// Owner: xtech agent (M3 part C; docs/rtl-m3.md §7.2, §9.2 item 4, §9.3). Selected by
// rtl/filelist_xilinx.f in place of rtl/tech/generic.
//
// Contract: identical to rtl/tech/generic/tsr_mac_lane.sv (same module name, parameters, ports,
// latency and reset / ce behaviour; tb/tech compares the two cycle by cycle, out_valid and y in
// every cycle):
//   y[t] = sum_{i < N} sext(w[i]) * sext(x[t][i])   (exact, two's complement, SUM_W bits)
//   fixed latency LAT >= 1 counted in edges with ce = 1, every register updates only at such
//   edges, reset clears the valid pipeline only, fully pipelined, no input register in front of
//   the first logic layer is assumed (the caller drives w and x from flops).
//
// Packing (WP486 INT8 packing, shared operand = the weight): tokens 2s (high) and 2s + 1 (low)
// of lane i share one DSP48E2 (slot s):
//     AD = sext(x[2s][i]) * 2^18 + sext(x[2s+1][i])   (27-bit pre-adder, A and D ports)
//     M  = AD * sext(w[i])                            (27 x 18 multiplier, B port)
//     P  = M + 2^17                                   (post-adder, constant C: sign correction)
//   The low product x[2s+1] w lies in [-16256, 16384], so L = x[2s+1] w + 2^17 lies in
//   [0, 2^18) and P = x[2s] w * 2^18 + L exactly: P[33:18] = x[2s] w and P[15:0] = x[2s+1] w
//   (16-bit two's complement, exact). Without the constant the high field would read
//   x[2s] w - 1 whenever the low product is negative (the borrow WP486 corrects in fabric); the
//   2^17 offset in the post-adder (a constant on the C port) folds that correction into the DSP,
//   so no fabric logic is spent on it. An odd T leaves token T - 1 alone in its slot: M = x w,
//   P = M. DSPs per instance: N x ceil(T / 2) (N = 32: T 1 / 2 / 3 / 8 -> 32 / 32 / 64 / 128).
//   Vivado 2024.2 (tb/tech/viv/tech_ooc.tcl, xcu200): exactly that many DSP48E2, the packed ones
//   as (C + ((D + A)' * B')')' with AMULTSEL = AD; the synthesised netlists match the generic
//   lane in post-synthesis simulation (tb/tech/viv/net_sim.sh).
//
// Pipeline: the logic layers are [AD] pre-adder -> [M] multiplier -> [P] post-adder ->
//   adder tree levels 1..LV (LV = clog2(N), N padded to 2^LV with zero products, level i sums are
//   16 + i bits) -> [OUT]. LAT - 1 internal registers are allocated in this order:
//     M (MREG), P (PREG), first tree cut, AD (ADREG; B gets its BREG 2nd stage), IN (AREG / DREG /
//     BREG 1st stage), further tree cuts, then plain output registers.
//   Tree cuts: TC of them after levels floor(c * LV / (TC + 1)), c = 1..TC (TC <= LV - 1).
//   N = 32:  LAT 1  no internal register (functional only)
//            LAT 2  M
//            LAT 3  M P | 5 levels -> OUT  (the core default MAC_LAT; operands -> pre-adder ->
//                   multiplier in one cycle and 12 logic levels P -> OUT: synthesis estimate
//                   only +0.29 / +0.62 ns before placement, not used for U200)
//            LAT 4  M P | 2 levels | 3 levels -> OUT
//            LAT 5  AD M P | 2 | 3       (U200 300 MHz: product inside the DSP, ADREG/MREG/PREG)
//            LAT 6  IN AD M P | 2 | 3    (+ AREG / DREG / BREG 2: also isolates the activation
//                                         RAM / weight register -> DSP routes)
//            LAT 7  IN AD M P | 1 | 2 | 2,  LAT 8 | 1 | 1 | 1 | 2,  LAT 9 a cut after every
//                   level, LAT >= 10 plain output registers behind [OUT].
//   Every register of the product side sits where DSP48E2 has one (A1 / D / B1, AD + B2, M, P),
//   written per slot so Vivado can absorb them; datapath registers have ce and no reset (DSP
//   CE pins, RST tied off).
// Cost per instance (N = 32): 32 x ceil(T / 2) DSP48E2, T adder trees of 31 adders (as generic),
//   tree-cut registers T x (8 x 18) bits at one cut, output T x 21 bits, LAT valid flops.

module tsr_mac_lane #(
  parameter  int unsigned N     = 32,   // products summed (lanes of one group)
  parameter  int unsigned T     = 1,    // tokens sharing the weights
  parameter  int unsigned LAT   = 3,    // fixed latency, >= 1
  localparam int unsigned SUM_W = 16 + $clog2(N)
) (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          ce,
  input  logic                          in_valid,
  input  logic [N-1:0][7:0]             w,      // signed 8-bit weights
  input  logic [T-1:0][N-1:0][7:0]      x,      // signed 8-bit activations, per token
  output logic                          out_valid,
  output logic [T-1:0][SUM_W-1:0]       y       // signed sums, per token
);

  if (N < 1 || T < 1 || LAT < 1) begin : g_chk
    $fatal(1, "tsr_mac_lane: N >= 1, T >= 1 and LAT >= 1 required");
  end

  localparam int unsigned LV   = (N > 1) ? $clog2(N) : 0;     // adder levels
  localparam int unsigned NP   = 1 << LV;                     // padded lane count
  localparam int unsigned NS   = (T + 1) / 2;                 // DSP slots per lane
  localparam int unsigned TCMX = (LV > 1) ? LV - 1 : 0;       // possible cuts inside the tree

  // ---- register allocation (header): M, P, first tree cut, AD, IN, more tree cuts, extras ----
  localparam int unsigned NREG  = LAT - 1;
  localparam bit          R_M   = NREG >= 1;
  localparam bit          R_P   = NREG >= 2;
  localparam int unsigned RA0   = (NREG > 2) ? NREG - 2 : 0;
  localparam int unsigned TC1   = (RA0 >= 1 && TCMX >= 1) ? 1 : 0;
  localparam int unsigned RA1   = RA0 - TC1;
  localparam bit          R_AD  = RA1 >= 1;
  localparam int unsigned RA2   = R_AD ? RA1 - 1 : RA1;
  localparam bit          R_IN  = RA2 >= 1;
  localparam int unsigned RA3   = R_IN ? RA2 - 1 : RA2;
  localparam int unsigned TC    = TC1 + ((RA3 < TCMX - TC1) ? RA3 : TCMX - TC1);
  localparam int unsigned NXTRA = RA3 - (TC - TC1);           // plain output registers

  // 1 if a tree cut register follows adder level i (1 <= i <= LV - 1).
  function automatic bit tcut_after(int i);
    for (int tc_c = 1; tc_c <= int'(TC); tc_c++) begin
      if (tc_c * int'(LV) / (int'(TC) + 1) == i) return 1'b1;
    end
    return 1'b0;
  endfunction

  // ---- products (layer 0): exact 16-bit two's complement, zero for the padding lanes ----
  logic [T-1:0][NP-1:0][15:0] prod;

  for (genvar s = 0; s < NS; s++) begin : g_s
    localparam bit PAIR = (2 * s + 1 < T);    // tokens 2s (high) and 2s + 1 (low) share the DSP
    for (genvar k = 0; k < N; k++) begin : g_k
      // operands
      logic signed [7:0]  a_d, d_d, b_d, a_i, d_i, b_i, b_a;
      logic signed [26:0] ad_d, ad_q;
      // use_dsp on the product only: on P it would also pull the tree adders it feeds into DSPs
      (* use_dsp = "yes" *) logic signed [34:0] m_d, m_q;
      logic signed [35:0] p_d, p_q;

      assign a_d = x[2*s][k];
      assign b_d = w[k];
      if (PAIR) begin : g_pair
        assign d_d = x[2*s+1][k];
      end else begin : g_one
        assign d_d = '0;
      end

      // IN: A1 / D / B1 registers
      if (R_IN) begin : g_in
        always_ff @(posedge clk) begin
          if (ce) begin
            a_i <= a_d;
            d_i <= d_d;
            b_i <= b_d;
          end
        end
      end else begin : g_no_in
        assign a_i = a_d;
        assign d_i = d_d;
        assign b_i = b_d;
      end

      // AD: pre-adder (pair) or the A operand alone (single), ADREG and B's second register
      if (PAIR) begin : g_ad_pair
        assign ad_d = $signed({{1{a_i[7]}}, a_i, 18'd0}) + $signed({{19{d_i[7]}}, d_i});
      end else begin : g_ad_one
        assign ad_d = $signed({{19{a_i[7]}}, a_i});
      end
      if (R_AD) begin : g_ad
        always_ff @(posedge clk) begin
          if (ce) begin
            ad_q <= ad_d;
            b_a  <= b_i;
          end
        end
      end else begin : g_no_ad
        assign ad_q = ad_d;
        assign b_a  = b_i;
      end

      // M: 27 x 8 signed product (|M| < 2^33)
      assign m_d = ad_q * b_a;
      if (R_M) begin : g_m
        always_ff @(posedge clk) begin
          if (ce) m_q <= m_d;
        end
      end else begin : g_no_m
        assign m_q = m_d;
      end

      // P: post-adder with the packing constant
      if (PAIR) begin : g_p_pair
        assign p_d = 36'(m_q) + 36'sd131072;
      end else begin : g_p_one
        assign p_d = 36'(m_q);
      end
      if (R_P) begin : g_p
        always_ff @(posedge clk) begin
          if (ce) p_q <= p_d;
        end
      end else begin : g_no_p
        assign p_q = p_d;
      end

      // extraction (wiring only)
      if (PAIR) begin : g_x_pair
        assign prod[2*s][k]   = p_q[33:18];
        assign prod[2*s+1][k] = p_q[15:0];
        wire unused_p = &{1'b0, p_q[35:34], p_q[17:16]};
      end else begin : g_x_one
        assign prod[2*s][k] = p_q[15:0];
        wire unused_p = &{1'b0, p_q[35:16], d_i};
      end
    end
  end

  for (genvar t = 0; t < T; t++) begin : g_pad
    for (genvar k = N; k < NP; k++) begin : g_k
      assign prod[t][k] = '0;
    end
  end

  // ---- adder tree: g_l[i].d = combinational result of layer i, g_l[i].q = after its cut ----
  for (genvar i = 0; i <= LV; i++) begin : g_l
    localparam int unsigned NI = NP >> i;
    localparam int unsigned WI = 16 + i;
    logic [T-1:0][NI-1:0][WI-1:0] d;
    logic [T-1:0][NI-1:0][WI-1:0] q;

    if (i == 0) begin : g_prod
      assign d = prod;
    end else begin : g_add
      for (genvar t = 0; t < T; t++) begin : g_t
        for (genvar k = 0; k < NI; k++) begin : g_k
          // (WI-1)-bit signed operands sign-extended to WI bits: the sum cannot overflow
          assign d[t][k] = $signed({g_l[i-1].q[t][2*k][WI-2], g_l[i-1].q[t][2*k]})
                         + $signed({g_l[i-1].q[t][2*k+1][WI-2], g_l[i-1].q[t][2*k+1]});
        end
      end
    end

    if (i >= 1 && i < LV && tcut_after(i)) begin : g_cut
      always_ff @(posedge clk) begin
        if (ce) q <= d;
      end
    end else begin : g_pass
      assign q = d;
    end
  end

  // ---- [OUT] and extra output registers; valid pipeline (LAT stages, reset) ----
  logic [LAT-1:0]               v_q;
  logic [T-1:0][SUM_W-1:0]      o_y [NXTRA + 1];

  always_ff @(posedge clk) begin
    if (!rst_n)  v_q <= '0;
    else if (ce) v_q <= LAT'({v_q, in_valid});
  end
  for (genvar t = 0; t < T; t++) begin : g_out
    always_ff @(posedge clk) begin
      if (ce) o_y[0][t] <= g_l[LV].q[t][0];      // 16 + LV = SUM_W bits
    end
  end
  for (genvar e = 1; e <= NXTRA; e++) begin : g_xtra
    always_ff @(posedge clk) begin
      if (ce) o_y[e] <= o_y[e-1];
    end
  end

  assign out_valid = v_q[LAT-1];
  assign y         = o_y[NXTRA];

endmodule
