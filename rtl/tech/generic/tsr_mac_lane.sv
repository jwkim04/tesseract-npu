// tsr_mac_lane.sv — generic (vendor-neutral) MAC lane group: N signed 8-bit products summed,
// for T tokens that share the weights. Technology wrapper (spec System structure: tsr_mac_lane).
//
// Owner: back agent (docs/rtl-m3.md §2 block (back), contract §7.2).
//
// Contract:
//   y[t] = sum_{i < N} sext(w[i]) * sext(x[t][i])   (exact, two's complement, SUM_W bits)
//   for every token t < T. w and x are signed 8-bit (the full range -128..127; Q4_K weights are
//   0..15 in a signed lane). SUM_W = 16 + clog2(N) holds the extreme N * (-128) * (-128)
//   (N = 32: 21 bits, spec Numeric rules, integer widths: "Q8_0 isum signed 21 bits").
// Timing: fixed latency LAT >= 1 with ce, exactly like tsr_fp_mul: every register (valid and data)
//   updates only at edges with ce = 1; in_valid/w/x are sampled only at such edges; out_valid/y
//   are registers available LAT advancing edges after the operands were taken (operands present
//   in the cycle of advancing edge n -> result on out_valid/y in the cycle after advancing edge
//   n + LAT - 1). Reset clears the valid pipeline only. Fully pipelined (one beat per cycle), no
//   combinational input -> output path. The MVM back end instantiates TSR_MVM_GROUPS = 4 of these
//   per cluster (N = 32 lanes each, 128 lanes per beat) with T = T_MAX.
// Implementation freedom (this is the reason it is a tech wrapper, spec MVM cluster
//   microarchitecture "DSP packing"): generic = products + pipelined adder tree, LAT chosen so that
//   each stage is <= ~6 LUT levels at 300 MHz (product register, then about two adder levels per
//   stage). rtl/tech/xilinx (M4) may use DSP48E2 cascades (one DSP per product at T = 1) and pack
//   two tokens per DSP for T >= 2 (same weight, two activations); the ports and the latency
//   parameter stay the same.
//
// Generic pipeline. Logic layers: layer 0 = the N 8 x 8 products (16 bit), layers 1..LV = the
// binary adder tree (LV = clog2(N); N is padded to 2^LV with zero products; layer i sums are
// 16 + i bits wide, so no intermediate can overflow). There is no register in front of the
// products: the caller drives w and x straight from flops (tsr_mvm_back: its placed-weight
// register wm_q and the activation buffer's RAM output register; ACT_RD_LAT + 1 adds a fabric
// register in front of x if the RAM-to-MAC route needs one). Registers:
//   cuts   NCUT = min(LAT - 1, LV) registers inside the chain: the product register (after
//          layer 0) first, then NCUT - 1 cuts spread evenly over the LV adder levels (after
//          level floor(c * LV / NCUT), c = 1 .. NCUT - 1). N = 32: LAT 2 products | 5 levels;
//          LAT 3 (the default) products | 2 levels | 3 levels (docs/rtl-m3.md §9.2 item 4:
//          product register, about two adder levels per stage); LAT 4 products | 1 | 2 | 2;
//          LAT 5 products | 1 | 1 | 1 | 2; LAT >= 6 a register after every layer
//   [OUT]  the final sum, out_valid                                   always
//   extra  LAT - NCUT - 1 plain output registers                      LAT > LV + 1
// tb/mvmback mac_lane_structure measures this placement on the layer signals g_l[i].d / .q.
// Generic cost per instance: N x T 8 x 8 multipliers, T trees of N - 1 adders (no DSP inference
// is forced; Vivado may map the products to DSP48E2 on its own: the product register is then
// MREG), registers at LAT 3: T x (N x 16 products + 8 x 18 level-2 sums + SUM_W) bits + valids.

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

  localparam int unsigned LV     = (N > 1) ? $clog2(N) : 0;      // adder levels
  localparam int unsigned NP     = 1 << LV;                      // padded lane count
  localparam int unsigned NCUT   = (LAT - 1 < LV) ? LAT - 1 : LV;  // cut registers inside the chain
  localparam int unsigned NXTRA  = LAT - 1 - NCUT;               // plain output registers

  // 1 if a cut register follows logic layer i (0 <= i < LV): the product register first, the
  // other cuts after adder levels floor(c * LV / NCUT), c = 1 .. NCUT - 1.
  function automatic bit cut_after(int i);
    if (i == 0) return NCUT > 0;
    for (int cut_c = 1; cut_c < int'(NCUT); cut_c++) begin
      if (cut_c * int'(LV) / int'(NCUT) == i) return 1'b1;
    end
    return 1'b0;
  endfunction

  // ---- layers: g_l[i].d = combinational result of layer i, g_l[i].q = after its optional cut ----
  for (genvar i = 0; i <= LV; i++) begin : g_l
    localparam int unsigned NI = NP >> i;
    localparam int unsigned WI = 16 + i;
    logic [T-1:0][NI-1:0][WI-1:0] d;
    logic [T-1:0][NI-1:0][WI-1:0] q;
    logic                         vd, vq;

    if (i == 0) begin : g_prod
      for (genvar t = 0; t < T; t++) begin : g_t
        for (genvar k = 0; k < NI; k++) begin : g_k
          if (k < N) begin : g_p
            // 16 x 16-bit signed product of the sign-extended bytes (exact: |w x| <= 2^14)
            assign d[t][k] = $signed({{8{w[k][7]}}, w[k]}) * $signed({{8{x[t][k][7]}}, x[t][k]});
          end else begin : g_z
            assign d[t][k] = '0;
          end
        end
      end
      assign vd = in_valid;
    end else begin : g_add
      for (genvar t = 0; t < T; t++) begin : g_t
        for (genvar k = 0; k < NI; k++) begin : g_k
          // (WI-1)-bit signed operands sign-extended to WI bits: the sum cannot overflow
          assign d[t][k] = $signed({g_l[i-1].q[t][2*k][WI-2], g_l[i-1].q[t][2*k]})
                         + $signed({g_l[i-1].q[t][2*k+1][WI-2], g_l[i-1].q[t][2*k+1]});
        end
      end
      assign vd = g_l[i-1].vq;
    end

    if (i < LV && cut_after(i)) begin : g_cut
      always_ff @(posedge clk) begin
        if (!rst_n)  vq <= 1'b0;
        else if (ce) vq <= vd;
      end
      always_ff @(posedge clk) begin
        if (ce) q <= d;
      end
    end else begin : g_pass
      assign vq = vd;
      assign q  = d;
    end
  end

  // ---- [OUT] and extra output registers ----
  logic                         o_v [NXTRA + 1];
  logic [T-1:0][SUM_W-1:0]      o_y [NXTRA + 1];

  always_ff @(posedge clk) begin
    if (!rst_n)  o_v[0] <= 1'b0;
    else if (ce) o_v[0] <= g_l[LV].vq;
  end
  for (genvar t = 0; t < T; t++) begin : g_out
    always_ff @(posedge clk) begin
      if (ce) o_y[0][t] <= g_l[LV].q[t][0];      // 16 + LV = SUM_W bits
    end
  end
  for (genvar e = 1; e <= NXTRA; e++) begin : g_xtra
    always_ff @(posedge clk) begin
      if (!rst_n)  o_v[e] <= 1'b0;
      else if (ce) o_v[e] <= o_v[e-1];
    end
    always_ff @(posedge clk) begin
      if (ce) o_y[e] <= o_y[e-1];
    end
  end

  assign out_valid = o_v[NXTRA];
  assign y         = o_y[NXTRA];

endmodule
