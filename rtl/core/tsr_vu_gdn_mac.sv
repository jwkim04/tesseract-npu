// tsr_vu_gdn_mac.sv — the DSP chain of the GDN dot-product array (docs/design/02-vector-unit.md §7.12.5).
//
// Four chains of LANES DSP48E2 slices, one slice per lane in each (K' high piece, K' low piece, Q high
// piece, Q low piece). Lane l computes the product of the aligned, signed row element
// (ax + dsgn, a 27-bit two's complement value: the XOR of the sign in fabric and the + sign in the
// DSP pre-adder) with its table piece and adds the partial sum of the previous lane:
//     p_l(t + 1) = p_(l-1)(t) + ad_l(t) * b_l(t)           (cascade; the first lane adds 0)
// The last lane's sum goes into one accumulator slice per chain:
//     acc(t + 1) = (acc_first ? 0 : acc(t)) + p_(LANES-1)(t)
// so after the last beat of a row acc holds the exact sum of all products of the row. Everything is
// integer arithmetic: the pieces are the entry v = th * 2^13 + tl (th signed 12 bits, tl in
// [0, 8191]); |acc| < 2^47 for rows of up to 256 elements, because the aligned element is below
// 2^26 and a piece below 2^13 in magnitude.
//
// The registers sit where the DSP48E2 has them: AREG / DREG (a_r, d_r), ADREG (ad_r), BREG1 / BREG2
// (b_*1, b_*2), MREG (m_*), PREG (p_*, acc_*), so Vivado absorbs all of them. Latency from the
// operand inputs (ax, dsgn, tw) to p of the same lane: 4 registers (A / D, AD, M, P); a table word
// enters BREG1 in the same cycle as ax enters AREG.
//
// All lanes of a chain must be in this one module: Vivado maps a chain whose P registers cross a
// module boundary through the C port (fabric routing) instead of PCIN, and then keeps the
// accumulators in fabric adders.

module tsr_vu_gdn_mac #(
  parameter int unsigned LANES = 8
) (
  input  logic                         clk,
  input  logic [LANES*27-1:0]          ax,
  input  logic [LANES-1:0]             dsgn,
  input  logic [LANES*64-1:0]          tw,          // per lane: [24:13] K' th, [12:0] K' tl, [56:45] Q th, [44:32] Q tl
  input  logic                         acc_first,   // the last lane's sum of this cycle is the first beat of a row
  output logic signed [47:0]           acc_kh,
  output logic signed [47:0]           acc_kl,
  output logic signed [47:0]           acc_qh,
  output logic signed [47:0]           acc_ql
);

  logic signed [26:0] a_r  [LANES];
  logic signed [26:0] d_r  [LANES];
  logic signed [26:0] ad_r [LANES];
  logic signed [12:0] kh_1 [LANES], kh_2 [LANES], qh_1 [LANES], qh_2 [LANES];
  logic signed [13:0] kl_1 [LANES], kl_2 [LANES], ql_1 [LANES], ql_2 [LANES];
  logic signed [47:0] mkh  [LANES], mkl  [LANES], mqh  [LANES], mql  [LANES];
  logic signed [47:0] pkh  [LANES], pkl  [LANES], pqh  [LANES], pql  [LANES];

  for (genvar l = 0; l < LANES; l++) begin : g
    always_ff @(posedge clk) begin
      a_r[l]  <= ax[27*l +: 27];
      d_r[l]  <= signed'({26'b0, dsgn[l]});
      ad_r[l] <= a_r[l] + d_r[l];
      kh_1[l] <= 13'(signed'(tw[64*l + 13 +: 12]));
      kl_1[l] <= signed'({1'b0, tw[64*l +: 13]});
      qh_1[l] <= 13'(signed'(tw[64*l + 45 +: 12]));
      ql_1[l] <= signed'({1'b0, tw[64*l + 32 +: 13]});
      kh_2[l] <= kh_1[l];
      kl_2[l] <= kl_1[l];
      qh_2[l] <= qh_1[l];
      ql_2[l] <= ql_1[l];
      mkh[l]  <= ad_r[l] * kh_2[l];
      mkl[l]  <= ad_r[l] * kl_2[l];
      mqh[l]  <= ad_r[l] * qh_2[l];
      mql[l]  <= ad_r[l] * ql_2[l];
      if (l == 0) begin
        pkh[l] <= mkh[l];
        pkl[l] <= mkl[l];
        pqh[l] <= mqh[l];
        pql[l] <= mql[l];
      end else begin
        pkh[l] <= pkh[l-1] + mkh[l];
        pkl[l] <= pkl[l-1] + mkl[l];
        pqh[l] <= pqh[l-1] + mqh[l];
        pql[l] <= pql[l-1] + mql[l];
      end
    end
  end

  always_ff @(posedge clk) begin
    acc_kh <= (acc_first ? 48'sd0 : acc_kh) + pkh[LANES-1];
    acc_kl <= (acc_first ? 48'sd0 : acc_kl) + pkl[LANES-1];
    acc_qh <= (acc_first ? 48'sd0 : acc_qh) + pqh[LANES-1];
    acc_ql <= (acc_first ? 48'sd0 : acc_ql) + pql[LANES-1];
  end

  // The table words carry 25 bits per half; the rest is not used.
  logic [LANES-1:0] unused_w;
  for (genvar l = 0; l < LANES; l++) begin : g_unused
    assign unused_w[l] = ^{tw[64*l + 25 +: 7], tw[64*l + 57 +: 7]};
  end
  wire unused_ok = &{1'b0, unused_w};

endmodule
