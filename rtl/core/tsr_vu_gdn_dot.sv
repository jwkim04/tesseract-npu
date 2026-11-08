// tsr_vu_gdn_dot.sv — the dot-product array of the GDN unit: LANES lanes (tsr_vu_gdn_dlane), the launch
// descriptor chain, the row accumulators and the result tags (docs/design/02-vector-unit.md §7.12.5).
//
// One launch (`go` with the slot, bank, row exponent Es and the beat count NB) reads a row of
// NB * LANES words from the row buffers and forms, in exact integer arithmetic, the two sums
//     I_K = sum_j A_j * TK_j,  I_Q = sum_j A_j * TQ_j
// of the aligned row elements A_j (26-bit window below the row frame, truncated; tsr_ref_gdn.h) with
// the entries of the head tables TK and TQ. Each sum comes as two chain results, I = hi * 2^13 + lo
// (high and low table pieces), the inputs of tsr_vu_gdn_fx2f. Rows may follow each other at the
// earliest every NB cycles (NB >= 4); the sums of a row are valid (res_valid) in one cycle only.
//
// Latency: go at cycle G, res_valid in cycle G + NB + LANES + RD_LAT + 6 (= G + NB + 17 for the
// default parameters).

module tsr_vu_gdn_dot #(
  parameter  int unsigned LANES  = 8,
  parameter  int unsigned NSLOT  = 16,
  parameter  int unsigned NB_MAX = 32,
  parameter  int unsigned RD_LAT = 3,
  parameter                STYLE  = "ultra",
  localparam int unsigned SW     = $clog2(NSLOT),
  localparam int unsigned NBW    = $clog2(NB_MAX),
  localparam int unsigned LW     = (LANES > 1) ? $clog2(LANES) : 1
) (
  input  logic                    clk,
  input  logic                    rst_n,
  // row buffer write: one beat (LANES words) to {slot, beat}
  input  logic                    b_we,
  input  logic [SW+NBW-1:0]       b_waddr,
  input  logic [LANES*32-1:0]     b_wdata,
  // table write: one 32-bit half word of one lane; [0] K' half, [1] Q half; address {bank, g}
  input  logic [1:0]              t_we,
  input  logic [LW-1:0]           t_lane,
  input  logic [NBW:0]            t_waddr,
  input  logic [31:0]             t_wdata,
  // launch
  input  logic                    go,
  input  logic [SW-1:0]           go_slot,
  input  logic                    go_bank,
  input  logic [7:0]              go_es,
  input  logic [NBW:0]            go_nb,
  // result
  output logic                    res_valid,
  output logic signed [47:0]      res_kh,
  output logic signed [47:0]      res_kl,
  output logic signed [47:0]      res_qh,
  output logic signed [47:0]      res_ql
);

  // ---------------------------------------------------------------------------------------------
  // Launch descriptor chain: lane l starts l cycles after lane 0 (which starts one cycle after go)
  // ---------------------------------------------------------------------------------------------
  logic              dv   [LANES];
  logic [SW-1:0]     dslt [LANES];
  logic              dbnk [LANES];
  logic [7:0]        des  [LANES];
  logic [NBW:0]      dnb  [LANES];
  logic              dpar [LANES];
  logic              par_q;

  always_ff @(posedge clk) begin
    if (!rst_n) par_q <= 1'b0;
    else if (go) par_q <= ~par_q;
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++) dv[l] <= 1'b0;
    end else begin
      dv[0] <= go;
      for (int l = 1; l < LANES; l++) dv[l] <= dv[l-1];
    end
  end
  always_ff @(posedge clk) begin
    dslt[0] <= go_slot;
    dbnk[0] <= go_bank;
    des[0]  <= go_es;
    dnb[0]  <= go_nb;
    dpar[0] <= par_q;
    for (int l = 1; l < LANES; l++) begin
      dslt[l] <= dslt[l-1];
      dbnk[l] <= dbnk[l-1];
      des[l]  <= des[l-1];
      dnb[l]  <= dnb[l-1];
      dpar[l] <= dpar[l-1];
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Lanes, the DSP chains and the row accumulators
  // ---------------------------------------------------------------------------------------------
  logic [LANES*27-1:0] ax;
  logic [LANES-1:0]    dsgn;
  logic [LANES*64-1:0] tw;
  logic [LANES-1:0]    tgv, tgf, tgl;

  for (genvar l = 0; l < LANES; l++) begin : g_lane
    tsr_vu_gdn_dlane #(.NSLOT(NSLOT), .NB_MAX(NB_MAX), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_lane (
      .clk, .rst_n,
      .d_valid(dv[l]), .d_slot(dslt[l]), .d_bank(dbnk[l]), .d_es(des[l]), .d_nb(dnb[l]), .d_par(dpar[l]),
      .b_we, .b_waddr, .b_wdata(b_wdata[32*l +: 32]),
      .t_we(t_we & {2{t_lane == LW'(l)}}), .t_waddr, .t_wdata,
      .ax(ax[27*l +: 27]), .dsgn(dsgn[l]), .tw(tw[64*l +: 64]),
      .tg_v(tgv[l]), .tg_first(tgf[l]), .tg_last(tgl[l]));
  end

  // The last lane's tags are aligned with its P register, which the accumulators add one cycle later.
  tsr_vu_gdn_mac #(.LANES(LANES)) u_mac (
    .clk, .ax, .dsgn, .tw, .acc_first(tgf[LANES-1]),
    .acc_kh(res_kh), .acc_kl(res_kl), .acc_qh(res_qh), .acc_ql(res_ql));

  always_ff @(posedge clk) begin
    if (!rst_n) res_valid <= 1'b0;
    else        res_valid <= tgv[LANES-1] & tgl[LANES-1];
  end

  // The lanes before the last do not use their tags.
  wire unused_ok = &{1'b0, tgv[LANES-2:0], tgf[LANES-2:0], tgl[LANES-2:0]};

endmodule
