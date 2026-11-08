// tsr_vu_gdn_upd.sv — the state update array of the GDN unit: LANES update lanes (tsr_vu_gdn_ulane),
// the second copy of the row buffer (RAM B2) and the table of the key (RAM KF)
// (docs/design/02-vector-unit.md §7.12.5).
//
// One launch (`go` with the slot, bank, beat count NB, alpha and e) reads the row from B2 and the
// key k from KF (element j = LANES * g + l of beat g in lane l, in the same layout as the dot array's
// tables), and writes S'_j = FA(FM(alpha, S_j), FM(e, k_j)) to the output port, LANES words per cycle.
// Rows may follow each other every NB cycles (NB >= 4). B2 holds the same words as the dot array's B1
// (the parent writes both from the ingest port); KF is written one lane word at a time.
//
// Timing: go in cycle G (the inputs alpha and e are sampled then); beat g is issued in cycle G + 1 + g,
// read data is valid RD_LAT cycles later, alpha / e reach the multipliers together with beat 0, and
// out_data of beat g is valid in cycle G + 1 + g + RD_LAT + FM_LAT + FA_LAT. out_last marks the last
// beat of a row.

module tsr_vu_gdn_upd #(
  parameter  int unsigned LANES  = 8,
  parameter  int unsigned NSLOT  = 16,
  parameter  int unsigned NB_MAX = 32,
  parameter  int unsigned RD_LAT = 3,
  parameter  int unsigned FM_LAT = 4,
  parameter  int unsigned FA_LAT = 5,
  parameter                STYLE  = "ultra",
  localparam int unsigned SW     = $clog2(NSLOT),
  localparam int unsigned NBW    = $clog2(NB_MAX),
  localparam int unsigned LW     = (LANES > 1) ? $clog2(LANES) : 1,
  localparam int unsigned LAT_U  = RD_LAT + FM_LAT + FA_LAT
) (
  input  logic                    clk,
  input  logic                    rst_n,
  // row buffer write: one beat (LANES words) to {slot, beat}
  input  logic                    b_we,
  input  logic [SW+NBW-1:0]       b_waddr,
  input  logic [LANES*32-1:0]     b_wdata,
  // key write: one word of one lane; address {bank, g}
  input  logic                    kf_we,
  input  logic [LW-1:0]           kf_lane,
  input  logic [NBW:0]            kf_waddr,
  input  logic [31:0]             kf_wdata,
  // launch
  input  logic                    go,
  input  logic [SW-1:0]           go_slot,
  input  logic                    go_bank,
  input  logic [NBW:0]            go_nb,
  input  logic [31:0]             go_alpha,
  input  logic [31:0]             go_e,
  // result
  output logic                    out_valid,
  output logic                    out_last,
  output logic [LANES*32-1:0]     out_data
);

  if (RD_LAT < 2) begin : g_chk_lat
    $fatal(1, "tsr_vu_gdn_upd: RD_LAT >= 2 required");
  end

  // Issue control (cycle G + 1 + g) ---------------------------------------------------------------
  logic              busy_q, bank_q;
  logic [SW-1:0]     slot_q;
  logic [NBW:0]      cnt_q, nb_q;
  logic              start_q;                 // the first issue cycle
  logic [SW-1:0]     slot_h;
  logic              bank_h;
  logic [NBW:0]      nb_h;
  // go is registered into *_h; the issue runs from them
  always_ff @(posedge clk) begin
    if (!rst_n) start_q <= 1'b0; else start_q <= go;
  end
  always_ff @(posedge clk) begin
    if (go) begin
      slot_h <= go_slot;
      bank_h <= go_bank;
      nb_h   <= go_nb;
    end
  end
  logic              issue, last_c, bank_c;
  logic [SW-1:0]     slot_c;
  logic [NBW-1:0]    g_c;
  assign issue  = start_q | busy_q;
  assign g_c    = start_q ? '0 : cnt_q[NBW-1:0];
  assign slot_c = start_q ? slot_h : slot_q;
  assign bank_c = start_q ? bank_h : bank_q;
  assign last_c = !start_q && (cnt_q == nb_q - 1'b1);
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      busy_q <= 1'b0;
    end else if (start_q) begin
      busy_q <= 1'b1;
    end else if (busy_q && last_c) begin
      busy_q <= 1'b0;
    end
  end
  always_ff @(posedge clk) begin
    if (start_q) begin
      cnt_q  <= (NBW + 1)'(1);
      nb_q   <= nb_h;
      slot_q <= slot_h;
      bank_q <= bank_h;
    end else if (busy_q) begin
      cnt_q <= cnt_q + 1'b1;
    end
  end

  // alpha and e: held at go, applied RD_LAT cycles after the first issue (with beat 0) ------------
  logic [31:0] alpha_h, e_h, alpha_q, e_q;
  logic [RD_LAT-1:0] go_d;                  // go_d[i]: go delayed by i + 1 cycles
  always_ff @(posedge clk) begin
    if (go) begin
      alpha_h <= go_alpha;
      e_h     <= go_e;
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) go_d <= '0; else go_d <= RD_LAT'({go_d, go});
  end
  always_ff @(posedge clk) begin
    if (go_d[RD_LAT-1]) begin
      alpha_q <= alpha_h;
      e_q     <= e_h;
    end
  end

  // Valid / last of the issued beat, delayed to the read data and on to the output
  logic [LAT_U-1:0] vv, ll;                 // vv[i]: issue valid delayed by i + 1 cycles
  always_ff @(posedge clk) begin
    if (!rst_n) vv <= '0; else vv <= LAT_U'({vv, issue});
    ll <= LAT_U'({ll, issue & last_c});
  end

  // RAMs and lanes ---------------------------------------------------------------------------------
  logic [LANES*32-1:0] s_rd, k_rd;
  logic [LANES-1:0]    uv;
  for (genvar l = 0; l < LANES; l++) begin : g_lane
    tsr_vu_gdn_ram #(.DEPTH(NSLOT * NB_MAX), .WIDTH(32), .NWE(1), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_b2 (
      .clk, .we(b_we), .waddr(b_waddr), .wdata(b_wdata[32*l +: 32]),
      .re(issue), .raddr({slot_c, g_c}), .rdata(s_rd[32*l +: 32]));
    tsr_vu_gdn_ram #(.DEPTH(2 * NB_MAX), .WIDTH(32), .NWE(1), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_kf (
      .clk, .we(kf_we & (kf_lane == LW'(l))), .waddr(kf_waddr), .wdata(kf_wdata),
      .re(issue), .raddr({bank_c, g_c}), .rdata(k_rd[32*l +: 32]));
    tsr_vu_gdn_ulane #(.FM_LAT(FM_LAT), .FA_LAT(FA_LAT)) u_ul (
      .clk, .rst_n, .in_valid(vv[RD_LAT-1]), .s(s_rd[32*l +: 32]), .k(k_rd[32*l +: 32]),
      .alpha(alpha_q), .e(e_q), .out_valid(uv[l]), .y(out_data[32*l +: 32]));
  end

  assign out_valid = uv[0];
  assign out_last  = ll[LAT_U-1];

  wire unused_ok = &{1'b0, uv[LANES-1:1], vv[LAT_U-1:RD_LAT], vv[RD_LAT-2:0], ll[LAT_U-2:0]};

endmodule
