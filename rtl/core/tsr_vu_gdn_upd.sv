// tsr_vu_gdn_upd.sv — the operand reader of the state update of the GDN engine: the second copy of the row
// buffer (RAM B2) and the table of the key (RAM KF) (docs/design/02-vector-unit.md §7.12.5).
//
// The state update S'_j = FA(FM(alpha, S_j), FM(e, k_j)) is not computed here. It runs on the front lanes of
// the vector unit (micro-op U_ROT of tsr_vu_lane with a = S_j, b = k_j, c1 = alpha, c2 = e; VU-30), which are
// FM_A and FM_B in parallel into FA, built from the same tsr_fp_mul and tsr_fp_add. This module reads the
// operands in the beat layout of the lanes and delivers them with the lane constants and the write
// descriptor of the beat.
//
// One launch (`go` with the slot, bank, beat count NB, alpha, e, the first destination line and the op tag)
// reads the row from B2 and the key k from KF (element j = LANES * g + l of beat g in lane l, in the layout of
// the dot array's tables). Beat g is issued in cycle G + 1 + g; the read data is valid RD_LAT cycles later,
// and u_v is high exactly then: u_s / u_k are the RAM output registers, u_alpha / u_e / u_tag hold the row's
// values from beat 0 to its last beat, u_line / u_part are the destination line and the half-line index of the
// beat. Rows may follow each other every NB cycles (NB >= 4). B2 holds the same words as the dot array's B1
// (the parent writes both from the ingest port); KF is written one lane word at a time.
//
// LANES = 8 sends a line in two beats (u_part = g[0], u_line = first line + g[NBW:1]); LANES = 16 in one.

module tsr_vu_gdn_upd #(
  parameter  int unsigned LANES  = 8,
  parameter  int unsigned NSLOT  = 64,
  parameter  int unsigned NB_MAX = 32,
  parameter  int unsigned RD_LAT = 3,
  parameter                STYLE  = "ultra",
  parameter  int unsigned LW_ADR = 24,
  localparam int unsigned SW     = $clog2(NSLOT),
  localparam int unsigned NBW    = $clog2(NB_MAX),
  localparam int unsigned LW     = (LANES > 1) ? $clog2(LANES) : 1,
  localparam int unsigned LSH    = (LANES == 16) ? 0 : 1,       // beats per line = 2^LSH
  localparam int unsigned LAW    = NBW - LSH                     // bits of a line index in a row
) (
  input  logic                    clk,
  input  logic                    rst_n,
  // row buffer write: one line (16 words) to {slot, line of the row}
  input  logic                    b_we,
  input  logic [SW+LAW-1:0]       b_waddr,
  input  logic [511:0]            b_wdata,
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
  input  logic [LW_ADR-1:0]       go_line,
  input  logic                    go_tag,
  // operand stream for the lanes
  output logic                    u_v,
  output logic [LANES*32-1:0]     u_s,
  output logic [LANES*32-1:0]     u_k,
  output logic [31:0]             u_alpha,
  output logic [31:0]             u_e,
  output logic [LW_ADR-1:0]       u_line,
  output logic                    u_part,
  output logic                    u_tag,
  output logic                    u_last
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
  logic [LW_ADR-1:0] line_h, line_q;
  // go is registered into *_h; the issue runs from them
  always_ff @(posedge clk) begin
    if (!rst_n) start_q <= 1'b0; else start_q <= go;
  end
  always_ff @(posedge clk) begin
    if (go) begin
      slot_h <= go_slot;
      bank_h <= go_bank;
      nb_h   <= go_nb;
      line_h <= go_line;
    end
  end
  logic              issue, last_c, bank_c;
  logic [SW-1:0]     slot_c;
  logic [NBW-1:0]    g_c;
  logic [LW_ADR-1:0] line_c;
  assign issue  = start_q | busy_q;
  assign g_c    = start_q ? '0 : cnt_q[NBW-1:0];
  assign slot_c = start_q ? slot_h : slot_q;
  assign bank_c = start_q ? bank_h : bank_q;
  assign last_c = !start_q && (cnt_q == nb_q - 1'b1);
  // destination line of the beat: first line + g / 2^LSH (the beat counter of beat g is g)
  assign line_c = (start_q ? line_h : line_q) + LW_ADR'(g_c >> LSH);
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
      line_q <= line_h;
    end else if (busy_q) begin
      cnt_q <= cnt_q + 1'b1;
    end
  end

  // alpha, e and the tag: held at go, applied RD_LAT cycles after the first issue (with beat 0) ------
  logic [31:0] alpha_h, e_h;
  logic        tag_h, tag_q;
  logic [RD_LAT-1:0] go_d;                  // go_d[i]: go delayed by i + 1 cycles
  always_ff @(posedge clk) begin
    if (go) begin
      alpha_h <= go_alpha;
      e_h     <= go_e;
      tag_h   <= go_tag;
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) go_d <= '0; else go_d <= RD_LAT'({go_d, go});
  end
  always_ff @(posedge clk) begin
    if (go_d[RD_LAT-1]) begin
      u_alpha <= alpha_h;
      u_e     <= e_h;
      tag_q   <= tag_h;
    end
  end
  assign u_tag = tag_q;

  // Valid / last / line / part of the issued beat, delayed to the read data -------------------------
  logic [RD_LAT-1:0] vv, ll, pp;            // vv[i]: issue valid delayed by i + 1 cycles
  logic [LW_ADR-1:0] lq [RD_LAT];
  always_ff @(posedge clk) begin
    if (!rst_n) vv <= '0; else vv <= RD_LAT'({vv, issue});
  end
  always_ff @(posedge clk) begin
    ll <= RD_LAT'({ll, issue & last_c});
    pp <= RD_LAT'({pp, (LSH == 1) ? g_c[0] : 1'b0});
    lq[0] <= line_c;
    for (int i = 1; i < RD_LAT; i++) lq[i] <= lq[i-1];
  end
  assign u_v    = vv[RD_LAT-1];
  assign u_last = ll[RD_LAT-1];
  assign u_part = pp[RD_LAT-1];
  assign u_line = lq[RD_LAT-1];

  // RAMs ---------------------------------------------------------------------------------------------
  for (genvar l = 0; l < LANES; l++) begin : g_lane
    logic [32*(1<<LSH)-1:0] lw, sw_;
    if (LSH == 1) begin : g_pair
      assign lw = {b_wdata[32*(l+LANES) +: 32], b_wdata[32*l +: 32]};
      assign u_s[32*l +: 32] = pp[RD_LAT-1] ? sw_[63:32] : sw_[31:0];       // the half of the beat
    end else begin : g_one
      assign lw = b_wdata[32*l +: 32];
      assign u_s[32*l +: 32] = sw_;
    end
    tsr_vu_gdn_ram #(.DEPTH(NSLOT * (NB_MAX >> LSH)), .WIDTH(32 << LSH), .NWE(1), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_b2 (
      .clk, .we(b_we), .waddr(b_waddr), .wdata(lw),
      .re(issue), .raddr({slot_c, g_c[NBW-1:LSH]}), .rdata(sw_));
    tsr_vu_gdn_ram #(.DEPTH(2 * NB_MAX), .WIDTH(32), .NWE(1), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_kf (
      .clk, .we(kf_we & (kf_lane == LW'(l))), .waddr(kf_waddr), .wdata(kf_wdata),
      .re(issue), .raddr({bank_c, g_c}), .rdata(u_k[32*l +: 32]));
  end

  wire unused_ok = &{1'b0, vv[RD_LAT-2:0], ll[RD_LAT-2:0], pp[RD_LAT-2:0]};

endmodule
