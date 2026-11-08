// tsr_vu_gdn_dlane.sv — one lane of the GDN dot-product array (docs/design/02-vector-unit.md §7.12.5).
//
// The lane holds element j = LANES * g + l of every row in its own row buffer (RAM B1, one 32-bit
// word per beat g) and the matching entries of the two head tables in RAM T (K' half and Q half,
// two banks). Per beat it
//   1. reads the row word and aligns its significand to the row frame: A = (M << 2) >> (Es - E),
//      a 26-bit magnitude, 0 for a shift of 26 or more and for a zero element, then the sign is
//      applied as a two's complement value: (A ^ {27{s}}) + s (the XOR in fabric, the + s in the DSP
//      pre-adder),
//   2. (tsr_vu_gdn_mac) multiplies by the four table operands in four DSP48E2 slices (K' high piece, K' low piece,
//      Q high piece, Q low piece; the pieces are the signed entry v = th * 2^13 + tl, th in
//      [-2048, 2047], tl in [0, 8191]) and adds the product to the partial sum that arrives from
//      the previous lane through the PCIN cascade: P_l(t + 1) = P_(l-1)(t) + product_l(t).
// The lanes start one cycle after one another (the launch descriptor travels down a register
// chain in tsr_vu_gdn_dot), so the cascade adds the 8 lanes' products of one beat in 8 cycles with
// no fabric adder and no data skew registers. All sums are exact integers: |P| < 2^47 for rows of up
// to 256 elements (26-bit magnitude times a 13-bit piece, 256 terms).
//
// Timing (cycle numbers from the cycle c = 0 in which this lane issues the read of beat g; the
// launch descriptor is `d_*`, valid in the cycle the lane starts):
//   c = 0                 row RAM read request {slot, g}
//   c = 3                 table RAM read request {bank, g} (the table RAM has a read latency of RD_LAT - 1)
//   c = RD_LAT            row word valid; decode, shift distance
//   c = RD_LAT + 1, + 2   alignment (shift 0..15, then 16)
//   c = RD_LAT + 3        DSP input registers (AREG / DREG; the table words arrive in BREG1)
//   c = RD_LAT + 4, + 5   pre-adder (ADREG), multiplier (MREG)
//   c = RD_LAT + 6        P registers; tg_* are the tags of this beat
// The DSP chain itself is tsr_vu_gdn_mac (all lanes in one module, so that Vivado infers the PCIN
// cascade and the accumulators: a chain split over module boundaries is mapped through the C port).
// Contents of RAM T are written by the parent; the write address is {bank, g}. RAM B1 is written by
// the parent from the ingest port (all lanes in one cycle).

module tsr_vu_gdn_dlane #(
  parameter  int unsigned NSLOT  = 16,
  parameter  int unsigned NB_MAX = 32,
  parameter  int unsigned RD_LAT = 3,
  parameter                STYLE  = "ultra",
  localparam int unsigned SW     = $clog2(NSLOT),
  localparam int unsigned NBW    = $clog2(NB_MAX)
) (
  input  logic                clk,
  input  logic                rst_n,
  // launch descriptor for this lane (valid in the first cycle of the row)
  input  logic                d_valid,
  input  logic [SW-1:0]       d_slot,
  input  logic                d_bank,
  input  logic [7:0]          d_es,
  input  logic [NBW:0]        d_nb,          // beats of the row, 2 .. NB_MAX
  input  logic                d_par,         // row parity: selects one of two Es registers
  // row buffer write (parent: all lanes at once)
  input  logic                b_we,
  input  logic [SW+NBW-1:0]   b_waddr,
  input  logic [31:0]         b_wdata,
  // table write: [0] K' half, [1] Q half; address {bank, g}
  input  logic [1:0]          t_we,
  input  logic [NBW:0]        t_waddr,
  input  logic [31:0]         t_wdata,
  // operands of the DSP chain (tsr_vu_gdn_mac registers them): the aligned element with its sign
  // folded in as (ax + dsgn), and the table word, both for the beat of cycle c = RD_LAT + 2
  output logic [26:0]         ax,
  output logic                dsgn,
  output logic [63:0]         tw,
  // tags of the beat whose products are in the P registers (used by the last lane)
  output logic                tg_v,
  output logic                tg_first,
  output logic                tg_last
);

  localparam int unsigned CP = RD_LAT + 6;       // cycle of the P registers
  localparam int unsigned TQ = 3;                // cycle of the table read request
  localparam int unsigned RD_LAT_T = RD_LAT - 1; // the table RAM has one stage less, its request is one cycle later

  // ---------------------------------------------------------------------------------------------
  // Issue control
  // ---------------------------------------------------------------------------------------------
  logic              busy_q, bank_q, par_q;
  logic [NBW:0]      cnt_q, nb_q;
  logic [SW-1:0]     slot_q;
  logic [7:0]        es_r [2];

  logic              start, issue, last_c, bank_c, par_c;
  logic [SW-1:0]     slot_c;
  logic [NBW-1:0]    g_c;
  assign start  = d_valid;
  assign issue  = start | busy_q;
  assign g_c    = start ? '0 : cnt_q[NBW-1:0];
  assign slot_c = start ? d_slot : slot_q;
  assign bank_c = start ? d_bank : bank_q;
  assign par_c  = start ? d_par : par_q;
  assign last_c = !start && (cnt_q == nb_q - 1'b1);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      busy_q <= 1'b0;
    end else if (start) begin
      busy_q <= 1'b1;
    end else if (busy_q && last_c) begin
      busy_q <= 1'b0;
    end
  end
  always_ff @(posedge clk) begin
    if (start) begin
      cnt_q  <= (NBW + 1)'(1);
      nb_q   <= d_nb;
      slot_q <= d_slot;
      bank_q <= d_bank;
      par_q  <= d_par;
      es_r[d_par] <= d_es;
    end else if (busy_q) begin
      cnt_q <= cnt_q + 1'b1;
    end
  end

  // ---------------------------------------------------------------------------------------------
  // Tag pipeline: tg_*[i] is the state of the beat i cycles after its issue
  // ---------------------------------------------------------------------------------------------
  logic              tv [CP+1];
  logic              tf [CP+1];
  logic              tl [CP+1];
  logic              tp [CP+1];
  logic              tb [TQ+1];
  logic [NBW-1:0]    tg [TQ+1];
  assign tv[0] = issue;
  assign tf[0] = start;
  assign tl[0] = issue & last_c;
  assign tp[0] = par_c;
  assign tb[0] = bank_c;
  assign tg[0] = g_c;
  always_ff @(posedge clk) begin
    for (int i = 1; i <= CP; i++) begin
      tf[i] <= tf[i-1];
      tl[i] <= tl[i-1];
      tp[i] <= tp[i-1];
    end
    for (int i = 1; i <= TQ; i++) begin
      tb[i] <= tb[i-1];
      tg[i] <= tg[i-1];
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int i = 1; i <= CP; i++) tv[i] <= 1'b0;
    end else begin
      for (int i = 1; i <= CP; i++) tv[i] <= tv[i-1];
    end
  end
  assign tg_v     = tv[CP];
  assign tg_first = tf[CP];
  assign tg_last  = tl[CP];

  // ---------------------------------------------------------------------------------------------
  // RAMs
  // ---------------------------------------------------------------------------------------------
  logic [31:0] b1_rdata;
  logic [63:0] tr_rdata;

  tsr_vu_gdn_ram #(.DEPTH(NSLOT * NB_MAX), .WIDTH(32), .NWE(1), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_b1 (
    .clk, .we(b_we), .waddr(b_waddr), .wdata(b_wdata),
    .re(issue), .raddr({slot_c, g_c}), .rdata(b1_rdata));

  tsr_vu_gdn_ram #(.DEPTH(2 * NB_MAX), .WIDTH(64), .NWE(2), .RD_LAT(RD_LAT_T), .STYLE(STYLE)) u_tr (
    .clk, .we(t_we), .waddr(t_waddr), .wdata({t_wdata, t_wdata}),
    .re(tv[TQ]), .raddr({tb[TQ], tg[TQ]}), .rdata(tr_rdata));

  // ---------------------------------------------------------------------------------------------
  // Alignment: c = RD_LAT (decode), + 1, + 2 -> AREG at c = RD_LAT + 3
  // ---------------------------------------------------------------------------------------------
  logic [8:0]  sh9;
  logic [7:0]  e_in, es_sel;
  logic        zero_in, big_c;
  logic [4:0]  sh_c;
  assign e_in    = b1_rdata[30:23];
  assign es_sel  = es_r[tp[RD_LAT]];
  assign sh9     = {1'b0, es_sel} - {1'b0, e_in};
  assign zero_in = (e_in == 8'd0) | ~tv[RD_LAT];
  assign big_c   = sh9[8] | (sh9[7:0] > 8'd25);
  assign sh_c    = sh9[4:0];

  // D1 (c = RD_LAT + 1)
  logic        d1_sign, d1_zero;
  logic [25:0] d1_m;
  logic [4:0]  d1_sh;
  always_ff @(posedge clk) begin
    d1_sign <= b1_rdata[31];
    d1_zero <= zero_in | big_c;
    d1_m    <= {1'b1, b1_rdata[22:0], 2'b00};
    d1_sh   <= sh_c;
  end

  // D2 (c = RD_LAT + 2): shift by 0 .. 15
  logic        d2_sign, d2_zero, d2_s16;
  logic [25:0] d2_m;
  always_ff @(posedge clk) begin
    d2_sign <= d1_sign;
    d2_zero <= d1_zero;
    d2_s16  <= d1_sh[4];
    d2_m    <= d1_m >> d1_sh[3:0];
  end

  // The element at the input of the DSP registers (c = RD_LAT + 2 -> AREG at c = RD_LAT + 3): shift by
  // 16, zero, sign. The table word is the RAM output of cycle c = RD_LAT + 2 (BREG1 follows).
  logic [25:0] d3_mag;
  logic        d3_s;
  assign d3_mag = d2_zero ? 26'd0 : (d2_s16 ? (d2_m >> 16) : d2_m);
  assign d3_s   = d2_sign & ~d2_zero;
  assign ax     = {1'b0, d3_mag} ^ {27{d3_s}};
  assign dsgn   = d3_s;
  assign tw     = tr_rdata;

endmodule
