// tsr_vu_gdn_coef.sv — the head setup engine of the GDN unit: it turns the vectors k, q, v of one head
// into the coefficients the row pipeline uses (docs/design/02-vector-unit.md §7.12.5; the definitions
// are in tsr_ref_gdn.h and §7.12.3).
//
// One word per cycle is accepted on `vec` in the order k (m words), q (m words), v (rows words), after a
// `start` with m, rows, alpha, beta and the bank. The engine
//   1. ab = FM(alpha, beta); kp_j = FM(ab, k_j); bv_i = FM(beta, v_i) on one shared multiplier,
//   2. writes the raw k_j to the key RAM (kf_*; the f32 factor of the state update),
//   3. keeps kp_j and q_j in a scratch RAM, with the largest biased exponent of each (emax_k, emax_q) and
//      a flag if any element is Inf or NaN,
//   4. aligns kp_j and q_j to their frame (24-bit window, round half up) and writes the table words
//      (tr_*): the entry v = th * 2^13 + tl as {th[11:0], tl[12:0]}, K' half then Q half,
//   5. writes bv_i (bv_*), and finally pulses `done` with the table exponents and flags.
// It is element-serial: about 4 m + rows + 40 cycles per head, overlapped by the parent with the rows of
// the previous head (two banks).

module tsr_vu_gdn_coef #(
  parameter  int unsigned LANES  = 8,
  parameter  int unsigned NB_MAX = 32,
  parameter  int unsigned R_MAX  = 1024,
  parameter  int unsigned RD_LAT = 3,
  parameter  int unsigned FM_LAT = 4,
  parameter                STYLE  = "ultra",
  localparam int unsigned NBW    = $clog2(NB_MAX),
  localparam int unsigned LW     = (LANES > 1) ? $clog2(LANES) : 1,
  localparam int unsigned MW     = $clog2(LANES * NB_MAX) + 1,     // bits of m (up to 256)
  localparam int unsigned RW     = $clog2(R_MAX) + 1,
  localparam int unsigned BVW    = $clog2(R_MAX) + 1,              // {bank, row}
  localparam int unsigned CW     = (RW > MW) ? RW : MW,
  localparam int unsigned JW     = MW - 1,                         // bits of an element index
  localparam int unsigned TA     = $clog2(2 * LANES * NB_MAX)      // scratch RAM address
) (
  input  logic                clk,
  input  logic                rst_n,
  // command
  input  logic                start,
  input  logic                bank,
  input  logic [MW-1:0]       m,
  input  logic [RW-1:0]       rows,
  input  logic [31:0]         alpha,
  input  logic [31:0]         beta,
  output logic                busy,
  // vector stream
  input  logic                vec_valid,
  input  logic [31:0]         vec_data,
  output logic                vec_ready,
  // key RAM (raw k_j): lane, address {bank, g}
  output logic                kf_we,
  output logic [LW-1:0]       kf_lane,
  output logic [NBW:0]        kf_waddr,
  output logic [31:0]         kf_wdata,
  // table RAM halves: [0] K', [1] Q; lane, address {bank, g}
  output logic [1:0]          tr_we,
  output logic [LW-1:0]       tr_lane,
  output logic [NBW:0]        tr_waddr,
  output logic [31:0]         tr_wdata,
  // bv RAM: address {bank, row}
  output logic                bv_we,
  output logic [BVW-1:0]      bv_waddr,
  output logic [31:0]         bv_wdata,
  // finished
  output logic                done,
  output logic [7:0]          done_ek,
  output logic [7:0]          done_eq,
  output logic                done_badk,
  output logic                done_badq
);

  typedef enum logic [3:0] {S_IDLE, S_AB, S_ABW, S_PK, S_PKD, S_PQ, S_PV, S_PVD, S_TK, S_TQ, S_FIN} st_e;
  st_e st_q;

  // command registers
  logic              bank_q;
  logic [MW-1:0]     m_q;
  logic [RW-1:0]     rows_q;
  logic [31:0]       alpha_q, beta_q, ab_q;
  logic [CW-1:0]     in_cnt, out_cnt, v_cnt;
  logic [MW-1:0]     rd_cnt, wr_cnt;
  logic [7:0]        emax_k, emax_q;
  logic              bad_k, bad_q;

  // shared multiplier -------------------------------------------------------------------------------
  logic        fm_v, fm_ov;
  logic [31:0] fm_a, fm_b, fm_y;
  logic        take;                         // a vec word is accepted this cycle
  assign vec_ready = (st_q == S_PK) | (st_q == S_PQ) | (st_q == S_PV);
  assign take      = vec_valid & vec_ready;
  assign fm_v = (st_q == S_AB) | (take & ((st_q == S_PK) | (st_q == S_PV)));
  assign fm_a = (st_q == S_AB) ? alpha_q : ((st_q == S_PK) ? ab_q : beta_q);
  assign fm_b = (st_q == S_AB) ? beta_q : vec_data;
  tsr_fp_mul #(.LAT(FM_LAT)) u_fm (
    .clk, .rst_n, .ce(1'b1), .in_valid(fm_v), .a(fm_a), .b(fm_b), .out_valid(fm_ov), .y(fm_y));

  // scratch RAM: kp_j at j, q_j at m + j (written from the FM output in PK and from the stream in PQ)
  logic        tmp_we, tmp_re;
  logic [TA-1:0] tmp_waddr, tmp_raddr;
  logic [31:0] tmp_wdata, tmp_rdata;
  assign tmp_we    = ((st_q == S_PK) | (st_q == S_PKD)) ? fm_ov : ((st_q == S_PQ) & take);
  assign tmp_waddr = ((st_q == S_PK) | (st_q == S_PKD)) ? TA'(out_cnt) : (TA'(m_q) + TA'(in_cnt));
  assign tmp_wdata = ((st_q == S_PK) | (st_q == S_PKD)) ? fm_y : vec_data;
  assign tmp_re    = ((st_q == S_TK) | (st_q == S_TQ)) & (rd_cnt != m_q);
  assign tmp_raddr = ((st_q == S_TQ) ? TA'(m_q) : '0) + TA'(rd_cnt);
  tsr_vu_gdn_ram #(.DEPTH(2 * LANES * NB_MAX), .WIDTH(32), .NWE(1), .RD_LAT(RD_LAT), .STYLE(STYLE)) u_tmp (
    .clk, .we(tmp_we), .waddr(tmp_waddr), .wdata(tmp_wdata), .re(tmp_re), .raddr(tmp_raddr), .rdata(tmp_rdata));

  // key RAM and bv RAM writes
  assign kf_we    = take & (st_q == S_PK);
  assign kf_lane  = in_cnt[LW-1:0];
  assign kf_waddr = {bank_q, in_cnt[LW +: NBW]};
  assign kf_wdata = vec_data;
  assign bv_we    = fm_ov & ((st_q == S_PV) | (st_q == S_PVD));
  assign bv_waddr = {bank_q, out_cnt[BVW-2:0]};
  assign bv_wdata = fm_y;

  // exponent / bad tracking: kp on the FM output, q on the stream
  logic [7:0] e_kp, e_q;
  assign e_kp = fm_y[30:23];
  assign e_q  = vec_data[30:23];

  // main FSM ----------------------------------------------------------------------------------------
  logic tbl_done;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q <= S_IDLE;
    end else begin
      case (st_q)
        S_IDLE: if (start) st_q <= S_AB;
        S_AB:   st_q <= S_ABW;
        S_ABW:  if (fm_ov) st_q <= S_PK;
        S_PK:   if (take && in_cnt == CW'(m_q) - 1'b1) st_q <= S_PKD;
        S_PKD:  if (out_cnt == CW'(m_q)) st_q <= S_PQ;
        S_PQ:   if (take && in_cnt == CW'(m_q) - 1'b1) st_q <= S_PV;
        S_PV:   if (take && v_cnt == CW'(rows_q) - 1'b1) st_q <= S_PVD;
        S_PVD:  if (out_cnt == CW'(rows_q)) st_q <= S_TK;
        S_TK:   if (tbl_done) st_q <= S_TQ;
        S_TQ:   if (tbl_done) st_q <= S_FIN;
        S_FIN:  st_q <= S_IDLE;
        default: st_q <= S_IDLE;
      endcase
    end
  end
  assign busy = (st_q != S_IDLE);

  always_ff @(posedge clk) begin
    if (st_q == S_IDLE && start) begin
      bank_q  <= bank;
      m_q     <= m;
      rows_q  <= rows;
      alpha_q <= alpha;
      beta_q  <= beta;
    end
    if (st_q == S_ABW && fm_ov) ab_q <= fm_y;
  end

  // counters
  always_ff @(posedge clk) begin
    case (st_q)
      S_IDLE: begin
        in_cnt <= '0; out_cnt <= '0; rd_cnt <= '0; wr_cnt <= '0; v_cnt <= '0;
        emax_k <= 8'd0; emax_q <= 8'd0; bad_k <= 1'b0; bad_q <= 1'b0;
      end
      S_PK, S_PKD: begin
        if (take && st_q == S_PK) in_cnt <= (in_cnt == CW'(m_q) - 1'b1) ? '0 : in_cnt + 1'b1;
        if (fm_ov) begin
          out_cnt <= out_cnt + 1'b1;
          if (e_kp == 8'hFF) bad_k <= 1'b1;
          else if (e_kp > emax_k) emax_k <= e_kp;
        end
      end
      S_PQ: begin
        if (take) begin
          in_cnt <= (in_cnt == CW'(m_q) - 1'b1) ? '0 : in_cnt + 1'b1;
          if (e_q == 8'hFF) bad_q <= 1'b1;
          else if (e_q > emax_q) emax_q <= e_q;
        end
        out_cnt <= '0;
      end
      S_PV, S_PVD: begin
        if (take && st_q == S_PV) v_cnt <= v_cnt + 1'b1;
        if (fm_ov) out_cnt <= out_cnt + 1'b1;
      end
      S_TK, S_TQ: begin
        if (tmp_re) rd_cnt <= rd_cnt + 1'b1;
        if (tr_we != 2'b00) wr_cnt <= wr_cnt + 1'b1;
        if (tbl_done) begin
          rd_cnt <= '0;
          wr_cnt <= '0;
        end
      end
      default: ;
    endcase
  end
  assign tbl_done = (wr_cnt == m_q) & ((st_q == S_TK) | (st_q == S_TQ));

  // table pipeline: RAM read (RD_LAT) -> T1 shift -> T2 round and sign -> T3 split -> write ----------
  logic [RD_LAT-1:0] jv;                      // read valid delayed by 1 .. RD_LAT (jv[RD_LAT-1]: the data is valid)
  logic [JW-1:0]     jd [RD_LAT];
  logic              half_k [RD_LAT];
  always_ff @(posedge clk) begin
    if (!rst_n) jv <= '0; else jv <= RD_LAT'({jv, tmp_re});
  end
  always_ff @(posedge clk) begin
    jd[0] <= rd_cnt[JW-1:0];
    half_k[0] <= (st_q == S_TK);
    for (int i = 1; i < RD_LAT; i++) begin
      jd[i] <= jd[i-1];
      half_k[i] <= half_k[i-1];
    end
  end
  wire [31:0] x_t = tmp_rdata;
  logic [7:0]  emax_sel, e_t;
  logic [7:0]  sh_t;
  logic        zero_t;
  assign emax_sel = (half_k[RD_LAT-1]) ? emax_k : emax_q;
  assign e_t      = x_t[30:23];
  assign zero_t   = (e_t == 8'd0) | (e_t == 8'hFF);
  assign sh_t     = emax_sel - e_t;
  // T1
  logic        t1_v, t1_s, t1_k;
  logic [JW-1:0] t1_j;
  logic [24:0] t1_m;
  always_ff @(posedge clk) begin
    if (!rst_n) t1_v <= 1'b0; else t1_v <= jv[RD_LAT-1];
  end
  always_ff @(posedge clk) begin
    t1_s <= x_t[31];
    t1_k <= half_k[RD_LAT-1];
    t1_j <= jd[RD_LAT-1];
    t1_m <= zero_t ? 25'd0 : (({1'b1, x_t[22:0], 1'b0}) >> ((sh_t > 8'd31) ? 5'd31 : sh_t[4:0]));
  end
  // T2
  logic          t2_v, t2_s, t2_k;
  logic [JW-1:0] t2_j;
  logic [23:0]   t2_mag;
  always_ff @(posedge clk) begin
    if (!rst_n) t2_v <= 1'b0; else t2_v <= t1_v;
  end
  always_ff @(posedge clk) begin
    t2_s   <= t1_s;
    t2_k   <= t1_k;
    t2_j   <= t1_j;
    t2_mag <= 24'((t1_m + 25'd1) >> 1);
  end
  // T3: signed entry and the word
  logic          t3_v, t3_k;
  logic [JW-1:0] t3_j;
  logic [24:0]   t3_w;
  logic [24:0]   t3_neg;
  assign t3_neg = 25'(-{1'b0, t2_mag});
  always_ff @(posedge clk) begin
    if (!rst_n) t3_v <= 1'b0; else t3_v <= t2_v;
  end
  always_ff @(posedge clk) begin
    t3_k <= t2_k;
    t3_j <= t2_j;
    t3_w <= (t2_s && t2_mag != 24'd0) ? t3_neg : {1'b0, t2_mag};      // two's complement entry, 25 bits
  end
  assign tr_we    = t3_v ? (t3_k ? 2'b01 : 2'b10) : 2'b00;
  assign tr_lane  = t3_j[LW-1:0];
  assign tr_waddr = {bank_q, t3_j[LW +: NBW]};
  assign tr_wdata = {7'd0, t3_w[24:13], t3_w[12:0]};

  // finished
  assign done      = (st_q == S_FIN);
  assign done_ek   = emax_k;
  assign done_eq   = emax_q;
  assign done_badk = bad_k;
  assign done_badq = bad_q;


endmodule
