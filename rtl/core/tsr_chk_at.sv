// tsr_chk_at.sv — the AT external checker classes of tsr_chk (CC §8.2: "opcodes from VU / AT plug in
// as classes"): KVST 0x14 (unit DMA), ATTQK 0x21 and ATTPV 0x22 (unit MVM). One op at a time: the
// error code in EX §8.1 order, ERR_INFO, the tagged operand extents and the unit payload
// (tsr_att_pkg::tsr_kvst_cmd_t / tsr_att_cmd_t, strides resolved).
//
// Owner: AT (W4-B "at"; docs/design/03-attention-kv.md "AT" §5.1-§5.3, §5.8). The external-class
// mux of tsr_chk's xc_* port (VU builder, rtl/core/tsr_xc_pkg.sv) routes the AT opcodes here
// (tsr_att_pkg::tsr_att_is_op); with no VU class the ports connect to tsr_chk's xc_* directly.
// Oracle: the simulator's chk_kvst / chk_attqk / chk_attpv (sw/sim/tsr_at.c), compared bit for bit
// (code, info, extents) by tb/at (test_at_chk), and the payload fields by the same bench.
//
// Checks (AT §5.8; tsr_at.c header), in this order:
//   KVST   reserved bits (TSR_KVST_RSV_W*) or sub >= 2: ILLEGAL_OP; E/64 = 0: K_RANGE; LG = 0:
//          OPERAND (0); c0 + n_rows > 16 C LG: OPERAND (c0); SPM_RANGE src with (n_rows - 1) x
//          stride + QA_LINES(A32, E) (stride 0 -> QA_LINES); DRAM_RANGE B with LG x 17 E/64.
//          Extents: SPM read src, DRAM write B (every channel).
//   ATTQK  reserved bits, or w7 != 0 without MAXOUT: ILLEGAL_OP; sub != 0: BAD_FMT; E/64 = 0 or
//          G E > K_MAX: K_RANGE; T_q GS > T_MAX: T_RANGE; OPERAND: E % D (E), hg0 + GS > G (hg0 +
//          GS), n_groups not in [1, 4095] (n), c_lo > c_hi or c_hi > 16 n (c_hi), pass rows > 128
//          (rows), 0 < rstride < n (rstride); SPM_RANGE dst (max row x rstride + n), x (T_q x
//          QA_LINES(A32, G E)), mdst (MAXOUT: ceil((max row + 1) / 16)); DRAM_RANGE B ((floor((u0 +
//          n - 1) / C) + 1) x GB). Extents: dst write, x read, B read (every channel), mdst write.
//   ATTPV  as ATTQK up to n; then OPERAND pass rows (rows), floor((u0 + n - 1) / C) >= LG (u0 + n
//          - 1), (E/D) 16 ceil(n / C) > PCAP (n), 0 < pstride < n (pstride); SPM_RANGE dst ((C - 1)
//          cstride + R D/16, with LSUM at least C cstride + C ceil(R / 16)), p (max row x pstride +
//          n); DRAM_RANGE B (((E/64 - 1) LG + l_max + 1) x 17). Extents: dst write, p read, B read.
// Range rule (spec §21): a + e <= limit in 48 bits; SPM limit SPM_BYTES / 64 lines, DRAM limit
// min(CH_BYTES / 64, 2^ADDR_BEAT_W) beats. ERR_INFO as listed (the start address for a range
// error); 0 for ILLEGAL_OP, BAD_FMT, K_RANGE, T_RANGE.
//
// Interface (the class side of tsr_chk's xc_* port): req_valid is a level held with req_insn from
// the request until the cycle of rsp_valid (tsr_chk's xc_wait_q); the class answers with one
// rsp_valid pulse, the rsp_* payload valid in that cycle and 0 in every other cycle (OR join,
// tsr_xc_pkg). A new request starts only from idle and after req_valid was seen low once.
//
// Pipeline: S0 capture, S1 decode and the small products, S2 the divisions by NUM_CH (exact
// multiply-by-reciprocal, tsr_att_divc) and second-level products, S3 max row / resolved strides,
// S4 extents, S5 end addresses, S6 limit compares, S7 priority -> rsp_valid: 8 cycles from the
// request to rsp_valid for every op (fixed; no serial divider). Every stage is register to register;
// the widest products (19 x 15, 17 x 22, 36 x 4 bits) map to one DSP48E2 each or a short LUT
// multiplier (CC §14.3: the checker is not on the core-clock critical list, CC §8.2).

module tsr_chk_at
  import tsr_pkg::*;
  import tsr_att_pkg::*;
#(
  parameter int unsigned     NUM_CH      = TSR_DEF_NUM_CH,
  parameter int unsigned     T_MAX       = TSR_DEF_T_MAX,
  parameter int unsigned     K_MAX       = TSR_DEF_K_MAX,
  parameter int unsigned     SPM_BYTES   = TSR_DEF_SPM_BYTES,
  parameter longint unsigned CH_BYTES    = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter int unsigned     XC_CMD_W    = 256
) (
  input  logic                     clk,
  input  logic                     rst_n,

  input  logic                     req_valid,
  input  tsr_insn_t                req_insn,

  output logic                     rsp_valid,
  output logic [3:0]               rsp_code,
  output logic [31:0]              rsp_info,
  output tsr_ext_t [TSR_EXT_N-1:0] rsp_ext,
  output logic [XC_CMD_W-1:0]      rsp_cmd
);

  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX) begin : g_chk_ch
    $fatal(1, "tsr_chk_at: NUM_CH must be in [1, 16]");
  end
  if (XC_CMD_W < $bits(tsr_att_cmd_t) || XC_CMD_W < $bits(tsr_kvst_cmd_t)) begin : g_chk_w
    $fatal(1, "tsr_chk_at: XC_CMD_W too small for the AT payloads");
  end

  localparam int unsigned AW = TSR_EXT_AW;                                  // 48
  localparam logic [AW-1:0] SPM_LIM  = AW'(SPM_BYTES / TSR_LINE_BYTES);
  localparam logic [AW-1:0] DRAM_LIM = AW'(tsr_dram_limit(CH_BYTES, ADDR_BEAT_W));
  localparam logic [31:0]   PCAP     = 32'(tsr_qa_lines(1'b1, 32'(K_MAX))) << 4;   // AT §5.8
  localparam logic [3:0]    CM1      = 4'(NUM_CH - 1);

  localparam logic [7:0] OP_KVST  = tsr_tables_pkg::TSR_OP_KVST;
  localparam logic [7:0] OP_ATTQK = tsr_tables_pkg::TSR_OP_ATTQK;
  localparam logic [7:0] OP_ATTPV = tsr_tables_pkg::TSR_OP_ATTPV;

  // reserved-bit masks per op (generated tables, EX §7.1)
  localparam tsr_insn_t RSV_KVST = {tsr_tables_pkg::TSR_KVST_RSV_W7, tsr_tables_pkg::TSR_KVST_RSV_W6,
    tsr_tables_pkg::TSR_KVST_RSV_W5, tsr_tables_pkg::TSR_KVST_RSV_W4, tsr_tables_pkg::TSR_KVST_RSV_W3,
    tsr_tables_pkg::TSR_KVST_RSV_W2, tsr_tables_pkg::TSR_KVST_RSV_W1, tsr_tables_pkg::TSR_KVST_RSV_W0};
  localparam tsr_insn_t RSV_ATTQK = {tsr_tables_pkg::TSR_ATTQK_RSV_W7, tsr_tables_pkg::TSR_ATTQK_RSV_W6,
    tsr_tables_pkg::TSR_ATTQK_RSV_W5, tsr_tables_pkg::TSR_ATTQK_RSV_W4, tsr_tables_pkg::TSR_ATTQK_RSV_W3,
    tsr_tables_pkg::TSR_ATTQK_RSV_W2, tsr_tables_pkg::TSR_ATTQK_RSV_W1, tsr_tables_pkg::TSR_ATTQK_RSV_W0};
  localparam tsr_insn_t RSV_ATTPV = {tsr_tables_pkg::TSR_ATTPV_RSV_W7, tsr_tables_pkg::TSR_ATTPV_RSV_W6,
    tsr_tables_pkg::TSR_ATTPV_RSV_W5, tsr_tables_pkg::TSR_ATTPV_RSV_W4, tsr_tables_pkg::TSR_ATTPV_RSV_W3,
    tsr_tables_pkg::TSR_ATTPV_RSV_W2, tsr_tables_pkg::TSR_ATTPV_RSV_W1, tsr_tables_pkg::TSR_ATTPV_RSV_W0};

  // -------------------------------------------------------------------------------------------
  // Stage control (one op at a time)
  // -------------------------------------------------------------------------------------------
  logic [7:0] st_q;              // st_q[k]: stage Sk holds the op
  logic       rsp_q;
  logic       seen_q;            // answered; waits for req_valid low (tsr_xc_pkg: one answer per request)
  wire        start = req_valid & (st_q == 8'd0) & ~rsp_q & ~seen_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q   <= '0;
      rsp_q  <= 1'b0;
      seen_q <= 1'b0;
    end else begin
      st_q   <= {st_q[6:0], start};
      rsp_q  <= st_q[7];
      if (st_q[7])        seen_q <= 1'b1;
      else if (!req_valid) seen_q <= 1'b0;
    end
  end

  // -------------------------------------------------------------------------------------------
  // S0: capture
  // -------------------------------------------------------------------------------------------
  tsr_insn_t w_q;
  always_ff @(posedge clk) begin
    if (start) w_q <= req_insn;
  end

  wire [7:0]  op    = w_q[0][7:0];
  wire [3:0]  sub   = w_q[0][11:8];
  wire [3:0]  flg   = w_q[0][15:12];
  wire [31:0] w1 = w_q[1], w2 = w_q[2], w3 = w_q[3], w4 = w_q[4], w5 = w_q[5], w6 = w_q[6], w7 = w_q[7];

  // -------------------------------------------------------------------------------------------
  // S1: decode, small products (from the S0 registers)
  // -------------------------------------------------------------------------------------------
  logic        is_kv_q, is_qk_q, is_pv_q, ill_q, fmt_q, maxout_q, lsum_q;
  logic [8:0]  e64_q, e64m1_q;   // E/64, E/64 - 1
  logic [1:0]  dc_q;
  logic [6:0]  g_q;              // G 1..64
  logic [5:0]  hg0_q;
  logic [3:0]  gs_q, tq_q;       // 1..8
  logic [14:0] rs_q;
  logic [15:0] u0_q, n_q;
  logic [6:0]  nr_q;             // KVST n_rows 1..64
  logic [21:0] ge_q;             // G E
  logic [6:0]  tg_q;             // T_q GS
  logic [8:0]  nseg_q;           // E / D (valid when E % D == 0)
  logic        emod_q;           // E % D != 0
  logic [6:0]  hgs_q;            // hg0 + GS
  logic [9:0]  kqa_q;            // KVST QA_LINES(A32, E) = E/64 + ceil(E/512)
  logic [13:0] gb_q;             // GB = 17 E/64
  logic [16:0] x1_q;             // u0 + n - 1 (n >= 1)
  logic [16:0] x2_q;             // n + C - 1
  logic        n_bad_q;          // n = 0 or n > 4095

  always_ff @(posedge clk) begin
    if (st_q[0]) begin
      is_kv_q  <= (op == OP_KVST);
      is_qk_q  <= (op == OP_ATTQK);
      is_pv_q  <= (op == OP_ATTPV);
      ill_q    <= (op == OP_KVST)  ? (((w_q & RSV_KVST) != '0) || sub >= 4'd2)
                : (op == OP_ATTQK) ? (((w_q & RSV_ATTQK) != '0) || (!flg[2] && w7 != 32'd0))
                : (op == OP_ATTPV) ? ((w_q & RSV_ATTPV) != '0)
                : 1'b1;
      fmt_q    <= (sub != 4'd0);
      maxout_q <= flg[2];
      lsum_q   <= flg[0];
      e64_q    <= (op == OP_KVST) ? w3[8:0] : w4[8:0];
      e64m1_q  <= ((op == OP_KVST) ? w3[8:0] : w4[8:0]) - 9'd1;
      dc_q     <= w4[10:9];
      g_q      <= 7'(w_q[0][21:16]) + 7'd1;
      hg0_q    <= w_q[0][27:22];
      gs_q     <= 4'(w4[16:14]) + 4'd1;
      tq_q     <= 4'(w4[13:11]) + 4'd1;
      rs_q     <= w4[31:17];
      u0_q     <= w5[15:0];
      n_q      <= w5[31:16];
      nr_q     <= 7'(w_q[0][21:16]) + 7'd1;
      ge_q     <= 22'(7'(w_q[0][21:16]) + 7'd1) * 22'({w4[8:0], 6'd0});
      tg_q     <= 7'(4'(w4[13:11]) + 4'd1) * 7'(4'(w4[16:14]) + 4'd1);
      nseg_q   <= w4[8:0] >> w4[10:9];
      emod_q   <= (w4[8:0] & ((9'd1 << w4[10:9]) - 9'd1)) != 9'd0;
      hgs_q    <= 7'(w_q[0][27:22]) + 7'(w4[16:14]) + 7'd1;
      kqa_q    <= 10'(w3[8:0]) + 10'((10'(w3[8:0]) + 10'd7) >> 3);
      gb_q     <= 14'(w3[8:0]) * 14'd17;
      x1_q     <= 17'(w5[15:0]) + 17'(w5[31:16]) - 17'd1;
      x2_q     <= 17'(w5[31:16]) + 17'(CM1);
      n_bad_q  <= (w5[31:16] == 16'd0) || (w5[31:16] > 16'(tsr_tables_pkg::TSR_ATT_MAX_GROUPS));
    end
  end
  // ATT ops: GB from w4's E/64 (KVST's was latched above from w3)
  logic [13:0] gba_q;
  always_ff @(posedge clk) begin
    if (st_q[0]) gba_q <= 14'(w4[8:0]) * 14'd17;
  end

  // -------------------------------------------------------------------------------------------
  // S2: divisions by NUM_CH, second products
  // -------------------------------------------------------------------------------------------
  logic        krng_q, trng_q;
  logic [15:0] nhead_q;          // G (E/D)
  logic [15:0] prows_q;          // (E/D) T_q GS
  logic [16:0] lmax_q;           // floor((u0 + n - 1) / C)
  logic [16:0] ncq_q;            // ceil(n / C)
  logic [31:0] kssp_q;           // KVST resolved stride
  logic [23:0] kcells_q;         // 16 C LG
  logic [29:0] kep_q;            // KVST LG x GB
  logic [32:0] kend_q;           // c0 + n_rows
  logic [4:0]  dsh_q;            // log2(D / 16) = 2 + dcode
  logic [12:0] tnm1_q;           // T_q (E/D) - 1
  logic [6:0]  hgs1_q;           // hg0 + GS - 1
  logic [22:0] kwlo_q, kwhi_q;   // KVST (n_rows - 1) x w5[15:0], x w5[31:16]
  logic [16:0] kqq_q;            // KVST (n_rows - 1) x QA_LINES(A32, E)

  always_ff @(posedge clk) begin
    if (st_q[1]) begin
      krng_q   <= (e64_q == 9'd0) || (!is_kv_q && 32'(ge_q) > 32'(K_MAX));
      trng_q   <= 32'(tg_q) > 32'(T_MAX);
      nhead_q  <= 16'(g_q) * 16'(nseg_q);
      prows_q  <= 16'(nseg_q) * 16'(tg_q);
      lmax_q   <= 17'(tsr_att_divc(21'(x1_q), NUM_CH));
      ncq_q    <= 17'(tsr_att_divc(21'(x2_q), NUM_CH));
      kssp_q   <= (w5 != 32'd0) ? w5 : 32'(kqa_q);
      kcells_q <= 24'(w3[31:16]) * 24'(16 * NUM_CH);
      kep_q    <= 30'(w3[31:16]) * 30'(gb_q);
      kend_q   <= 33'(w4) + 33'(nr_q);
      dsh_q    <= 5'd2 + 5'(dc_q);
      // the S3 products split into single multiplies (no DSP cascade or post-adder chain per stage)
      tnm1_q   <= 13'(4'(tq_q) * 9'(nseg_q)) - 13'd1;     // nseg >= 1 (E % D checked; E/64 = 0 fails)
      hgs1_q   <= 7'(hg0_q) + 7'(gs_q) - 7'd1;
      kwlo_q   <= 23'(nr_q - 7'd1) * 23'(w5[15:0]);
      kwhi_q   <= 23'(nr_q - 7'd1) * 23'(w5[31:16]);
      kqq_q    <= 17'(nr_q - 7'd1) * 17'(kqa_q);
    end
  end

  // -------------------------------------------------------------------------------------------
  // S3: max row, resolved strides, R, QA lines
  // -------------------------------------------------------------------------------------------
  logic [19:0] mrow_q;           // r(T_q - 1, E/D - 1, GS - 1) (E/D >= 1)
  logic [15:0] rsp_q2;           // resolved rstride / pstride
  logic [19:0] r_q;              // R = T_q n_head
  logic [16:0] qax_q;            // QA_LINES(A32, G E)
  logic [38:0] kes_q;            // KVST (n_rows - 1) x stride
  logic [29:0] lgqa_q;           // ATTPV (E/64 - 1) x LG
  logic        pcap_bad_q;
  logic        lmax_bad_q;

  always_ff @(posedge clk) begin
    if (st_q[2]) begin
      // (T_q - 1) n_head + (E/D - 1) G + hg0 + GS - 1 = G (T_q E/D - 1) + hg0 + GS - 1: one product
      mrow_q     <= 20'(g_q) * 20'(tnm1_q) + 20'(hgs1_q);
      rsp_q2     <= (rs_q != 15'd0) ? 16'(rs_q) : n_q;
      r_q        <= 20'(tq_q) * 20'(nhead_q);
      qax_q      <= 17'(ge_q >> 6) + 17'((ge_q + 22'd511) >> 9);
      kes_q      <= (w5 != 32'd0) ? (39'(kwhi_q) << 16) + 39'(kwlo_q) : 39'(kqq_q);   // (n_rows - 1) x stride
      lgqa_q     <= 30'(e64m1_q) * 30'(w6[15:0]);
      pcap_bad_q <= ({6'd0, 26'(nseg_q) * 26'(ncq_q), 4'd0} > {4'd0, PCAP});
      lmax_bad_q <= 32'(lmax_q) >= 32'(w6[15:0]);
    end
  end

  // -------------------------------------------------------------------------------------------
  // S4: extents (48 bits)
  // -------------------------------------------------------------------------------------------
  logic [AW-1:0] e_q [4];        // operand extents in field order
  logic [AW-1:0] rl_q;           // ATTPV R D / 16
  logic [AW-1:0] r16_q;          // ceil(R / 16)

  always_ff @(posedge clk) begin
    if (st_q[3]) begin
      e_q[0] <= '0;
      e_q[1] <= '0;
      e_q[2] <= '0;
      e_q[3] <= '0;
      if (is_kv_q) begin
        e_q[0] <= AW'(kes_q) + AW'(kqa_q);                                  // src
        e_q[1] <= AW'(kep_q);                                               // plane
      end else begin
        e_q[0] <= AW'(mrow_q) * AW'(rsp_q2) + AW'(n_q);                     // ATTQK dst / (ATTPV: p)
        e_q[1] <= AW'(tq_q) * AW'(qax_q);                                   // ATTQK x
        e_q[2] <= (AW'(lmax_q) + AW'(1)) * AW'(gba_q);                      // ATTQK plane
        e_q[3] <= (AW'(mrow_q) + AW'(16)) >> 4;                              // ATTQK mdst
      end
      rl_q  <= AW'(r_q) << dsh_q;
      r16_q <= (AW'(r_q) + AW'(15)) >> 4;
    end
  end

  // ATTPV extents: dst, p, plane (from S4 values); cstride resolved
  logic [AW-1:0] pv_cs, pv_ed;
  always_comb begin
    pv_cs = (w7 != 32'd0) ? AW'(w7) : rl_q;
    pv_ed = AW'(CM1) * pv_cs + rl_q;
    if (lsum_q && (AW'(NUM_CH) * pv_cs + AW'(NUM_CH) * r16_q > pv_ed)) pv_ed = AW'(NUM_CH) * pv_cs + AW'(NUM_CH) * r16_q;
  end

  logic [AW-1:0] x_q [4];        // final extents in field order (dst / src, x / p, plane, mdst)
  logic [31:0]   cs_q;           // ATTPV resolved cstride (payload)
  always_ff @(posedge clk) begin
    if (st_q[4]) begin
      x_q[0] <= is_pv_q ? pv_ed : e_q[0];
      x_q[1] <= is_pv_q ? e_q[0] : e_q[1];
      x_q[2] <= is_pv_q ? (AW'(lgqa_q) + AW'(lmax_q) + AW'(1)) * AW'(17) : e_q[2];
      x_q[3] <= e_q[3];
      if (is_kv_q) begin
        x_q[0] <= e_q[0];
        x_q[1] <= e_q[1];
        x_q[2] <= '0;
        x_q[3] <= '0;
      end
      cs_q <= 32'(pv_cs);
    end
  end

  // -------------------------------------------------------------------------------------------
  // S5: end addresses; S6: limit compares
  // -------------------------------------------------------------------------------------------
  logic [AW-1:0] a_q [4];        // operand starts
  logic [AW-1:0] end_q [4];
  logic [3:0]    oor_q;

  // operand starts: KVST src = w1, plane = w2; ATTQK dst = w1, x = w2, plane = w3, mdst = w7;
  // ATTPV dst = w1, p = w2, plane = w3
  always_ff @(posedge clk) begin
    if (st_q[5]) begin
      a_q[0] <= AW'(w1);
      a_q[1] <= AW'(w2);
      a_q[2] <= AW'(w3);
      a_q[3] <= AW'(w7);
      end_q[0] <= AW'(w1) + x_q[0];
      end_q[1] <= AW'(w2) + x_q[1];
      end_q[2] <= AW'(w3) + x_q[2];
      end_q[3] <= AW'(w7) + x_q[3];
    end
    if (st_q[6]) begin
      oor_q[0] <= end_q[0] > SPM_LIM;
      oor_q[1] <= end_q[1] > (is_kv_q ? DRAM_LIM : SPM_LIM);
      oor_q[2] <= end_q[2] > DRAM_LIM;
      oor_q[3] <= end_q[3] > SPM_LIM;
    end
  end

  // -------------------------------------------------------------------------------------------
  // S7: priority, info, extents, payload
  // -------------------------------------------------------------------------------------------
  logic [3:0]  c_code;
  logic [31:0] c_info;
  tsr_ext_t [TSR_EXT_N-1:0] c_ext;
  tsr_kvst_cmd_t kcmd;
  tsr_att_cmd_t  acmd;

  always_comb begin
    c_code = TSR_ERR_NONE;
    c_info = 32'd0;
    if (ill_q) begin
      c_code = TSR_ERR_ILLEGAL_OP;
    end else if (is_kv_q) begin
      if (e64_q == 9'd0)                       c_code = TSR_ERR_K_RANGE;
      else if (w3[31:16] == 16'd0)             begin c_code = TSR_ERR_OPERAND; c_info = 32'd0; end
      else if (kend_q > 33'(kcells_q))         begin c_code = TSR_ERR_OPERAND; c_info = w4; end
      else if (oor_q[0])                       begin c_code = TSR_ERR_SPM_RANGE; c_info = w1; end
      else if (oor_q[1])                       begin c_code = TSR_ERR_DRAM_RANGE; c_info = w2; end
    end else begin
      if (fmt_q)                               c_code = TSR_ERR_BAD_FMT;
      else if (krng_q)                         c_code = TSR_ERR_K_RANGE;
      else if (trng_q)                         c_code = TSR_ERR_T_RANGE;
      else if (emod_q)                         begin c_code = TSR_ERR_OPERAND; c_info = 32'({e64_q, 6'd0}); end
      else if (hgs_q > g_q)                    begin c_code = TSR_ERR_OPERAND; c_info = 32'(hgs_q); end
      else if (n_bad_q)                        begin c_code = TSR_ERR_OPERAND; c_info = 32'(n_q); end
      else if (is_qk_q) begin
        if (w6[15:0] > w6[31:16] || 20'(w6[31:16]) > (20'(n_q) << 4))
                                               begin c_code = TSR_ERR_OPERAND; c_info = 32'(w6[31:16]); end
        else if (prows_q > 16'(tsr_tables_pkg::TSR_ATT_MAX_PASS_ROWS))
                                               begin c_code = TSR_ERR_OPERAND; c_info = 32'(prows_q); end
        else if (rs_q != 15'd0 && 16'(rs_q) < n_q)
                                               begin c_code = TSR_ERR_OPERAND; c_info = 32'(rs_q); end
        else if (oor_q[0])                     begin c_code = TSR_ERR_SPM_RANGE; c_info = w1; end
        else if (oor_q[1])                     begin c_code = TSR_ERR_SPM_RANGE; c_info = w2; end
        else if (maxout_q && oor_q[3])         begin c_code = TSR_ERR_SPM_RANGE; c_info = w7; end
        else if (oor_q[2])                     begin c_code = TSR_ERR_DRAM_RANGE; c_info = w3; end
      end else begin
        if (prows_q > 16'(tsr_tables_pkg::TSR_ATT_MAX_PASS_ROWS))
                                               begin c_code = TSR_ERR_OPERAND; c_info = 32'(prows_q); end
        else if (lmax_bad_q)                   begin c_code = TSR_ERR_OPERAND; c_info = 32'(x1_q); end
        else if (pcap_bad_q)                   begin c_code = TSR_ERR_OPERAND; c_info = 32'(n_q); end
        else if (rs_q != 15'd0 && 16'(rs_q) < n_q)
                                               begin c_code = TSR_ERR_OPERAND; c_info = 32'(rs_q); end
        else if (oor_q[0])                     begin c_code = TSR_ERR_SPM_RANGE; c_info = w1; end
        else if (oor_q[1])                     begin c_code = TSR_ERR_SPM_RANGE; c_info = w2; end
        else if (oor_q[2])                     begin c_code = TSR_ERR_DRAM_RANGE; c_info = w3; end
      end
    end

    // extents in field order (all non-empty for a passing op)
    c_ext = '0;
    if (is_kv_q) begin
      c_ext[0] = '{v: 1'b1, dram: 1'b0, wr: 1'b0, ch: 8'd0, lo: a_q[0], hi: end_q[0]};
      c_ext[1] = '{v: 1'b1, dram: 1'b1, wr: 1'b1, ch: TSR_EXT_CH_ALL, lo: a_q[1], hi: end_q[1]};
    end else begin
      c_ext[0] = '{v: 1'b1, dram: 1'b0, wr: 1'b1, ch: 8'd0, lo: a_q[0], hi: end_q[0]};
      c_ext[1] = '{v: 1'b1, dram: 1'b0, wr: 1'b0, ch: 8'd0, lo: a_q[1], hi: end_q[1]};
      c_ext[2] = '{v: 1'b1, dram: 1'b1, wr: 1'b0, ch: TSR_EXT_CH_ALL, lo: a_q[2], hi: end_q[2]};
      if (is_qk_q && maxout_q)
        c_ext[3] = '{v: 1'b1, dram: 1'b0, wr: 1'b1, ch: 8'd0, lo: a_q[3], hi: end_q[3]};
    end

    kcmd = '{kv: sub[0], n_rows: nr_q, src: w1, stride: kssp_q, base: w2, e64: e64_q, lg: w3[31:16],
             c0: w4};
    acmd = '{pv: is_pv_q, flags: is_pv_q ? {2'b00, lsum_q} : flg[2:0], e64: e64_q, dcode: dc_q,
             g_m1: 6'(g_q - 7'd1), hg0: hg0_q, gs_m1: 3'(gs_q - 4'd1), tq_m1: 3'(tq_q - 4'd1),
             dst: w1, x: w2, base: w3, rs: 15'(rsp_q2), u0: u0_q, n: n_q[11:0],
             aux0: w6[15:0], aux1: is_pv_q ? 16'd0 : w6[31:16], w7: is_pv_q ? cs_q : w7};
  end

  // rsp_* are 0 outside the rsp_valid cycle (the classes' answers are ORed, tsr_xc_pkg::tsr_xc_or)
  always_ff @(posedge clk) begin
    if (!rst_n || !st_q[7]) begin
      rsp_code <= '0;
      rsp_info <= '0;
      rsp_ext  <= '0;
      rsp_cmd  <= '0;
    end else begin
      rsp_code <= c_code;
      rsp_info <= c_info;
      rsp_ext  <= (c_code == TSR_ERR_NONE) ? c_ext : '0;
      rsp_cmd  <= is_kv_q ? XC_CMD_W'(kcmd) : XC_CMD_W'(acmd);
    end
  end

  assign rsp_valid = rsp_q;

  wire unused_ok = &{1'b0, w_q[0][31:28], flg[3]};

`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      assert ($onehot0(st_q)) else $error("%m: more than one op in the pipeline (%b)", st_q);
      if (st_q[1]) assert (tsr_att_is_op(op)) else $error("%m: opcode %02x is not an AT op", op);
    end
  end
`endif

endmodule
