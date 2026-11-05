// tsr_mvm_cluster.sv — one MVM cluster (spec MVM cluster microarchitecture: the MVM engine of one channel): front end,
// activation buffer, back end, and the glue between them and the broadcast / result pipes.
//
// Owner: top agent (docs/rtl-m3.md §2 block (top); contract §3.1, §3.2, §3.5, §6.2).
//
// Contract:
//   - broadcast input (after tsr_mvm's pipe, no backpressure, docs/rtl-m3.md §3.1): per MVM with
//     N >= 1 exactly one bc_cmd (tsr_mvm_ccmd_t), then T * QA_LINES bc_line items, the final one
//     with last = 1; the next MVM's bc_cmd comes only after the previous MVM finished everywhere
//     (tsr_mvm waits for every expected last result). The cluster accepts every item in any
//     state: bc_line always goes into the activation buffer (tsr_mvm_actbuf write port).
//   - setup on bc_cmd: first = CH * R (R = rows_per_ch, 64-bit product), rows_c =
//     tsr_mvm_rows_c(n, R, first), beats_c = rows_c * row_beats (multi-cycle arithmetic allowed);
//     if rows_c >= 1: one tsr_mvm_fcmd_t {fmt, k, row_beats, w_off, rows_c, beats_c} to the front
//     end (which starts streaming weights at once, overlapping the broadcast), cfg_fmt /
//     cfg_d_lines to the back end. rows_c = 0: the cluster stays silent (no memory traffic, no
//     result) and only absorbs the broadcast.
//   - activation gating: front -> back beats are passed only after the bc_line with last = 1 of
//     this MVM has been written into the activation buffer (and until the MVM's last result).
//   - results (R1/R2, through tsr_mvm's result pipe): one item per row in row order:
//     res_row = first + r (global row index n), res_last = (r == rows_c - 1), res_y = the back
//     end's T_MAX results.
//   - memory: the front end's WS read source port, unchanged (docs/rtl-m3.md §4).
//   - idle: no MVM in progress in this cluster (setup done or not started, front and back idle).
// Placement: one cluster per channel, in the SLR of that channel's DDR controller (spec Placement);
//   only the broadcast pipe, the result pipe and the memory port leave it.
//
// Interface addition (additive, top agent): parameter WS_RD_LAT (weight FIFO RAM read latency,
//   docs/rtl-m3.md §1, default TSR_DEF_WS_RD_LAT) is passed to tsr_mvm_front.
//
// Attention (W4-B at; AT §4.2, §5.2, §5.3, §8.1, §8.2; parameter HAS_ATT). An attention op arrives as
//   bc_acmd (tsr_att_ccmd_t on the broadcast pipe) followed by its lines:
//   - geometry: the op's groups u in [u0, u0 + n) with u mod C = CH are local groups lambda = 0 ..
//     ng - 1 of this channel, local index l_first + lambda (l_first = u0 / C or + 1, ng from
//     (u0 + n) / C, both from the command's q0 / m0 / qn / mn), u - u0 = off0 + lambda C.
//   - slots: t < tact = T_q GS is query q(t) = t / GS, head offset i(t) = t mod GS; row r(q, g, i)
//     = q n_head + g G + hg0 + i.
//   - ATTQK: the lines are the T_q Q vectors (tok = q), each written into the GS slot columns of its
//     query (actbuf token mask); the K plane stream is the channel's groups, contiguous: front
//     command w_off = B + l_first GB, ng E/64 "rows" of K = 1024 (AT §8.1). The back end computes
//     the SEGs (att_cfg / att_tbase), tsr_mvm_aqk assembles the score lines (-inf outside the per-
//     query bounds lo_q / hi_q) into jobs: line = dst + r rstride + (u - u0). A group's first beat
//     enters only while fewer than 2 groups are buffered (aqk's line-buffer halves).
//   - ATTPV: the lines are the p rows in the order (g, u - u0, t) (tsr_mvm); this cluster keeps the
//     lines of its own groups (u mod C = CH), line (g, lambda) of slot t at its slot-buffer line
//     g ng + lambda, and counts what it has (p_g / p_l) for the PV unit. The V stream is E/64 ranges
//     (slices) of ng chunks at stride 17 LG from B + 17 l_first (tsr_mvm_ws ranges). tsr_mvm_pv
//     does the rest (the back end only lends it the activation reads and the MAC lanes) and its
//     jobs: O lines dst + CH cstride + r D/16 + 4 (s mod D/64) + m, the row sums at dst + C cstride
//     + CH ceil(R/16) + r/16. Every cluster writes its partials, also with ng = 0 (zeros).
//   - aj_*: the attention jobs (gather job format) of this cluster, the last one flagged.
//   - att_stall: an ATT beat waits for the line buffer (ATTQK) or its p line (ATTPV) (PERF
//     ATT_PATH_STALL, AT §10).
//
// Microarchitecture:
//   - bc_line -> one register (aw_*: the activation-buffer write port; docs/rtl-m3.md §9.2 item 1:
//     one more register in front of the 3 x T_MAX activation RAMs) -> tsr_mvm_actbuf.
//     act_ready_q := 0 at bc_cmd, := 1 at the edge that writes the line with last = 1.
//   - setup pipeline (one-hot su_q; the front command is valid 8 cycles after the bc_cmd cycle):
//       su0: CH * R as two partial products (CH & 3) * R and (CH >> 2) * R (constant shifts / adds)
//       su1: first = sum (36 bits)
//       su2: N > first, N - first
//       su3: rows_c = min(R, N - first) or 0
//       su4..su6: beats = rows_c * row_beats (32 x 16 multiplier with 3 pipeline registers: maps
//                 to cascaded DSPs with M / P registers; ~2 DSP48E2 per cluster); the front
//                 command becomes valid with the last product register (rows_c >= 1) and is held
//                 until the front end takes it (R2)
//     The front end starts streaming at once; the setup hides behind the broadcast (T * QA_LINES
//     lines) and the memory round trip.
//   - beat gating (docs/rtl-m3.md §3.2): beat_valid_back = beat_valid & act_ready_q,
//     beat_ready_front = beat_ready & act_ready_q (flops, one LUT).
//   - result tagging: row_q (= first + r) and left_q (rows still to come) count result
//     handshakes; res_last_q = (left_q == 1). res_valid / res_y / res_ready pass straight between
//     the back end and the result pipe (no combinational input -> output path: the back end's
//     res_valid does not depend on res_ready, R2).
//   - attention setup (as_q one-hot, 4 steps; every product between flops), then the rowbase table
//     builder (ATTQK: dst + r rstride per pass row, one entry per cycle, done long before the first
//     group's drain) and the start pulses of the back end, aqk / pv.
//   Simulation checks: protocol of the broadcast (one MVM at a time), results only for this
//   cluster's rows, every row's result exactly once.
//   Cost (RTL structure): ~1.1 K FF (527 activation-buffer write register, ~400 setup) and the
//   32 x 16 product (2 DSP48E2 expected) besides the front end, activation buffer and back end;
//   attention: ~1.5 K FF of setup / slot tables, tsr_mvm_aqk, tsr_mvm_pv.
//   Measured (Vivado 2024.2 OOC synthesis, xcu200, 300 MHz, CH 0 of 4, T_MAX 8, MAC_LAT 9, W4-B at):
//   HAS_ATT 0: 67.8 K LUT, 65.2 K FF, 674 DSP, 200 RAMB36, 0.6 K LUTRAM; HAS_ATT 1: 103.1 K LUT
//   (+35.3 K), 92.7 K FF (+27.4 K), 713 DSP (+39), 220 RAMB36 + 16 RAMB18, 4.3 K LUTRAM: tsr_mvm_pv
//   21.3 K LUT / 24.6 K FF (capture bank 19.5 K FF), tsr_mvm_aqk 2.3 K LUT, activation buffer +3.3 K
//   LUT (per-token addresses), back end +3.6 K LUT; post-synthesis WNS +0.217 ns (HAS_ATT 0: +1.017).

module tsr_mvm_cluster
  import tsr_pkg::*;
  import tsr_att_pkg::*;
#(
  parameter  bit          LOCAL_INTERMEDIATES = 1'b0,
  parameter  int unsigned CH            = 0,     // channel index c of this cluster
  parameter  int unsigned NUM_CH        = TSR_DEF_NUM_CH,
  parameter  bit          HAS_ATT       = 1'b0,
  parameter  int unsigned K_MAX         = TSR_DEF_K_MAX,
  parameter  int unsigned T_MAX         = TSR_DEF_T_MAX,
  parameter  int unsigned ADDR_BEAT_W   = TSR_DEF_ADDR_BEAT_W,
  parameter  int unsigned BEAT_BITS     = TSR_DEF_BEAT_BITS,
  parameter  int unsigned WS_FIFO_BEATS = TSR_DEF_WS_FIFO_BEATS,
  parameter  int unsigned ACT_RD_LAT    = TSR_DEF_ACT_RD_LAT,
  parameter  int unsigned MAC_LAT       = TSR_DEF_MAC_LAT,
  parameter  int unsigned MUL_LAT       = TSR_DEF_FMUL_LAT,
  parameter  int unsigned ADD_LAT       = TSR_DEF_FADD_LAT,
  parameter  int unsigned WS_RD_LAT     = TSR_DEF_WS_RD_LAT,
  parameter  int unsigned SPM_AW        = 16,
  localparam int unsigned JW            = 1 + SPM_AW + 16 + 16 + TSR_LINE_BITS
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // ---- broadcast stream (tsr_mvm -> cluster, no backpressure) ----
  input  logic                          bc_cmd_valid,
  input  tsr_mvm_ccmd_t                 bc_cmd,
  input  logic                          bc_acmd_valid,    // attention command (HAS_ATT)
  input  tsr_att_ccmd_t                 bc_acmd,
  input  logic                          bc_line_valid,
  input  tsr_mvm_bcline_t               bc_line,

  // ---- read source port (tsr_mem_arb source TSR_SRC_WS of channel CH) ----
  output logic                          rd_req_valid,
  input  logic                          rd_req_ready,
  output logic [ADDR_BEAT_W-1:0]        rd_req_addr,
  output logic [TSR_LEN_W-1:0]          rd_req_len,
  output logic [TSR_SRC_ID_W-1:0]       rd_req_id,
  input  logic                          rd_rsp_valid,
  output logic                          rd_rsp_ready,
  input  logic [BEAT_BITS-1:0]          rd_rsp_data,
  input  logic                          rd_rsp_last,
  input  logic [TSR_SRC_ID_W-1:0]       rd_rsp_id,

  // ---- row results (cluster -> tsr_mvm gather) ----
  output logic                          res_valid,
  input  logic                          res_ready,
  output logic [31:0]                   res_row,
  output logic                          res_last,
  output logic [T_MAX-1:0][31:0]        res_y,

  // ---- attention jobs (cluster -> the gather's job pipe of this cluster) ----
  output logic                          aj_valid,
  input  logic                          aj_ready,
  output logic [JW-1:0]                 aj,
  output logic                          att_stall,
  input  tsr_local_score_req_t           local_score_req,
  output logic [TSR_LINE_BITS-1:0]       local_score_data,

  output logic                          idle
);

  if (CH >= TSR_NUM_CH_MAX || (HAS_ATT && CH >= NUM_CH)) begin : g_chk_ch   // NUM_CH: attention only
    $fatal(1, "tsr_mvm_cluster: CH must be < 16 (and < NUM_CH with HAS_ATT)");
  end

  localparam int unsigned QS_WORDS  = tsr_actbuf_qs_words(K_MAX);
  localparam int unsigned DBS_LINES = tsr_actbuf_dbs_lines(K_MAX);
  localparam int unsigned QS_AW     = (QS_WORDS > 1) ? $clog2(QS_WORDS) : 1;
  localparam int unsigned DBS_AW    = (DBS_LINES > 1) ? $clog2(DBS_LINES) : 1;
  localparam int unsigned NS        = (T_MAX + 1) / 2;
  localparam logic [35:0] CH_LO     = 36'(CH & 3);
  localparam logic [35:0] CH_HI     = 36'(CH >> 2);

  // -------------------------------------------------------------------------------------------
  // Broadcast: activation-buffer write register and activation gating
  // -------------------------------------------------------------------------------------------
  logic            aw_v_q;
  tsr_mvm_bcline_t aw_q;
  logic [T_MAX-1:0] aw_tm_q;          // tokens the line goes to (MVM: tok; ATTQK / ATTPV below)
  logic            act_ready_q;
  logic [T_MAX-1:0] att_tm;           // attention token mask of the incoming line
  logic            att_own;           // ATTPV: the incoming line belongs to this cluster
  tsr_mvm_bcline_t att_line;          // the incoming line with the attention address (ATTPV)
  logic            at_qk_q, at_pv_q;  // the current op is ATTQK / ATTPV
  logic                 pv_local, pv_write;
  tsr_mvm_bcline_t      pv_line;
  logic [T_MAX-1:0]     pv_tm;


  always_ff @(posedge clk) begin
    if (!rst_n) begin
      aw_v_q      <= 1'b0;
      act_ready_q <= 1'b0;
    end else begin
      aw_v_q <= pv_local ? pv_write : bc_line_valid & (~at_pv_q | att_own);
      if (bc_cmd_valid || bc_acmd_valid) act_ready_q <= 1'b0;
      else if (aw_v_q && aw_q.last)      act_ready_q <= 1'b1;
    end
  end
  // aw_q loads every cycle (no clock enable on 527 flops, M3 part C timing): every use is
  // qualified by aw_v_q (actbuf write enables, `last`), so only lines with valid matter.
  always_ff @(posedge clk) begin
    aw_q    <= pv_local ? pv_line : at_pv_q ? att_line : bc_line;
    aw_tm_q <= pv_local ? pv_tm : (at_qk_q | at_pv_q) ? att_tm : (T_MAX'(1) << bc_line.tok);
  end

  // -------------------------------------------------------------------------------------------
  // Setup (docs/rtl-m3.md §3.2)
  // -------------------------------------------------------------------------------------------
  logic [6:0]                 su_q;        // one-hot setup step
  logic                       busy_q;      // MVM in progress (bc_cmd .. last result / no rows)
  logic                       fmt_q;
  logic [15:0]                k_q, rb_q;
  logic [31:0]                n_q, r_q, woff_q;
  logic [7:0]                 dl_q;
  logic [35:0]                plo_q, phi_q, first_q, left_q;
  logic                       gt_q;
  logic [31:0]                rows_q;
  logic                       has_rows_q;
  logic [47:0]                mul1_q, mul2_q, mul3_q;
  logic                       fc_v_q;
  logic                       f_cmd_ready;

  // result tagging
  logic [31:0]                row_q, left_rows_q;
  logic                       res_last_q;

  wire res_fire = res_valid & res_ready;
  wire fc_fire  = fc_v_q & f_cmd_ready;

  // attention front command (from the attention setup below)
  logic                       afc_v;       // pulse: send the attention front command
  tsr_mvm_fcmd_t              afc;
  logic [TSR_MVM_BEATS_W-1:0] afc_rbeats;
  logic [31:0]                afc_stride;
  logic                       fc_att_q;    // the pending front command is an attention stream
  tsr_mvm_fcmd_t              afc_q;
  logic [TSR_MVM_BEATS_W-1:0] afc_rbeats_q;
  logic [31:0]                afc_stride_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      su_q        <= '0;
      busy_q      <= 1'b0;
      fmt_q       <= 1'b0;                 // back-end configuration: defined after reset
      dl_q        <= '0;
      fc_v_q      <= 1'b0;
      fc_att_q    <= 1'b0;
      has_rows_q  <= 1'b0;
      row_q       <= '0;
      left_rows_q <= '0;
      res_last_q  <= 1'b0;
    end else begin
      su_q <= {su_q[5:0], bc_cmd_valid};
      if (bc_cmd_valid) begin
        busy_q <= 1'b1;
        fmt_q  <= bc_cmd.fmt;
        dl_q   <= bc_cmd.d_lines;
      end
      if (bc_acmd_valid) fmt_q <= 1'b0;    // attention: Q8_0 (KV format 0)
      if (su_q[3]) begin
        has_rows_q  <= gt_q;
        row_q       <= first_q[31:0];
      end
      if (su_q[4]) begin
        left_rows_q <= rows_q;
        res_last_q  <= (rows_q == 32'd1);
        if (!has_rows_q) busy_q <= 1'b0;   // rows_c = 0: only the broadcast is absorbed
      end
      if (su_q[6] && has_rows_q) begin
        fc_v_q   <= 1'b1;
        fc_att_q <= 1'b0;
      end else if (afc_v) begin
        fc_v_q   <= 1'b1;
        fc_att_q <= 1'b1;
      end else if (fc_fire) begin
        fc_v_q   <= 1'b0;
      end
      if (res_fire) begin
        row_q       <= row_q + 32'd1;
        left_rows_q <= left_rows_q - 32'd1;
        res_last_q  <= (left_rows_q == 32'd2);
        if (res_last_q) busy_q <= 1'b0;
      end
    end
  end

  // Setup datapath (no reset: qualified by su_q).
  always_ff @(posedge clk) begin
    if (bc_cmd_valid) begin
      k_q    <= bc_cmd.k;
      n_q    <= bc_cmd.n;
      r_q    <= bc_cmd.rows_per_ch;
      rb_q   <= bc_cmd.row_beats;
      woff_q <= bc_cmd.w_off;
    end
    // su0: CH * R = (CH & 3) * R + ((CH >> 2) * R) << 2 (constant factors <= 3)
    plo_q   <= CH_LO * 36'(r_q);
    phi_q   <= CH_HI * 36'(r_q);
    // su1
    first_q <= plo_q + (phi_q << 2);
    // su2
    gt_q    <= (36'(n_q) > first_q);
    left_q  <= 36'(n_q) - first_q;
    // su3: rows_c = max(0, min(R, N - first)) (tsr_mvm_rows_c)
    rows_q  <= !gt_q ? 32'd0 : ((left_q < 36'(r_q)) ? left_q[31:0] : r_q);
    // su4..su6: beats_c = rows_c * row_beats
    mul1_q  <= 48'(rows_q) * 48'(rb_q);
    mul2_q  <= mul1_q;
    mul3_q  <= mul2_q;
    if (afc_v) begin
      afc_q        <= afc;
      afc_rbeats_q <= afc_rbeats;
      afc_stride_q <= afc_stride;
    end
  end

  // -------------------------------------------------------------------------------------------
  // Front end, activation buffer, back end
  // -------------------------------------------------------------------------------------------
  tsr_mvm_fcmd_t fc;
  always_comb begin
    if (fc_att_q) begin
      fc = afc_q;
    end else begin
      fc.fmt       = fmt_q;
      fc.k         = k_q;
      fc.row_beats = rb_q;
      fc.w_off     = woff_q;
      fc.rows      = rows_q;
      fc.beats     = mul3_q[TSR_MVM_BEATS_W-1:0];
    end
  end

  logic          f_beat_valid, f_beat_ready, f_idle;
  tsr_mvm_beat_t f_beat;
  logic          b_beat_ready, b_idle;
  logic          qs_re, dbs_re;
  logic [T_MAX-1:0][QS_AW-1:0]  qs_addr;
  logic [T_MAX-1:0][DBS_AW-1:0] dbs_addr;
  logic [T_MAX-1:0][1023:0] qs_data;
  logic [T_MAX-1:0][511:0]  dbs_data;

  tsr_mvm_front #(
    .K_MAX(K_MAX), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .WS_FIFO_BEATS(WS_FIFO_BEATS),
    .WS_RD_LAT(WS_RD_LAT)
  ) u_front (
    .clk, .rst_n,
    .cmd_valid(fc_v_q), .cmd_ready(f_cmd_ready), .cmd(fc), .cmd_att(fc_att_q),
    .cmd_rbeats(fc_att_q ? afc_rbeats_q : fc.beats), .cmd_stride(fc_att_q ? afc_stride_q : 32'd0),
    .rd_req_valid, .rd_req_ready, .rd_req_addr, .rd_req_len, .rd_req_id,
    .rd_rsp_valid, .rd_rsp_ready, .rd_rsp_data, .rd_rsp_last, .rd_rsp_id,
    .beat_valid(f_beat_valid), .beat_ready(f_beat_ready), .beat(f_beat),
    .idle(f_idle)
  );

  // beat routing (header): MVM / ATTQK -> back end (activation gating; ATTQK also the line-buffer
  // gate), ATTPV -> PV unit
  logic qk_gate_q, pv_ready;
  assign f_beat_ready = at_pv_q ? pv_ready : (b_beat_ready & act_ready_q & (~at_qk_q | qk_gate_q));
  wire   b_beat_valid = f_beat_valid & act_ready_q & ~at_pv_q & (~at_qk_q | qk_gate_q);

  tsr_mvm_actbuf #(.K_MAX(K_MAX), .T_MAX(T_MAX), .RD_LAT(ACT_RD_LAT)) u_act (
    .clk, .rst_n,
    .wr_valid(aw_v_q), .wr_tmask(aw_tm_q), .wr_line(aw_q),
    .qs_re, .qs_addr, .qs_data, .dbs_re, .dbs_addr, .dbs_data
  );

  tsr_att_bcfg_t                acfg_q;
  logic [T_MAX-1:0][10:0]       tbase_q;
  logic                         att_start_q;
  logic                         seg_valid;
  logic [T_MAX-1:0][31:0]       seg_y;
  tsr_att_rd_t                  pv_rd;
  logic                         pv_mac_v, pv_mac_first;
  logic [63:0][7:0]             pv_mac_q;
  logic [TSR_MVM_GROUPS-1:0][NS-1:0][24:0] pv_mac_a;
  logic [TSR_MVM_GROUPS-1:0][NS-1:0][31:0][37:0] pv_acc;

  tsr_mvm_back #(
    .HAS_ATT(HAS_ATT), .K_MAX(K_MAX), .T_MAX(T_MAX), .ACT_RD_LAT(ACT_RD_LAT), .MAC_LAT(MAC_LAT),
    .MUL_LAT(MUL_LAT), .ADD_LAT(ADD_LAT)
  ) u_back (
    .clk, .rst_n,
    .cfg_fmt(fmt_q), .cfg_d_lines(dl_q),
    .att_cfg(acfg_q), .att_tbase(tbase_q), .att_start(att_start_q), .seg_valid, .seg_y,
    .pv_rd, .pv_mac_v, .pv_mac_first, .pv_mac_q, .pv_mac_a, .pv_acc,
    .beat_valid(b_beat_valid), .beat_ready(b_beat_ready), .beat(f_beat),
    .qs_re, .qs_addr, .qs_data, .dbs_re, .dbs_addr, .dbs_data,
    .res_valid, .res_ready, .res_y,
    .idle(b_idle)
  );

  assign res_row  = row_q;
  assign res_last = res_last_q;

  // -------------------------------------------------------------------------------------------
  // Attention (HAS_ATT)
  // -------------------------------------------------------------------------------------------
  logic att_act_q;             // an attention op is in progress in this cluster

  if (HAS_ATT) begin : g_att
    localparam int unsigned TW = (T_MAX > 1) ? $clog2(T_MAX) : 1;
    logic          att_fin;
    tsr_att_ccmd_t ac_q;
    logic [4:0]    as_q;                 // one-hot setup steps
    // geometry
    logic [16:0]   lf_q, le_q;           // l_first, l_end (exclusive)
    logic [12:0]   ng_q;
    logic [4:0]    off0_q;
    logic [13:0]   gb_q;                 // 17 E/64
    logic [3:0]    gs_q, tq_q, tact_q;
    logic [6:0]    g_q;                  // G
    // slots
    logic [T_MAX-1:0][2:0]   qt_q;       // query of slot t
    logic [T_MAX-1:0][3:0]   it_q;       // head offset i(t)
    logic [7:0][T_MAX-1:0]   qm_q;       // ATTQK: slots of query q (actbuf token mask)
    logic [T_MAX-1:0][17:0]  r0_q;       // q(t) n_head + hg0 + i(t)
    logic [7:0][17:0]        lo_q, hi_q; // ATTQK bounds per query
    // ATTPV output bases
    logic [T_MAX-1:0][SPM_AW-1:0] ob0_q;
    logic [SPM_AW-1:0]       og_q, lbase_q;
    logic [SPM_AW-1:0]       chcs_q, ccs_q, chrl_q;
    // front command terms (as_q[1]; as_q[2] only adds and shifts)
    logic [21:0]             arows_q;    // ng E/64
    logic [31:0]             awofs_q;    // l_first GB (ATTQK) / 17 l_first (ATTPV)

    wire [3:0] c_gs  = 4'(bc_acmd.c.gs_m1) + 4'd1;
    wire [3:0] c_tq  = 4'(bc_acmd.c.tq_m1) + 4'd1;

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        at_qk_q     <= 1'b0;
        at_pv_q     <= 1'b0;
        as_q        <= '0;
        att_act_q   <= 1'b0;
        att_start_q <= 1'b0;
        acfg_q      <= '0;
      end else begin
        as_q        <= {as_q[3:0], bc_acmd_valid};
        att_start_q <= as_q[3];
        if (bc_cmd_valid) begin
          at_qk_q <= 1'b0;
          at_pv_q <= 1'b0;
          acfg_q  <= '0;
        end
        if (bc_acmd_valid) begin
          at_qk_q   <= ~bc_acmd.c.pv;
          at_pv_q   <= bc_acmd.c.pv;
          att_act_q <= 1'b1;
          acfg_q    <= '{att: ~bc_acmd.c.pv, pv: bc_acmd.c.pv, dcode: bc_acmd.c.dcode,
                         nb: {bc_acmd.c.e64, 1'b0},
                         hstep: (11'(bc_acmd.c.g_m1) + 11'd1) << (3'(bc_acmd.c.dcode) + 3'd1)};
        end
        if (att_fin) att_act_q <= 1'b0;
      end
    end

    // ---- setup datapath ----
    always_ff @(posedge clk) begin
      if (bc_acmd_valid) begin
        ac_q  <= bc_acmd;
        gs_q  <= c_gs;
        tq_q  <= c_tq;
        tact_q <= bc_acmd.tact;
        g_q   <= 7'(bc_acmd.c.g_m1) + 7'd1;
        gb_q  <= 14'(bc_acmd.c.e64) * 14'd17;
        lf_q  <= (32'(CH) >= 32'(bc_acmd.m0)) ? 17'(bc_acmd.q0) : 17'(bc_acmd.q0) + 17'd1;
        le_q  <= (32'(CH) <  32'(bc_acmd.mn)) ? bc_acmd.qn + 17'd1 : bc_acmd.qn;
        off0_q <= (32'(CH) >= 32'(bc_acmd.m0)) ? 5'(CH) - 5'(bc_acmd.m0) : 5'(NUM_CH + CH) - 5'(bc_acmd.m0);
        // slots: q(t) = t / GS, i(t) = t mod GS; query masks
        for (int t = 0; t < int'(T_MAX); t++) begin
          qt_q[t] <= 3'(t / int'(c_gs));
          it_q[t] <= 4'(t % int'(c_gs));
        end
        for (int q = 0; q < 8; q++) begin
          for (int t = 0; t < int'(T_MAX); t++) qm_q[q][t] <= (t / int'(c_gs) == q);
        end
        // ATTQK bounds: lo_q = max(0, c_lo - SLIDE (T_q - 1 - q)), hi_q = c_hi - CAUSAL (T_q - 1 - q)
        for (int q = 0; q < 8; q++) begin
          automatic int dq = int'(c_tq) - 1 - q;
          automatic int lo = int'(bc_acmd.c.aux0) - (bc_acmd.c.flags[TSR_ATT_FL_SLIDE] ? dq : 0);
          lo_q[q] <= (lo < 0) ? 18'd0 : 18'(lo);
          hi_q[q] <= 18'(int'(bc_acmd.c.aux1) - (bc_acmd.c.flags[TSR_ATT_FL_CAUSAL] ? dq : 0));
        end
      end
      if (as_q[0]) begin
        ng_q <= 13'(le_q - lf_q);
        for (int t = 0; t < int'(T_MAX); t++) begin
          r0_q[t] <= 18'(qt_q[t]) * 18'(ac_q.n_head) + 18'(ac_q.c.hg0) + 18'(it_q[t]);
          tbase_q[t] <= 11'((11'(ac_q.c.hg0) + 11'(it_q[t])) << (3'(ac_q.c.dcode) + 3'd1));
        end
        chcs_q <= SPM_AW'(32'(CH) * ac_q.c.w7);
        ccs_q  <= SPM_AW'(32'(NUM_CH) * ac_q.c.w7);
        chrl_q <= SPM_AW'(32'(CH) * 32'(ac_q.rl));
        og_q   <= SPM_AW'(32'(g_q) << (3'(ac_q.c.dcode) + 3'd2));
      end
      if (as_q[1]) begin
        arows_q <= 22'(ng_q) * 22'(ac_q.c.e64);
        awofs_q <= ac_q.c.pv ? 32'({lf_q, 4'd0}) + 32'(lf_q) : 32'(lf_q) * 32'(gb_q);
        for (int t = 0; t < int'(T_MAX); t++) begin
          ob0_q[t] <= SPM_AW'(ac_q.c.dst) + chcs_q + SPM_AW'(32'(r0_q[t]) << (3'(ac_q.c.dcode) + 3'd2));
        end
        lbase_q <= SPM_AW'(ac_q.c.dst) + ccs_q + chrl_q;
      end
    end

    // ---- front command (as_q[2]): the K plane groups / the V plane slices ----
    // (17 x = 16 x + x: no multiplier between the setup flops and the front command register)
    always_comb begin
      afc.fmt       = 1'b0;
      afc.k         = 16'd1024;
      afc.row_beats = 16'd17;
      afc.rows      = 32'(arows_q);
      afc.beats     = TSR_MVM_BEATS_W'({arows_q, 4'd0}) + TSR_MVM_BEATS_W'(arows_q);
      afc.w_off     = ac_q.c.base + awofs_q;
      afc_rbeats    = ac_q.c.pv ? TSR_MVM_BEATS_W'({ng_q, 4'd0}) + TSR_MVM_BEATS_W'(ng_q) : afc.beats;
      afc_stride    = 32'({ac_q.c.aux0, 4'd0}) + 32'(ac_q.c.aux0);
    end
    assign afc_v = as_q[2] & (ng_q != 13'd0);

    // ---- ATTQK: Q line token masks; ATTPV: p line ownership, slot-buffer address, received ----
    // p lines arrive in the order (g, uo = u - u0, t < tact); this cluster owns uo with
    // (u0 + uo) mod C = CH, i.e. uo = off0 + lambda C
    logic [8:0]  pg_q;              // head group of the next line
    logic [12:0] puo_q;             // uo
    logic [3:0]  pt_q;              // slot
    logic [3:0]  pum_q;             // (u0 + uo) mod C
    logic [12:0] pl_q;              // own lambda of the current uo
    logic [13:0] pab_q;             // g ng (slot-buffer line of lambda 0)
    logic [12:0] rcv_g_q, rcv_l_q;  // complete own items: (g, lambda) received
    logic [12:0] rg1_q, rl1_q, rg2_q, rl2_q;    // the same 2 cycles later (the actbuf write delay)
    wire  own = (pum_q == 4'(CH));
    wire  [13:0] pa = pab_q + 14'(pl_q);
    always_comb begin
      att_line      = bc_line;
      att_line.dbs  = (32'(pa) >= 2 * QS_WORDS);
      att_line.addr = att_line.dbs ? 10'(32'(pa) - 2 * QS_WORDS) : 10'(pa);
      att_own       = own;
      att_tm        = at_pv_q ? (T_MAX'(1) << pt_q) : qm_q[bc_line.tok];
    end
    always_ff @(posedge clk) begin
      if (!rst_n || bc_acmd_valid) begin
        pg_q    <= '0;
        puo_q   <= '0;
        pt_q    <= '0;
        pum_q   <= bc_acmd_valid ? bc_acmd.m0 : 4'd0;
        pl_q    <= '0;
        pab_q   <= '0;
        rcv_g_q <= '0;
        rcv_l_q <= '0;
      end else if (bc_line_valid && at_pv_q) begin
        if (pt_q == tact_q - 4'd1) begin
          pt_q <= '0;
          if (own) begin
            pl_q    <= pl_q + 13'd1;
            rcv_l_q <= pl_q + 13'd1;
          end
          pum_q <= (32'(pum_q) == NUM_CH - 1) ? 4'd0 : pum_q + 4'd1;
          if (puo_q == 13'(ac_q.c.n) - 13'd1) begin
            puo_q   <= '0;
            pum_q   <= ac_q.m0;
            pl_q    <= '0;
            pg_q    <= pg_q + 9'd1;
            pab_q   <= pab_q + 14'(ng_q);
            rcv_g_q <= 13'(pg_q) + 13'd1;
            rcv_l_q <= '0;
          end else begin
            puo_q <= puo_q + 13'd1;
          end
        end else begin
          pt_q <= pt_q + 4'd1;
        end
      end
    end
    always_ff @(posedge clk) begin
      rg1_q <= rcv_g_q;
      rl1_q <= rcv_l_q;
      rg2_q <= rg1_q;
      rl2_q <= rl1_q;
    end

    // ---- ATTQK: rowbase table (dst + r rstride, e = g tact + t), line-buffer gate ----
    logic        tb_act_q, tb_ph_q, tb_done_q;   // builder: phase 0 rb0[t], phase 1 the table
    logic [3:0]  tb_t_q;
    logic [8:0]  tb_g_q;
    logic [6:0]  tb_e_q;
    logic [SPM_AW-1:0] tb_gofs_q, gstep_q;
    logic [T_MAX-1:0][SPM_AW-1:0] rb0_q;
    logic        tb_we;
    logic [SPM_AW-1:0] tb_data;
    always_ff @(posedge clk) begin
      if (!rst_n || bc_acmd_valid) begin
        tb_act_q  <= 1'b0;
        tb_done_q <= 1'b0;
      end else if (as_q[2] && at_qk_q) begin
        tb_act_q  <= 1'b1;
        tb_ph_q   <= 1'b0;
        tb_t_q    <= '0;
        tb_g_q    <= '0;
        tb_e_q    <= '0;
        tb_gofs_q <= '0;
      end else if (tb_act_q) begin
        if (!tb_ph_q) begin
          tb_t_q <= tb_t_q + 4'd1;
          if (tb_t_q == tact_q - 4'd1) begin
            tb_ph_q <= 1'b1;
            tb_t_q  <= '0;
          end
        end else begin
          tb_e_q <= tb_e_q + 7'd1;
          if (tb_t_q == tact_q - 4'd1) begin
            tb_t_q    <= '0;
            tb_g_q    <= tb_g_q + 9'd1;
            tb_gofs_q <= tb_gofs_q + gstep_q;
            if (tb_g_q == ac_q.nseg - 9'd1) begin
              tb_act_q  <= 1'b0;
              tb_done_q <= 1'b1;
            end
          end else begin
            tb_t_q <= tb_t_q + 4'd1;
          end
        end
      end
    end
    always_ff @(posedge clk) begin
      if (as_q[1]) gstep_q <= SPM_AW'(32'(g_q) * 32'(ac_q.c.rs));
      if (tb_act_q && !tb_ph_q) rb0_q[TW'(tb_t_q)] <= SPM_AW'(ac_q.c.dst) + SPM_AW'(32'(r0_q[TW'(tb_t_q)]) * 32'(ac_q.c.rs));
    end
    assign tb_we   = tb_act_q & tb_ph_q;
    assign tb_data = rb0_q[TW'(tb_t_q)] + tb_gofs_q;

    // line-buffer gate: a group's first beat enters only while < 2 groups are buffered
    logic [15:0] gbt_q;            // beats of the current group accepted (0 .. 16 E/64 - 1)
    logic [12:0] gin_q;            // groups whose first beat was accepted
    logic [12:0] gdone;
    wire         qk_acc = f_beat_valid & f_beat_ready & at_qk_q;
    wire [15:0]  gbeats = 16'({ac_q.c.e64, 4'd0});     // 16 E/64 beats per group
    wire         g_lastb = (gbt_q == gbeats - 16'd1);
    wire [15:0]  gbt_n  = qk_acc ? (g_lastb ? 16'd0 : gbt_q + 16'd1) : gbt_q;
    wire [12:0]  gin_n  = gin_q + 13'(qk_acc & (gbt_q == 16'd0));
    always_ff @(posedge clk) begin
      if (!rst_n || bc_acmd_valid) begin
        gbt_q     <= '0;
        gin_q     <= '0;
        qk_gate_q <= 1'b0;
      end else begin
        gbt_q     <= gbt_n;
        gin_q     <= gin_n;
        qk_gate_q <= (gbt_n != 16'd0) | ((gin_n - gdone) < 13'd2);
      end
    end

    // ---- ATTQK assembler ----
    logic        q_jv, q_jr, q_busy;
    logic [JW-1:0] q_j;
    tsr_mvm_aqk #(.T_MAX(T_MAX), .NUM_CH(NUM_CH), .SPM_AW(SPM_AW)) u_aqk (
      .clk, .rst_n,
      .start(att_start_q & at_qk_q), .tact(tact_q), .nseg(ac_q.nseg), .prows(ac_q.prows), .ng(ng_q),
      .off0(off0_q), .qt(qt_q), .lo(lo_q), .hi(hi_q),
      .tb_we, .tb_addr(tb_e_q), .tb_data, .tb_done(tb_done_q),
      .seg_valid, .seg_y,
      .job_valid(q_jv), .job_ready(q_jr), .job(q_j),
      .groups_done(gdone), .busy(q_busy)
    );

    // ---- ATTPV unit ----
    logic        p_jv, p_jr, p_done, p_stall, p_busy;
    logic [JW-1:0] p_j;
    tsr_mvm_beat_t pv_beat;
    assign pv_beat = f_beat;
    tsr_mvm_pv #(
      .K_MAX(K_MAX), .T_MAX(T_MAX), .SPM_AW(SPM_AW), .ACT_RD_LAT(ACT_RD_LAT), .MAC_LAT(MAC_LAT),
      .MUL_LAT(MUL_LAT), .ADD_LAT(ADD_LAT)
    ) u_pv (
      .clk, .rst_n,
      .start(att_start_q & at_pv_q), .tact(tact_q), .tq(tq_q), .gs(gs_q), .e64(ac_q.c.e64),
      .dcode(ac_q.c.dcode), .nseg(ac_q.nseg), .ng(ng_q), .ngl(ng_q), .lsum(ac_q.c.flags[TSR_ATT_FL_LSUM]),
      .g_heads(g_q), .hg0(ac_q.c.hg0), .n_head(ac_q.n_head), .ob0(ob0_q), .og(og_q), .lbase(lbase_q),
      .p_g(rcv_g), .p_l(rcv_l), .done(p_done),
      .beat_valid(f_beat_valid & at_pv_q), .beat_ready(pv_ready), .beat(pv_beat),
      .rd(pv_rd), .qs_data, .dbs_data,
      .mac_v(pv_mac_v), .mac_first(pv_mac_first), .mac_q(pv_mac_q), .mac_a(pv_mac_a), .pv_acc,
      .job_valid(p_jv), .job_ready(p_jr), .job(p_j),
      .stall(p_stall), .busy(p_busy)
    );

    logic [JW-1:0] score_job;
    logic score_valid, score_ready;
    logic [12:0] rcv_g, rcv_l;
    if (LOCAL_INTERMEDIATES) begin : g_local_score
      typedef logic [TSR_LOCAL_SCORE_AW-1:0] score_addr_t;
      logic [2:0][TSR_LINE_BITS-1:0] data_by_slot;
      logic [TSR_LINE_BITS-1:0] pv_data, vu_data;
      tsr_mvm_bcline_t pv_line_q;
      logic [12:0] rg3_q, rl3_q;

      wire [1:0] slot = ac_q.c.local_slot;
      wire score_write = at_qk_q && slot != 0 && q_jv && q_jr;
      wire [SPM_AW-1:0] score_offset = q_j[JW-2 -: SPM_AW] - SPM_AW'(ac_q.c.dst);
      wire pv_read = pv_local && bc_line_valid && own;
      assign pv_local = at_pv_q && slot != 0;
      for (genvar bank = 0; bank < 3; bank++) begin : g_epoch
        wire sel = slot == 2'(bank + 1);
        wire vu_sel = local_score_req.slot == 2'(bank + 1);
        tsr_sram_1r1w #(.DEPTH(TSR_LOCAL_SCORE_LINES), .WIDTH(TSR_LINE_BITS)) u_ram (
          .clk,
          .we((score_write && sel) || (local_score_req.wr && vu_sel)),
          .waddr(score_write && sel ? score_addr_t'(score_offset >> 2) : local_score_req.wa),
          .wdata(score_write && sel ? q_j[TSR_LINE_BITS-1:0] : local_score_req.data), .wbe('1),
          .re((pv_read && sel) || (local_score_req.rd && vu_sel)),
          .raddr(pv_read && sel ? score_addr_t'(bc_line.data[SPM_AW-1:2]) : local_score_req.ra),
          .rdata(data_by_slot[bank])
        );
`ifndef SYNTHESIS
        logic checks_on_q;
        initial checks_on_q = 1'b0;
        always_ff @(posedge clk) if (!rst_n) checks_on_q <= 1'b1;
        else if (checks_on_q) begin
          assert (!(score_write && sel && local_score_req.wr && vu_sel));
          assert (!(pv_read && sel && local_score_req.rd && vu_sel));
        end
`endif
      end
      always_comb begin
        pv_data = '0;
        vu_data = '0;
        for (int bank = 0; bank < 3; bank++) begin
          if (slot == 2'(bank + 1)) pv_data = data_by_slot[bank];
          if (local_score_req.slot == 2'(bank + 1)) vu_data = data_by_slot[bank];
        end
        pv_line = pv_line_q;
        pv_line.data = pv_data;
      end
      assign local_score_data = vu_data;
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          pv_write <= 1'b0;
          rg3_q <= '0;
          rl3_q <= '0;
        end else begin
          pv_write <= pv_read;
          rg3_q <= rg2_q;
          rl3_q <= rl2_q;
        end
        pv_line_q <= att_line;
        pv_tm <= att_tm;
      end
      assign rcv_g = pv_local ? rg3_q : rg2_q;
      assign rcv_l = pv_local ? rl3_q : rl2_q;
      // A line reserves one compact statistics entry before accepting a
      // score write. The balanced tree runs freely into that reserved space,
      // while the gather may apply backpressure at the head.
      localparam int unsigned META_W = JW - TSR_LINE_BITS;
      localparam int unsigned STAT_DEPTH = 8;
      logic [3:0] stat_count_q, stat_wr_q, stat_rd_q;
      logic [4:0] stat_valid_q;
      logic [4:0][META_W-1:0] stat_tag_q;
      logic [15:0][32:0] k0_q;
      logic [7:0][32:0] k1_q;
      logic [3:0][32:0] k2_q;
      logic [1:0][32:0] k3_q;
      logic [32:0] k4_q;
      logic [META_W+31:0] stats [STAT_DEPTH];
      wire stats_valid = stat_wr_q != stat_rd_q;
      wire stats_take = stats_valid && aj_ready;
      wire [META_W+31:0] stats_head = stats[stat_rd_q[2:0]];

      function automatic logic [32:0] larger(logic [32:0] lhs, logic [32:0] rhs);
        return lhs >= rhs ? lhs : rhs;
      endfunction

      assign score_valid = slot == 0 ? q_jv : stats_valid;
      assign score_ready = slot == 0 ? aj_ready : stat_count_q < 4'(STAT_DEPTH);
      always_comb begin
        score_job = q_j;
        if (slot != 0) begin
          score_job = {stats_head[META_W+31:32], TSR_LINE_BITS'(stats_head[31:0])};
          score_job[TSR_LINE_BITS+14] = 1'b1;
        end
      end
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          stat_count_q <= '0;
          stat_wr_q <= '0;
          stat_rd_q <= '0;
          stat_valid_q <= '0;
        end else begin
          stat_count_q <= stat_count_q + 4'(score_write) - 4'(stats_take);
          if (stats_take) stat_rd_q <= stat_rd_q + 1'b1;
          if (stat_valid_q[4]) stat_wr_q <= stat_wr_q + 1'b1;
          stat_valid_q <= {stat_valid_q[3:0], score_write};
        end
        stat_tag_q[0] <= q_j[JW-1:TSR_LINE_BITS];
        for (int k = 1; k < 5; k++) stat_tag_q[k] <= stat_tag_q[k-1];
        for (int w = 0; w < 16; w++) k0_q[w] <= tsr_att_maxkey(q_j[32*w +: 32]);
        for (int w = 0; w < 8; w++) k1_q[w] <= larger(k0_q[2*w], k0_q[2*w+1]);
        for (int w = 0; w < 4; w++) k2_q[w] <= larger(k1_q[2*w], k1_q[2*w+1]);
        for (int w = 0; w < 2; w++) k3_q[w] <= larger(k2_q[2*w], k2_q[2*w+1]);
        k4_q <= larger(k3_q[0], k3_q[1]);
        if (stat_valid_q[4]) stats[stat_wr_q[2:0]] <= {stat_tag_q[4], tsr_att_keyval(k4_q)};
      end
`ifndef SYNTHESIS
      logic stats_checks_q;
      initial stats_checks_q = 1'b0;
      always_ff @(posedge clk) begin
        if (!rst_n) stats_checks_q <= 1'b1;
        else if (stats_checks_q) begin
          assert (stat_count_q <= 4'(STAT_DEPTH));
          // Gather retirement, including the last statistics packet, must
          // precede a new command's broadcast to this cluster.
          if (bc_cmd_valid || bc_acmd_valid)
            assert (stat_count_q == 0 && stat_valid_q == 0 && !stats_valid);
        end
      end
`endif
    end else begin : g_shared_score
      assign pv_local = 1'b0;
      assign pv_write = 1'b0;
      assign pv_line = '0;
      assign pv_tm = '0;
      assign local_score_data = '0;
      assign rcv_g = rg2_q;
      assign rcv_l = rl2_q;
      assign score_job = q_j;
      assign score_valid = q_jv;
      assign score_ready = aj_ready;
      wire unused_local = &{1'b0, local_score_req};
    end
    assign aj_valid = at_pv_q ? p_jv : score_valid;
    assign aj       = at_pv_q ? p_j : score_job;
    assign q_jr     = score_ready & ~at_pv_q;
    assign p_jr     = aj_ready & at_pv_q;
    assign att_stall = (at_qk_q & f_beat_valid & act_ready_q & ~qk_gate_q) | (at_pv_q & p_stall);

    // the op is finished here: ATTQK after the last job (or at once without groups), ATTPV at done
    wire qk_last = q_jv & q_jr & q_j[JW-1];
    assign att_fin = att_act_q & ((at_qk_q & ((ng_q == 13'd0 && as_q[4]) | qk_last)) | (at_pv_q & p_done));

    wire unused_att = &{1'b0, q_busy, p_busy, le_q[16:13], lf_q[16:13], ac_q.r, ac_q.c.u0, ac_q.c.x,
                        ac_q.c.aux1, ac_q.q0, ac_q.qn, ac_q.mn, it_q, r0_q, ac_q.tact};
  end else begin : g_noatt
    assign at_qk_q     = 1'b0;
    assign at_pv_q     = 1'b0;
    assign pv_local = 1'b0;
    assign pv_write = 1'b0;
    assign pv_line = '0;
    assign pv_tm = '0;
    assign local_score_data = '0;
    wire unused_local = &{1'b0, local_score_req};
    assign att_act_q   = 1'b0;
    assign att_start_q = 1'b0;
    assign acfg_q      = '0;
    assign tbase_q     = '0;
    assign afc_v       = 1'b0;
    assign afc         = '0;
    assign afc_rbeats  = '0;
    assign afc_stride  = '0;
    assign att_tm      = '0;
    assign att_own     = 1'b0;
    assign att_line    = bc_line;
    assign qk_gate_q   = 1'b0;
    assign pv_ready    = 1'b0;
    assign pv_rd       = '0;
    assign pv_mac_v    = 1'b0;
    assign pv_mac_first = 1'b0;
    assign pv_mac_q    = '0;
    assign pv_mac_a    = '0;
    assign aj_valid    = 1'b0;
    assign aj          = '0;
    assign att_stall   = 1'b0;
    wire unused_att = &{1'b0, bc_acmd_valid, bc_acmd, aj_ready, seg_valid, seg_y, pv_acc};
  end

  assign idle = ~busy_q & ~fc_v_q & f_idle & b_idle & ~att_act_q;

  wire unused_ok = &{1'b0, mul3_q[47:TSR_MVM_BEATS_W], left_q[35:32]};

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      assert (!(bc_cmd_valid && bc_line_valid)) else $error("%m: bc_cmd and bc_line in one cycle");
      if (bc_cmd_valid) begin
        assert (!busy_q && su_q == '0 && !att_act_q) else $error("%m: bc_cmd while an MVM is in progress");
        assert (bc_cmd.n != 32'd0 && bc_cmd.rows_per_ch != 32'd0) else $error("%m: bc_cmd with N = 0");
        assert (32'(bc_cmd.row_beats) == tsr_row_beats(bc_cmd.fmt, 32'(bc_cmd.k)))
          else $error("%m: row_beats %0d != ROW_BEATS", bc_cmd.row_beats);
        assert (32'(bc_cmd.d_lines) == tsr_qa_d_lines(bc_cmd.fmt, 32'(bc_cmd.k)))
          else $error("%m: d_lines %0d != QA D lines", bc_cmd.d_lines);
      end
      if (bc_acmd_valid) begin
        assert (HAS_ATT && !busy_q && !att_act_q) else $error("%m: attention command while busy (or without HAS_ATT)");
      end
      if (su_q[4]) begin
        assert (rows_q == tsr_mvm_rows_c(n_q, r_q, 64'(CH) * 64'(r_q)))
          else $error("%m: rows_c %0d", rows_q);
      end
      if (fc_fire && !fc_att_q) begin
        assert (64'(fc.beats) == 64'(rows_q) * 64'(rb_q)) else $error("%m: beats != rows_c * row_beats");
      end
      if (res_valid) begin
        assert (busy_q && has_rows_q && left_rows_q != 32'd0 && su_q == '0)
          else $error("%m: result without a pending row");
      end
      if (f_beat_valid && f_beat_ready && !at_pv_q) begin
        assert (act_ready_q) else $error("%m: beat passed before the activations");
      end
      if (afc_v) assert (!fc_v_q) else $error("%m: attention front command while one is pending");
    end
  end
`endif

endmodule
