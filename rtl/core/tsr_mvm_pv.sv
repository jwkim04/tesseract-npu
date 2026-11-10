// tsr_mvm_pv.sv — the ATTPV datapath of one MVM cluster (AT §6.4, §6.5, §8.2): P·V with
// block-floating-point weights per 64-cell channel segment, accumulated exactly in the MAC lanes'
// PV mode, plus the row sums (LSUM).
//
// Oracle: tsr_ref_attpv (sw/sim/tsr_ref_at.c pv_segment / lsum_segment)
// through the simulator's ATTPV; tb/at compares every partial and row-sum word.
//
// Input: the front end's tagged beats of the V plane, slice by slice (s = 0 .. E/64 - 1), per slice the
// channel's local groups lambda = 0 .. ng - 1 in order, per group the 16 cells j (one quant beat each:
// d_a / d_b = fp16 -> f32 of the scales of blocks 2s, 2s + 1 of cell j, w[0..63] = its 64 V bytes; the
// meta beats stay in the front end). Segments are runs of 4 local groups (64 cells; the last may be
// shorter); slice s belongs to head group g = s >> dcode, its rows are r(q(t), g, i(t)) for the
// active slots t < tact. p of row (g, t) for group lambda is the activation-buffer line g ng + lambda
// of slot t (the cluster writes the p broadcast there); cell j is word j.
//
// Numerics (bit for bit, AT §6.4 / §6.5), per slot t and segment S:
//   w_{j,e} = fmul(p_j, d_{j,e}) for cells with DAZ(p_j) != +-0, else the cell is skipped (w = 0);
//   nf_e = some w_{j,e} non-finite; E_S = max exponent of the finite nonzero w of both blocks; any;
//   W = bf_int(w, E_S) (25 bits); A_L = sum_j W_{j, L/32} q_j[L] (exact, the MAC lanes); c_L = NaN if
//   nf_{L/32}, +0 if !any, else F(A_L, E_S - 23); O[L] = fadd(O[L], c_L) from +0 per slice.
//   LSUM (first slice of each head group, LSUM set): E_p over DAZ(p) of the segment's cells, P_j =
//   bf_int(DAZ(p_j), E_p), cl = NaN / +0 / F(sum P, E_p - 23), l = fadd(l, cl) from +0.
//
// Pipeline:
//   pass 1 (one cell per cycle, beat_ready a flop): activation read of the p line (pv_rd -> the back
//     end's address registers) -> p of every slot -> 2 tsr_fp_mul per slot -> exponent / flag
//     trackers (through a registered per-cell summary) and the delay FIFO (q bytes written at the
//     accept, w0 / w1 / DAZ(p) of every slot MUL_LAT + ACT_RD_LAT + 2 cycles later at the same
//     index); the segment's info (E_S, flags, E_p ...) into the segment FIFO at its last cell. A beat waits for its p line (p_rcv counters) and for
//     room in the delay FIFO (QD cells).
//   pass 2 (one cell per cycle once the segment's info is known): FIFO read -> two aligner stages
//     (bf_int of w0, w1 and DAZ(p) per slot) -> pv_mac_* (the back end registers them into the MAC
//     lanes; PV_LAT later the lanes' accumulators hold the cell) and the LSUM integer sums. Segments
//     follow each other without a gap (the next segment's first cell restarts the accumulators one
//     cycle after the previous one's capture); only a segment's last cell waits until the capture
//     bank is free at its capture (no earlier segment still to be captured, the bank empty or its
//     drain ending in time).
//   capture: (last cell) + 1 + PV_LAT the 64 x T_MAX accumulators, the segment info and the LSUM
//     sums move to the capture bank.
//   drain: 8 convert / add pipes, pipe k = lanes 8 l + k (l < 8) of every active slot (8 tact cycles per
//     segment): F() in 3 stages, fadd with O (LUTRAM per pipe), O written back; one LSUM
//     converter + adder per segment (tact steps); a slice's last segment, once drained, emits its
//     4 tact O lines (dst + CH cstride + r D/16 + 4 (s mod D/64) + m); a head group's first slice
//     stores its l values for the row-sum emitter. The O store holds two slices (slice parity): the
//     next slice drains while the previous one is output; a drain waits only for the output of the
//     slice two before it.
//   end: with LSUM, tsr_mvm_rvec writes the channel's l vector (dst + C cstride + CH ceil(R/16) + r/16,
//     word r mod 16; byte enables). ng = 0: every slice's O lines and the l vector are +0.
// Cost (RTL structure, T_MAX 8): 16 tsr_fp_mul, 24 aligners, delay FIFO 128 x (512 + 768) bits
//   (RAMB36), capture bank 512 x 38 + segment info (~20 K FF), 8 + 1 F() converters, 9 tsr_fp_add,
//   O store 2 x 512 x 32 (LUTRAM), l table 128 x 32. Vivado OOC synthesis (xcu200, MAC_LAT 9):
//   21.3 K LUT, 24.6 K FF, 2.3 K LUTRAM, 20 RAMB36, 34 DSP.

module tsr_mvm_pv
  import tsr_pkg::*;
  import tsr_att_pkg::*;
#(
  parameter  int unsigned K_MAX      = TSR_DEF_K_MAX,
  parameter  int unsigned T_MAX      = TSR_DEF_T_MAX,
  parameter  int unsigned SPM_AW     = 16,
  parameter  int unsigned ACT_RD_LAT = TSR_DEF_ACT_RD_LAT,
  parameter  int unsigned MAC_LAT    = TSR_DEF_MAC_LAT,
  parameter  int unsigned MUL_LAT    = TSR_DEF_FMUL_LAT,
  parameter  int unsigned ADD_LAT    = TSR_DEF_FADD_LAT,
  parameter  bit          FAST_CLK   = 1'b0,    // the row-vector emitter in three stages (tsr_mvm_rvec PIPE)
  localparam int unsigned NS         = (T_MAX + 1) / 2,
  localparam int unsigned JW         = 1 + SPM_AW + 16 + 16 + TSR_LINE_BITS
) (
  input  logic                         clk,
  input  logic                         rst_n,

  // ---- op setup (stable from start to done) ----
  input  logic                         start,
  input  logic [3:0]                   tact,
  input  logic [3:0]                   tq,
  input  logic [3:0]                   gs,
  input  logic [8:0]                   e64,          // slices
  input  logic [1:0]                   dcode,        // slices per head group = 2^dcode
  input  logic [8:0]                   nseg,
  input  logic [12:0]                  ng,           // local groups of this channel
  input  logic [12:0]                  ngl,          // lines per head group in the slot buffer (= ng)
  input  logic                         lsum,
  input  logic [6:0]                   g_heads,
  input  logic [5:0]                   hg0,
  input  logic [14:0]                  n_head,
  input  logic [T_MAX-1:0][SPM_AW-1:0] ob0,          // dst + CH cstride + r(q(t), 0, i(t)) D/16
  input  logic [SPM_AW-1:0]            og,           // G D/16: next head group
  input  logic [SPM_AW-1:0]            lbase,        // dst + C cstride + CH ceil(R/16)
  input  logic [12:0]                  p_g,          // p lines received: head groups complete
  input  logic [12:0]                  p_l,          // and groups of head group p_g
  output logic                         done,         // pulse: every job of the op taken

  // ---- beats (front end) ----
  input  logic                         beat_valid,
  output logic                         beat_ready,
  input  tsr_mvm_beat_t                beat,

  // ---- activation buffer (through the back end) ----
  output tsr_att_rd_t                  rd,
  input  logic [T_MAX-1:0][1023:0]     qs_data,
  input  logic [T_MAX-1:0][511:0]      dbs_data,

  // ---- MAC lanes (through the back end) ----
  output logic                         mac_v,
  output logic                         mac_first,
  output logic [63:0][7:0]             mac_q,
  output logic [TSR_MVM_GROUPS-1:0][NS-1:0][24:0] mac_a,
  input  logic [TSR_MVM_GROUPS-1:0][NS-1:0][31:0][37:0] pv_acc,

  // ---- jobs ----
  output logic                         job_valid,
  input  logic                         job_ready,
  output logic [JW-1:0]                job,

  output logic                         stall,        // a beat waits for its p line
  output logic                         busy
);

  localparam int unsigned QS_WORDS = tsr_actbuf_qs_words(K_MAX);
  localparam int unsigned QD       = 128;            // delay FIFO cells
  localparam int unsigned QDW      = 7;
  localparam int unsigned SQ       = 8;              // segment info FIFO
  localparam int unsigned PV_LAT   = tsr_mac_pv_lat(MAC_LAT, TSR_MVM_GROUP_LANES);
  localparam int unsigned FRL      = 3;              // delay FIFO RAM read latency (stage 3: a fabric register before the aligners)
  localparam int unsigned P1       = 1 + ACT_RD_LAT; // p line at the PV unit (cycles after the accept)
  localparam int unsigned P3       = P1 + 1 + MUL_LAT;   // w at the cell summary / FIFO w write
  localparam int unsigned P4       = P3 + 1;             // cell summary at the trackers
  localparam int unsigned CAPD     = 2 + 1 + PV_LAT; // aligner stages + back-end register + MAC
  localparam int unsigned CVX      = FAST_CLK ? 1 : 0;   // FAST_CLK: fs3 in two stages (docs/platform-u200.md §8.10)
  localparam int unsigned CVP      = FAST_CLK ? 1 : 0;   // FAST_CLK: the lane / slot select in a stage of its own
  localparam int unsigned CVL      = 3 + CVX + CVP;  // F() stages
  localparam logic [31:0] F_NAN    = TSR_CANON_NAN;   // (not NAN: a C macro in the Verilator model)
  localparam int unsigned TW       = (T_MAX > 1) ? $clog2(T_MAX) : 1;   // slot index into T_MAX arrays

  if (T_MAX < 1 || T_MAX > 8) begin : g_chk
    $fatal(1, "tsr_mvm_pv: T_MAX in [1, 8]");
  end

  // ===========================================================================================
  // Pass 1
  // ===========================================================================================
  logic [3:0]  j_q;            // cell in the group
  logic [12:0] l_q;            // local group
  logic [8:0]  s_q;            // slice
  logic [12:0] a_q;            // g ng (slot-buffer line of group 0 of the head group)
  logic        first_cell_q;   // the next accepted cell starts a segment
  logic        ready_q;
  logic [7:0]  fq_n_q;         // delay FIFO cells: held + in flight (credit)
  logic [QDW-1:0] fq_wp_q;     // write index of the next accepted cell

  // Position registers with their precomputed neighbors (no adder or variable shift between the
  // counters and ready_q): l + 1, g + 1, ng - 1, the slice within its head group, last-slice flags.
  logic [12:0] lq1_q;          // l_q + 1
  logic [12:0] g_q, gq1_q;     // head group of the next cell (s >> dcode), + 1
  logic [12:0] ngm1_q;         // ng - 1
  logic [2:0]  slc_q;          // slice within its head group (s mod D/64)
  logic        hgl_q;          // slc_q is the head group's last slice
  logic        slast_q;        // s_q = E/64 - 1
  logic        stall_q;
  wire [2:0]   nslm1 = 3'((4'd1 << dcode) - 4'd1);               // D/64 - 1

  wire acc   = beat_valid & ready_q;
  wire c_lst = (j_q == 4'd15);                                  // last cell of the group
  wire g_lst = (l_q == ngm1_q);                                  // last group of the slice
  wire seg_e = c_lst & ((l_q[1:0] == 2'd3) | g_lst);             // segment end
  wire sl_e  = c_lst & g_lst;                                    // slice end
  wire [2:0]  sl_of = slc_q;                                     // slice within the head group

  // p line of the next position available: A(g, l) = (g, l) received; the four candidates are
  // compared in parallel and selected by the move of this cycle
  function automatic logic avail_at(logic [12:0] g, logic [12:0] l, logic [12:0] pg, logic [12:0] pl);
    return (g < pg) | ((g == pg) & (l < pl));
  endfunction
  wire av_same = avail_at(g_q, l_q, p_g, p_l);                   // no move / next cell of the group
  wire av_grp  = avail_at(g_q, lq1_q, p_g, p_l);                 // next group of the slice
  wire av_slc  = avail_at(g_q, 13'd0, p_g, p_l);                 // next slice, same head group
  wire av_hg   = avail_at(gq1_q, 13'd0, p_g, p_l);               // next head group
  wire avail_n = ~(acc & c_lst) ? av_same : ~g_lst ? av_grp : ~hgl_q ? av_slc : av_hg;
  logic rd2_go;                 // pass 2 reads a cell (frees one FIFO entry)
  wire [7:0] fq_n_n = fq_n_q + 8'(acc) - 8'(rd2_go);
  logic act_q;                  // the op's beats are not all in yet

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      j_q          <= '0;
      l_q          <= '0;
      lq1_q        <= 13'd1;
      s_q          <= '0;
      g_q          <= '0;
      gq1_q        <= 13'd1;
      slc_q        <= '0;
      hgl_q        <= 1'b0;
      slast_q      <= 1'b0;
      ngm1_q       <= '0;
      a_q          <= '0;
      first_cell_q <= 1'b1;
      ready_q      <= 1'b0;
      stall_q      <= 1'b0;
      fq_n_q       <= '0;
      fq_wp_q      <= '0;
      act_q        <= 1'b0;
    end else if (start) begin
      j_q          <= '0;
      l_q          <= '0;
      lq1_q        <= 13'd1;
      s_q          <= '0;
      g_q          <= '0;
      gq1_q        <= 13'd1;
      slc_q        <= '0;
      hgl_q        <= (dcode == 2'd0);
      slast_q      <= (e64 == 9'd1);
      ngm1_q       <= ng - 13'd1;
      a_q          <= '0;
      fq_wp_q      <= '0;            // with the pass-2 read index (reset at start as well)
      first_cell_q <= 1'b1;
      ready_q      <= 1'b0;
      stall_q      <= 1'b0;
      act_q        <= (ng != 13'd0);
    end else begin
      fq_n_q  <= fq_n_n;
      stall_q <= act_q & beat_valid & ~ready_q & ~av_same;
      if (acc) begin
        j_q          <= j_q + 4'd1;
        fq_wp_q      <= fq_wp_q + 1'b1;
        first_cell_q <= seg_e;
        if (c_lst) begin
          if (g_lst) begin                                   // slice end
            l_q     <= '0;
            lq1_q   <= 13'd1;
            s_q     <= s_q + 9'd1;
            slast_q <= (s_q + 9'd2 == e64);
            if (slast_q) act_q <= 1'b0;
            if (hgl_q) begin                                 // head group end
              slc_q <= '0;
              hgl_q <= (nslm1 == 3'd0);
              g_q   <= gq1_q;
              gq1_q <= gq1_q + 13'd1;
              a_q   <= a_q + ngl;
            end else begin
              slc_q <= slc_q + 3'd1;
              hgl_q <= (slc_q + 3'd1 == nslm1);
            end
          end else begin
            l_q   <= lq1_q;
            lq1_q <= lq1_q + 13'd1;
          end
        end
      end
      // FIFO credit from flops only (fq_n_q + 1 < QD - 2: one entry more conservative than fq_n_n)
      ready_q <= act_q & ~(acc & sl_e & slast_q) & avail_n & (fq_n_q < 8'(QD - 3));
    end
  end
  assign beat_ready = ready_q;
  assign stall      = stall_q;                // registered (PERF ATT_PATH_STALL, one cycle late)

  // activation read of the p line: slot-buffer line a = g ng + lambda (QS line a < 2 QS_WORDS, else
  // DBS line a - 2 QS_WORDS)
  wire [13:0] pa = 14'(a_q) + 14'(l_q);
  wire        pa_dbs = (32'(pa) >= 2 * QS_WORDS);
  assign rd = '{v: acc, dbs: pa_dbs, word: pa_dbs ? 10'd0 : 10'(pa >> 1),
                line: pa_dbs ? 11'(32'(pa) - 2 * QS_WORDS) : 11'd0};

  // tag pipeline of pass 1
  typedef struct packed {
    logic [3:0]  j;
    logic        dbs;
    logic        bank;          // QS: line a odd
    logic        first;         // first cell of a segment
    logic        se;            // last cell of a segment
    logic        sfirst;        // segment is the first of its slice
    logic        slast;         // segment is the last of its slice
    logic        lsum_en;       // LSUM for this slice (first slice of its head group)
    logic [8:0]  s;
    logic [QDW-1:0] idx;        // delay FIFO index
    logic [31:0] da, db;
  } t1_t;
  t1_t  t1 [P4+1];
  logic [P4:1] v1;
  logic first_seg_of_slice_q;   // the next segment starts its slice

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      v1 <= '0;
      first_seg_of_slice_q <= 1'b1;
    end else begin
      v1 <= {v1[P4-1:1], acc};
      if (acc && seg_e) first_seg_of_slice_q <= sl_e;
    end
  end
  always_ff @(posedge clk) begin
    t1[1] <= '{j: j_q, dbs: pa_dbs, bank: pa[0], first: first_cell_q, se: seg_e,
               sfirst: first_seg_of_slice_q, slast: sl_e, lsum_en: lsum & (sl_of == 3'd0), s: s_q,
               idx: fq_wp_q, da: beat.d_a, db: beat.d_b};
    for (int i = 2; i <= int'(P4); i++) t1[i] <= t1[i-1];
  end

  // q bytes into the delay FIFO at the accept
  logic [511:0] fq_q_rd;
  logic [QDW-1:0] fq_rp_q;
  logic [511:0] beat_q;
  always_comb for (int l = 0; l < 64; l++) beat_q[8*l +: 8] = beat.w[l];
  tsr_sram_1r1w #(.DEPTH(QD), .WIDTH(512), .RD_LAT(FRL)) u_fq (
    .clk, .we(acc), .waddr(fq_wp_q), .wdata(beat_q), .wbe({64{1'b1}}),
    .re(rd2_go), .raddr(fq_rp_q), .rdata(fq_q_rd)
  );

  // p of every slot at P1 -> registers (stage P1 + 1 = the fmul inputs)
  logic [T_MAX-1:0][31:0] p_q;
  always_ff @(posedge clk) begin
    for (int t = 0; t < int'(T_MAX); t++) begin
      p_q[t] <= t1[P1].dbs ? dbs_data[t][32*t1[P1].j +: 32]
                           : qs_data[t][512*t1[P1].bank + 32*t1[P1].j +: 32];
    end
  end
  // DAZ(p) and the skip flag travel with the products
  logic [T_MAX-1:0][31:0] pz_d;
  logic [T_MAX-1:0]       sk_d;
  logic [T_MAX-1:0][31:0] pz_p;
  logic [T_MAX-1:0]       sk_p;
  always_comb begin
    for (int t = 0; t < int'(T_MAX); t++) begin
      pz_d[t] = tsr_att_daz(p_q[t]);
      sk_d[t] = (pz_d[t][30:0] == 31'd0) | (32'(t) >= 32'(tact));
    end
  end
  tsr_mvm_dly #(.W(33 * T_MAX), .N(MUL_LAT)) u_dly_pz (.clk, .rst_n, .d({sk_d, pz_d}), .q({sk_p, pz_p}));

  logic [T_MAX-1:0][1:0][31:0] w_p;
  for (genvar t = 0; t < T_MAX; t++) begin : g_wm
    logic v0, v1u;
    tsr_fp_mul #(.LAT(MUL_LAT)) u_m0 (
      .clk, .rst_n, .ce(1'b1), .in_valid(v1[P1+1]), .a(p_q[t]), .b(t1[P1+1].da), .out_valid(v0), .y(w_p[t][0])
    );
    tsr_fp_mul #(.LAT(MUL_LAT)) u_m1 (
      .clk, .rst_n, .ce(1'b1), .in_valid(v1[P1+1]), .a(p_q[t]), .b(t1[P1+1].db), .out_valid(v1u), .y(w_p[t][1])
    );
    wire unused_v = &{1'b0, v0, v1u};
  end

  // skipped cells: w = 0 (not counted, not NaN)
  logic [T_MAX-1:0][1:0][31:0] w_s;
  always_comb begin
    for (int t = 0; t < int'(T_MAX); t++) begin
      for (int e = 0; e < 2; e++) w_s[t][e] = sk_p[t] ? 32'd0 : w_p[t][e];
    end
  end

  // trackers (per slot): exponent max of the finite nonzero w (both blocks) and of DAZ(p), flags
  typedef struct packed {
    logic [T_MAX-1:0][7:0] es;     // biased E_S (valid with any)
    logic [T_MAX-1:0]      any;
    logic [T_MAX-1:0][1:0] nf;     // a non-finite w in block e
    logic [T_MAX-1:0][7:0] ep;     // biased E_p
    logic [T_MAX-1:0]      anyp;
    logic [T_MAX-1:0]      nfp;
    logic [7:0]            ncell;  // cells of the segment
    logic                  sfirst, slast, lsum_en;
    logic [8:0]            s;
  } seg_t;
  seg_t tr_q;
  seg_t tr_n;
  // cell summary (P3 -> register -> P4): per slot the larger exponent of the finite nonzero w of the
  // two blocks (0: none), the non-finite flags, and DAZ(p)'s exponent (0: zero or non-finite)
  logic [T_MAX-1:0][7:0] cs_es_q, cs_ep_q;
  logic [T_MAX-1:0]      cs_any_q, cs_anyp_q, cs_nfp_q;
  logic [T_MAX-1:0][1:0] cs_nf_q;
  always_ff @(posedge clk) begin
    for (int t = 0; t < int'(T_MAX); t++) begin
      automatic logic [7:0] x0 = w_s[t][0][30:23], x1 = w_s[t][1][30:23], xp = pz_p[t][30:23];
      automatic logic [7:0] f0 = (x0 == 8'hFF) ? 8'd0 : x0, f1 = (x1 == 8'hFF) ? 8'd0 : x1;
      cs_es_q[t]   <= (f1 > f0) ? f1 : f0;
      cs_any_q[t]  <= (f0 != 8'd0) | (f1 != 8'd0);
      cs_nf_q[t]   <= {x1 == 8'hFF, x0 == 8'hFF};
      cs_ep_q[t]   <= (xp == 8'hFF) ? 8'd0 : xp;
      cs_anyp_q[t] <= (xp != 8'hFF) & (xp != 8'd0);
      cs_nfp_q[t]  <= (xp == 8'hFF);
    end
  end
  // trackers (P4): the segment's maxima and flags, restarted at its first cell
  always_comb begin
    tr_n = tr_q;
    if (t1[P4].first) begin
      tr_n.es = '0; tr_n.any = '0; tr_n.nf = '0; tr_n.ep = '0; tr_n.anyp = '0; tr_n.nfp = '0;
      tr_n.ncell = '0;
    end
    tr_n.ncell = tr_n.ncell + 8'd1;
    tr_n.sfirst  = t1[P4].sfirst;
    tr_n.slast   = t1[P4].slast;
    tr_n.lsum_en = t1[P4].lsum_en;
    tr_n.s       = t1[P4].s;
    for (int t = 0; t < int'(T_MAX); t++) begin
      tr_n.nf[t]   = tr_n.nf[t] | cs_nf_q[t];
      tr_n.any[t]  = tr_n.any[t] | cs_any_q[t];
      if (cs_es_q[t] > tr_n.es[t]) tr_n.es[t] = cs_es_q[t];
      // LSUM: every cell (a zero p adds nothing and has no exponent)
      tr_n.nfp[t]  = tr_n.nfp[t] | cs_nfp_q[t];
      tr_n.anyp[t] = tr_n.anyp[t] | cs_anyp_q[t];
      if (cs_ep_q[t] > tr_n.ep[t]) tr_n.ep[t] = cs_ep_q[t];
    end
  end
  always_ff @(posedge clk) begin
    if (v1[P4]) tr_q <= tr_n;
  end

  // w0, w1, DAZ(p) of every slot into the delay FIFO's w RAM at the cell's index
  localparam int unsigned WW = T_MAX * 96;
  logic [WW-1:0] fw_wd, fw_rd;
  always_comb begin
    for (int t = 0; t < int'(T_MAX); t++) fw_wd[96*t +: 96] = {pz_p[t], w_s[t][1], w_s[t][0]};
  end
  tsr_sram_1r1w #(.DEPTH(QD), .WIDTH(WW), .RD_LAT(FRL)) u_fw (
    .clk, .we(v1[P3]), .waddr(t1[P3].idx), .wdata(fw_wd), .wbe({(WW/8){1'b1}}),
    .re(rd2_go), .raddr(fq_rp_q), .rdata(fw_rd)
  );

  // segment info FIFO (pushed at a segment's last cell, popped at its capture)
  seg_t        sq_mem [SQ];
  logic [2:0]  sq_wp_q, sq_rp_q;
  logic [3:0]  sq_n_q;
  wire         sq_push = v1[P4] & t1[P4].se;
  logic        sq_pop;
  always_ff @(posedge clk) begin
    if (sq_push) sq_mem[sq_wp_q] <= tr_n;
  end

  // ===========================================================================================
  // Pass 2: FIFO -> aligners -> MAC lanes
  // ===========================================================================================
  logic        p2_act_q;        // a segment is being read
  logic [7:0]  p2_left_q;       // its cells still to read
  logic        p2_first_q;      // the next read is the segment's first cell
  logic [2:0]  p2_seg_q;        // segment info entry being read (= sq_rp_q + in-flight segments)
  seg_t        p2_info_q;       // its info
  logic        fin_ok;          // the segment's last cell may be read: the bank is free at its capture
  wire         p2_go;           // a cell is read in this cycle
  wire  sq_ne  = (sq_n_q != 4'd0);
  logic [3:0]  p2_started_q;    // segments started in pass 2 and not yet captured
  wire  p2_start = ~p2_act_q & (sq_n_q > p2_started_q);
  assign p2_go  = p2_act_q & ((p2_left_q != 8'd1) | fin_ok);
  assign rd2_go = p2_go;

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      p2_act_q     <= 1'b0;
      p2_left_q    <= '0;
      p2_first_q   <= 1'b0;
      p2_seg_q     <= '0;
      fq_rp_q      <= '0;
    end else begin
      if (p2_start) begin
        p2_act_q   <= 1'b1;
        p2_info_q  <= sq_mem[p2_seg_q];
        p2_left_q  <= sq_mem[p2_seg_q].ncell;
        p2_first_q <= 1'b1;
      end else if (p2_go) begin
        p2_first_q <= 1'b0;
        p2_left_q  <= p2_left_q - 8'd1;
        if (p2_left_q == 8'd1) begin
          p2_act_q <= 1'b0;
          p2_seg_q <= p2_seg_q + 3'd1;
        end
      end
      if (rd2_go) fq_rp_q <= fq_rp_q + 1'b1;
    end
  end

  // pass-2 tags: read cycle R -> data at R + FRL -> A1 -> A2 (mac_*)
  typedef struct packed {
    logic first, last;
  } t2_t;
  logic [FRL:0] v2;
  t2_t  t2 [FRL+1];
  seg_t i2 [FRL+1];             // the segment info travels with the cells (needed at A1)
  always_ff @(posedge clk) begin
    if (!rst_n) v2 <= '0;
    else        v2 <= {v2[FRL-1:0], rd2_go};
  end
  always_ff @(posedge clk) begin
    t2[0] <= '{first: p2_first_q, last: (p2_left_q == 8'd1)};
    i2[0] <= p2_info_q;
    for (int i = 1; i <= int'(FRL); i++) begin
      t2[i] <= t2[i-1];
      i2[i] <= i2[i-1];
    end
  end
  // FRL - 1 = the index of the stage whose data is the RAM output
  wire  d2_v = v2[FRL-1];
  t2_t  d2_t;
  seg_t d2_i;
  assign d2_t = t2[FRL-1];
  assign d2_i = i2[FRL-1];

  // A1: shift amount, magnitude shift (guard / sticky kept); A2: RNE + sign (tsr_att_bf_int split)
  typedef struct packed {
    logic        z;             // zero result
    logic        neg;
    logic [24:0] v;             // m >> sh with one guard bit at [0]
    logic        st;            // sticky below the guard
  } al_t;
  function automatic al_t al1(logic [31:0] w, logic [7:0] es);
    al_t   r;
    logic [8:0]  sh;
    logic [48:0] m;
    r = '0;
    r.neg = w[31];
    sh = 9'(es) - 9'(w[30:23]);
    if (w[30:23] == 8'd0 || sh >= 9'd25) begin
      r.z = 1'b1;
      return r;
    end
    m    = {1'b1, w[22:0], 25'd0} >> sh;        // 24-bit m shifted, 25 fraction bits kept
    r.v  = m[49-1 -: 25];                        // integer part (24 bits) + guard
    r.st = |m[23:0];
    return r;
  endfunction
  function automatic logic [24:0] al2(al_t a);
    logic [23:0] q;
    logic        g;
    if (a.z) return 25'd0;
    q = a.v[24:1];
    g = a.v[0];
    q = q + 24'(g && (a.st || q[0]));
    return a.neg ? -25'({1'b0, q}) : 25'({1'b0, q});
  endfunction

  al_t  [T_MAX-1:0][2:0] a1_r;       // [slot][w0, w1, p]
  logic [511:0]          a1_qb, a2_qb;
  t2_t                   a1_t, a2_t;
  logic                  a1_v, a2_v;
  logic [T_MAX-1:0][2:0][24:0] a2_w;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      a1_v <= 1'b0;
      a2_v <= 1'b0;
    end else begin
      a1_v <= d2_v;
      a2_v <= a1_v;
    end
  end
  always_ff @(posedge clk) begin
    for (int t = 0; t < int'(T_MAX); t++) begin
      a1_r[t][0] <= al1(fw_rd[96*t +: 32], d2_i.es[t]);
      a1_r[t][1] <= al1(fw_rd[96*t + 32 +: 32], d2_i.es[t]);
      a1_r[t][2] <= al1(fw_rd[96*t + 64 +: 32], d2_i.ep[t]);
    end
    a1_qb <= fq_q_rd;
    a1_t  <= d2_t;
    for (int t = 0; t < int'(T_MAX); t++) begin
      for (int k = 0; k < 3; k++) a2_w[t][k] <= al2(a1_r[t][k]);
    end
    a2_qb <= a1_qb;
    a2_t  <= a1_t;
  end

  // MAC operands: slot t of lane group g: g < 2 slot s = t (lanes 32 g ..), g >= 2 slot t = NS + s
  always_comb begin
    mac_v     = a2_v;
    mac_first = a2_t.first;
    for (int l = 0; l < 64; l++) mac_q[l] = a2_qb[8*l +: 8];
    for (int g = 0; g < TSR_MVM_GROUPS; g++) begin
      for (int s = 0; s < int'(NS); s++) begin
        automatic int t = (g < 2) ? s : int'(NS) + s;
        mac_a[g][s] = (t < int'(T_MAX)) ? a2_w[t][g % 2] : 25'd0;
      end
    end
  end

  // LSUM integer sums (per slot), restarted at a segment's first cell; ls_h_q holds a segment's sums
  // from its last cell to its capture (1 + PV_LAT cycles < the 16 cells of the next segment)
  logic [T_MAX-1:0][31:0] ls_q, ls_n, ls_h_q;
  always_comb begin
    for (int t = 0; t < int'(T_MAX); t++) ls_n[t] = (a2_t.first ? 32'd0 : ls_q[t]) + 32'($signed(a2_w[t][2]));
  end
  always_ff @(posedge clk) begin
    if (a2_v) ls_q <= ls_n;
    if (a2_v && a2_t.last) ls_h_q <= ls_n;
  end
  if (PV_LAT + 2 > 16) begin : g_chk_lsh
    $fatal(1, "tsr_mvm_pv: the LSUM hold register needs 1 + PV_LAT < 16 (a segment has >= 16 cells)");
  end

  // ===========================================================================================
  // Capture bank
  // ===========================================================================================
  // capture strobe: the segment's last cell left A2 (mac_v) -> + 1 (back-end register) + PV_LAT
  logic cap_go;
  tsr_mvm_dly #(.W(1), .N(1 + PV_LAT), .RST(1'b1)) u_dly_cap (.clk, .rst_n, .d(a2_v & a2_t.last), .q(cap_go));
  logic [T_MAX-1:0][63:0][37:0] cb_a;   // A_L per slot and lane
  logic [T_MAX-1:0][31:0]       cb_ls;  // LSUM sums
  seg_t        cb_i;
  logic        cb_full_q;               // captured, drain pending / running
  logic        dr_act_q;                // draining
  logic [6:0]  dr_left_q;               // drain cycles left

  always_ff @(posedge clk) begin
    if (cap_go) begin
      for (int g = 0; g < TSR_MVM_GROUPS; g++) begin
        for (int s = 0; s < int'(NS); s++) begin
          automatic int t = (g < 2) ? s : int'(NS) + s;
          if (t < int'(T_MAX)) begin
            for (int k = 0; k < 32; k++) cb_a[t][32*(g % 2) + k] <= pv_acc[g][s][k];
          end
        end
      end
      cb_ls <= ls_h_q;
      cb_i  <= sq_mem[sq_rp_q];
    end
  end
  assign sq_pop = cap_go;

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      sq_wp_q      <= '0;
      sq_rp_q      <= '0;
      sq_n_q       <= '0;
      p2_started_q <= '0;
    end else begin
      if (sq_push) sq_wp_q <= sq_wp_q + 3'd1;
      if (sq_pop)  sq_rp_q <= sq_rp_q + 3'd1;
      sq_n_q       <= sq_n_q + 4'(sq_push) - 4'(sq_pop);
      p2_started_q <= p2_started_q + 4'(p2_start) - 4'(sq_pop);
    end
  end

  // a segment's last cell may be read when the bank is free at its capture (FRL + 3 + PV_LAT cycles
  // later): no earlier segment still to be captured (p2_started_q counts this one), and the bank
  // empty or draining with at most CAPD = 3 + PV_LAT cycles left
  assign fin_ok = (p2_started_q == 4'd1) & (~cb_full_q | (dr_act_q & (8'(dr_left_q) <= 8'(CAPD))));

  // ===========================================================================================
  // Drain: 8 pipes x 8 lanes, (t, l) for t < tact, l < 8
  // ===========================================================================================
  logic [3:0] dr_t_q;
  logic [2:0] dr_l_q;
  // the drain's lane select: pipe k of the step (slot dr_t_q, lane group dr_l_q) reads lane 8 dr_l_q + k
  logic [7:0][37:0] cb_sel;
  logic [7:0]       nf_sel;
  tsr_mvm_pv_lsel #(.T_MAX(T_MAX)) u_lsel (.l(dr_l_q), .t(TW'(dr_t_q)), .a(cb_a), .nf(cb_i.nf), .y(cb_sel), .f(nf_sel));
  // a drain starts once the bank is full, the slice output (O being read) is not running, and the
  // previous drain started >= DGAP cycles ago (its O / l read-modify-writes are written back before
  // the same (t, l) is read again: only tact = 1 can come closer)
  localparam int unsigned DGAP = CVL + ADD_LAT + 3;
  logic [4:0] dr_since_q;
  // o_hold_q[p]: the slice of parity p in the O store is complete and not yet output (set at its last
  // segment's drain start, cleared when its output's last line is taken): a drain of the next slice
  // of that parity (two slices later) waits
  logic [1:0] o_hold_q;
  wire        dr_start = cb_full_q & ~dr_act_q & ~o_hold_q[cb_i.s[0]] & (dr_since_q >= 5'(DGAP));
  logic       dr_last_q;                 // last step of the drain

  // F(A, E_S - 23) / NaN / +0, three stages (tsr_att_bf_f split), then fadd
  typedef struct packed {
    logic        nan, zero, s;
    logic [37:0] mag;
    logic [7:0]  es;
  } f1_t;
  typedef struct packed {
    logic        nan, zero, s;
    logic [37:0] nrm;
    logic [5:0]  msb;
    logic [7:0]  es;
  } f2_t;
  function automatic f1_t fs1(logic [37:0] a, logic [7:0] es, logic nan, logic any);
    f1_t r;
    r.nan  = nan;
    r.zero = ~any | (a == 38'd0);
    r.s    = a[37];
    r.mag  = a[37] ? (~a + 38'd1) : a;
    r.es   = es;
    return r;
  endfunction
  function automatic f2_t fs2(f1_t x);
    f2_t r;
    int  m;
    r.nan = x.nan;
    r.zero = x.zero;
    r.s = x.s;
    r.es = x.es;
    m = 0;
    for (int i = 0; i < 38; i++) if (x.mag[i]) m = i;
    r.msb = 6'(m);
    r.nrm = x.mag << (37 - m);
    return r;
  endfunction
  // fs3 in two halves for FAST_CLK: fs3a rounds the mantissa and forms the exponent before the rounding carry,
  // fs3b adds the carry, range-checks the exponent and packs the result; fs3 = fs3b o fs3a
  typedef struct packed {
    logic        nan, zero, s;
    logic [24:0] mr;
    logic [9:0]  ex;
  } f3_t;
  function automatic f3_t fs3a(f2_t x);
    f3_t         r;
    logic [23:0] m;
    logic        g, st, up;
    m  = x.nrm[37:14];
    g  = x.nrm[13];
    st = |x.nrm[12:0];
    up = g & (st | m[0]);
    r.nan  = x.nan;
    r.zero = x.zero;
    r.s    = x.s;
    r.mr   = {1'b0, m} + 25'(up);
    r.ex   = 10'(int'(x.msb) + int'(x.es) - 23);
    return r;
  endfunction
  function automatic logic [31:0] fs3b(f3_t x);
    int ex;
    if (x.nan) return F_NAN;
    if (x.zero) return 32'd0;
    ex = int'($signed(x.ex)) + (x.mr[24] ? 1 : 0);
    if (ex < 1) return {x.s, 31'd0};
    if (ex > 254) return {x.s, 8'hFF, 23'd0};
    return {x.s, 8'(ex), x.mr[24] ? x.mr[23:1] : x.mr[22:0]};
  endfunction
  function automatic logic [31:0] fs3(f2_t x);
    return fs3b(fs3a(x));
  endfunction

  // O stores: pipe k, lane l, slot t (LUTRAM, async read)
  // O store: pipe k holds lanes L = 8 l + k, entry {slice parity, l, slot} (128 x 32 per pipe, LUTRAM:
  // one write, three asynchronous reads: the drain's O read and the two halves of an output line)
  logic [7:0][31:0]      om_rmw;          // O of (cv_par, cv_l, cv_t) for the drain's add
  logic [7:0][1:0][31:0] om_out;          // output line m of slot ob_t: lanes 16 m + 8 h + k
  logic [3:0]  cv_t [CVL+1];
  logic [2:0]  cv_l [CVL+1];
  logic        cv_v [CVL+1];
  logic        cv_first [CVL+1];
  logic        cv_par [CVL+1];          // slice parity (O store half)
  f1_t [7:0]   c1_q;
  f2_t [7:0]   c2_q;
  logic [7:0][31:0] c3_q, o_old;
  logic [7:0]  add_v;
  logic [7:0][31:0] add_y;

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      dr_act_q  <= 1'b0;
      dr_t_q    <= '0;
      dr_l_q    <= '0;
      dr_left_q <= '0;
      cb_full_q <= 1'b0;
      dr_last_q <= 1'b0;
    end else begin
      dr_last_q <= 1'b0;
      if (cap_go) cb_full_q <= 1'b1;
      if (dr_start) begin
        dr_act_q  <= 1'b1;
        dr_t_q    <= '0;
        dr_l_q    <= '0;
        dr_left_q <= 7'(8 * tact);
      end else if (dr_act_q) begin
        dr_left_q <= dr_left_q - 7'd1;
        if (dr_l_q == 3'd7) begin
          dr_l_q <= '0;
          dr_t_q <= dr_t_q + 4'd1;
        end else begin
          dr_l_q <= dr_l_q + 3'd1;
        end
        if (dr_left_q == 7'd1) begin
          dr_act_q  <= 1'b0;
          cb_full_q <= cap_go;                // a new capture in this cycle stays
          dr_last_q <= 1'b1;
        end
      end
    end
  end

  // the select stage of the F() pipeline: the lane group / slot select of the capture bank (FAST_CLK: registered first)
  logic [7:0][37:0] sel_a;
  logic [7:0][7:0]  sel_es;
  logic [7:0]       sel_nf, sel_any;
  if (CVP == 0) begin : g_sel0
    for (genvar k = 0; k < 8; k++) begin : g_k
      assign sel_a[k]   = cb_sel[k];
      assign sel_es[k]  = cb_i.es[TW'(dr_t_q)];
      assign sel_nf[k]  = nf_sel[k];
      assign sel_any[k] = cb_i.any[TW'(dr_t_q)];
    end
  end else begin : g_sel1
    always_ff @(posedge clk) begin
      for (int k = 0; k < 8; k++) begin
        sel_a[k]   <= cb_sel[k];
        sel_es[k]  <= cb_i.es[TW'(dr_t_q)];
        sel_nf[k]  <= nf_sel[k];
        sel_any[k] <= cb_i.any[TW'(dr_t_q)];
      end
    end
  end

  // convert pipeline (one step per cycle for all 8 pipes)
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int i = 0; i <= int'(CVL); i++) cv_v[i] <= 1'b0;
    end else begin
      cv_v[0] <= dr_act_q;
      for (int i = 1; i <= int'(CVL); i++) cv_v[i] <= cv_v[i-1];
    end
  end
  always_ff @(posedge clk) begin
    cv_t[0] <= dr_t_q;
    cv_l[0] <= dr_l_q;
    cv_first[0] <= cb_i.sfirst;
    cv_par[0]   <= cb_i.s[0];
    for (int i = 1; i <= int'(CVL); i++) begin
      cv_t[i] <= cv_t[i-1];
      cv_l[i] <= cv_l[i-1];
      cv_first[i] <= cv_first[i-1];
      cv_par[i]   <= cv_par[i-1];
    end
    // stage 1 c1_q, stage 2 c2_q, stage 3 c3_q and o_old (from the stage-2 tags cv_*[1]); the
    // add issues in stage 3 (cv_*[2])
    for (int k = 0; k < 8; k++) begin
      c1_q[k] <= fs1(sel_a[k], sel_es[k], sel_nf[k], sel_any[k]);
      c2_q[k] <= fs2(c1_q[k]);
      o_old[k] <= cv_first[CVL-2] ? 32'd0 : om_rmw[k];
    end
  end
  if (CVX == 0) begin : g_c3
    always_ff @(posedge clk) begin
      for (int k = 0; k < 8; k++) c3_q[k] <= fs3(c2_q[k]);
    end
  end else begin : g_c3x
    f3_t [7:0] c3a_q;
    always_ff @(posedge clk) begin
      for (int k = 0; k < 8; k++) begin
        c3a_q[k] <= fs3a(c2_q[k]);
        c3_q[k]  <= fs3b(c3a_q[k]);
      end
    end
  end
  for (genvar k = 0; k < 8; k++) begin : g_cvadd
    tsr_fp_add #(.LAT(ADD_LAT)) u_add (
      .clk, .rst_n, .ce(1'b1), .in_valid(cv_v[CVL-1]), .a(o_old[k]), .b(c3_q[k]), .sub(1'b0),
      .out_valid(add_v[k]), .y(add_y[k])
    );
  end
  logic [3:0] aw_t;
  logic [2:0] aw_l;
  logic       aw_p;
  logic       aw_v;
  tsr_mvm_dly #(.W(8), .N(ADD_LAT)) u_dly_aw (.clk, .rst_n, .d({cv_par[CVL-1], cv_t[CVL-1], cv_l[CVL-1]}),
                                              .q({aw_p, aw_t, aw_l}));
  tsr_mvm_dly #(.W(1), .N(ADD_LAT), .RST(1'b1)) u_dly_awv (.clk, .rst_n, .d(cv_v[CVL-1]), .q(aw_v));
  logic [1:0] ob_m;                     // (the output's line m, declared with the output below)
  logic       ob_p;
  logic [2:0] ob_t3;
  for (genvar k = 0; k < 8; k++) begin : g_om
    logic [31:0] om [128];
    always_ff @(posedge clk) begin
      if (aw_v) om[{aw_p, aw_l, 3'(aw_t)}] <= add_y[k];
    end
    assign om_rmw[k]    = om[{cv_par[CVL-2], cv_l[CVL-2], 3'(cv_t[CVL-2])}];
    assign om_out[k][0] = om[{ob_p, ob_m, 1'b0, ob_t3}];
    assign om_out[k][1] = om[{ob_p, ob_m, 1'b1, ob_t3}];
  end

  // LSUM: one converter + adder, slot by slot during the drain
  logic [3:0]  lq_t_q;
  logic        lq_act_q;
  f1_t         l1_q;
  f2_t         l2_q;
  logic [31:0] l3_q, l_old;
  logic [3:0]  lt [CVL+1];
  logic        lv [CVL+1];
  logic [T_MAX-1:0][31:0] l_cur;        // running l of the head group's first slice, per slot
  logic        l_add_v;
  logic [31:0] l_add_y;
  logic [3:0]  l_add_t;
  wire  l_go = lq_act_q;
  logic [37:0] lsel_a;
  logic [7:0]  lsel_es;
  logic        lsel_nf, lsel_any;
  if (CVP == 0) begin : g_lsel0
    assign lsel_a   = 38'($signed(cb_ls[TW'(lq_t_q)]));
    assign lsel_es  = cb_i.ep[TW'(lq_t_q)];
    assign lsel_nf  = cb_i.nfp[TW'(lq_t_q)];
    assign lsel_any = cb_i.anyp[TW'(lq_t_q)];
  end else begin : g_lsel1
    always_ff @(posedge clk) begin
      lsel_a   <= 38'($signed(cb_ls[TW'(lq_t_q)]));
      lsel_es  <= cb_i.ep[TW'(lq_t_q)];
      lsel_nf  <= cb_i.nfp[TW'(lq_t_q)];
      lsel_any <= cb_i.anyp[TW'(lq_t_q)];
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      lq_act_q <= 1'b0;
      lq_t_q   <= '0;
      for (int i = 0; i <= int'(CVL); i++) lv[i] <= 1'b0;
    end else begin
      if (dr_start && cb_i.lsum_en) begin
        lq_act_q <= 1'b1;
        lq_t_q   <= '0;
      end else if (lq_act_q) begin
        lq_t_q <= lq_t_q + 4'd1;
        if (lq_t_q == tact - 4'd1) lq_act_q <= 1'b0;
      end
      lv[0] <= l_go;
      for (int i = 1; i <= int'(CVL); i++) lv[i] <= lv[i-1];
    end
  end
  always_ff @(posedge clk) begin
    lt[0] <= lq_t_q;
    for (int i = 1; i <= int'(CVL); i++) lt[i] <= lt[i-1];
    l1_q  <= fs1(lsel_a, lsel_es, lsel_nf, lsel_any);
    l2_q  <= fs2(l1_q);
    l_old <= cb_i.sfirst ? 32'd0 : l_cur[TW'(lt[CVL-2])];
  end
  if (CVX == 0) begin : g_l3
    always_ff @(posedge clk) l3_q <= fs3(l2_q);
  end else begin : g_l3x
    f3_t l3a_q;
    always_ff @(posedge clk) begin
      l3a_q <= fs3a(l2_q);
      l3_q  <= fs3b(l3a_q);
    end
  end
  tsr_fp_add #(.LAT(ADD_LAT)) u_ladd (
    .clk, .rst_n, .ce(1'b1), .in_valid(lv[CVL-1]), .a(l_old), .b(l3_q), .sub(1'b0),
    .out_valid(l_add_v), .y(l_add_y)
  );
  tsr_mvm_dly #(.W(4), .N(ADD_LAT)) u_dly_lt (.clk, .rst_n, .d(lt[CVL-1]), .q(l_add_t));
  always_ff @(posedge clk) begin
    if (l_add_v) l_cur[TW'(l_add_t)] <= l_add_y;
  end

  // the drained segment is complete (O writes and LSUM adds in) DTAIL cycles after its last step
  localparam int unsigned DTAIL = CVL + ADD_LAT + 3;
  typedef struct packed {
    logic       slast, lsum_en;
    logic [8:0] s;
  } dn_t;
  dn_t  dn_d, done_i;
  logic dr_last_d, seg_done;
  always_ff @(posedge clk) begin
    if (dr_start) dn_d <= '{slast: cb_i.slast, lsum_en: cb_i.lsum_en, s: cb_i.s};
  end
  tsr_mvm_dly #(.W(1), .N(DTAIL), .RST(1'b1)) u_dly_sd (.clk, .rst_n, .d(dr_last_q), .q(dr_last_d));
  tsr_mvm_dly #(.W($bits(dn_t)), .N(DTAIL)) u_dly_si (.clk, .rst_n, .d(dn_d), .q(done_i));
  assign seg_done = dr_last_d & ~start;
  always_ff @(posedge clk) begin
    if (!rst_n || start) dr_since_q <= 5'(DGAP);
    else if (dr_start)   dr_since_q <= '0;
    else if (dr_since_q != 5'h1F) dr_since_q <= dr_since_q + 5'd1;
  end

  // l table (row-sum emitter), e = g tact + t, written at the end of a head group's first slice
  logic [31:0] lt_mem [128];
  logic [127:0] lt_vld_q;
  logic [6:0]  lt_wa_q;
  logic [3:0]  lt_wt_q;
  logic        lt_wr_q;
  logic [7:0]  lt_gbase;
  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      lt_wr_q  <= 1'b0;
      lt_vld_q <= '0;
    end else begin
      if (seg_done && done_i.lsum_en && done_i.slast) begin
        lt_wr_q <= 1'b1;
        lt_wt_q <= '0;
        lt_wa_q <= 7'(lt_gbase);
      end else if (lt_wr_q) begin
        lt_vld_q[lt_wa_q] <= 1'b1;
        lt_wa_q <= lt_wa_q + 7'd1;
        lt_wt_q <= lt_wt_q + 4'd1;
        if (lt_wt_q == tact - 4'd1) lt_wr_q <= 1'b0;
      end
    end
  end
  always_ff @(posedge clk) begin
    if (lt_wr_q) lt_mem[lt_wa_q] <= l_cur[TW'(lt_wt_q)];
  end
  assign lt_gbase = 8'(32'(done_i.s >> dcode) * 32'(tact));   // e of row (g, t = 0)

  // ===========================================================================================
  // O output: a slice's last segment completed (or, with ng = 0, every slice in turn): 4 tact jobs;
  // after the last slice, with LSUM, the row-sum lines (tsr_mvm_rvec); done when the last job left
  // ===========================================================================================
  typedef enum logic [2:0] { O_WAIT, O_RUN, O_LS, O_END } ost_t;
  ost_t        o_st_q;
  logic [8:0]  ob_s_q;            // slice being output
  logic [3:0]  ob_t_q;
  logic [1:0]  ob_m_q;
  logic        ob_zero_q;         // ng = 0: zero lines
  logic [SPM_AW-1:0] ob_gofs_q;  // g og of the slice
  logic        jv_q;
  logic [JW-1:0] j_q2;
  logic        rv_job_v, rv_done;
  logic [JW-1:0] rv_job;
  logic        rv_start_q;
  logic [6:0]  rv_addr;
  logic        done_q;

  wire j_free  = ~jv_q | job_ready;
  wire ob_step = (o_st_q == O_RUN) & j_free;
  wire [SPM_AW-1:0] ob_line = ob0[TW'(ob_t_q)] + ob_gofs_q
                            + SPM_AW'({(ob_s_q & ((9'd1 << dcode) - 9'd1)), 2'b00}) + SPM_AW'(ob_m_q);
  logic [TSR_LINE_BITS-1:0] ob_data;
  assign ob_m  = ob_m_q;
  assign ob_p  = ob_s_q[0];
  assign ob_t3 = 3'(ob_t_q);
  always_comb begin
    for (int w = 0; w < 16; w++) ob_data[32*w +: 32] = ob_zero_q ? 32'd0 : om_out[w % 8][w / 8];
  end
  wire ob_last_line = (ob_t_q == tact - 4'd1) & (ob_m_q == 2'd3);
  wire ob_final     = ob_last_line & (ob_s_q == e64 - 9'd1);
  wire rv_take      = (o_st_q == O_LS) & rv_job_v & j_free;
  wire o_trig_s     = ~ob_zero_q & seg_done & done_i.slast;
  logic [1:0] o_pend_q;           // complete slices not yet output (<= 2: o_hold_q)
  wire        o_take = (o_st_q == O_WAIT) & ~ob_zero_q & ((o_pend_q != 2'd0) | o_trig_s);

  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      o_st_q     <= O_WAIT;
      // A reset leaves no op to output. ng is whatever the setup flops hold then (0 after configuration); a set
      // ob_zero_q would start the zero lines of an op that never began and offer them to the gather as soon as the
      // first ATTPV switches the cluster to PV mode.
      ob_zero_q  <= rst_n & (ng == 13'd0);
      ob_s_q     <= '0;
      jv_q       <= 1'b0;
      rv_start_q <= 1'b0;
      done_q     <= 1'b0;
    end else begin
      rv_start_q <= 1'b0;
      done_q     <= 1'b0;
      if (jv_q && job_ready) jv_q <= 1'b0;
      unique case (o_st_q)
        O_WAIT: if (ob_zero_q || o_take) begin  // slices complete in order: ob_s_q is the next one
          o_st_q     <= O_RUN;
          ob_t_q     <= '0;
          ob_m_q     <= '0;
          ob_gofs_q  <= SPM_AW'(32'(ob_s_q >> dcode) * 32'(og));
        end
        O_RUN: if (ob_step) begin
          jv_q   <= 1'b1;
          j_q2   <= {ob_final & ~lsum, ob_line, 16'hFFFF, 16'd0, ob_data};
          ob_m_q <= ob_m_q + 2'd1;
          if (ob_m_q == 2'd3) ob_t_q <= ob_t_q + 4'd1;
          if (ob_last_line) begin                // (o_hold_q[ob_s_q[0]] falls with it, below)
            ob_s_q     <= ob_s_q + 9'd1;
            if (ob_s_q == e64 - 9'd1) begin
              o_st_q     <= lsum ? O_LS : O_END;
              rv_start_q <= lsum;
            end else begin
              o_st_q     <= O_WAIT;
            end
          end
        end
        O_LS: begin
          if (rv_take) begin
            jv_q <= 1'b1;
            j_q2 <= rv_job;
          end
          if (rv_done) o_st_q <= O_END;
        end
        O_END: if (!jv_q || job_ready) begin
          done_q <= 1'b1;
          o_st_q <= O_WAIT;
          ob_zero_q <= 1'b0;
        end
        default: o_st_q <= O_WAIT;
      endcase
    end
  end
  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      o_hold_q <= '0;
      o_pend_q <= '0;
    end else begin
      for (int p = 0; p < 2; p++) begin
        if (dr_start && cb_i.slast && cb_i.s[0] == 1'(p))         o_hold_q[p] <= 1'b1;
        else if (ob_step && ob_last_line && ob_s_q[0] == 1'(p))  o_hold_q[p] <= 1'b0;
      end
      o_pend_q <= o_pend_q + 2'(o_trig_s) - 2'(o_take);
    end
  end
  logic op_done_q;                               // the op is complete (until the next start)
  always_ff @(posedge clk) begin
    if (!rst_n || start) op_done_q <= 1'b0;
    else if (done_q)     op_done_q <= 1'b1;
  end

  tsr_mvm_rvec #(.SPM_AW(SPM_AW), .PIPE(FAST_CLK)) u_rvec (
    .clk, .rst_n,
    .start(rv_start_q), .tq, .gs, .tact, .nseg, .g_heads, .hg0, .n_head, .base(lbase), .last_en(1'b1),
    .val_addr(rv_addr), .val_data(lt_vld_q[rv_addr] ? lt_mem[rv_addr] : 32'd0),
    .job_valid(rv_job_v), .job_ready(rv_take), .job(rv_job), .done(rv_done)
  );

  assign job_valid = jv_q;
  assign job       = j_q2;
  assign done      = done_q;
  logic started_q;
  assign busy      = started_q & ~op_done_q;
  always_ff @(posedge clk) begin
    if (!rst_n)      started_q <= 1'b0;
    else if (start)  started_q <= 1'b1;
    else if (done_q) started_q <= 1'b0;
  end

  // every entry of the convert pipeline's tag arrays (the last ones, and with CVL 3 also entries between, are not read)
  logic unused_cv;
  always_comb begin
    unused_cv = 1'b0;
    for (int i = 0; i <= int'(CVL); i++) begin
      unused_cv = unused_cv ^ lt[i][0] ^ lv[i] ^ cv_v[i] ^ cv_t[i][0] ^ cv_l[i][0] ^ cv_first[i];
    end
  end
  wire unused_ok = &{1'b0, sq_ne, beat.last, beat.row_last, beat.blk, beat.two, beat.half, beat.sc, beat.m,
                     beat.w[127:64], add_v, d2_t.last, i2[FRL], t2[FRL], dr_since_q, unused_cv, d2_i, cb_i, aw_t[3],
                     l_add_t, lt_gbase[7], lt_wt_q, ob_t_q};

`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && a_on_q && !start) begin
      assert (fq_n_q <= 8'(QD)) else $error("%m: delay FIFO overflow");
      assert (sq_n_q <= 4'(SQ)) else $error("%m: segment FIFO overflow");
      if (cap_go) assert (!cb_full_q || (dr_act_q && dr_left_q == 7'd1)) else $error("%m: capture into a busy bank");
      assert (!jv_q || started_q) else $error("%m: output job offered while no ATTPV op runs");
    end
  end
`endif

endmodule

// The ATTPV drain's lane select (AT §8.2): pipe k of a drain step reads lane 8 l + k of slot t of the capture bank and
// the non-finite flag of that lane's block (lane / 32). The lane number is the concatenation {l, k}; an integer
// expression for it (`automatic int L = 8 * int'(l) + k`, then `a[t][6'(L)]` and `nf[t][L / 32]`) is synthesized by
// Vivado 2024.2 without the top bit of l, so that lanes 32 to 63 read as lanes 0 to 31 (docs/platform-u200.md §8.8;
// tb/at/viv/lsel_net.sh simulates the netlist of both forms).
/* verilator lint_off DECLFILENAME */
module tsr_mvm_pv_lsel #(
  parameter  int unsigned T_MAX = 4,
  localparam int unsigned TW    = (T_MAX > 1) ? $clog2(T_MAX) : 1
) (
  input  logic [2:0]                    l,
  input  logic [TW-1:0]                 t,
  input  logic [T_MAX-1:0][63:0][37:0]  a,
  input  logic [T_MAX-1:0][1:0]         nf,
  output logic [7:0][37:0]              y,
  output logic [7:0]                    f
);
  always_comb begin
    for (int k = 0; k < 8; k++) begin
      automatic logic [5:0] lane = {l, 3'(k)};
      y[k] = a[t][lane];
      f[k] = nf[t][lane[5]];
    end
  end
endmodule
/* verilator lint_on DECLFILENAME */
