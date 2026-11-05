// tsr_dma_kvst.sv — the KVST engine of the DMA unit (AT §5.1, §8.3): stores n_rows QA(A32, E)
// vectors of the SPM as q8_0 K or V cell rows into a KV plane in device DRAM (AT §4.3 / §4.4).
//
// Owner: AT (W4-B "at"). Instantiated by tsr_dma (HAS_KVST = 1), which hands it the memory source
// ports of every channel and the SPM read port while a KVST runs (one DMA command at a time).
// Oracle: tsr_ref_kvst (sw/sim/tsr_ref_at.c) through the simulator's ex_kvst; tb/at test_at_kvst
// compares every plane byte and every SPM / DRAM access range.
//
// Semantics (AT §5.1): cell c0 + i receives row i (QA vector at src + i stride): its quant bytes
// QS[0, E) and its block scales fp16(D[b]) (tsr_fp_pkg::tsr_f32_to_fp16 = tsr_f32_to_fp16, exact
// for QACT output), placed per AT §4.3 (K: beta = r E/32 + b, chunk kappa = beta >> 5 at beat
// B + l GB + 17 kappa, meta bytes [2o, 2o+2), quant beat + 1 + o/2, bytes [32(o & 1), +32)) or
// §4.4 (V: chunk (s LG + l) 17, meta bytes [4r, 4r+4) = d of blocks 2s, 2s+1, quant beat + 1 + r).
// Group u lives in channel u mod C at local index l = u / C. Every other plane byte is unchanged:
// a meta beat the op covers only partly (K: a chunk shared with cells outside the op; V: unless the
// op writes all 16 cells of the group) uses a word mask; a fully covered meta beat and every quant beat enable all words.
//
// Work order (deterministic; the result does not depend on it): group segments (the op's cells in
// one group, ascending), per segment its items in DRAM order: K per chunk kappa a META item then a
// QRUN item (the chunk's quant beats of the op's cells, one contiguous run, <= 16 beats); V per slice
// s a META item then a QRUN item (cells r_lo..r_hi-1 of the slice, contiguous beats).
//
// Microarchitecture (one walker; every output is a flop):
//   - setup: u_lo = c0 / 16, its channel / local index (exact division by NUM_CH, tsr_att_divc),
//     the segment bases (B + l GB for K, B + 17 l for V), advanced per segment (no multiplier in
//     the walk).
//   - META: the walker issues the D-line reads of the meta beat (K: one per run of blocks of one
//     cell inside one D line, <= 16 blocks; V: one per cell), back to back while the response
//     buffer (RB entries) has room (R4: SPM reads have no backpressure); the placer converts two
//     words per cycle (2 x tsr_f32_to_fp16) into the 32 fp16 slots of meta_q. The new words and their
//     mask enter the write queue together with a one-beat write request.
//   - QRUN: SPM reads of the run's QS lines (K: consecutive QS lines of a cell, then the next cell;
//     V: line s of each cell) into the data queue (credit: DQ entries), and the run's write
//     request(s) (split at 64-beat blocks).
//   - data queue (DQ entries) -> wr_dat of the item's channel, wr_dat_last at each burst end;
//     write requests from a request register (wr_req and wr_dat independent, rule M5).
//   - done: the walk finished, the data queue and the request register are empty and every write
//     response is in (sequential semantics: a later op sees the plane).

module tsr_dma_kvst
  import tsr_pkg::*;
  import tsr_att_pkg::*;
  import tsr_fp_pkg::*;
#(
  parameter  int unsigned NUM_CH      = TSR_DEF_NUM_CH,
  parameter  int unsigned ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter  int unsigned SPM_AW      = 16,
  parameter  int unsigned SPM_RD_LAT  = TSR_DEF_SPM_RD_LAT,
  localparam int unsigned LB          = TSR_LINE_BITS
) (
  input  logic                         clk,
  input  logic                         rst_n,

  input  logic                         start,       // pulse, only while !busy
  input  tsr_kvst_cmd_t                cmd,
  output logic                         busy,
  output logic                         done,        // registered pulse

  output logic [NUM_CH-1:0]            wr_req_valid,
  input  logic [NUM_CH-1:0]            wr_req_ready,
  output logic [ADDR_BEAT_W-1:0]       wr_req_addr,
  output logic [TSR_LEN_W-1:0]         wr_req_len,
  output logic [NUM_CH-1:0]            wr_dat_valid,
  input  logic [NUM_CH-1:0]            wr_dat_ready,
  output logic [LB-1:0]                wr_dat_data,
  output logic [LB/32-1:0]              wr_dat_mask,
  output logic                         wr_dat_last,
  input  logic [NUM_CH-1:0]            wr_rsp_valid,

  output logic                         spm_rd_req_valid,
  input  logic                         spm_rd_req_ready,
  output logic [SPM_AW-1:0]            spm_rd_req_line,
  input  logic                         spm_rd_rsp_valid,
  input  logic [LB-1:0]                spm_rd_rsp_data,

  output logic                         perf_rmw      // explicit core RMW events (zero on the masked path)
);

  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX) begin : g_chk_ch
    $fatal(1, "tsr_dma_kvst: NUM_CH must be in [1, 16]");
  end

  localparam int unsigned CHW = (NUM_CH > 1) ? $clog2(NUM_CH) : 1;
  localparam int unsigned AW  = (ADDR_BEAT_W > 7) ? ADDR_BEAT_W : 7;   // internal address width
  localparam int unsigned RB  = 4;                  // D-line response buffer entries
  localparam int unsigned DQ  = (SPM_RD_LAT + 4 > 8) ? 16 : 8;   // data queue entries
  localparam int unsigned DQW = $clog2(DQ);

  // -------------------------------------------------------------------------------------------
  // Command registers and setup
  // -------------------------------------------------------------------------------------------
  tsr_kvst_cmd_t c_q;
  logic [9:0]    nb_q;          // E / 32 blocks per cell row
  logic [13:0]   gb_q;          // 17 E/64 beats per group
  logic [20:0]   lg17_q;        // 17 LG
  logic [20:0]   u_lo_q, u_hi_q, u_q;
  logic [3:0]    rlo0_q, rhi1_q;  // first segment's r_lo, last segment's r_hi - 1
  logic [20:0]   llo_q;         // u_lo / C
  logic [CHW-1:0] ch_q;         // channel of the current segment
  logic [AW-1:0] sbase_q;       // segment base: K B + l GB, V B + 17 l
  logic [AW-1:0] sofs_q;        // its offset l GB / 17 l (S_I2), added in S_I3

  typedef enum logic [3:0] {
    S_IDLE, S_I1, S_I2, S_I3, S_SEG, S_SEG2, S_MPRE, S_MREQ, S_MPUSH, S_QRUN, S_NEXT, S_FIN
  } st_t;
  st_t st_q;

  // segment geometry
  logic [3:0]  rlo_q;           // first cell of the segment in its group
  logic [4:0]  rhi_q;           // one past the last (1..16)
  logic [13:0] blo_q, bhi_q;    // K: beta range [r_lo nb, r_hi nb)

  // walk position: K chunk / V slice
  logic [8:0]  kap_q;           // K chunk index inside the group (< E/64) / V slice s (< E/64)
  logic [8:0]  klast_q;         // K last chunk of the segment / V E/64 - 1
  logic [AW-1:0] maddr_q;       // address of the current item's meta beat
  // K: block cursor (cell r, block b of its row, the cell's SPM line); V: cell cursor
  logic [4:0]  r_q;             // cell inside the group
  logic [9:0]  b_q;             // K block inside the cell row
  logic [31:0] line_q;          // SPM line of the cell's QA vector
  logic [13:0] beta_q;          // K beta of the cursor
  // saved cursor (chunk start / slice start) for the QRUN re-walk
  logic [4:0]  sr_q;
  logic [9:0]  sb_q;
  logic [31:0] sline_q;
  logic [13:0] sbeta_q;
  logic [31:0] vline0_q;        // V: SPM line of the segment's first cell

  wire is_v = c_q.kv;
  wire [13:0] cb0 = (blo_q > {kap_q, 5'd0}) ? blo_q : {kap_q, 5'd0};        // K chunk beta range
  wire [13:0] cb1 = (bhi_q < ({kap_q, 5'd0} + 14'd32)) ? bhi_q : ({kap_q, 5'd0} + 14'd32);
  wire [5:0]  olo = 6'(cb0 - {kap_q, 5'd0});
  wire [5:0]  ohi = 6'(cb1 - {kap_q, 5'd0});
  // the item's chunk bounds, registered in S_MPRE (the walk then uses no adder chain from kap_q)
  logic [5:0]  olo_q, ohi_q;
  logic [9:0]  torow_q, storow_q;   // K: blocks to the row end (nb - b), at the item start
  logic [13:0] tock_q, stock_q;     // K: blocks to the chunk end (cb1 - beta), at the item start

  // -------------------------------------------------------------------------------------------
  // D-line reads (META) and QRUN SPM reads share the SPM read port; responses are routed by a tag
  // pipeline (the read latency is fixed).
  // -------------------------------------------------------------------------------------------
  typedef struct packed {
    logic [CHW-1:0] ch;        // QRUN: the beat's channel (the walker may move on before it returns)
    logic       qrun;          // 1: QRUN beat -> data queue; 0: D line -> response buffer
    logic [2:0] wp0;           // D line: first word pair
    logic [3:0] np_m1;         // D line: pairs - 1
    logic [3:0] sp0;           // D line: first meta slot pair
    logic       last;          // QRUN: burst end
  } rtag_t;

  logic             sr_v_q;            // SPM read request register
  logic [SPM_AW-1:0] sr_line_q;
  rtag_t            sr_tag_q;
  wire              sr_fire = sr_v_q & spm_rd_req_ready;
  logic [SPM_RD_LAT-1:0] tv_q;        // tag pipeline
  rtag_t            tg_q [SPM_RD_LAT];
  wire              rsp_v = spm_rd_rsp_valid & busy;
  rtag_t            rsp_tag;
  assign rsp_tag = tg_q[SPM_RD_LAT-1];

  always_ff @(posedge clk) begin
    if (!rst_n) tv_q <= '0;
    else        tv_q <= SPM_RD_LAT'({tv_q, sr_fire});
  end
  always_ff @(posedge clk) begin
    tg_q[0] <= sr_tag_q;
    for (int i = 1; i < int'(SPM_RD_LAT); i++) tg_q[i] <= tg_q[i-1];
  end

  // response buffer for D lines (RB entries) and its descriptors
  logic [LB-1:0] rb_mem [RB];
  rtag_t         rb_tag [RB];
  logic [1:0]    rb_wp_q, rb_rp_q;
  logic [2:0]    rb_cnt_q;             // held
  logic [2:0]    rb_inf_q;             // D-line reads requested, response not in
  // data queue (DQ entries): {data, ch one-hot index, last}
  logic [LB-1:0] dq_mem [DQ];
  logic [LB/32-1:0] dq_mask [DQ];
  logic [CHW-1:0] dq_ch [DQ];
  logic          dq_last [DQ];
  logic [DQW:0]  dq_wp_q, dq_rp_q;
  logic [DQW:0]  dq_inf_q;            // QRUN reads requested, response not in
  wire  [DQW:0]  dq_cnt = dq_wp_q - dq_rp_q;

  // -------------------------------------------------------------------------------------------
  // Placer: one D line at a time from the response buffer, two words per cycle into meta_q
  // -------------------------------------------------------------------------------------------
  logic [15:0][31:0] meta_q;           // slot pair p = fp16 slots 2p (low half), 2p + 1
  logic [15:0]       mmask_q;          // slot pairs written by this op
  logic              pl_act_q;
  logic [LB-1:0]     pl_line_q;
  logic [2:0]        pl_wp_q;
  logic [3:0]        pl_n_q, pl_sp_q;
  logic [5:0]        placed_q;         // D-line responses placed for the current META
  logic [5:0]        dreq_q;           // D-line reads issued for the current META

  wire rb_ne    = (rb_cnt_q != 3'd0);
  wire pl_start = rb_ne & ~pl_act_q;
  wire pl_end   = pl_act_q & (pl_n_q == 4'd0);
  wire [63:0] pl_pair = pl_line_q[64*pl_wp_q +: 64];
  wire [15:0] h0 = tsr_f32_to_fp16(pl_pair[31:0]);
  wire [15:0] h1 = tsr_f32_to_fp16(pl_pair[63:32]);

  // -------------------------------------------------------------------------------------------
  // Write requests, write responses
  // -------------------------------------------------------------------------------------------
  logic          wq_v_q;
  logic [AW-1:0] wq_addr_q;
  logic [TSR_LEN_W-1:0] wq_len_q;
  logic [15:0]   wr_pend_q;
  wire wq_fire = wq_v_q & |(wr_req_ready & (NUM_CH'(1) << ch_q));
  wire wq_free = ~wq_v_q | wq_fire;
  logic [15:0] wr_rsp_n;
  always_comb begin
    wr_rsp_n = '0;
    for (int c = 0; c < int'(NUM_CH); c++) wr_rsp_n += 16'(wr_rsp_valid[c] & busy);
  end

  // -------------------------------------------------------------------------------------------
  // Walk
  // -------------------------------------------------------------------------------------------
  // META D-line run (K): blocks from the cursor to min(next 16-block boundary of the row, row end,
  // chunk end); V: one cell per read.
  wire [9:0]  k_tob16 = 10'd16 - {6'd0, b_q[3:0]};
  wire [9:0]  k_torow = torow_q;
  wire [13:0] k_tock  = tock_q;
  wire [9:0]  k_run   = (k_tob16 < k_torow) ? ((14'(k_tob16) < k_tock) ? k_tob16 : 10'(k_tock))
                                            : ((14'(k_torow) < k_tock) ? k_torow : 10'(k_tock));
  wire        k_rowe  = (k_torow <= k_tob16) & (14'(k_torow) <= k_tock);   // b + k_run = nb
  wire        m_rd_more = is_v ? (r_q != rhi_q) : (tock_q != 14'd0);
  // rb_inf_q / dq_inf_q count a read from its load into the request register (m_issue / q_issue)
  wire        d_room    = (32'(rb_cnt_q) + 32'(rb_inf_q)) < RB;
  // QRUN: one QS line per beat; K: the pair at the cursor, V: line s of cell r
  wire        q_room    = (32'(dq_cnt) + 32'(dq_inf_q)) < DQ;
  wire        sr_free   = ~sr_v_q | sr_fire;
  // QRUN burst bookkeeping: beats of the run left, current beat address
  logic [4:0]    qn_q;               // beats of the run not yet requested from the SPM
  logic [AW-1:0] qaddr_q;            // DRAM address of the next QRUN beat (for wr_dat_last)
  logic [4:0]    qreq_left_q;        // beats not yet covered by a write request
  logic [AW-1:0] qreq_addr_q;

  wire [6:0] q_to_blk = 7'd64 - {1'b0, qreq_addr_q[5:0]};
  wire [6:0] q_blen   = (7'(qreq_left_q) < q_to_blk) ? 7'(qreq_left_q) : q_to_blk;

  // SPM read issue (one register, shared): META D lines in S_MREQ, QRUN lines in S_QRUN
  wire m_issue = (st_q == S_MREQ) & m_rd_more & d_room & sr_free;
  wire q_issue = (st_q == S_QRUN) & (qn_q != 5'd0) & q_room & sr_free;

  // META ready to push: all scale reads placed, write request free, data queue room
  wire m_ready = (placed_q == dreq_q) & ~pl_act_q & (rb_cnt_q == 3'd0) & (rb_inf_q == 3'd0)
               & wq_free & ((32'(dq_cnt) + 32'(dq_inf_q)) < DQ);
  wire m_push  = (st_q == S_MPUSH) & m_ready;

  wire walk_done = (st_q == S_FIN);
  wire all_quiet = walk_done & ~wq_v_q & (dq_cnt == '0) & (dq_inf_q == '0) & (wr_pend_q == 16'd0)
                 & ~sr_v_q & (tv_q == '0);

  logic done_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q        <= S_IDLE;
      done_q      <= 1'b0;
      sr_v_q      <= 1'b0;
      rb_wp_q     <= '0;
      rb_rp_q     <= '0;
      rb_cnt_q    <= '0;
      rb_inf_q    <= '0;
      dq_wp_q     <= '0;
      dq_rp_q     <= '0;
      dq_inf_q    <= '0;
      pl_act_q    <= 1'b0;
      placed_q    <= '0;
      dreq_q      <= '0;
      wq_v_q      <= 1'b0;
      wr_pend_q   <= '0;
      mmask_q     <= '0;
      qn_q        <= '0;
      qreq_left_q <= '0;
    end else begin
      done_q <= 1'b0;

      // ---- SPM read register ----
      if (m_issue || q_issue) sr_v_q <= 1'b1;
      else if (sr_fire)       sr_v_q <= 1'b0;

      // ---- response routing ----
      rb_inf_q <= rb_inf_q + 3'(m_issue) - 3'(rsp_v && !rsp_tag.qrun);
      dq_inf_q <= dq_inf_q + (DQW+1)'(q_issue) - (DQW+1)'(rsp_v & rsp_tag.qrun);
      if (rsp_v && !rsp_tag.qrun) rb_wp_q <= rb_wp_q + 2'd1;
      rb_cnt_q <= rb_cnt_q + 3'(rsp_v && !rsp_tag.qrun) - 3'(pl_start);
      if (pl_start) rb_rp_q <= rb_rp_q + 2'd1;

      // ---- placer ----
      if (pl_start) begin
        pl_act_q <= 1'b1;
        pl_wp_q  <= rb_tag[rb_rp_q].wp0;
        pl_n_q   <= rb_tag[rb_rp_q].np_m1;
        pl_sp_q  <= rb_tag[rb_rp_q].sp0;
      end else if (pl_act_q) begin
        if (pl_n_q == 4'd0) pl_act_q <= 1'b0;
        pl_n_q  <= pl_n_q - 4'd1;
        pl_wp_q <= pl_wp_q + 3'd1;
        pl_sp_q <= pl_sp_q + 4'd1;
      end
      if (pl_act_q) mmask_q[pl_sp_q] <= 1'b1;
      if (pl_end)   placed_q <= placed_q + 6'd1;
      if (m_issue)  dreq_q <= dreq_q + 6'd1;

      // ---- data queue ----
      if (m_push || (rsp_v && rsp_tag.qrun)) dq_wp_q <= dq_wp_q + 1'b1;
      if (wr_dat_valid != '0 && |(wr_dat_ready & wr_dat_valid)) dq_rp_q <= dq_rp_q + 1'b1;

      // ---- write requests / responses ----
      if (wq_fire) wq_v_q <= 1'b0;
      wr_pend_q <= wr_pend_q + 16'(wq_fire) - wr_rsp_n;   // only KVST writes are in flight; several
                                                         // channels may answer in one cycle

      // ---- walker ----
      unique case (st_q)
        S_IDLE: if (start) st_q <= S_I1;
        S_I1: st_q <= S_I2;                                     // u_lo / C
        S_I2: st_q <= S_I3;                                     // channel, segment base
        S_I3: st_q <= S_SEG;
        S_SEG: st_q <= S_SEG2;                                  // segment geometry
        S_SEG2: st_q <= S_MPRE;                                // segment start: first item
        S_MPRE: st_q <= S_MREQ;                                 // the item's chunk bounds
        S_MREQ: if (!m_rd_more && !m_issue) st_q <= S_MPUSH;
        S_MPUSH: if (m_push) st_q <= S_QRUN;
        // the run's write request(s) and SPM reads; the segment's channel stays until its last
        // request is accepted (wq_v_q = 0)
        S_QRUN: if (qn_q == 5'd0 && qreq_left_q == 5'd0 && !wq_v_q) st_q <= S_NEXT;
        S_NEXT: st_q <= (kap_q == klast_q && u_q == u_hi_q) ? S_FIN :
                        (kap_q == klast_q) ? S_SEG : S_MPRE;
        S_FIN: if (all_quiet) begin
          st_q   <= S_IDLE;
          done_q <= 1'b1;
        end
        default: st_q <= S_IDLE;
      endcase

      // write request issue: META (1 beat) at its push, QRUN bursts in S_QRUN
      if (m_push) begin
        wq_v_q <= 1'b1;
      end else if (st_q == S_QRUN && wq_free && qreq_left_q != 5'd0) begin
        wq_v_q      <= 1'b1;
        qreq_left_q <= qreq_left_q - 5'(q_blen);
      end
      if (st_q == S_MPUSH && m_push) begin
        qreq_left_q <= is_v ? 5'(rhi_q - 5'(rlo_q)) : 5'((ohi_q - olo_q) >> 1);
        qn_q        <= is_v ? 5'(rhi_q - 5'(rlo_q)) : 5'((ohi_q - olo_q) >> 1);
      end
      if (q_issue) qn_q <= qn_q - 5'd1;

      if (st_q == S_SEG2 || st_q == S_NEXT) begin               // item / segment start: a new META
        mmask_q  <= '0;
        placed_q <= '0;
        dreq_q   <= '0;
      end
    end
  end

  // ---- walker datapath (no reset; qualified by st_q) ----
  logic [20:0] llo_c;
  logic [4:0]  ch_c;
  always_comb begin
    llo_c = tsr_att_divc(u_lo_q, NUM_CH);
    ch_c  = 5'(u_lo_q - llo_c * 21'(NUM_CH));
  end

  always_ff @(posedge clk) begin
    if (start && st_q == S_IDLE) begin
      c_q     <= cmd;
      nb_q    <= {cmd.e64, 1'b0};
      gb_q    <= 14'(cmd.e64) * 14'd17;
      lg17_q  <= 21'(cmd.lg) * 21'd17;
      u_lo_q  <= 21'(cmd.c0 >> 4);
      u_hi_q  <= 21'((cmd.c0 + 32'(cmd.n_rows) - 32'd1) >> 4);
      rlo0_q  <= cmd.c0[3:0];
      rhi1_q  <= 4'(cmd.c0 + 32'(cmd.n_rows) - 32'd1);
      line_q  <= cmd.src;
      vline0_q <= cmd.src;
    end
    if (st_q == S_I1) llo_q <= llo_c;
    if (st_q == S_I3) sbase_q <= AW'(c_q.base) + sofs_q;
    if (st_q == S_I2) begin
      ch_q    <= CHW'(ch_c);
      u_q     <= u_lo_q;
      sofs_q  <= c_q.kv ? AW'({llo_q, 4'd0}) + AW'(llo_q) : AW'(llo_q) * AW'(gb_q);   // l_lo 17 / l_lo GB
      rlo_q   <= rlo0_q;
    end
    // segment geometry (S_SEG): the segment of group u_q with r_lo = rlo_q
    if (st_q == S_SEG) begin
      rhi_q      <= (u_q == u_hi_q) ? 5'(rhi1_q) + 5'd1 : 5'd16;
      blo_q      <= 14'(rlo_q) * 14'(nb_q);
      bhi_q      <= ((u_q == u_hi_q) ? 14'(5'(rhi1_q) + 5'd1) : 14'd16) * 14'(nb_q);
    end
    if (st_q == S_SEG2) begin
      // first item of the segment
      kap_q    <= is_v ? 9'd0 : 9'(blo_q >> 5);
      klast_q  <= is_v ? 9'(c_q.e64 - 9'd1) : 9'((bhi_q - 14'd1) >> 5);
      maddr_q  <= is_v ? sbase_q : sbase_q + AW'(blo_q >> 5) * AW'(17);
      r_q      <= 5'(rlo_q);
      b_q      <= '0;
      beta_q   <= blo_q;
      if (is_v) line_q <= vline0_q;
      sr_q     <= 5'(rlo_q);
      sb_q     <= '0;
      sbeta_q  <= blo_q;
      torow_q  <= nb_q;
      storow_q <= nb_q;
      sline_q  <= is_v ? vline0_q : line_q;
    end

    // item start (S_MPRE): chunk bounds of kap_q; beta_q = sbeta_q here
    if (st_q == S_MPRE) begin
      olo_q    <= olo;
      ohi_q    <= ohi;
      tock_q   <= cb1 - beta_q;
      stock_q  <= cb1 - beta_q;
    end

    // META D-line reads
    if (m_issue) begin
      sr_line_q <= is_v ? SPM_AW'(line_q + 32'(c_q.e64) + 32'(kap_q >> 3))
                        : SPM_AW'(line_q + 32'(c_q.e64) + 32'(b_q >> 4));
      sr_tag_q  <= is_v ? '{ch: ch_q, qrun: 1'b0, wp0: kap_q[2:0], np_m1: 4'd0, sp0: r_q[3:0], last: 1'b0}
                        : '{ch: ch_q, qrun: 1'b0, wp0: b_q[3:1], np_m1: 4'((k_run >> 1) - 10'd1),
                            sp0: 4'((beta_q - {kap_q, 5'd0}) >> 1), last: 1'b0};
      if (is_v) begin
        r_q    <= r_q + 5'd1;
        line_q <= line_q + c_q.stride;
      end else begin
        beta_q <= beta_q + 14'(k_run);
        tock_q <= tock_q - 14'(k_run);
        if (k_rowe) begin
          b_q     <= '0;
          torow_q <= nb_q;
          r_q     <= r_q + 5'd1;
          line_q  <= line_q + c_q.stride;
        end else begin
          b_q     <= b_q + k_run;
          torow_q <= torow_q - k_run;
        end
      end
    end
    // META -> QRUN: rewind the cursor to the item start
    if (m_push) begin
      r_q     <= sr_q;
      b_q     <= sb_q;
      beta_q  <= sbeta_q;
      torow_q <= storow_q;
      tock_q  <= stock_q;
      line_q  <= sline_q;
      qaddr_q <= is_v ? maddr_q + AW'(1) + AW'(rlo_q) : maddr_q + AW'(1) + AW'(olo_q >> 1);
      qreq_addr_q <= is_v ? maddr_q + AW'(1) + AW'(rlo_q) : maddr_q + AW'(1) + AW'(olo_q >> 1);
    end
    if (st_q == S_QRUN && wq_free && qreq_left_q != 5'd0) begin
      wq_addr_q   <= qreq_addr_q;
      wq_len_q    <= TSR_LEN_W'(q_blen);
      qreq_addr_q <= qreq_addr_q + AW'(q_blen);
    end else if (m_push) begin
      wq_addr_q <= maddr_q;
      wq_len_q  <= TSR_LEN_W'(1);
    end
    // QRUN SPM reads
    if (q_issue) begin
      sr_line_q <= is_v ? SPM_AW'(line_q + 32'(kap_q)) : SPM_AW'(line_q + 32'(b_q >> 1));
      sr_tag_q  <= '{ch: ch_q, qrun: 1'b1, wp0: 3'd0, np_m1: 4'd0, sp0: 4'd0,
                     last: (qaddr_q[5:0] == 6'h3F) || (qn_q == 5'd1)};
      qaddr_q   <= qaddr_q + AW'(1);
      if (is_v) begin
        r_q    <= r_q + 5'd1;
        line_q <= line_q + c_q.stride;
      end else begin
        beta_q <= beta_q + 14'd2;
        tock_q <= tock_q - 14'd2;
        if (torow_q == 10'd2) begin                             // b + 2 = nb
          b_q     <= '0;
          torow_q <= nb_q;
          r_q     <= r_q + 5'd1;
          line_q  <= line_q + c_q.stride;
        end else begin
          b_q     <= b_q + 10'd2;
          torow_q <= torow_q - 10'd2;
        end
      end
    end
    // next item / segment
    if (st_q == S_NEXT) begin
      if (kap_q != klast_q) begin
        kap_q   <= kap_q + 9'd1;
        maddr_q <= maddr_q + (is_v ? AW'(lg17_q) : AW'(17));
        // K: the cursor continues (it sits at the next chunk's first block); V: next slice
        if (is_v) begin
          r_q    <= 5'(rlo_q);
          line_q <= vline0_q;
        end
        sr_q    <= is_v ? 5'(rlo_q) : r_q;
        sb_q    <= is_v ? '0 : b_q;
        sbeta_q <= is_v ? '0 : beta_q;
        storow_q <= torow_q;
        sline_q <= is_v ? vline0_q : line_q;
      end else if (u_q != u_hi_q) begin
        // next segment: group u + 1, r_lo = 0; the cell line continues (K) / the slice walk ended
        // at cell r_hi of the last slice (V)
        u_q   <= u_q + 21'd1;
        rlo_q <= 4'd0;
        if (is_v) vline0_q <= line_q;
        if (32'(ch_q) == NUM_CH - 1) begin
          ch_q    <= '0;
          sbase_q <= sbase_q + (is_v ? AW'(17) : AW'(gb_q));
        end else begin
          ch_q    <= ch_q + 1'b1;
        end
      end
    end
  end
  // the K walk keeps its cursor across chunks: S_SEG2 re-reads line_q (K) as the segment start

  // response buffer / data queue writes, placer datapath
  always_ff @(posedge clk) begin
    if (rsp_v && !rsp_tag.qrun) begin
      rb_mem[rb_wp_q] <= spm_rd_rsp_data;
      rb_tag[rb_wp_q] <= rsp_tag;
    end
    if (pl_start) pl_line_q <= rb_mem[rb_rp_q];
    if (pl_act_q) meta_q[pl_sp_q] <= {h1, h0};
    if (m_push) begin
      dq_mem[dq_wp_q[DQW-1:0]]  <= meta_q;
      dq_mask[dq_wp_q[DQW-1:0]] <= mmask_q;
      dq_ch[dq_wp_q[DQW-1:0]]   <= ch_q;
      dq_last[dq_wp_q[DQW-1:0]] <= 1'b1;
    end else if (rsp_v && rsp_tag.qrun) begin
      dq_mem[dq_wp_q[DQW-1:0]]  <= spm_rd_rsp_data;
      dq_mask[dq_wp_q[DQW-1:0]] <= '1;
      dq_ch[dq_wp_q[DQW-1:0]]   <= rsp_tag.ch;
      dq_last[dq_wp_q[DQW-1:0]] <= rsp_tag.last;
    end
  end

  // ---- outputs ----
  wire dq_ne = (dq_cnt != '0);
  assign busy             = (st_q != S_IDLE);
  assign done             = done_q;
  assign wr_req_valid     = wq_v_q ? (NUM_CH'(1) << ch_q) : '0;
  assign wr_req_addr      = ADDR_BEAT_W'(wq_addr_q);
  assign wr_req_len       = wq_len_q;
  assign wr_dat_valid     = dq_ne ? (NUM_CH'(1) << dq_ch[dq_rp_q[DQW-1:0]]) : '0;
  assign wr_dat_data      = dq_mem[dq_rp_q[DQW-1:0]];
  assign wr_dat_mask      = dq_mask[dq_rp_q[DQW-1:0]];
  assign wr_dat_last      = dq_last[dq_rp_q[DQW-1:0]];
  assign spm_rd_req_valid = sr_v_q;
  assign spm_rd_req_line  = sr_line_q;
  assign perf_rmw         = 1'b0;

  wire unused_ok = &{1'b0, c_q.n_rows, c_q.src, c_q.lg, c_q.c0, rb_tag[0].last, rb_tag[0].qrun, rb_tag[0].ch, ch_c, wq_addr_q, gb_q, lg17_q, llo_q};

`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (start) assert (st_q == S_IDLE) else $error("%m: start while busy");
      assert (32'(rb_cnt_q) + 32'(rb_inf_q) <= RB) else $error("%m: D-line response buffer overflow (held %0d in flight %0d st %0d)", rb_cnt_q, rb_inf_q, st_q);
      assert (32'(dq_cnt) + 32'(dq_inf_q) <= DQ) else $error("%m: data queue overflow");
      if (m_push) assert (mmask_q != '0) else $error("%m: empty META mask");
      if (wq_fire) assert (wq_len_q != '0 && 32'(wq_addr_q[5:0]) + 32'(wq_len_q) <= 64)
        else $error("%m: write burst addr %0h len %0d breaks the 4 KiB rule", wq_addr_q, wq_len_q);
      if (st_q == S_IDLE) assert (!sr_v_q && !wq_v_q && dq_cnt == '0)
        else $error("%m: traffic while idle");
    end
  end
`endif

endmodule
