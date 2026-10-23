// tsr_mvm_gather.sv — MVM result gather (spec MVM 클러스터 stage 10 "출력"): row results of the
// NUM_CH clusters -> y lines in the SPM (partial lines with byte enables), ACC read-modify-write,
// zero tails.
//
// Owner: top agent (docs/rtl-m3.md §2 block (top); contract §6.3).
//
// Contract (normative text docs/rtl-m3.md §6.3; tsr_sim.c ex_mvm write loop):
//   - start (pulse) with the MVM fields (stable from start until done): acc, t (tokens), n (N >= 1),
//     dst_line, dst_stride (resolved, never 0), expect_ch[c] = channel c has rows. dst_stride <
//     ceil(N/16) with t > 1 makes token ranges overlap: undefined result, but the gather must still
//     finish normally (no hang; spec 구현 명확화 "겹치는 피연산자").
//   - res_* per cluster (R1/R2): rows res_row = n_row in increasing order, res_last on the
//     cluster's final row. Element n_row of token tk (tk < t) goes to SPM line
//     dst_line + tk*dst_stride + n_row/16, f32 lane n_row % 16; with acc it is
//     fadd(old, y) (tsr_fp_add, old = the line's value before the MVM), else y.
//   - each cluster's consecutive rows inside one SPM line are assembled and written as ONE partial
//     line per token (spm_wr_be = 4 bytes per present lane); the line holding row N-1 also
//     carries the zero tail (lanes > (N-1) % 16 written as 0, also with acc). Every y byte is
//     written exactly once; no byte outside [dst + tk*stride, + ceil(4N/64) lines) is written.
//   - acc: the old line is read through spm_rd_* (fixed SPM_RD_LAT, no backpressure: space for
//     the response is reserved before the read, R4) and only the owner's lanes are written back,
//     so partial lines of different clusters in the same SPM line never race (§6.3).
//   - done: registered 1-cycle pulse after every expected cluster delivered its last row and the
//     last SPM write was accepted (all writes accepted => visible, docs/rtl-m2.md §5); never in
//     the start cycle. res_ready may be low only for local reasons (queue full).
// Throughput: up to one SPM line write per cycle (ACC: one read + one write per cycle, 16
//   tsr_fp_add lanes); see docs/rtl-m3.md §6.3.
//
// Microarchitecture:
//
//   res[c] ──► assembler c: two buffers (ping-pong) of T_MAX x 16 lanes x 32 bits
//                │  fill buffer wsel: row n -> lane n%16 of every token; the group is complete at
//                │  lane 15 or res_last; per buffer: lane mask own, tail mask, SPM line of the next
//                │  job (dst + n/16 at the group's first row, + stride per emitted token)
//                ▼  emit buffer esel: one job per token tk < t (req[c] = full[c][esel])
//   round robin (registered pick sel_q over req, last taken cluster lowest) ── take ──►
//      non-ACC: job {line, be, y masked to own lanes (tail lanes 0)} -> write queue
//      ACC    : job -> issue slice (tsr_rslice) -> SPM read -> pipe SPM_RD_LAT + 1 (old line
//               registered) -> 16 tsr_fp_add (old + y) -> {line, be, own ? sum : 0} -> write queue
//   write queue (WQ_D entries + output register) ──► spm_wr_* (valid, line, be, data from flops)
//
//   - Credits (R4): cnt_q counts jobs from the take to the accepted SPM write; a take needs
//     cnt < WQ_D (registered space_q), so the write queue never overflows and the ACC read
//     response (no backpressure) always has room. WQ_D >= the credit loop (take -> issue -> read
//     -> SPM_RD_LAT + 1 -> ADD_LAT -> queue -> output register -> pop -> space_q), so the gather
//     sustains one job per cycle when the SPM write port accepts one line per cycle.
//   - Hazards (docs/rtl-m3.md §6.3): two jobs on one SPM line belong to different clusters with
//     disjoint lane sets; each writes only its lanes (byte enables), so any order of the ACC
//     reads and writes gives the ex_mvm result. Different tokens never share a line (stride >=
//     y_lines when t > 1; otherwise undefined).
//   - res_ready[c] (registered) = active and the buffer the next row goes to is not full. A
//     buffer is full from its group's completion to the take of its last token's job.
//   - Logic depth: take = flops through 2-3 LUTs; job mux = cluster (sel_q, NUM_CH:1) x buffer
//     (esel) x token (etk) of flops, then one AND for the lane mask; write-queue output from
//     an output register. The ACC adders take a registered old line.
//   Latency (ideal SPM, one requester): non-ACC row -> SPM write accept: completion edge, pick 1,
//     take 1, queue 1, output register 1; ACC adds the issue slice, SPM_RD_LAT + 1 and ADD_LAT.
//   Measured (tb/mvm/test_mvm_gather.py, results offered every cycle, SPM always ready):
//     1.000 line writes per cycle, plain and ACC, whenever the write port is the bottleneck
//     (16 clusters x T 8, SPM_RD_LAT 4, ADD_LAT 8); otherwise the source rate (NUM_CH * T / 16
//     lines per cycle) is met exactly. First write 19 cycles after a line's first row (16 rows at
//     one per cycle + 3); ACC 23 / 27 / 28 / 33 cycles at (SPM_RD_LAT, ADD_LAT) = (1,1) / (3,3) /
//     (2,5) / (4,8). Fixed-priority arbitration instead of the round robin halves the rate (0.50).
//   Cost (from the RTL structure; no synthesis run in this step), U200 (4 ch, T_MAX 1, 4 MiB SPM,
//     SPM_RD_LAT 4, ADD_LAT 5): assembly buffers 4 x 2 x 512 FF, lane masks / lines 384 FF, ACC
//     issue slice 1.1 K FF, ACC read pipe 2.8 K FF (no reset, SRL-capable), old line 512 FF,
//     write queue 16 x 592 bits (distributed RAM) + 593 FF output register, 16 tsr_fp_add:
//     ~9.9 K FF total. Assembly buffers scale with NUM_CH x 2 x T_MAX x 512 bits.

module tsr_mvm_gather
  import tsr_pkg::*;
#(
  parameter  int unsigned NUM_CH     = TSR_DEF_NUM_CH,
  parameter  int unsigned T_MAX      = TSR_DEF_T_MAX,
  parameter  int unsigned SPM_BYTES  = TSR_DEF_SPM_BYTES,
  parameter  int unsigned SPM_RD_LAT = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned ADD_LAT    = TSR_DEF_FADD_LAT,
  localparam int unsigned SPM_LINES  = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned SPM_AW     = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1
) (
  input  logic                                   clk,
  input  logic                                   rst_n,

  // ---- MVM fields (from tsr_mvm; stable from start until done) ----
  input  logic                                   start,
  input  logic                                   acc,
  input  logic [3:0]                             t,
  input  logic [31:0]                            n,
  input  logic [SPM_AW-1:0]                      dst_line,
  input  logic [31:0]                            dst_stride,
  input  logic [NUM_CH-1:0]                      expect_ch,
  output logic                                   done,

  // ---- row results per cluster (after the result pipes) ----
  input  logic [NUM_CH-1:0]                      res_valid,
  output logic [NUM_CH-1:0]                      res_ready,
  input  logic [NUM_CH-1:0][31:0]                res_row,
  input  logic [NUM_CH-1:0]                      res_last,
  input  logic [NUM_CH-1:0][T_MAX-1:0][31:0]     res_y,

  // ---- SPM (client TSR_SPM_CL_MVM; tsr_mvm muxes the read port with the broadcast) ----
  output logic                                   spm_rd_req_valid,
  input  logic                                   spm_rd_req_ready,
  output logic [SPM_AW-1:0]                      spm_rd_req_line,
  input  logic                                   spm_rd_rsp_valid,
  input  logic [TSR_LINE_BITS-1:0]               spm_rd_rsp_data,
  output logic                                   spm_wr_valid,
  input  logic                                   spm_wr_ready,
  output logic [SPM_AW-1:0]                      spm_wr_line,
  output logic [TSR_LINE_BITS-1:0]               spm_wr_data,
  output logic [TSR_SPM_BE_W-1:0]                spm_wr_be
);

  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX || T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH) begin : g_chk
    $fatal(1, "tsr_mvm_gather: NUM_CH in [1, 16], T_MAX in [1, 8]");
  end
  if (SPM_RD_LAT < 1 || ADD_LAT < 1 || ADD_LAT > 8) begin : g_chk_lat
    $fatal(1, "tsr_mvm_gather: SPM_RD_LAT >= 1, ADD_LAT in [1, 8]");
  end

  // -------------------------------------------------------------------------------------------
  // Sizes
  // -------------------------------------------------------------------------------------------
  localparam int unsigned CW    = (NUM_CH > 1) ? $clog2(NUM_CH) : 1;   // cluster index
  localparam int unsigned TKW   = (T_MAX > 1) ? $clog2(T_MAX) : 1;     // token index
  localparam int unsigned WQ_D  = 1 << $clog2(SPM_RD_LAT + ADD_LAT + 6);  // write queue entries
  localparam int unsigned WQ_AW = $clog2(WQ_D);
  localparam int unsigned CNT_W = WQ_AW + 1;                           // 0 .. WQ_D
  localparam int unsigned WQ_W  = SPM_AW + TSR_SPM_BE_W + TSR_LINE_BITS;   // {line, be, data}
  localparam int unsigned IS_W  = SPM_AW + 16 + 16 + TSR_LINE_BITS;        // {line, own, tail, y}
  localparam int unsigned MT_W  = SPM_AW + 16 + 16;                        // {line, own, tail}

  typedef logic [15:0][31:0] line_t;    // 16 f32 lanes (lane l = bits [32l +: 32])

  function automatic logic [TSR_SPM_BE_W-1:0] lane_be(logic [15:0] lanes);
    logic [TSR_SPM_BE_W-1:0] be;
    for (int unsigned l = 0; l < 16; l++) be[4*l +: 4] = {4{lanes[l]}};
    return be;
  endfunction

  function automatic line_t lane_mask(line_t d, logic [15:0] lanes);
    line_t o;
    for (int unsigned l = 0; l < 16; l++) o[l] = lanes[l] ? d[l] : 32'd0;
    return o;
  endfunction

  // -------------------------------------------------------------------------------------------
  // MVM configuration (registered at start)
  // -------------------------------------------------------------------------------------------
  logic              act_q;          // holding an MVM (start .. done)
  logic              acc_q;
  logic [TKW-1:0]    tm1_q;          // t - 1
  logic [31:0]       nm1_q;          // N - 1
  logic [SPM_AW-1:0] dst_q;
  logic [SPM_AW-1:0] stride_q;       // low bits suffice: dst + tk * stride < SPM lines for tk < t
  logic [NUM_CH-1:0] exp_q;          // channels with rows
  logic              done_q;

  // -------------------------------------------------------------------------------------------
  // Assemblers (per cluster: two buffers)
  // -------------------------------------------------------------------------------------------
  logic [NUM_CH-1:0][1:0][T_MAX-1:0] [15:0][31:0] buf_q;     // datapath, no reset
  logic [NUM_CH-1:0][1:0][15:0]                   own_q;     // lanes of the group's rows
  logic [NUM_CH-1:0][1:0][15:0]                   tail_q;    // zero-tail lanes (row N-1 in group)
  logic [NUM_CH-1:0][1:0][SPM_AW-1:0]             line_q;    // SPM line of the next job
  logic [NUM_CH-1:0][1:0]                         full_q;    // group complete, jobs pending
  logic [NUM_CH-1:0]                              wsel_q;    // buffer being filled
  logic [NUM_CH-1:0]                              fillv_q;   // ... holds >= 1 row
  logic [NUM_CH-1:0]                              esel_q;    // buffer being emitted
  logic [NUM_CH-1:0][TKW-1:0]                     etk_q;     // token of its next job
  logic [NUM_CH-1:0]                              rdy_q;     // res_ready
  logic [NUM_CH-1:0]                              got_last_q;

  // -------------------------------------------------------------------------------------------
  // Arbiter, job, credits
  // -------------------------------------------------------------------------------------------
  logic [CW-1:0]     sel_q, ptr_q;   // picked cluster / last taken cluster
  logic              sel_v_q;
  logic [CNT_W-1:0]  cnt_q;          // jobs taken and not yet written (accepted by the SPM)
  logic              space_q;        // cnt_q < WQ_D
  logic              iss_s_ready;    // ACC issue slice can take a job

  logic [NUM_CH-1:0] req;
  always_comb begin
    for (int unsigned c = 0; c < NUM_CH; c++) req[c] = full_q[c][esel_q[c]];
  end

  wire take = sel_v_q & req[sel_q] & space_q & (~acc_q | iss_s_ready);

  // job of the picked cluster: buffer esel, token etk
  logic              jb;
  logic [TKW-1:0]    jt;
  logic [SPM_AW-1:0] j_line;
  logic [15:0]       j_own, j_tail;
  line_t             j_y;
  always_comb begin
    jb     = esel_q[sel_q];
    jt     = etk_q[sel_q];
    j_line = line_q[sel_q][jb];
    j_own  = own_q[sel_q][jb];
    j_tail = tail_q[sel_q][jb];
    j_y    = buf_q[sel_q][jb][jt];
  end

  // round robin over req; the cluster taken in this cycle gets the lowest priority
  logic [CW-1:0] ptr_n, pick;
  logic          pick_v;
  logic [CW:0]   rr_sum;             // ptr_n + i < 2 * NUM_CH
  logic [CW-1:0] rr_idx;             // (ptr_n + i) mod NUM_CH
  always_comb begin
    ptr_n  = take ? sel_q : ptr_q;
    pick   = '0;
    pick_v = 1'b0;
    rr_sum = '0;
    rr_idx = '0;
    for (int unsigned i = NUM_CH; i >= 1; i--) begin    // i = 1 (after ptr_n) has priority
      rr_sum = {1'b0, ptr_n} + (CW+1)'(i);
      rr_idx = CW'((rr_sum >= (CW+1)'(NUM_CH)) ? rr_sum - (CW+1)'(NUM_CH) : rr_sum);
      if (req[rr_idx]) begin
        pick   = rr_idx;
        pick_v = 1'b1;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Per-cluster next state
  // -------------------------------------------------------------------------------------------
  logic [NUM_CH-1:0]       acc_row, cmpl, tk_take, tk_last, wsel_n;
  logic [NUM_CH-1:0][1:0]  full_n;
  logic [NUM_CH-1:0][3:0]  lane;
  logic [NUM_CH-1:0]       is_nm1;
  always_comb begin
    for (int unsigned c = 0; c < NUM_CH; c++) begin
      lane[c]    = res_row[c][3:0];
      acc_row[c] = res_valid[c] & rdy_q[c];
      cmpl[c]    = acc_row[c] & ((lane[c] == 4'hF) | res_last[c]);
      is_nm1[c]  = (res_row[c] == nm1_q);
      tk_take[c] = take & (sel_q == CW'(c));
      tk_last[c] = (etk_q[c] == tm1_q);
      full_n[c]  = full_q[c];
      if (tk_take[c] && tk_last[c]) full_n[c][esel_q[c]] = 1'b0;
      if (cmpl[c])                  full_n[c][wsel_q[c]] = 1'b1;
      wsel_n[c]  = wsel_q[c] ^ cmpl[c];
    end
  end

  // all groups written: nothing assembled, nothing pending, every expected last row seen
  wire all_last  = &(got_last_q | ~exp_q);
  wire all_empty = ~|full_q & ~|fillv_q & (cnt_q == '0);
  wire fin       = act_q & all_last & all_empty;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      act_q      <= 1'b0;
      acc_q      <= 1'b0;
      tm1_q      <= '0;
      nm1_q      <= '0;
      dst_q      <= '0;
      stride_q   <= '0;
      exp_q      <= '0;
      done_q     <= 1'b0;
      full_q     <= '0;
      wsel_q     <= '0;
      fillv_q    <= '0;
      esel_q     <= '0;
      etk_q      <= '0;
      rdy_q      <= '0;
      got_last_q <= '0;
      sel_q      <= '0;
      ptr_q      <= '0;
      sel_v_q    <= 1'b0;
    end else begin
      done_q  <= fin;
      sel_q   <= pick;
      sel_v_q <= pick_v;
      ptr_q   <= ptr_n;
      if (start) begin
        act_q      <= 1'b1;
        acc_q      <= acc;
        tm1_q      <= TKW'(t - 4'd1);
        nm1_q      <= n - 32'd1;
        dst_q      <= dst_line;
        stride_q   <= dst_stride[SPM_AW-1:0];
        exp_q      <= expect_ch;
        full_q     <= '0;
        wsel_q     <= '0;
        fillv_q    <= '0;
        esel_q     <= '0;
        etk_q      <= '0;
        rdy_q      <= '1;
        got_last_q <= '0;
      end else begin
        if (fin) act_q <= 1'b0;
        full_q <= full_n;
        wsel_q <= wsel_n;
        for (int unsigned c = 0; c < NUM_CH; c++) begin
          rdy_q[c] <= act_q & ~fin & ~full_n[c][wsel_n[c]];
          if (acc_row[c]) fillv_q[c] <= ~cmpl[c];
          if (acc_row[c] && res_last[c]) got_last_q[c] <= 1'b1;
          if (tk_take[c]) begin
            etk_q[c] <= tk_last[c] ? '0 : etk_q[c] + 1'b1;
            if (tk_last[c]) esel_q[c] <= ~esel_q[c];
          end
        end
      end
    end
  end

  // Datapath of the assemblers (no reset; qualified by full / fillv).
  always_ff @(posedge clk) begin
    for (int unsigned c = 0; c < NUM_CH; c++) begin
      if (acc_row[c]) begin
        for (int unsigned tk = 0; tk < T_MAX; tk++) buf_q[c][wsel_q[c]][tk][lane[c]] <= res_y[c][tk];
        own_q[c][wsel_q[c]]  <= (fillv_q[c] ? own_q[c][wsel_q[c]] : 16'h0) | (16'h1 << lane[c]);
        tail_q[c][wsel_q[c]] <= is_nm1[c] ? (16'hFFFE << lane[c]) : 16'h0;
        if (!fillv_q[c]) line_q[c][wsel_q[c]] <= dst_q + SPM_AW'(res_row[c] >> 4);
      end
      if (tk_take[c]) line_q[c][esel_q[c]] <= line_q[c][esel_q[c]] + stride_q;
    end
  end

  // -------------------------------------------------------------------------------------------
  // ACC path: issue slice -> SPM read -> old line register -> 16 x tsr_fp_add
  // -------------------------------------------------------------------------------------------
  logic              iss_v;
  logic [IS_W-1:0]   iss_d;
  logic              rp_v;
  logic [IS_W-1:0]   rp_d;
  line_t             old_q;
  logic [15:0]       sum_v;
  line_t             sum_y;
  logic              ap_v;
  logic [MT_W-1:0]   ap_d;

  tsr_rslice #(.W(IS_W)) u_iss (
    .clk, .rst_n,
    .s_valid(take & acc_q), .s_ready(iss_s_ready), .s_data({j_line, j_own, j_tail, j_y}),
    .m_valid(iss_v), .m_ready(spm_rd_req_ready), .m_data(iss_d)
  );
  assign spm_rd_req_valid = iss_v;
  assign spm_rd_req_line  = iss_d[IS_W-1 -: SPM_AW];
  wire rd_fire = iss_v & spm_rd_req_ready;

  // job of each read, aligned with its response + 1 (old line register)
  tsr_vpipe #(.VW(1), .W(IS_W), .DEPTH(SPM_RD_LAT + 1)) u_rdp (
    .clk, .rst_n, .in_valid(rd_fire), .in_data(iss_d), .out_valid(rp_v), .out_data(rp_d)
  );
  always_ff @(posedge clk) old_q <= spm_rd_rsp_data;

  line_t rp_y;
  assign rp_y = rp_d[TSR_LINE_BITS-1:0];
  for (genvar l = 0; l < 16; l++) begin : g_add
    tsr_fp_add #(.LAT(ADD_LAT)) u_add (
      .clk, .rst_n, .ce(1'b1), .in_valid(rp_v), .a(old_q[l]), .b(rp_y[l]), .sub(1'b0),
      .out_valid(sum_v[l]), .y(sum_y[l])
    );
  end
  tsr_vpipe #(.VW(1), .W(MT_W), .DEPTH(ADD_LAT)) u_addp (
    .clk, .rst_n, .in_valid(rp_v), .in_data(rp_d[IS_W-1 -: MT_W]), .out_valid(ap_v), .out_data(ap_d)
  );

  // -------------------------------------------------------------------------------------------
  // Write queue: WQ_D entries + output register (spm_wr_* from flops)
  // -------------------------------------------------------------------------------------------
  logic [SPM_AW-1:0] ap_line;
  logic [15:0]       ap_own, ap_tail;
  assign {ap_line, ap_own, ap_tail} = ap_d;

  logic              push;
  logic [WQ_W-1:0]   push_d;
  always_comb begin
    if (acc_q) begin
      push   = ap_v;
      push_d = {ap_line, lane_be(ap_own | ap_tail), lane_mask(sum_y, ap_own)};
    end else begin
      push   = take;
      push_d = {j_line, lane_be(j_own | j_tail), lane_mask(j_y, j_own)};
    end
  end

  logic [WQ_W-1:0]   wq_mem [WQ_D];
  logic [WQ_AW:0]    wq_wp_q, wq_rp_q;     // with wrap bit
  logic              out_v_q;
  logic [WQ_W-1:0]   out_q;

  wire pop     = out_v_q & spm_wr_ready;
  wire wq_ne   = (wq_wp_q != wq_rp_q);
  wire out_ld  = wq_ne & (~out_v_q | pop);
  wire [CNT_W-1:0] cnt_n = cnt_q + CNT_W'(take) - CNT_W'(pop);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wq_wp_q <= '0;
      wq_rp_q <= '0;
      out_v_q <= 1'b0;
      cnt_q   <= '0;
      space_q <= 1'b1;
    end else begin
      if (push)   wq_wp_q <= wq_wp_q + 1'b1;
      if (out_ld) wq_rp_q <= wq_rp_q + 1'b1;
      if (out_ld)   out_v_q <= 1'b1;
      else if (pop) out_v_q <= 1'b0;
      cnt_q   <= cnt_n;
      space_q <= (cnt_n < CNT_W'(WQ_D));
    end
  end

  always_ff @(posedge clk) begin
    if (push)   wq_mem[wq_wp_q[WQ_AW-1:0]] <= push_d;
    if (out_ld) out_q <= wq_mem[wq_rp_q[WQ_AW-1:0]];
  end

  assign spm_wr_valid = out_v_q;
  assign {spm_wr_line, spm_wr_be, spm_wr_data} = out_q;

  assign done      = done_q;
  assign res_ready = rdy_q;

  wire unused_ok = &{1'b0, dst_stride[31:SPM_AW], sum_v};

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  logic [NUM_CH-1:0][31:0] a_next_q;   // next expected row per cluster
  logic [NUM_CH-1:0]       a_seen_q;   // a row of this MVM was seen
  logic                    a_rsv_q;    // SPM read response valid, one cycle late
  always_ff @(posedge clk) begin
    a_rsv_q <= rst_n & spm_rd_rsp_valid;
    if (!rst_n || start) begin
      a_seen_q <= '0;
    end else begin
      for (int unsigned c = 0; c < NUM_CH; c++) begin
        if (acc_row[c]) begin
          a_seen_q[c] <= 1'b1;
          a_next_q[c] <= res_row[c] + 32'd1;
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (start) begin
        assert (!act_q) else $error("%m: start while an MVM is held");
        assert (t >= 4'd1 && 32'(t) <= T_MAX) else $error("%m: t = %0d out of [1, T_MAX]", t);
        assert (n != 32'd0) else $error("%m: start with N = 0");
        assert (expect_ch[0]) else $error("%m: channel 0 must have rows");
      end
      for (int unsigned c = 0; c < NUM_CH; c++) begin
        if (acc_row[c]) begin
          assert (act_q) else $error("%m: result of cluster %0d outside an MVM", c);
          assert (exp_q[c] && !got_last_q[c])
            else $error("%m: unexpected result from cluster %0d (expected %0b, last seen %0b)", c,
                        exp_q[c], got_last_q[c]);
          assert (res_row[c] <= nm1_q) else $error("%m: cluster %0d row %0d >= N", c, res_row[c]);
          assert (!a_seen_q[c] || res_row[c] == a_next_q[c])
            else $error("%m: cluster %0d row %0d, expected %0d", c, res_row[c], a_next_q[c]);
          assert (!is_nm1[c] || res_last[c]) else $error("%m: row N-1 without res_last");
        end
      end
      if (act_q && !acc_q) begin
        assert (!iss_v && !rp_v) else $error("%m: ACC read without ACC");
      end
      assert (rp_v == a_rsv_q) else $error("%m: SPM read response not aligned with the ACC read pipe");
      assert (!(push && (wq_wp_q[WQ_AW-1:0] == wq_rp_q[WQ_AW-1:0]) && (wq_wp_q[WQ_AW] != wq_rp_q[WQ_AW])))
        else $error("%m: write queue overflow");
      assert (cnt_q <= CNT_W'(WQ_D)) else $error("%m: credit count overflow");
      if (acc_q) begin
        assert (sum_v == {16{ap_v}}) else $error("%m: fp adder valid misaligned");
      end
      if (pop) begin
        assert (act_q) else $error("%m: SPM write outside an MVM");
      end
    end
  end
`endif

endmodule
