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
// Microarchitecture (M3 part C: per-cluster assemblers + central part, docs/rtl-m3.md §14):
//
//   start + fields ─► per cluster: tsr_vpipe(JOB_PIPE, keep) ─► assembler c (tsr_mvm_gasm)
//   res[c] ──────────────────────────────────────────────────► assembler c: two buffers
//                │  (ping-pong) of T_MAX x 16 lanes x 32 bits; fill buffer wsel: row n -> lane
//                │  n%16 of every token; the group is complete at lane 15 or res_last; per
//                │  buffer: lane mask own, tail mask, SPM line of the next job (dst + n/16 at the
//                │  group's first row, + stride per emitted token)
//                ▼  emit buffer esel: one job per token tk < t, {last, line, own, tail, y & own}
//              tsr_rpipe(JOB_PIPE) per cluster (valid / ready)
//                ▼
//   central: round robin (registered pick sel_q over the job heads, last taken cluster lowest)
//      ── take ──► non-ACC: {line, be(own | tail), y} -> write queue
//                  ACC    : job -> issue slice (tsr_rslice) -> SPM read -> pipe SPM_RD_LAT + 1
//                           (old line registered) -> 16 tsr_fp_add (old + y) -> {line, be,
//                           own ? sum : 0} -> write queue
//   write queue (WQ_D entries + output register) ──► spm_wr_* (valid, line, be, data from flops)
//
//   - The assemblers hold everything that grows with T_MAX (NUM_CH x 2 x T_MAX x 512 flops); a
//     job is a 2 x T_MAX : 1 select from flops with registered selects (esel / etk), local to its
//     cluster. JOB_PIPE (tsr_mvm: = RES_PIPE) puts register slices between each assembler and
//     the central part, so on the U200 an assembler sits next to its cluster (another SLR) and
//     only jobs cross. The central part selects among NUM_CH pipe heads (one NUM_CH : 1 select).
//     Before M3 part C the job data was one NUM_CH x 2 x T_MAX : 1 select at the centre: 4 ch /
//     T_MAX 8 routing congestion and -0.8 ns there on the U200 (docs/rtl-m3.md §14).
//   - An assembler takes rows only after its start arrived (res_ready = 0 until then; rows of a
//     new MVM wait in the result pipes), and it is busy from that start to its last job (the job
//     carrying the cluster's last row, `last`); clusters without rows (expect_ch[c] = 0) stay idle.
//   - Credits (R4): cnt_q counts jobs from the take to the accepted SPM write; a take needs
//     cnt < WQ_D (registered space_q), so the write queue never overflows and the ACC read
//     response (no backpressure) always has room. WQ_D >= the credit loop (take -> issue -> read
//     -> SPM_RD_LAT + 1 -> ADD_LAT -> queue -> output register -> pop -> space_q), so the gather
//     sustains one job per cycle when the SPM write port accepts one line per cycle.
//   - done: registered pulse when every expected cluster's last job has been taken and every
//     taken job's SPM write was accepted (cnt_q = 0).
//   - Hazards (docs/rtl-m3.md §6.3): two jobs on one SPM line belong to different clusters with
//     disjoint lane sets; each writes only its lanes (byte enables), so any order of the ACC
//     reads and writes gives the ex_mvm result. Different tokens never share a line (stride >=
//     y_lines when t > 1; otherwise undefined).
//   - res_ready[c] (registered, in the assembler) = active and the buffer the next row goes to
//     is not full. A buffer is full from its group's completion to the emit of its last token.
//   - Logic depth: take = flops through 2-3 LUTs; job data = NUM_CH:1 select of the pipe heads
//     (with JOB_PIPE = 0 of the assemblers' job outputs: 2 x T_MAX : 1 of flops + mask), then the
//     write queue (distributed RAM); write-queue output from an output register reloaded whenever
//     it is free. The ACC adders take a registered old line.
//   Latency (ideal SPM, one requester, JOB_PIPE = 0): non-ACC row -> SPM write accept: completion
//     edge, pick 1, take 1, queue 1, output register 1; ACC adds the issue slice, SPM_RD_LAT + 1
//     and ADD_LAT; JOB_PIPE adds its stages.
//   Measured (tb/mvm/test_mvm_gather.py, JOB_PIPE = 0, results offered every cycle, SPM always
//     ready): 1.000 line writes per cycle, plain and ACC, whenever the write port is the
//     bottleneck (16 clusters x T 8, SPM_RD_LAT 4, ADD_LAT 8); otherwise the source rate (NUM_CH *
//     T / 16 lines per cycle) is met exactly. First write 19 cycles after a line's first row (16
//     rows at one per cycle + 3); ACC 23 / 27 / 28 / 33 cycles at (SPM_RD_LAT, ADD_LAT) = (1,1) /
//     (3,3) / (2,5) / (4,8). Fixed-priority arbitration instead of the round robin halves the
//     rate (0.50).
//   Cost: assembly buffers NUM_CH x 2 x T_MAX x 512 FF (+ masks / lines 2 x 48 FF per cluster),
//     job pipes JOB_PIPE x 2 x (49 + SPM_AW + 512) FF per cluster, start pipes JOB_PIPE x (68 +
//     2 SPM_AW) FF per cluster, ACC issue slice 1.1 K FF, ACC read pipe 2.8 K FF (no reset,
//     SRL-capable), old line 512 FF, write queue WQ_D x 592 bits (distributed RAM) + 593 FF output
//     register, 16 tsr_fp_add. Measured before the split (Vivado 2024.2 synthesis, docs/rtl-m3.md
//     §14): 4 ch T_MAX 1 11.0 K LUT / 12.4 K FF, 4 ch T_MAX 8 24.1 K / 41.1 K.

// The assembler below lives in this file (one filelist entry, like tsr_rslice in tsr_mem_arb.sv);
// targeted waiver for the file-name rule only.
/* verilator lint_off DECLFILENAME */
// ================================================================================================
// tsr_mvm_gasm — assembler of one cluster (instantiated by tsr_mvm_gather only).
//   start (pulse) with active (= the cluster has rows) and the MVM fields of that cycle: tm1 =
//   t - 1, nm1 = N - 1, dst, stride (registered here). Rows res_* in order (R1/R2); jobs out
//   (valid / ready, R1/R2): for each complete group, one job per token tk = 0 .. t-1 in order,
//   job = {last (the cluster's final job), line (dst + tk*stride + n/16), own (lanes of the
//   group's rows), tail (zero-tail lanes, row N-1 in the group), y (token tk, lanes outside own
//   forced to 0)}.
// ================================================================================================
module tsr_mvm_gasm
  import tsr_pkg::*;
#(
  parameter  int unsigned T_MAX  = TSR_DEF_T_MAX,
  parameter  int unsigned SPM_AW = 16,
  localparam int unsigned TKW    = (T_MAX > 1) ? $clog2(T_MAX) : 1,
  localparam int unsigned JW     = 1 + SPM_AW + 16 + 16 + TSR_LINE_BITS   // {last, line, own, tail, y}
) (
  input  logic                    clk,
  input  logic                    rst_n,

  input  logic                    start,
  input  logic                    active,
  input  logic [TKW-1:0]          tm1,
  input  logic [31:0]             nm1,
  input  logic [SPM_AW-1:0]       dst,
  input  logic [SPM_AW-1:0]       stride,

  input  logic                    res_valid,
  output logic                    res_ready,
  input  logic [31:0]             res_row,
  input  logic                    res_last,
  input  logic [T_MAX-1:0][31:0]  res_y,

  output logic                    job_valid,
  input  logic                    job_ready,
  output logic [JW-1:0]           job
);

  if (T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH || SPM_AW < 1) begin : g_chk
    $fatal(1, "tsr_mvm_gasm: T_MAX in [1, 8], SPM_AW >= 1");
  end

  typedef logic [15:0][31:0] line_t;

  logic              act_q;        // start (active) .. the last job emitted
  logic [TKW-1:0]    tm1_q;
  logic [31:0]       nm1_q;
  logic [SPM_AW-1:0] dst_q, stride_q;
  logic [1:0][T_MAX-1:0][15:0][31:0] buf_q;    // datapath, no reset
  logic [1:0][15:0]       own_q, tail_q;
  logic [1:0][SPM_AW-1:0] line_q;
  logic [1:0]        full_q;       // group complete, jobs pending
  logic [1:0]        fin_q;        // the group holds the cluster's last row
  logic              wsel_q, fillv_q, esel_q, rdy_q;
  logic [TKW-1:0]    etk_q;

  wire       acc_row = res_valid & rdy_q;
  wire [3:0] lane    = res_row[3:0];
  wire       cmpl    = acc_row & ((lane == 4'hF) | res_last);
  wire       is_nm1  = (res_row == nm1_q);
  wire       emit    = job_valid & job_ready;
  wire       tk_last = (etk_q == tm1_q);
  wire       jlast   = fin_q[esel_q] & tk_last;

  logic [1:0] full_n;
  logic       wsel_n;
  always_comb begin
    full_n = full_q;
    if (emit && tk_last) full_n[esel_q] = 1'b0;
    if (cmpl)            full_n[wsel_q] = 1'b1;
    wsel_n = wsel_q ^ cmpl;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      act_q   <= 1'b0;
      tm1_q   <= '0;
      nm1_q   <= '0;
      dst_q   <= '0;
      stride_q <= '0;
      full_q  <= '0;
      fin_q   <= '0;
      wsel_q  <= 1'b0;
      fillv_q <= 1'b0;
      esel_q  <= 1'b0;
      etk_q   <= '0;
      rdy_q   <= 1'b0;
    end else if (start) begin
      act_q   <= active;
      tm1_q   <= tm1;
      nm1_q   <= nm1;
      dst_q   <= dst;
      stride_q <= stride;
      full_q  <= '0;
      fin_q   <= '0;
      wsel_q  <= 1'b0;
      fillv_q <= 1'b0;
      esel_q  <= 1'b0;
      etk_q   <= '0;
      rdy_q   <= active;
    end else begin
      if (emit && jlast) act_q <= 1'b0;
      full_q <= full_n;
      wsel_q <= wsel_n;
      rdy_q  <= act_q & ~(emit & jlast) & ~full_n[wsel_n];
      if (acc_row) fillv_q <= ~cmpl;
      if (cmpl)    fin_q[wsel_q] <= res_last;
      if (emit) begin
        etk_q <= tk_last ? '0 : etk_q + 1'b1;
        if (tk_last) esel_q <= ~esel_q;
      end
    end
  end

  // Datapath (no reset; qualified by full / fillv). A row is written into the fill buffer while
  // it is not full; a job reads the emit buffer while it is full, so the two never meet.
  always_ff @(posedge clk) begin
    if (acc_row) begin
      for (int unsigned tk = 0; tk < T_MAX; tk++) buf_q[wsel_q][tk][lane] <= res_y[tk];
      own_q[wsel_q]  <= (fillv_q ? own_q[wsel_q] : 16'h0) | (16'h1 << lane);
      tail_q[wsel_q] <= is_nm1 ? (16'hFFFE << lane) : 16'h0;
      if (!fillv_q) line_q[wsel_q] <= dst_q + SPM_AW'(res_row >> 4);
    end
    if (emit) line_q[esel_q] <= line_q[esel_q] + stride_q;
  end

  line_t jy;
  always_comb begin
    for (int unsigned l = 0; l < 16; l++) jy[l] = own_q[esel_q][l] ? buf_q[esel_q][etk_q][l] : 32'd0;
  end
  assign job_valid = full_q[esel_q];
  assign job       = {jlast, line_q[esel_q], own_q[esel_q], tail_q[esel_q], jy};
  assign res_ready = rdy_q;

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks (rows in order, inside an MVM, last on row N-1)
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q, a_seen_q, a_got_last_q;
  logic [31:0] a_next_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      a_seen_q     <= 1'b0;
      a_got_last_q <= 1'b0;
    end else if (acc_row) begin
      a_seen_q <= 1'b1;
      a_next_q <= res_row + 32'd1;
      if (res_last) a_got_last_q <= 1'b1;
    end
  end
  always_ff @(posedge clk) begin
    if (rst_n && a_on_q && !start) begin
      if (acc_row) begin
        assert (act_q && !a_got_last_q) else $error("%m: row %0d outside this cluster's MVM", res_row);
        assert (res_row <= nm1_q) else $error("%m: row %0d >= N", res_row);
        assert (!a_seen_q || res_row == a_next_q) else $error("%m: row %0d, expected %0d", res_row, a_next_q);
        assert (!is_nm1 || res_last) else $error("%m: row N-1 without res_last");
      end
      assert (!(full_q[0] && full_q[1] && rdy_q)) else $error("%m: ready with both buffers full");
    end
  end
`endif

endmodule
/* verilator lint_on DECLFILENAME */

// ================================================================================================
module tsr_mvm_gather
  import tsr_pkg::*;
#(
  parameter  int unsigned NUM_CH     = TSR_DEF_NUM_CH,
  parameter  int unsigned T_MAX      = TSR_DEF_T_MAX,
  parameter  int unsigned SPM_BYTES  = TSR_DEF_SPM_BYTES,
  parameter  int unsigned SPM_RD_LAT = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned ADD_LAT    = TSR_DEF_FADD_LAT,
  parameter  int unsigned JOB_PIPE   = 0,     // register slices assembler c -> central (implementation)
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
  localparam int unsigned JW    = 1 + IS_W;                                // {last, line, own, tail, y}
  localparam int unsigned SW    = 1 + TKW + 32 + 2 * SPM_AW;               // {active, tm1, nm1, dst, stride}

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
  // MVM configuration (registered at start; the assemblers get their own copies)
  // -------------------------------------------------------------------------------------------
  logic              act_q;          // holding an MVM (start .. done)
  logic              acc_q;
  logic [NUM_CH-1:0] exp_q;          // channels with rows
  logic [NUM_CH-1:0] got_last_q;     // the cluster's last job was taken
  logic              done_q;

  // -------------------------------------------------------------------------------------------
  // Per cluster: start pipe -> assembler -> job pipe
  // -------------------------------------------------------------------------------------------
  logic [NUM_CH-1:0]          jv, jr;       // job pipe heads: valid, ready (= take of that cluster)
  logic [NUM_CH-1:0][JW-1:0]  jd;

  wire [TKW-1:0]    s_tm1    = TKW'(t - 4'd1);
  wire [31:0]       s_nm1    = n - 32'd1;
  wire [SPM_AW-1:0] s_stride = dst_stride[SPM_AW-1:0];   // dst + tk * stride < SPM lines for tk < t

  for (genvar c = 0; c < NUM_CH; c++) begin : g_asm
    logic              a_start;
    logic [SW-1:0]     a_f;
    logic              a_jv, a_jr;
    logic [JW-1:0]     a_jd;
    // Identical across the clusters but for the active bit: kept separate (each next to its
    // assembler, which may sit in another SLR than the central part).
    tsr_vpipe #(.VW(1), .W(SW), .DEPTH(JOB_PIPE), .KEEP(1'b1)) u_sp (
      .clk, .rst_n, .in_valid(start), .in_data({expect_ch[c], s_tm1, s_nm1, dst_line, s_stride}),
      .out_valid(a_start), .out_data(a_f)
    );
    tsr_mvm_gasm #(.T_MAX(T_MAX), .SPM_AW(SPM_AW)) u_asm (
      .clk, .rst_n,
      .start(a_start), .active(a_f[SW-1]), .tm1(a_f[SW-2 -: TKW]), .nm1(a_f[2*SPM_AW +: 32]),
      .dst(a_f[SPM_AW +: SPM_AW]), .stride(a_f[0 +: SPM_AW]),
      .res_valid(res_valid[c]), .res_ready(res_ready[c]), .res_row(res_row[c]),
      .res_last(res_last[c]), .res_y(res_y[c]),
      .job_valid(a_jv), .job_ready(a_jr), .job(a_jd)
    );
    tsr_rpipe #(.W(JW), .DEPTH(JOB_PIPE)) u_jp (
      .clk, .rst_n, .s_valid(a_jv), .s_ready(a_jr), .s_data(a_jd),
      .m_valid(jv[c]), .m_ready(jr[c]), .m_data(jd[c])
    );
  end

  // -------------------------------------------------------------------------------------------
  // Arbiter, job, credits
  // -------------------------------------------------------------------------------------------
  logic [CW-1:0]     sel_q, ptr_q;   // picked cluster / last taken cluster
  logic              sel_v_q;
  logic [CNT_W-1:0]  cnt_q;          // jobs taken and not yet written (accepted by the SPM)
  logic              space_q;        // cnt_q < WQ_D
  logic              iss_s_ready;    // ACC issue slice can take a job

  wire take = sel_v_q & jv[sel_q] & space_q & (~acc_q | iss_s_ready);

  // job of the picked cluster: the head of its job pipe
  logic              j_last;
  logic [SPM_AW-1:0] j_line;
  logic [15:0]       j_own, j_tail;
  line_t             j_y;            // lanes outside own already 0 (assembler)
  assign {j_last, j_line, j_own, j_tail, j_y} = jd[sel_q];

  always_comb begin
    for (int unsigned c = 0; c < NUM_CH; c++) jr[c] = take & (sel_q == CW'(c));
  end

  // round robin over the job heads; the cluster taken in this cycle gets the lowest priority
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
      if (jv[rr_idx]) begin
        pick   = rr_idx;
        pick_v = 1'b1;
      end
    end
  end

  // all written: every expected cluster's last job taken, every taken job written
  wire all_last = &(got_last_q | ~exp_q);
  wire fin      = act_q & all_last & (cnt_q == '0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      act_q      <= 1'b0;
      acc_q      <= 1'b0;
      exp_q      <= '0;
      got_last_q <= '0;
      done_q     <= 1'b0;
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
        exp_q      <= expect_ch;
        got_last_q <= '0;
      end else begin
        if (fin) act_q <= 1'b0;
        if (take && j_last) got_last_q[sel_q] <= 1'b1;
      end
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
      push_d = {j_line, lane_be(j_own | j_tail), j_y};
    end
  end

  logic [WQ_W-1:0]   wq_mem [WQ_D];
  logic [WQ_AW:0]    wq_wp_q, wq_rp_q;     // with wrap bit
  logic              out_v_q;
  logic [WQ_W-1:0]   out_q;

  wire pop      = out_v_q & spm_wr_ready;
  wire wq_ne    = (wq_wp_q != wq_rp_q);
  wire out_free = ~out_v_q | pop;         // output register free after this edge
  wire out_ld   = wq_ne & out_free;
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

  // out_q reloads whenever the output register is free, not only on out_ld (M3 part C timing,
  // docs/rtl-m3.md §14: its 592-flop clock enable no longer waits for the queue pointer compare):
  // while out_v_q = 0 it holds don't-care data; whenever out_v_q = 1 it holds exactly the entry
  // out_ld loaded, held until pop as before.
  always_ff @(posedge clk) begin
    if (push)     wq_mem[wq_wp_q[WQ_AW-1:0]] <= push_d;
    if (out_free) out_q <= wq_mem[wq_rp_q[WQ_AW-1:0]];
  end

  assign spm_wr_valid = out_v_q;
  assign {spm_wr_line, spm_wr_be, spm_wr_data} = out_q;

  assign done = done_q;

  wire unused_ok = &{1'b0, dst_stride[31:SPM_AW], sum_v};

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks (row order / range per cluster: tsr_mvm_gasm)
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  logic a_rsv_q;    // SPM read response valid, one cycle late
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;
  always_ff @(posedge clk) a_rsv_q <= rst_n & spm_rd_rsp_valid;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (start) begin
        assert (!act_q) else $error("%m: start while an MVM is held");
        assert (t >= 4'd1 && 32'(t) <= T_MAX) else $error("%m: t = %0d out of [1, T_MAX]", t);
        assert (n != 32'd0) else $error("%m: start with N = 0");
        assert (expect_ch[0]) else $error("%m: channel 0 must have rows");
      end
      for (int unsigned c = 0; c < NUM_CH; c++) begin
        if (jv[c]) begin
          assert (act_q && exp_q[c] && !got_last_q[c])
            else $error("%m: job of cluster %0d outside its MVM (act %0b expected %0b last taken %0b)", c,
                        act_q, exp_q[c], got_last_q[c]);
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
      if (done_q) begin
        assert (jv == '0) else $error("%m: job pending at done");
      end
    end
  end
`endif

endmodule
