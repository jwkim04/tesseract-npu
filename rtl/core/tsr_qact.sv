// tsr_qact.sv — QACT unit: T f32 vectors in the SPM -> T QA vectors (A32 or A256) in the SPM.
//
// Spec: docs/rtl-core.md §2, block (e); dispatch §3.3, numerics §8.1.
//
// Dispatch: cmd_valid/cmd_ready accept one tsr_qact_cmd_t (cmd_ready = idle, from the state
//   register). Operands are legal (the dispatch checks K, T and both SPM ranges; src_stride already
//   resolved; qa_lines = QA_LINES(kind, K)). `done` is a registered 1-cycle pulse in the cycle after
//   the edge at which tsr_spm accepted the last QA line write of the instruction (never in the
//   accepting cycle). At that point the unit is quiescent: every SPM read has returned, every
//   internal queue is empty, both dividers are idle.
// Semantics: bit-exact tsr_qact() (sw/sim/tsr_ref.c) per token t = 0..T-1, as tsr_sim.c ex_qact:
//   source f32 x[0..K-1] at SPM line src_line + t*src_stride (16 f32 per line, little-endian);
//   destination QA image at dst_line + t*qa_lines: QS (ceil(K/64) lines, int8), then D
//   (ceil(4*nblk/64) lines, f32 per block), then BS (A256 only, ceil(2*K/16/64) lines, int16 per
//   16 q). Every line of the image is written exactly once, in full; bytes after the last valid
//   element of each region are written as 0 (spec, Implementation clarifications, "Tail bytes").
//   A32 per 32-block : amax = max |DAZ(x)| over non-NaN x (0 if none); d = fdiv(amax, 127);
//                      id = (d != +-0) ? fdiv(1, d) : +0; q = q8_round_away(fmul(x, id));
//                      D = fp16_to_f32(f32_to_fp16(d)).
//   A256 per 256-block: max = DAZ(x) of the FIRST non-NaN element with the largest |DAZ(x)|;
//                      if amax == 0: d = +0, q = 0; else iscale = fdiv(-127, max),
//                      q = q8_round_even(fmul(iscale, x)), d = fdiv(1, iscale);
//                      BS[j] = sum of q[16j .. 16j+15] (int16).
//   fdiv/fmul/q8_round_*/amax search are tsr_fp_div / tsr_fp_mul / tsr_fp_pkg (block (c)).
//   dst and src never overlap in legal programs (spec: undefined otherwise; this unit reads ahead
//   of its writes, so an overlapping program gets no particular result).
// SPM side: client TSR_SPM_CL_QACT of tsr_spm. Reads: responses arrive SPM_RD_LAT cycles after
//   the accepting edge without backpressure; buffer space is reserved before a request is raised
//   (docs/rtl-core.md R4). Writes: full lines (be all ones), valid/line/data held until wr_ready.
//   Every output is a register (no combinational input->output path, R5).
//
// Microarchitecture (one instruction at a time; all stages in order, no reordering anywhere):
//
//   S1 reader     raises one SPM read per cycle (request register rd_v/rd_line, R2) for the source
//                 lines token by token. A line is requested only when a line-buffer slot is
//                 reserved (buf_res < BUF_LINES), and the first line of a block only when an
//                 amax-queue entry is reserved (aq_res < AQ_DEPTH).
//   RSP           the response line is registered (rs_*: the SPM read bus is wide and shared),
//                 then written into the line buffer (tsr_sram_1r1w, BUF_LINES x 512, ring) and fed
//                 to the amax search.
//   AMAX          tsr_f32_amax_cand of the 16 elements, then a 4-level tsr_f32_amax_merge tree
//                 (one level per register stage, element order kept: the merge is associative, so
//                 the tree equals the reference's sequential first-largest scan), then a running
//                 merge over the lines of the block. The block result {mag, neg} is pushed into
//                 the amax queue AQ (AQ_DEPTH entries) with the block's last line.
//   S2 dividers   two tsr_fp_div in series, each holding one block (slot registers x_* / y_*):
//                   A32 : DA = amax / 127 = d,       DB = 1 / d = id      (id := +0 if d == +-0)
//                   A256: DA = -127 / max = iscale,  DB = 1 / iscale = d  (d := +0 if amax == 0)
//                 DA works on block i+1 while DB works on block i: one block per 16 cycles (the
//                 15-cycle divide + the handover). f32_to_fp16(d) (A32) is a free-running register
//                 behind slot Y (y_h16), settled long before DB finishes.
//   S3 quantizer  per block: {scale, d, zero} is loaded from DB / slot Y (cur_*), then the block's
//                 lines are read from the line buffer, one per cycle, while a processed-line queue
//                 credit is free (pq_cred: PQ_DEPTH minus lines in flight or queued). Fixed-latency
//                 pipeline, never stalled (16 x tsr_fp_mul with ce = 1):
//                   I0 buffer read | I1 multiplier inputs (x_j, scale) | +MUL_LAT products |
//                   Q8 tsr_f32_to_q8 (half-even for A256; 0 for an all-zero A256 block) |
//                   BS1 4 sums of 4 q | push {q[16], BS = sum of the 16 q, d} into PQ.
//                 One source line = 16 q = a quarter QS line and exactly one BS value.
//   W writer      pops PQ in order and assembles QS (4 lines / QS line), BS (32 / BS line) and D
//                 (16 blocks / D line) in three line registers; a slot-0 write clears the rest of
//                 the line (zero tails). A completed line becomes a pending write (p_qs/p_bs/p_d,
//                 with its SPM line) and goes through the output register out_* (priority QS, BS,
//                 D). A new PQ entry is popped only when no other write is pending (or the only one
//                 leaves this cycle with out_* empty), so the assembly registers are never
//                 overwritten before they are written out.
//   done          registered at the edge that transfers the last write (last line of the last
//                 token popped, nothing pending, out_* accepted): high in the cycle after it.
//
// Throughput (measured, tb/qact qact_throughput / qact_kmax; SPM always ready, defaults):
//   A256 1.10 cycles per source line (K 32768: 2248 cycles; the four buffered blocks cover the
//   amax -> DA -> DB -> quantize latency of a block); A32 16.0 cycles per 32-block (= 8 per line,
//   divider bound; K 32768: 16422 cycles). Accept -> done for one block: A256 83, A32 54 cycles.
//   docs/rtl-core.md §8.1 asks for ~2 cycles per line outside the per-block divides.
// Timing (300 MHz): every SPM-facing output is a flop; the dispatch fields and the response bus
//   are registered before any logic; one amax merge level per stage; the multiplier inputs come
//   straight from the buffer's read register and a scale register; f32_to_q8 is one stage between
//   registers; behind spm_wr_ready there is one LUT (the output register's load enable); the
//   writer's pop decision depends only on registers. Vivado 2024.2 out of context,
//   xcu200-fsgd2104-2-e, 3.333 ns, I/O delay 1.6 ns (tb/qact/viv/qact_ooc.tcl), defaults:
//     WNS +0.124 ns, WHS +0.007 ns. Worst path: synthesis merges the 16 identical scale input
//     registers of the multipliers into one that drives all 32 DSPs; with
//     -keep_equivalent_registers WNS is +0.216 ns (worst: rs_v -> the 8 BRAM write enables,
//     routing only). Deepest logic: y_keep -> y_h16 (f32_to_fp16), 9-10 levels, +0.66 ns.
//   Utilization: 5522 LUT (147 as LUTRAM: AQ, PQ), 6416 FF, 32 DSP48E2, 8 RAMB36 (line buffer;
//   64 deep is too shallow for URAM).
// Room for T_MAX 8 (M7): A32 throughput is set by the single divider pair; more pairs (round robin,
//   in order) would scale it if QACT ever matters next to MVM.

module tsr_qact
  import tsr_pkg::*;
#(
  parameter  int unsigned K_MAX      = TSR_DEF_K_MAX,
  parameter  int unsigned T_MAX      = TSR_DEF_T_MAX,
  parameter  int unsigned SPM_BYTES  = TSR_DEF_SPM_BYTES,
  parameter  int unsigned SPM_RD_LAT = TSR_DEF_SPM_RD_LAT,
  // Implementation parameters (tsr_core keeps the defaults).
  parameter  int unsigned MUL_LAT    = 4,    // tsr_fp_mul latency (4 = the 300 MHz configuration)
  parameter  int unsigned BUF_LINES  = 64,   // source line buffer: power of two >= 16 (one A256 block)
  parameter  int unsigned AQ_DEPTH   = 4,    // amax queue entries: power of two >= 2
  parameter  int unsigned PQ_DEPTH   = 16,   // processed-line queue entries: power of two >= 2
  localparam int unsigned SPM_LINES  = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned SPM_AW     = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1
) (
  input  logic                         clk,
  input  logic                         rst_n,

  // ---- dispatch (the dispatcher, Y4) ----
  input  logic                         cmd_valid,
  output logic                         cmd_ready,
  input  tsr_qact_cmd_t                cmd,
  output logic                         done,

  // ---- SPM client (tsr_spm client TSR_SPM_CL_QACT) ----
  output logic                         spm_rd_req_valid,
  input  logic                         spm_rd_req_ready,
  output logic [SPM_AW-1:0]            spm_rd_req_line,
  input  logic                         spm_rd_rsp_valid,
  input  logic [TSR_LINE_BITS-1:0]     spm_rd_rsp_data,
  output logic                         spm_wr_valid,
  input  logic                         spm_wr_ready,
  output logic [SPM_AW-1:0]            spm_wr_line,
  output logic [TSR_LINE_BITS-1:0]     spm_wr_data,
  output logic [TSR_SPM_BE_W-1:0]      spm_wr_be
);
  import tsr_fp_pkg::*;

  // -------------------------------------------------------------------------------------------
  // Configuration
  // -------------------------------------------------------------------------------------------
  localparam int unsigned NL_W   = $clog2(K_MAX / 16 + 1);  // source lines per token: K/16 <= K_MAX/16
  localparam int unsigned BUF_AW = $clog2(BUF_LINES);
  localparam int unsigned BUF_CW = BUF_AW + 1;
  localparam int unsigned AQ_AW  = $clog2(AQ_DEPTH);
  localparam int unsigned AQ_CW  = AQ_AW + 1;
  localparam int unsigned PQ_AW  = $clog2(PQ_DEPTH);
  localparam int unsigned PQ_CW  = PQ_AW + 1;
  localparam int unsigned PQ_W   = 128 + 16 + 32;           // {q[16] int8, BS int16, d f32}

  if (K_MAX < 256 || K_MAX > 65280 || (K_MAX % 256) != 0) begin : g_chk_kmax
    $fatal(1, "tsr_qact: K_MAX must be a multiple of 256 in [256, 65280]");
  end
  if (T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH) begin : g_chk_tmax
    $fatal(1, "tsr_qact: T_MAX must be in [1, 8]");
  end
  if (SPM_BYTES < TSR_LINE_BYTES || (SPM_BYTES % TSR_LINE_BYTES) != 0) begin : g_chk_spm
    $fatal(1, "tsr_qact: SPM_BYTES must be a positive multiple of 64");
  end
  if (MUL_LAT < 1 || SPM_RD_LAT < 1) begin : g_chk_lat
    $fatal(1, "tsr_qact: MUL_LAT >= 1 and SPM_RD_LAT >= 1 required");
  end
  if (BUF_LINES < 16 || (BUF_LINES & (BUF_LINES - 1)) != 0) begin : g_chk_buf
    $fatal(1, "tsr_qact: BUF_LINES must be a power of two >= 16");
  end
  if (AQ_DEPTH < 2 || (AQ_DEPTH & (AQ_DEPTH - 1)) != 0 ||
      PQ_DEPTH < 2 || (PQ_DEPTH & (PQ_DEPTH - 1)) != 0) begin : g_chk_q
    $fatal(1, "tsr_qact: AQ_DEPTH and PQ_DEPTH must be powers of two >= 2");
  end

  // Sum of four int8 (as 10-bit two's complement) and of four such sums (as int16).
  function automatic logic [9:0] sum4_s8(logic [7:0] a, logic [7:0] b, logic [7:0] c, logic [7:0] d);
    return {{2{a[7]}}, a} + {{2{b[7]}}, b} + {{2{c[7]}}, c} + {{2{d[7]}}, d};
  endfunction
  function automatic logic [15:0] sum4_s10(logic [9:0] a, logic [9:0] b, logic [9:0] c, logic [9:0] d);
    return {{6{a[9]}}, a} + {{6{b[9]}}, b} + {{6{c[9]}}, c} + {{6{d[9]}}, d};
  endfunction

  // -------------------------------------------------------------------------------------------
  // Unit state and the command registers
  // -------------------------------------------------------------------------------------------
  typedef enum logic {
    S_IDLE = 1'b0,
    S_RUN  = 1'b1
  } state_e;

  state_e st;
  wire    accept = cmd_valid && cmd_ready;
  assign  cmd_ready = (st == S_IDLE);

  logic              c_kind;      // 0: A32, 1: A256
  logic [15:0]       c_k;         // K
  logic [NL_W-1:0]   c_nlm1;      // source lines per token - 1 (K/16 - 1)
  logic [3:0]        c_tm1;       // T - 1
  logic [SPM_AW-1:0] c_stride;    // source stride (lines); only used when T >= 2
  logic [SPM_AW-1:0] c_qa;        // QA_LINES (destination stride)
  logic [SPM_AW-1:0] c_doff;      // QS lines = offset of the D region in a QA image
  logic [SPM_AW-1:0] c_bsoff;     // QS + D lines = offset of the BS region
  logic              acc_d1, acc_d2;  // 1 and 2 cycles after the accepting edge (setup)

  wire [SPM_AW-1:0] a_dst = cmd.dst_line[SPM_AW-1:0];
  wire [SPM_AW-1:0] a_src = cmd.src_line[SPM_AW-1:0];

  // The dispatch fields are registered without logic; the QA region offsets (tsr_pkg helpers, =
  // tsr_ref.c tsr_qa_*) follow in the first setup cycle, the writer's region bases in the second
  // (the writer's first PQ entry is dozens of cycles away).
  always_ff @(posedge clk) begin
    if (accept) begin
      c_kind   <= cmd.kind;
      c_k      <= cmd.k;
      c_nlm1   <= NL_W'(cmd.k[15:4] - 12'd1);
      c_tm1    <= cmd.t - 4'd1;
      c_stride <= cmd.src_stride[SPM_AW-1:0];
      c_qa     <= SPM_AW'(cmd.qa_lines);
    end
    if (acc_d1) begin
      c_doff  <= SPM_AW'(tsr_qa_qs_lines({16'd0, c_k}));
      c_bsoff <= SPM_AW'(tsr_qa_qs_lines({16'd0, c_k}) + tsr_qa_d_lines(c_kind, {16'd0, c_k}));
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      acc_d1 <= 1'b0;
      acc_d2 <= 1'b0;
    end else begin
      acc_d1 <= accept;
      acc_d2 <= acc_d1;
    end
  end

  // -------------------------------------------------------------------------------------------
  // S1: reader (SPM read requests)
  // -------------------------------------------------------------------------------------------
  logic              rd_v;
  logic [SPM_AW-1:0] rd_line;
  logic              r_act;       // source lines left to request
  logic [3:0]        r_tok;       // token being requested
  logic [NL_W-1:0]   r_g;         // line within the token
  logic [SPM_AW-1:0] r_ptr;       // SPM line of the next request
  logic [SPM_AW-1:0] r_base;      // first source line of the token
  logic [BUF_CW-1:0] buf_res;     // line-buffer slots reserved (requested, not yet read by S3)
  logic [AQ_CW-1:0]  aq_res;      // AQ entries reserved (blocks requested, not yet popped by DA)
  logic              q_issue;     // S3 reads one buffered line (frees its slot)
  logic              aq_pop;      // DA takes the AQ head

  wire               r_first  = c_kind ? (r_g[3:0] == 4'd0) : !r_g[0];
  wire               r_lastln = (r_g == c_nlm1);
  wire               r_can    = r_act && (buf_res != BUF_CW'(BUF_LINES)) &&
                                (!r_first || (aq_res != AQ_CW'(AQ_DEPTH)));
  wire               r_load   = r_can && (!rd_v || spm_rd_req_ready);
  wire [SPM_AW-1:0]  r_nbase  = r_base + c_stride;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_v    <= 1'b0;
      r_act   <= 1'b0;
      r_tok   <= '0;
      r_g     <= '0;
      r_ptr   <= '0;
      r_base  <= '0;
      buf_res <= '0;
      aq_res  <= '0;
    end else begin
      if (accept) begin
        r_act  <= 1'b1;
        r_tok  <= '0;
        r_g    <= '0;
        r_ptr  <= a_src;
        r_base <= a_src;
      end else if (r_load) begin
        if (r_lastln) begin
          r_g    <= '0;
          r_tok  <= r_tok + 4'd1;
          r_base <= r_nbase;
          r_ptr  <= r_nbase;
          if (r_tok == c_tm1) r_act <= 1'b0;
        end else begin
          r_g   <= r_g + NL_W'(1);
          r_ptr <= r_ptr + SPM_AW'(1);
        end
      end
      if (r_load)                rd_v <= 1'b1;
      else if (spm_rd_req_ready) rd_v <= 1'b0;
      buf_res <= buf_res + BUF_CW'(r_load) - BUF_CW'(q_issue);
      aq_res  <= aq_res + AQ_CW'(r_load && r_first) - AQ_CW'(aq_pop);
    end
  end

  always_ff @(posedge clk) begin
    if (r_load) rd_line <= r_ptr;
  end

  assign spm_rd_req_valid = rd_v;
  assign spm_rd_req_line  = rd_line;

  // -------------------------------------------------------------------------------------------
  // RSP: response register, line buffer write, block position
  // -------------------------------------------------------------------------------------------
  logic                     rs_v;
  logic [TSR_LINE_BITS-1:0] rs_d;
  logic [3:0]               rs_pos;   // response index mod 16 (blocks are 2 or 16 lines, aligned)
  logic [BUF_AW-1:0]        buf_wp, buf_rp;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rs_v   <= 1'b0;
      rs_pos <= '0;
      buf_wp <= '0;
    end else begin
      rs_v <= spm_rd_rsp_valid;
      if (accept)    rs_pos <= '0;
      else if (rs_v) rs_pos <= rs_pos + 4'd1;
      if (rs_v) buf_wp <= buf_wp + BUF_AW'(1);
    end
  end

  always_ff @(posedge clk) begin
    if (spm_rd_rsp_valid) rs_d <= spm_rd_rsp_data;
  end

  wire rs_first = c_kind ? (rs_pos == 4'd0)  : !rs_pos[0];
  wire rs_last  = c_kind ? (rs_pos == 4'd15) : rs_pos[0];

  logic [TSR_LINE_BITS-1:0] buf_rd;

  tsr_sram_1r1w #(
    .DEPTH(BUF_LINES), .WIDTH(TSR_LINE_BITS), .RD_LAT(1)
  ) u_buf (
    .clk,
    .we(rs_v), .waddr(buf_wp), .wdata(rs_d), .wbe({TSR_SPM_BE_W{1'b1}}),
    .re(q_issue), .raddr(buf_rp), .rdata(buf_rd)
  );

  // -------------------------------------------------------------------------------------------
  // AMAX: candidates + merge tree (one level per stage) + running merge over the block's lines
  // -------------------------------------------------------------------------------------------
  logic          a1_v, a1_f, a1_l, a2_v, a2_f, a2_l, a3_v, a3_f, a3_l, a4_v, a4_f, a4_l;
  tsr_f32_amax_t a1_m [8];
  tsr_f32_amax_t a2_m [4];
  tsr_f32_amax_t a3_m [2];
  tsr_f32_amax_t a4_m;
  tsr_f32_amax_t acc_q, acc_n;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      a1_v <= 1'b0;
      a2_v <= 1'b0;
      a3_v <= 1'b0;
      a4_v <= 1'b0;
    end else begin
      a1_v <= rs_v;
      a2_v <= a1_v;
      a3_v <= a2_v;
      a4_v <= a3_v;
    end
  end

  always_ff @(posedge clk) begin
    {a1_f, a1_l} <= {rs_first, rs_last};
    {a2_f, a2_l} <= {a1_f, a1_l};
    {a3_f, a3_l} <= {a2_f, a2_l};
    {a4_f, a4_l} <= {a3_f, a3_l};
    for (int i = 0; i < 8; i++) begin
      a1_m[i] <= tsr_f32_amax_merge(tsr_f32_amax_cand(rs_d[64*i +: 32]),
                                    tsr_f32_amax_cand(rs_d[64*i+32 +: 32]));
    end
    for (int i = 0; i < 4; i++) a2_m[i] <= tsr_f32_amax_merge(a1_m[2*i], a1_m[2*i+1]);
    for (int i = 0; i < 2; i++) a3_m[i] <= tsr_f32_amax_merge(a2_m[2*i], a2_m[2*i+1]);
    a4_m <= tsr_f32_amax_merge(a3_m[0], a3_m[1]);
    if (a4_v) acc_q <= acc_n;
  end

  assign acc_n = a4_f ? a4_m : tsr_f32_amax_merge(acc_q, a4_m);

  // -------------------------------------------------------------------------------------------
  // AQ: amax queue (block results, in order)
  // -------------------------------------------------------------------------------------------
  logic          aq_push;
  tsr_f32_amax_t aq_mem [AQ_DEPTH];
  logic [AQ_AW-1:0] aq_wp, aq_rp;
  logic [AQ_CW-1:0] aq_cnt;
  tsr_f32_amax_t aq_head;

  assign aq_push = a4_v && a4_l;
  assign aq_head = aq_mem[aq_rp];

  always_ff @(posedge clk) begin
    if (aq_push) aq_mem[aq_wp] <= acc_n;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      aq_wp  <= '0;
      aq_rp  <= '0;
      aq_cnt <= '0;
    end else begin
      if (aq_push) aq_wp <= aq_wp + AQ_AW'(1);
      if (aq_pop)  aq_rp <= aq_rp + AQ_AW'(1);
      aq_cnt <= aq_cnt + AQ_CW'(aq_push) - AQ_CW'(aq_pop);
    end
  end

  // -------------------------------------------------------------------------------------------
  // S2: dividers. Slot X = the block in DA (engine or output), slot Y = the block in DB.
  // -------------------------------------------------------------------------------------------
  logic        da_in_ready, da_out_valid, da_out_ready;
  logic        db_in_ready, db_out_valid, db_out_ready;
  logic [31:0] da_y, db_y;
  logic        x_v, x_zero;
  logic        y_v, y_zero, y_dz;
  logic [31:0] y_keep;            // DA's result: A256 iscale, A32 d
  logic [15:0] y_h16;             // f32_to_fp16(y_keep), one cycle behind y_keep

  wire         da_in_valid = (aq_cnt != '0) && !x_v;
  wire [31:0]  da_a = c_kind ? TSR_F32_M127 : {1'b0, aq_head.mag};
  wire [31:0]  da_b = c_kind ? tsr_f32_amax_val(aq_head) : TSR_F32_127;
  wire         da_acc = da_in_valid && da_in_ready;
  assign       aq_pop = da_acc;

  tsr_fp_div u_da (
    .clk, .rst_n,
    .in_valid(da_in_valid), .in_ready(da_in_ready), .a(da_a), .b(da_b),
    .out_valid(da_out_valid), .out_ready(da_out_ready), .y(da_y)
  );

  wire         db_in_valid = da_out_valid && !y_v;
  assign       da_out_ready = db_in_ready && !y_v;
  wire         ab_x = db_in_valid && db_in_ready;

  tsr_fp_div u_db (
    .clk, .rst_n,
    .in_valid(db_in_valid), .in_ready(db_in_ready), .a(TSR_F32_ONE), .b(da_y),
    .out_valid(db_out_valid), .out_ready(db_out_ready), .y(db_y)
  );

  logic blk_load;                 // S3 takes DB's result and slot Y

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      x_v <= 1'b0;
      y_v <= 1'b0;
    end else begin
      if (da_acc)    x_v <= 1'b1;
      else if (ab_x) x_v <= 1'b0;
      if (ab_x)          y_v <= 1'b1;
      else if (blk_load) y_v <= 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    if (da_acc) x_zero <= (aq_head.mag == 31'd0);
    if (ab_x) begin
      y_zero <= x_zero;
      y_dz   <= !c_kind && (da_y[30:0] == 31'd0);
      y_keep <= da_y;
    end
    // Free running: the block in slot Y is loaded by S3 only after DB's ~15-cycle divide, so
    // y_h16 always matches y_keep by then (checked in simulation).
    y_h16 <= tsr_f32_to_fp16(y_keep);
  end

  // -------------------------------------------------------------------------------------------
  // S3: quantizer (block register, line issue, fixed-latency pipeline)
  // -------------------------------------------------------------------------------------------
  logic             cur_v, cur_zero;
  logic [4:0]       cur_left;     // lines of the block still to issue
  logic [31:0]      cur_scale;    // A32: id, A256: iscale
  logic [31:0]      cur_dval;     // the block's D value
  logic [PQ_CW-1:0] pq_cred;      // PQ slots neither occupied nor reserved by lines in flight
  logic             w_pop;        // writer pops PQ

  assign q_issue  = cur_v && (pq_cred != '0);
  wire   cur_last = (cur_left == 5'd1);
  assign blk_load = db_out_valid && (!cur_v || (q_issue && cur_last));
  assign db_out_ready = blk_load;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cur_v    <= 1'b0;
      cur_left <= '0;
      buf_rp   <= '0;
      pq_cred  <= PQ_CW'(PQ_DEPTH);
    end else begin
      if (blk_load) begin
        cur_v    <= 1'b1;
        cur_left <= c_kind ? 5'd16 : 5'd2;
      end else if (q_issue) begin
        cur_left <= cur_left - 5'd1;
        if (cur_last) cur_v <= 1'b0;
      end
      if (q_issue) buf_rp <= buf_rp + BUF_AW'(1);
      pq_cred <= pq_cred - PQ_CW'(q_issue) + PQ_CW'(w_pop);
    end
  end

  always_ff @(posedge clk) begin
    if (blk_load) begin
      cur_zero  <= y_zero;
      cur_scale <= c_kind ? y_keep : (y_dz ? 32'd0 : db_y);
      cur_dval  <= c_kind ? (y_zero ? 32'd0 : db_y) : tsr_fp16_to_f32(y_h16);
    end
  end

  // I1: multiplier operands (x from the buffer's read register, scale from p1_scale).
  logic        p1_v, p1_zero;
  logic [31:0] p1_scale, p1_dval;

  always_ff @(posedge clk) begin
    if (!rst_n) p1_v <= 1'b0;
    else        p1_v <= q_issue;
  end
  always_ff @(posedge clk) begin
    p1_zero  <= cur_zero;
    p1_scale <= cur_scale;
    p1_dval  <= cur_dval;
  end

  logic [15:0]       m_v;
  logic [15:0][31:0] m_y;

  for (genvar j = 0; j < 16; j++) begin : g_mul
    tsr_fp_mul #(.LAT(MUL_LAT)) u_mul (
      .clk, .rst_n, .ce(1'b1), .in_valid(p1_v),
      .a(buf_rd[32*j +: 32]), .b(p1_scale),
      .out_valid(m_v[j]), .y(m_y[j])
    );
  end

  // Side band alongside the multipliers (MUL_LAT stages).
  logic        sd_zero [MUL_LAT];
  logic [31:0] sd_dval [MUL_LAT];

  always_ff @(posedge clk) begin
    sd_zero[0] <= p1_zero;
    sd_dval[0] <= p1_dval;
    for (int i = 1; i < int'(MUL_LAT); i++) begin
      sd_zero[i] <= sd_zero[i-1];
      sd_dval[i] <= sd_dval[i-1];
    end
  end

  // Q8: f32 -> int8; BS1: four partial sums; then the PQ push (final BS sum).
  logic             q2_v, q3_v;
  logic [15:0][7:0] q2_q, q3_q;
  logic [31:0]      q2_dval, q3_dval;
  logic [3:0][9:0]  q3_s;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      q2_v <= 1'b0;
      q3_v <= 1'b0;
    end else begin
      q2_v <= m_v[0];
      q3_v <= q2_v;
    end
  end

  always_ff @(posedge clk) begin
    for (int j = 0; j < 16; j++) begin
      if (sd_zero[MUL_LAT-1]) q2_q[j] <= 8'd0;
      else                    q2_q[j] <= tsr_f32_to_q8(m_y[j], c_kind);
    end
    q2_dval <= sd_dval[MUL_LAT-1];
    q3_q    <= q2_q;
    q3_dval <= q2_dval;
    for (int i = 0; i < 4; i++) q3_s[i] <= sum4_s8(q2_q[4*i], q2_q[4*i+1], q2_q[4*i+2], q2_q[4*i+3]);
  end

  // -------------------------------------------------------------------------------------------
  // PQ: processed-line queue {q[16], BS, d}
  // -------------------------------------------------------------------------------------------
  logic [PQ_W-1:0]  pq_mem [PQ_DEPTH];
  logic [PQ_AW-1:0] pq_wp, pq_rp;
  logic [PQ_CW-1:0] pq_cnt;
  logic [PQ_W-1:0]  pq_head;
  wire              pq_push = q3_v;

  assign pq_head = pq_mem[pq_rp];

  always_ff @(posedge clk) begin
    if (pq_push) pq_mem[pq_wp] <= {q3_q, sum4_s10(q3_s[0], q3_s[1], q3_s[2], q3_s[3]), q3_dval};
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      pq_wp  <= '0;
      pq_rp  <= '0;
      pq_cnt <= '0;
    end else begin
      if (pq_push) pq_wp <= pq_wp + PQ_AW'(1);
      if (w_pop)   pq_rp <= pq_rp + PQ_AW'(1);
      pq_cnt <= pq_cnt + PQ_CW'(pq_push) - PQ_CW'(w_pop);
    end
  end

  wire [127:0] h_q  = pq_head[PQ_W-1 -: 128];
  wire [15:0]  h_bs = pq_head[47:32];
  wire [31:0]  h_d  = pq_head[31:0];

  // -------------------------------------------------------------------------------------------
  // W: writer (line assembly, pending writes, output register)
  // -------------------------------------------------------------------------------------------
  logic                     w_fin;                  // last line of the last token popped
  logic [3:0]               w_tok;
  logic [NL_W-1:0]          w_g;                    // line within the token
  logic [SPM_AW-1:0]        w_base, w_dbase, w_bsbase;  // token's QS / D / BS region start
  logic [TSR_LINE_BITS-1:0] qs_asm, bs_asm, d_asm;
  logic                     p_qs, p_bs, p_d;        // pending line writes
  logic [SPM_AW-1:0]        l_qs, l_bs, l_d;        // their SPM lines
  logic                     out_v;
  logic [SPM_AW-1:0]        out_line;
  logic [TSR_LINE_BITS-1:0] out_data;

  // out_* loads the highest-priority pending line when it is empty or being accepted: the load
  // enable is the only logic behind spm_wr_ready, the line / data selection uses registers only.
  wire        out_ld   = (!out_v || spm_wr_ready) && (p_qs || p_bs || p_d);
  wire        e_qs     = out_ld && p_qs;
  wire        e_bs     = out_ld && !p_qs && p_bs;
  wire        e_d      = out_ld && !p_qs && !p_bs && p_d;
  wire [1:0]  npend    = 2'(p_qs) + 2'(p_bs) + 2'(p_d);
  assign      w_pop    = (pq_cnt != '0) && ((npend == 2'd0) || (npend == 2'd1 && !out_v));

  wire [15:0] w_gx     = 16'(w_g);
  wire        w_last   = (w_g == c_nlm1);
  wire        w_blkend = c_kind ? (w_gx[3:0] == 4'hF) : w_gx[0];
  wire [1:0]  w_sq     = w_gx[1:0];                         // slot in the QS line
  wire [4:0]  w_sb     = w_gx[4:0];                         // slot in the BS line
  wire [3:0]  w_sd     = c_kind ? w_gx[7:4] : w_gx[4:1];    // slot in the D line (block mod 16)
  wire [SPM_AW-1:0] w_di = c_kind ? SPM_AW'(w_gx >> 8) : SPM_AW'(w_gx >> 5);  // D line in the token
  // The last write of the instruction is accepted at this edge.
  wire        a_fin    = w_fin && (npend == 2'd0) && out_v && spm_wr_ready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w_fin    <= 1'b0;
      w_tok    <= '0;
      w_g      <= '0;
      w_base   <= '0;
      w_dbase  <= '0;
      w_bsbase <= '0;
      p_qs     <= 1'b0;
      p_bs     <= 1'b0;
      p_d      <= 1'b0;
      out_v    <= 1'b0;
    end else begin
      if (e_qs) p_qs <= 1'b0;
      if (e_bs) p_bs <= 1'b0;
      if (e_d)  p_d  <= 1'b0;
      if (accept) begin
        w_fin    <= 1'b0;
        w_tok    <= '0;
        w_g      <= '0;
        w_base   <= a_dst;
      end else if (acc_d2) begin
        w_dbase  <= w_base + c_doff;
        w_bsbase <= w_base + c_bsoff;
      end else if (w_pop) begin
        p_qs <= (w_sq == 2'd3) || w_last;
        p_bs <= c_kind && ((w_sb == 5'd31) || w_last);
        p_d  <= w_blkend && ((w_sd == 4'd15) || w_last);
        if (w_last) begin
          w_g      <= '0;
          w_tok    <= w_tok + 4'd1;
          w_base   <= w_base + c_qa;
          w_dbase  <= w_dbase + c_qa;
          w_bsbase <= w_bsbase + c_qa;
          if (w_tok == c_tm1) w_fin <= 1'b1;
        end else begin
          w_g <= w_g + NL_W'(1);
        end
      end
      if (out_ld)            out_v <= 1'b1;
      else if (spm_wr_ready) out_v <= 1'b0;
    end
  end

  always_ff @(posedge clk) begin
    if (w_pop) begin
      for (int s = 0; s < 4; s++) begin
        if (2'(s) == w_sq)      qs_asm[128*s +: 128] <= h_q;
        else if (w_sq == 2'd0)  qs_asm[128*s +: 128] <= '0;
      end
      if (c_kind) begin
        for (int s = 0; s < 32; s++) begin
          if (5'(s) == w_sb)      bs_asm[16*s +: 16] <= h_bs;
          else if (w_sb == 5'd0)  bs_asm[16*s +: 16] <= '0;
        end
      end
      if (w_blkend) begin
        for (int s = 0; s < 16; s++) begin
          if (4'(s) == w_sd)      d_asm[32*s +: 32] <= h_d;
          else if (w_sd == 4'd0)  d_asm[32*s +: 32] <= '0;
        end
      end
      l_qs <= w_base + SPM_AW'(w_gx >> 2);
      l_bs <= w_bsbase + SPM_AW'(w_gx >> 5);
      l_d  <= w_dbase + w_di;
    end
    if (out_ld) begin
      out_line <= p_qs ? l_qs : (p_bs ? l_bs : l_d);
      out_data <= p_qs ? qs_asm : (p_bs ? bs_asm : d_asm);
    end
  end

  assign spm_wr_valid = out_v;
  assign spm_wr_line  = out_line;
  assign spm_wr_data  = out_data;
  assign spm_wr_be    = {TSR_SPM_BE_W{1'b1}};

  // -------------------------------------------------------------------------------------------
  // Unit FSM and done
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st   <= S_IDLE;
      done <= 1'b0;
    end else begin
      done <= 1'b0;
      if (accept) begin
        st <= S_RUN;
      end else if (st == S_RUN && a_fin) begin
        st   <= S_IDLE;
        done <= 1'b1;
      end
    end
  end

  // Unused: m_v[15:1] (all multipliers share one valid), the bits of cmd not needed after the
  // checks (upper address bits, k[3:0], qa_lines above SPM_AW; the dispatch guarantees their values).
  wire unused_ok = &{1'b0, m_v[15:1], cmd};

`ifndef SYNTHESIS
  // -------------------------------------------------------------------------------------------
  // Simulation checks (armed by the first clock edge with rst_n low; before it the state is
  // arbitrary under X randomisation).
  // -------------------------------------------------------------------------------------------
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  wire chk = chk_on_q && rst_n;

  // Read responses: exactly SPM_RD_LAT cycles after each accepted request, none otherwise.
  logic y_fresh;                    // slot Y was written at the previous edge
  always_ff @(posedge clk) y_fresh <= ab_x;

  logic [SPM_RD_LAT-1:0] lat_sr;
  logic [31:0]           rd_out;    // requests accepted, responses not yet received
  wire                   rd_fire = rd_v && spm_rd_req_ready;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      lat_sr <= '0;
      rd_out <= '0;
    end else begin
      lat_sr <= SPM_RD_LAT'({lat_sr, rd_fire});
      rd_out <= rd_out + 32'(rd_fire) - 32'(spm_rd_rsp_valid);
    end
  end

  // Accepted command: the guarantees of docs/rtl-core.md §3.3.
  function automatic logic cmd_legal(tsr_qact_cmd_t c);
    longint unsigned lim, bs, dst_end, src_end;
    lim = longint'(SPM_LINES);
    bs  = c.kind ? 64'd256 : 64'd32;
    dst_end = longint'(c.dst_line) + longint'(c.t) * longint'(c.qa_lines);
    src_end = longint'(c.src_line) + (longint'(c.t) - 64'd1) * longint'(c.src_stride) +
              (longint'(c.k) >> 4);
    return (c.k != 16'd0) && (longint'(c.k) <= longint'(K_MAX)) && ((longint'(c.k) % bs) == 64'd0) &&
           (c.t != 4'd0) && (longint'(c.t) <= longint'(T_MAX)) &&
           (c.qa_lines == 16'(tsr_qa_lines(c.kind, {16'd0, c.k}))) &&
           (dst_end <= lim) && (src_end <= lim);
  endfunction

  always @(posedge clk) begin
    if (chk) begin
      if (accept) begin
        assert (cmd_legal(cmd))
          else $error("tsr_qact: illegal command kind %0d dst %0d src %0d k %0d t %0d stride %0d qa %0d",
                      cmd.kind, cmd.dst_line, cmd.src_line, cmd.k, cmd.t, cmd.src_stride, cmd.qa_lines);
      end
      assert (spm_rd_rsp_valid == lat_sr[SPM_RD_LAT-1])
        else $error("tsr_qact: SPM read response not exactly SPM_RD_LAT = %0d cycles after its request",
                    SPM_RD_LAT);
      assert (!(spm_rd_rsp_valid && rd_out == 32'd0))
        else $error("tsr_qact: SPM read response without a request");
      assert (buf_res <= BUF_CW'(BUF_LINES)) else $error("tsr_qact: line buffer over-reserved");
      assert (!(q_issue && buf_res == '0)) else $error("tsr_qact: buffer read of an unreserved line");
      assert (aq_cnt <= aq_res && aq_res <= AQ_CW'(AQ_DEPTH)) else $error("tsr_qact: AQ accounting");
      assert (!(aq_push && aq_cnt == AQ_CW'(AQ_DEPTH))) else $error("tsr_qact: AQ overflow");
      assert (pq_cnt + pq_cred <= PQ_CW'(PQ_DEPTH)) else $error("tsr_qact: PQ accounting");
      assert (!(pq_push && pq_cnt == PQ_CW'(PQ_DEPTH))) else $error("tsr_qact: PQ overflow");
      assert (!(da_out_valid && !x_v)) else $error("tsr_qact: DA result without slot X");
      assert (!(w_pop && (acc_d1 || acc_d2))) else $error("tsr_qact: writer active during setup");
      assert (!(blk_load && y_fresh)) else $error("tsr_qact: slot Y loaded before y_h16 settled");
      assert (!(db_out_valid && !y_v)) else $error("tsr_qact: DB result without slot Y");
      assert (!(st == S_IDLE && (rd_v || r_act || rs_v || a1_v || a4_v || aq_cnt != '0 || x_v || y_v ||
                                 cur_v || p1_v || q2_v || q3_v || pq_cnt != '0 || out_v ||
                                 p_qs || p_bs || p_d)))
        else $error("tsr_qact: activity while idle");
      if (done) begin
        assert (buf_res == '0 && aq_res == '0 && pq_cred == PQ_CW'(PQ_DEPTH) && rd_out == 32'd0 &&
                buf_wp == buf_rp && !da_out_valid && !db_out_valid && da_in_ready && db_in_ready)
          else $error("tsr_qact: done while not quiescent");
      end
    end
  end
`endif

endmodule
