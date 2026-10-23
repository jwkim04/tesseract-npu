// tsr_fetch.sv — instruction prefetcher on channel 0.
//
// Owner: cmdproc agent (docs/rtl-m2.md §2, block (a); fetch rules §3.1).
//
// Contract:
//   start (1-cycle pulse, only while idle) latches cmd_addr (64-bit beat address) and cmd_count.
//   The unit then delivers instructions pc = 0, 1, ..., cmd_count-1 in order on the insn stream
//   (valid/ready); instruction pc lives in ch0 beat cmd_addr + pc/2 at bits [256*(pc%2) +: 256].
//   Beats at or beyond the DRAM limit (min(CH_BYTES/64, 2^ADDR_BEAT_W), compared in 64 bits so
//   that CMD_ADDR_HI != 0 is out of range) are NEVER read: the instruction(s) they would hold are
//   delivered with range_err = 1 and insn = 0 (tsr_cmdproc reports DRAM_RANGE when it executes
//   that pc). Nothing is read for pc >= cmd_count (prefetch stays inside CMD_COUNT).
//   Prefetch: up to BUF_BEATS beats buffered; a burst is requested only when buffer space for all
//   of its beats is reserved (rule R4), so rd_rsp_ready stays 1. Bursts: <= 64 beats, inside one
//   64-beat block, below the limit. Read id 0 (local).
//   stop (1-cycle pulse) abandons the remaining instructions: no new requests; a request already
//   offered stays offered until accepted (valid is never withdrawn); responses of requests
//   already issued are drained and discarded. idle = 1 when no request is outstanding and no run
//   is active (after reset, after stop + drain, or before start). tsr_cmdproc waits for idle
//   before it reports the end of a run, so STATUS.DONE implies a quiescent memory port.
//   A read response while no beat is outstanding (possible only if the core was reset while the
//   memory side still had fetch bursts in flight) is dropped: it is not buffered and does not
//   move the outstanding count, so the unit stays consistent and the next run is unaffected.
//
// Implementation:
//   F_SETUP0/1 (two cycles after start): n_ok = number of in-range beats of the program
//     = (cmd_addr < LIM) ? min(ceil(cmd_count/2), LIM - cmd_addr) : 0 (41-bit arithmetic,
//     LIM <= 2^40). Instruction pc >= 2*n_ok (and < cmd_count) is delivered as range_err.
//   Requests: the next burst covers min(beats left, beats to the 64-beat boundary) beats if that
//     many buffer slots are free, else the free slots once at least half the buffer is free.
//     Sizing is pipelined for timing (cap_q -> plan_q -> issue; >= 3 cycles between requests,
//     far above the instruction rate); a plan is conservative because free slots only grow until
//     the next issue. All request outputs are registers.
//   Buffer: BUF_BEATS x BEAT_BITS FIFO (distributed RAM, asynchronous read); insn is selected
//     from the head beat by pc[0]; the head is popped after its second instruction or after the
//     program's last instruction.
//   insn_valid / insn depend only on registers (no path from insn_ready).

module tsr_fetch
  import tsr_pkg::*;
#(
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter longint unsigned CH_BYTES    = TSR_DEF_CH_BYTES,
  parameter int unsigned     BEAT_BITS   = TSR_DEF_BEAT_BITS,
  parameter int unsigned     BUF_BEATS   = 16
) (
  input  logic                        clk,
  input  logic                        rst_n,

  // ---- control from tsr_cmdproc ----
  input  logic                        start,
  input  logic [63:0]                 cmd_addr,
  input  logic [31:0]                 cmd_count,
  input  logic                        stop,
  output logic                        idle,

  // ---- instruction stream to tsr_cmdproc ----
  output logic                        insn_valid,
  input  logic                        insn_ready,
  output tsr_fetch_insn_t             insn,

  // ---- memory source port (channel 0 tsr_mem_arb, source TSR_SRC_FETCH) ----
  output logic                        rd_req_valid,
  input  logic                        rd_req_ready,
  output logic [ADDR_BEAT_W-1:0]      rd_req_addr,
  output logic [TSR_LEN_W-1:0]        rd_req_len,
  output logic [TSR_SRC_ID_W-1:0]     rd_req_id,
  input  logic                        rd_rsp_valid,
  output logic                        rd_rsp_ready,
  input  logic [BEAT_BITS-1:0]        rd_rsp_data,
  input  logic                        rd_rsp_last,
  input  logic [TSR_SRC_ID_W-1:0]     rd_rsp_id
);

  // -------------------------------------------------------------------------------------------
  // Constants
  // -------------------------------------------------------------------------------------------
  localparam longint unsigned LIM   = tsr_dram_limit(CH_BYTES, ADDR_BEAT_W);   // <= 2^40
  localparam int unsigned     LW    = 41;                                      // holds LIM
  localparam logic [LW-1:0]   LIM_W = LW'(LIM);
  localparam int unsigned     PTR_W = (BUF_BEATS > 1) ? $clog2(BUF_BEATS) : 1;
  localparam int unsigned     CNT_W = $clog2(BUF_BEATS + 1);
  localparam logic [CNT_W-1:0] BUF_N  = CNT_W'(BUF_BEATS);
  localparam logic [CNT_W-1:0] HALF_N = CNT_W'(BUF_BEATS / 2);

  if (BEAT_BITS != 2 * TSR_INSN_BITS) begin : g_chk_beat
    $fatal(1, "tsr_fetch: a beat must hold exactly two instructions (BEAT_BITS = 512)");
  end
  if (BUF_BEATS < 2 || BUF_BEATS > TSR_MAX_BURST_BEATS || (BUF_BEATS & (BUF_BEATS - 1)) != 0) begin : g_chk_buf
    $fatal(1, "tsr_fetch: BUF_BEATS must be a power of two in [2, 64]");
  end
  if (ADDR_BEAT_W < 1 || ADDR_BEAT_W > 40 || LIM < 1) begin : g_chk_lim
    $fatal(1, "tsr_fetch: ADDR_BEAT_W must be in [1, 40] and the DRAM limit >= 1 beat");
  end

  typedef enum logic [2:0] { F_IDLE, F_SETUP0, F_SETUP1, F_RUN, F_DRAIN } fstate_t;

  // -------------------------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------------------------
  fstate_t          st_q;
  logic [63:0]      base_q;
  logic [31:0]      count_q;
  logic             inr_q;              // base < LIM
  logic [LW-1:0]    room_q;             // LIM - base (when inr_q)
  logic [31:0]      nbeats_q;           // ceil(count / 2)

  logic [ADDR_BEAT_W-1:0] req_addr_q;   // next beat to request
  logic [31:0]      req_left_q;         // in-range beats still to request
  logic [6:0]       cap_q;              // min(req_left, 64 - req_addr % 64) (one cycle old)
  logic             plan_ok_q;          // a burst of plan_len_q beats may be requested
  logic [6:0]       plan_len_q;
  logic [1:0]       hold_q;             // cycles until cap_q / plan_q reflect the last issue

  logic             rq_valid_q;
  logic [ADDR_BEAT_W-1:0] rq_addr_q;
  logic [TSR_LEN_W-1:0]   rq_len_q;

  logic [BEAT_BITS-1:0] buf_mem [BUF_BEATS];
  logic [PTR_W-1:0] wptr_q, rptr_q;
  logic [CNT_W-1:0] fcnt_q;             // beats in the buffer
  logic [CNT_W-1:0] infl_q;             // beats requested (offered or accepted), not yet received

  logic [31:0]      pc_q;               // next instruction to deliver
  logic [31:0]      insn_left_q;        // instructions not yet delivered
  logic             have_insn_q;        // insn_left_q != 0
  logic             last_q;             // insn_left_q == 1
  logic [31:0]      ok_left_q;          // in-range beats not yet popped
  logic             ok_have_q;          // ok_left_q != 0

  // -------------------------------------------------------------------------------------------
  // Request sizing
  // -------------------------------------------------------------------------------------------
  wire [5:0]       addr_lo  = 6'(req_addr_q);                    // req_addr % 64
  wire [6:0]       to_bound = 7'd64 - {1'b0, addr_lo};
  wire [6:0]       cap_d    = (req_left_q < 32'(to_bound)) ? req_left_q[6:0] : to_bound;
  wire [CNT_W-1:0] free_n   = BUF_N - fcnt_q - infl_q;
  wire             cap_fits = ({1'b0, cap_q} <= 8'(free_n));
  wire             plan_ok  = (cap_q != 7'd0) && (cap_fits || (free_n >= HALF_N));
  wire [6:0]       plan_len = cap_fits ? cap_q : 7'(free_n);
  wire             can_req  = (st_q == F_RUN) && !stop && !rq_valid_q && (hold_q == 2'd0) && plan_ok_q;
  wire [6:0]       req_len  = plan_len_q;

  // -------------------------------------------------------------------------------------------
  // Delivery
  // -------------------------------------------------------------------------------------------
  wire [BEAT_BITS-1:0] head = buf_mem[rptr_q];
  assign insn_valid = (st_q == F_RUN) && have_insn_q && (!ok_have_q || (fcnt_q != '0));

  always_comb begin
    insn.range_err = !ok_have_q;
    insn.pc        = pc_q;
    insn.insn      = '0;
    if (ok_have_q) insn.insn = pc_q[0] ? head[2*TSR_INSN_BITS-1:TSR_INSN_BITS] : head[TSR_INSN_BITS-1:0];
  end

  wire deliver  = insn_valid && insn_ready && !stop;
  wire pop_beat = deliver && ok_have_q && (pc_q[0] || last_q);
  // rd_rsp_ready = 1. A beat counts only while one is outstanding: a stray response (nothing
  // outstanding) is dropped instead of wrapping infl_q and entering the buffer.
  wire rsp      = rd_rsp_valid && (infl_q != '0);
  wire push     = rsp && (st_q == F_RUN) && !stop;

  // -------------------------------------------------------------------------------------------
  // Control
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q        <= F_IDLE;
      rq_valid_q  <= 1'b0;
      hold_q      <= 2'd2;
      plan_ok_q   <= 1'b0;
      wptr_q      <= '0;
      rptr_q      <= '0;
      fcnt_q      <= '0;
      infl_q      <= '0;
      have_insn_q <= 1'b0;
      ok_have_q   <= 1'b0;
      req_left_q  <= 32'd0;
    end else begin
      // ---- memory request / response bookkeeping (every state) ----
      if (rq_valid_q && rd_req_ready) rq_valid_q <= 1'b0;
      infl_q <= infl_q + (can_req ? CNT_W'(req_len) : '0) - (rsp ? CNT_W'(1) : '0);
      if (can_req) begin
        rq_valid_q <= 1'b1;
        rq_addr_q  <= req_addr_q;
        rq_len_q   <= req_len;
        req_addr_q <= req_addr_q + ADDR_BEAT_W'(req_len);
        req_left_q <= req_left_q - 32'(req_len);
      end
      cap_q      <= cap_d;
      plan_ok_q  <= plan_ok;
      plan_len_q <= plan_len;
      if (can_req)              hold_q <= 2'd2;
      else if (hold_q != 2'd0)  hold_q <= hold_q - 2'd1;

      // ---- buffer ----
      if (push) wptr_q <= wptr_q + PTR_W'(1);
      if (pop_beat) rptr_q <= rptr_q + PTR_W'(1);
      fcnt_q <= fcnt_q + (push ? CNT_W'(1) : '0) - (pop_beat ? CNT_W'(1) : '0);

      // ---- delivery ----
      if (deliver) begin
        pc_q        <= pc_q + 32'd1;
        insn_left_q <= insn_left_q - 32'd1;
        have_insn_q <= !last_q;
        last_q      <= (insn_left_q == 32'd2);
      end
      if (pop_beat) begin
        ok_left_q <= ok_left_q - 32'd1;
        ok_have_q <= (ok_left_q != 32'd1);
      end

      // ---- sequencing ----
      unique case (st_q)
        F_IDLE: begin
          if (start) begin
            base_q  <= cmd_addr;
            count_q <= cmd_count;
            st_q    <= F_SETUP0;
          end
        end
        F_SETUP0: begin
          inr_q    <= (base_q[63:LW] == '0) && (base_q[LW-1:0] < LIM_W);
          room_q   <= LIM_W - base_q[LW-1:0];
          nbeats_q <= {1'b0, count_q[31:1]} + {31'd0, count_q[0]};
          st_q     <= F_SETUP1;
        end
        F_SETUP1: begin
          // n_ok = in range ? min(nbeats, room) : 0
          if (!inr_q) begin
            req_left_q <= 32'd0;
            ok_left_q  <= 32'd0;
            ok_have_q  <= 1'b0;
          end else if (room_q < LW'(nbeats_q)) begin
            req_left_q <= room_q[31:0];
            ok_left_q  <= room_q[31:0];
            ok_have_q  <= (room_q != '0);
          end else begin
            req_left_q <= nbeats_q;
            ok_left_q  <= nbeats_q;
            ok_have_q  <= (nbeats_q != 32'd0);
          end
          req_addr_q  <= base_q[ADDR_BEAT_W-1:0];
          pc_q        <= 32'd0;
          insn_left_q <= count_q;
          have_insn_q <= (count_q != 32'd0);
          last_q      <= (count_q == 32'd1);
          hold_q      <= 2'd2;
          st_q        <= F_RUN;
        end
        F_RUN: ;
        F_DRAIN: begin
          if (infl_q == '0) st_q <= F_IDLE;
        end
        default: st_q <= F_IDLE;
      endcase

      // ---- stop: discard the buffer, drain what is in flight ----
      if (stop && st_q != F_IDLE) begin
        wptr_q      <= '0;
        rptr_q      <= '0;
        fcnt_q      <= '0;
        have_insn_q <= 1'b0;
        ok_have_q   <= 1'b0;
        req_left_q  <= 32'd0;
        st_q        <= F_DRAIN;
      end
    end
  end

  // Buffer storage: no reset (distributed RAM).
  always_ff @(posedge clk) begin
    if (push) buf_mem[wptr_q] <= rd_rsp_data;
  end

  // -------------------------------------------------------------------------------------------
  // Outputs
  // -------------------------------------------------------------------------------------------
  assign idle         = (st_q == F_IDLE);
  assign rd_req_valid = rq_valid_q;
  assign rd_req_addr  = rq_addr_q;
  assign rd_req_len   = rq_len_q;
  assign rd_req_id    = '0;
  assign rd_rsp_ready = 1'b1;

  // The response id / last are informational: one local id, in-order beats (asserted below).
  wire unused_ok = &{1'b0, rd_rsp_last, rd_rsp_id};

`ifndef SYNTHESIS
  // Checked only after a reset was applied (the state is random before; the initial block runs
  // before the first clock edge, after Verilator's X randomisation).
  // +tsr_allow_stray_rsp: for the test that injects responses with nothing outstanding on
  // purpose (tb/cmdproc cp_stray_rsp); everywhere else such a response is a routing bug.
  logic rst_seen, chk_stray;
  initial rst_seen = 1'b0;
  initial chk_stray = !$test$plusargs("tsr_allow_stray_rsp");
  always @(posedge clk) if (!rst_n) rst_seen <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && rst_seen) begin
      if (rd_rsp_valid && chk_stray)
        assert (infl_q != '0) else $error("tsr_fetch: read response with nothing outstanding");
      if (rd_rsp_valid) assert (rd_rsp_id == '0) else $error("tsr_fetch: read response id %0d", rd_rsp_id);
      if (start) assert (st_q == F_IDLE) else $error("tsr_fetch: start while not idle");
      if (push)  assert (fcnt_q != BUF_N || pop_beat) else $error("tsr_fetch: buffer overflow");
    end
  end
`endif

endmodule
