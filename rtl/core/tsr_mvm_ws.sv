// tsr_mvm_ws.sv — MVM weight streamer of one cluster (spec MVM cluster microarchitecture stage 1 "Stream requester")
// plus its weight FIFO: the WS read source of one channel's tsr_mem_arb.
//
// Owner: front agent (docs/rtl-m3.md §2 block (front), protocol §4).
//
// Contract:
//   - cmd_valid/cmd_ready (R1/R2): one command per MVM with rows_c >= 1: request exactly
//     cmd_beats beats starting at beat cmd_w_off of this channel (the channel's rows are one
//     contiguous range: W_OFF + r * ROW_BEATS for r < rows_c, spec Symmetric allocation). cmd_ready is a
//     register: 1 only while idle (no beat requested-but-not-delivered, FIFO empty).
//     cmd_beats = rows_c * ROW_BEATS is in [2, 2^40 - 1] for every legal MVM, so TSR_MVM_BEATS_W
//     = 40 bits hold it: rows_c <= N <= 16 * (2^26 - 1) = 2^30 - 16 (dst SPM range check, SPM_BYTES
//     < 2^32), and a count of exactly 2^40 would need ROW_BEATS a power of two <= 1024 with
//     rows_c >= 2^30. The maximum 2^40 - 1 (Q8_0 K 61696: ROW_BEATS 1025, 1072694271 rows, DRAM
//     limit 2^40) uses every bit (tb/mvmfront front_max_count).
//   - read requests (docs/rtl-m2.md §4 M1-M6, docs/rtl-m3.md §4): local id 0 for every burst
//     (memory-side id {TSR_SRC_WS, 0} = 8: one id, so the data return in request order); bursts
//     in address order, len = min(64 - addr % 64, beats left): <= 64 beats and never across a
//     4 KiB (64-beat) block; the sum of all lens is exactly cmd_beats (no over-fetch:
//     PERF_CH_RD_BEATS must equal rows_c * ROW_BEATS, spec, Implementation clarifications, "Performance counters").
//   - reserve before request (R4/M6): a burst is offered only when FIFO_BEATS minus (beats
//     buffered + beats requested and not yet received) >= its len; space is taken when the burst
//     is loaded into the request register (valid may not be withdrawn, R2). Hence rd_rsp_ready is
//     constant 1.
//   - out_valid/out_ready/out_data: the beats in address order (data unchanged), one per cycle
//     when the memory delivers and the consumer is ready; out_* follow R1/R2.
//   - idle: no command in progress, nothing outstanding, FIFO empty (for done/assertions).
//
// Microarchitecture:
//   request generator   rq_addr / rq_left (beats still to request) -> rq_len = min(64 - addr % 64,
//                       left) is registered one cycle after every change (len stage), then the
//                       burst is loaded into the request register (rd_req_*, flops) when the
//                       register is free and the FIFO has rq_len unreserved slots. One burst per
//                       two cycles at most (>= 32 beats per cycle of request bandwidth for every
//                       burst but the first and the last of a command: plenty).
//   weight FIFO         tsr_sram_1r1w FIFO_BEATS x BEAT_BITS, RD_LAT; written by every response
//                       beat (rd_rsp_ready = 1), read in order.
//   output queue (OQ)   OQ_D = RD_LAT + 2 registers: a RAM read is issued only while
//                       (OQ entries + reads in flight) < OQ_D (registered credit), so the RAM
//                       pipeline never needs to stall; out_valid is a flop, out_data is the OQ
//                       head (OQ_D:1 mux of flops). OQ_D = RD_LAT + 2 sustains one beat per cycle
//                       with out_ready = 1 (issue -> head RD_LAT + 1 cycles, credit return 1).
//   accounting          space = FIFO_BEATS - (reserved and not received + held in the RAM);
//                       a slot is returned when its beat is read from the RAM into the OQ path.
// Ranges (W4-B at, AT §8.2: the V plane is streamed slice by slice): the command's beats are
//   cmd_beats / cmd_rbeats ranges of cmd_rbeats beats each, range k starting at cmd_w_off + k x
//   cmd_stride (bursts never cross a range end). cmd_rbeats >= cmd_beats is one contiguous range:
//   the MVM's and ATTQK's streams, identical to the behaviour without ranges.
// No combinational path from any input to any output: rd_req_*, out_valid, cmd_ready, idle are
//   flops; out_data is a mux of flops selected by a flop; rd_rsp_ready is constant.
// Throughput: sustains 1 beat/cycle if the memory round trip (request handshake at this port ->
//   first data beat at this port) is <= FIFO_BEATS - 64 - (internal latency, ~8) cycles; default
//   512 covers ~440 cycles (1.47 us at 300 MHz). FIFO: tsr_sram_1r1w, FIFO_BEATS x 512 bits
//   (8 BRAM36 at 512).

module tsr_mvm_ws
  import tsr_pkg::*;
#(
  parameter int unsigned ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter int unsigned BEAT_BITS   = TSR_DEF_BEAT_BITS,
  parameter int unsigned FIFO_BEATS  = TSR_DEF_WS_FIFO_BEATS,   // power of two, >= 64
  parameter int unsigned RD_LAT      = TSR_DEF_WS_RD_LAT        // weight FIFO RAM read latency, 1..4
) (
  input  logic                        clk,
  input  logic                        rst_n,

  // ---- command ----
  input  logic                        cmd_valid,
  output logic                        cmd_ready,
  input  logic [31:0]                 cmd_w_off,     // first beat (range-checked: fits ADDR_BEAT_W)
  input  logic [TSR_MVM_BEATS_W-1:0]  cmd_beats,     // 2 .. 2^40 - 1 (contract above)
  input  logic [TSR_MVM_BEATS_W-1:0]  cmd_rbeats,    // beats per range (>= cmd_beats: one range)
  input  logic [31:0]                 cmd_stride,    // beats from one range start to the next

  // ---- read source port (tsr_mem_arb source TSR_SRC_WS of this channel) ----
  output logic                        rd_req_valid,
  input  logic                        rd_req_ready,
  output logic [ADDR_BEAT_W-1:0]      rd_req_addr,
  output logic [TSR_LEN_W-1:0]        rd_req_len,
  output logic [TSR_SRC_ID_W-1:0]     rd_req_id,
  input  logic                        rd_rsp_valid,
  output logic                        rd_rsp_ready,
  input  logic [BEAT_BITS-1:0]        rd_rsp_data,
  input  logic                        rd_rsp_last,
  input  logic [TSR_SRC_ID_W-1:0]     rd_rsp_id,

  // ---- weight beats, in address order ----
  output logic                        out_valid,
  input  logic                        out_ready,
  output logic [BEAT_BITS-1:0]        out_data,

  output logic                        idle
);

  if (BEAT_BITS != 512) begin : g_chk_beat
    $fatal(1, "tsr_mvm_ws: BEAT_BITS must be 512");
  end
  if (FIFO_BEATS < 64 || (FIFO_BEATS & (FIFO_BEATS - 1)) != 0) begin : g_chk_fifo
    $fatal(1, "tsr_mvm_ws: FIFO_BEATS must be a power of two >= 64 (one maximal burst)");
  end
  if (ADDR_BEAT_W < 1 || ADDR_BEAT_W > 40) begin : g_chk_aw
    $fatal(1, "tsr_mvm_ws: ADDR_BEAT_W must be in [1, 40]");
  end
  if (RD_LAT < 1 || RD_LAT > 4) begin : g_chk_lat
    $fatal(1, "tsr_mvm_ws: RD_LAT must be in [1, 4]");
  end

  localparam int unsigned AQ_W  = (ADDR_BEAT_W > 7) ? ADDR_BEAT_W : 7;  // request address counter
  localparam int unsigned FA_W  = $clog2(FIFO_BEATS);                   // FIFO RAM address
  localparam int unsigned FC_W  = FA_W + 1;                             // 0..FIFO_BEATS
  localparam int unsigned OQ_D  = RD_LAT + 2;                           // output queue entries
  localparam int unsigned OQ_AW = $clog2(OQ_D);
  localparam int unsigned OQ_CW = $clog2(OQ_D + 1);
  localparam int unsigned BW    = TSR_MVM_BEATS_W;
  localparam logic [FC_W-1:0]  FIFO_FULL = FC_W'(FIFO_BEATS);
  localparam logic [OQ_AW-1:0] OQ_LAST   = OQ_AW'(OQ_D - 1);

  // -------------------------------------------------------------------------------------------
  // Command and request generator
  // -------------------------------------------------------------------------------------------
  logic                   cmd_ready_q;   // idle
  logic [AQ_W-1:0]        rq_addr_q;     // next beat to request
  logic [BW-1:0]          rq_left_q;     // beats still to request
  logic [BW-1:0]          rq_rleft_q;    // beats still to request in the current range
  logic [AQ_W-1:0]        rq_rbase_q;    // start of the current range
  logic [BW-1:0]          rbeats_q;
  logic [31:0]            stride_q;
  logic                   rq_lv_q;       // rq_len_q is valid for rq_addr_q / rq_left_q
  logic [TSR_LEN_W-1:0]   rq_len_q;      // length of the next burst
  logic                   req_v_q;       // request register (rd_req_*)
  logic [ADDR_BEAT_W-1:0] req_addr_q;
  logic [TSR_LEN_W-1:0]   req_len_q;
  logic [FC_W-1:0]        space_q;       // FIFO slots neither held nor reserved

  wire cmd_fire = cmd_valid & cmd_ready_q;
  wire rq_more  = |rq_left_q;

  // len = min(64 - addr % 64, left): the distance to the next 4 KiB block is 1..64.
  wire [6:0]           to_blk  = 7'd64 - {1'b0, rq_addr_q[5:0]};
  wire                 left_lt = ~|rq_rleft_q[BW-1:7] & (rq_rleft_q[6:0] < to_blk);
  wire [TSR_LEN_W-1:0] len_c   = left_lt ? rq_rleft_q[6:0] : to_blk;
  wire                 rng_end = (rq_rleft_q == BW'(rq_len_q));     // the loaded burst ends its range

  wire req_free = ~req_v_q | rd_req_ready;
  wire load     = rq_lv_q & req_free & (space_q >= FC_W'(rq_len_q));

  // -------------------------------------------------------------------------------------------
  // Weight FIFO (RAM) and output queue
  // -------------------------------------------------------------------------------------------
  logic [FA_W:0]          wptr_q, rptr_q;   // with wrap bit
  logic [RD_LAT-1:0]      rv_q;             // RAM read pipeline valid
  logic [OQ_CW-1:0]       oq_cnt_q;         // OQ entries + RAM reads in flight (credit)
  logic [OQ_CW-1:0]       oq_n_q;           // OQ entries
  logic [OQ_AW-1:0]       oq_wp_q, oq_rp_q;
  logic                   out_valid_q;
  logic [BEAT_BITS-1:0]   oq_mem [OQ_D];
  logic [BEAT_BITS-1:0]   ram_rdata;

  wire ram_we = rd_rsp_valid;                                     // rd_rsp_ready = 1 (reserved)
  wire ram_re = (wptr_q != rptr_q) & (oq_cnt_q < OQ_CW'(OQ_D));
  wire oq_wr  = rv_q[RD_LAT-1];
  wire pop    = out_valid_q & out_ready;

  wire [OQ_CW-1:0] oq_n_d = oq_n_q + OQ_CW'(oq_wr) - OQ_CW'(pop);

  tsr_sram_1r1w #(.DEPTH(FIFO_BEATS), .WIDTH(BEAT_BITS), .RD_LAT(RD_LAT)) u_fifo (
    .clk,
    .we(ram_we), .waddr(wptr_q[FA_W-1:0]), .wdata(rd_rsp_data), .wbe({(BEAT_BITS / 8){1'b1}}),
    .re(ram_re), .raddr(rptr_q[FA_W-1:0]), .rdata(ram_rdata)
  );

  // Everything requested has been delivered (FIFO, OQ and RAM pipeline empty, nothing reserved).
  wire all_done = ~rq_more & ~rq_lv_q & ~req_v_q & (space_q == FIFO_FULL) & (oq_cnt_q == '0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cmd_ready_q <= 1'b1;
      rq_left_q   <= '0;
      rq_lv_q     <= 1'b0;
      req_v_q     <= 1'b0;
      space_q     <= FIFO_FULL;
      wptr_q      <= '0;
      rptr_q      <= '0;
      rv_q        <= '0;
      oq_cnt_q    <= '0;
      oq_n_q      <= '0;
      oq_wp_q     <= '0;
      oq_rp_q     <= '0;
      out_valid_q <= 1'b0;
    end else begin
      cmd_ready_q <= cmd_fire ? 1'b0 : (cmd_ready_q | all_done);

      // request generator: len stage, then load
      if (cmd_fire) begin
        rq_left_q <= cmd_beats;
        rq_lv_q   <= 1'b0;
      end else if (load) begin
        rq_left_q <= rq_left_q - BW'(rq_len_q);
        rq_lv_q   <= 1'b0;
      end else if (!rq_lv_q && rq_more) begin
        rq_lv_q   <= 1'b1;
      end

      if (load)              req_v_q <= 1'b1;
      else if (rd_req_ready) req_v_q <= 1'b0;

      space_q <= space_q - (load ? FC_W'(rq_len_q) : '0) + FC_W'(ram_re);

      // FIFO pointers, RAM read pipeline, output queue
      if (ram_we) wptr_q <= wptr_q + 1'b1;
      if (ram_re) rptr_q <= rptr_q + 1'b1;
      rv_q        <= RD_LAT'({rv_q, ram_re});
      oq_cnt_q    <= oq_cnt_q + OQ_CW'(ram_re) - OQ_CW'(pop);
      oq_n_q      <= oq_n_d;
      out_valid_q <= (oq_n_d != '0);
      if (oq_wr) oq_wp_q <= (oq_wp_q == OQ_LAST) ? '0 : oq_wp_q + 1'b1;
      if (pop)   oq_rp_q <= (oq_rp_q == OQ_LAST) ? '0 : oq_rp_q + 1'b1;
    end
  end

  // Datapath registers (qualified by the control flops above; no reset).
  always_ff @(posedge clk) begin
    if (cmd_fire) begin
      rq_addr_q  <= AQ_W'(cmd_w_off);
      rq_rbase_q <= AQ_W'(cmd_w_off);
      rq_rleft_q <= (cmd_rbeats < cmd_beats) ? cmd_rbeats : cmd_beats;
      rbeats_q   <= cmd_rbeats;
      stride_q   <= cmd_stride;
    end else if (load) begin
      if (rng_end) begin                     // next range (if any) at rbase + stride
        rq_addr_q  <= rq_rbase_q + AQ_W'(stride_q);
        rq_rbase_q <= rq_rbase_q + AQ_W'(stride_q);
        rq_rleft_q <= rbeats_q;
      end else begin
        rq_addr_q  <= rq_addr_q + AQ_W'(rq_len_q);
        rq_rleft_q <= rq_rleft_q - BW'(rq_len_q);
      end
    end
    if (!rq_lv_q)  rq_len_q  <= len_c;
    if (load) begin
      req_addr_q <= ADDR_BEAT_W'(rq_addr_q);
      req_len_q  <= rq_len_q;
    end
    if (oq_wr) oq_mem[oq_wp_q] <= ram_rdata;
  end

  assign cmd_ready    = cmd_ready_q;
  assign idle         = cmd_ready_q;
  assign rd_req_valid = req_v_q;
  assign rd_req_addr  = req_addr_q;
  assign rd_req_len   = req_len_q;
  assign rd_req_id    = '0;
  assign rd_rsp_ready = 1'b1;
  assign out_valid    = out_valid_q;
  assign out_data     = oq_mem[oq_rp_q];

  // rd_rsp_last: bursts are counted by beats (the FIFO accounting); rd_rsp_id: checked below.
  wire unused_ok = &{1'b0, rd_rsp_last, rd_rsp_id, stride_q};

  // -------------------------------------------------------------------------------------------
  // Simulation checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  // Checks are armed once a reset has been seen (the time-0 state may be random).
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  // Beats requested and not yet received (FIFO_BEATS = space + in flight + held in the RAM).
  wire [FC_W-1:0] held     = FC_W'(wptr_q - rptr_q);
  wire [FC_W:0]   inflight = (FC_W + 1)'(FIFO_BEATS) - (FC_W + 1)'(space_q) - (FC_W + 1)'(held);

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (cmd_fire) begin
        assert (cmd_beats != '0)
          else $error("%m: command with 0 beats (commands only for rows_c >= 1; legal counts are < 2^40)");
        assert (41'(cmd_w_off) + 41'(cmd_beats) <= (41'd1 << ADDR_BEAT_W))
          else $error("%m: command beats [0x%0h, +%0d) beyond 2^ADDR_BEAT_W", cmd_w_off, cmd_beats);
      end
      if (load) begin
        assert (rq_len_q != '0 && 32'(rq_len_q) <= TSR_MAX_BURST_BEATS
                && 32'(rq_addr_q[5:0]) + 32'(rq_len_q) <= TSR_BURST_ALIGN
                && BW'(rq_len_q) <= rq_left_q)
          else $error("%m: burst addr 0x%0h len %0d (left %0d) breaks the burst rule",
                      rq_addr_q, rq_len_q, rq_left_q);
      end
      if (rd_rsp_valid) begin
        assert (rd_rsp_id == '0)
          else $error("%m: response with local id %0d (the WS source only uses id 0)", rd_rsp_id);
        assert (inflight != '0)
          else $error("%m: response beat without an outstanding request (FIFO overflow)");
      end
      assert (held <= FC_W'(FIFO_BEATS)) else $error("%m: FIFO holds more than FIFO_BEATS beats");
    end
  end
`endif

endmodule
