// tsr_dma.sv — LD / ST engine: moves lines between the SPM and one DRAM channel.
//
// Spec: docs/rtl-core.md §2, block (b); dispatch §3.2; memory port rules §4.
//
// Dispatch: cmd_valid/cmd_ready accept one tsr_dma_cmd_t (cmd_ready = idle). Operands are legal
//   (the dispatch checks them; n_lines >= 1). `done` pulses one cycle when the instruction's effects
//   are complete: LD — the last SPM line write has been accepted by tsr_spm; ST — every DRAM write
//   response of the instruction has been received (sequential semantics: a later instruction
//   sees the data). done is never in the same cycle as the accepting handshake.
// LD (DRAM -> SPM): read beats dram_beat .. dram_beat+n_lines-1 of channel `ch`, write SPM lines
//   spm_line .. spm_line+n_lines-1 (beat i -> line spm_line+i, full lines, wr_be all ones).
// ST (SPM -> DRAM): read those SPM lines, write those beats.
// Bursts: split at 64 beats and at 64-beat (4 KiB) boundaries: len = min(left, 64 - (addr % 64)).
//   The per-channel limit holds by construction (range-checked operands).
// Memory side: one source port per channel (packed [NUM_CH]); only port `ch` is used by a command,
//   the others stay idle. Reads use local id 0; responses of one id return in order. Writes: offer
//   wr_req and wr_dat independently (never wait for wr_req_ready before driving wr_dat_valid, nor
//   the reverse); data beats in request order; wr_dat_last on the last beat of each burst.
// SPM side: single-port client TSR_SPM_CL_DMA of tsr_spm through tsr_spm_cport (read responses
//   arrive SPM_RD_LAT cycles after the request is accepted, no backpressure).
//
// Microarchitecture
//   One engine, one command at a time. Three decoupled pipelines share a beat queue (QD =
//   LDQ_BEATS entries, default 512: a tsr_sram_1r1w of QD x 512, i.e. 8 RAMB36 in 512 x 72
//   columns on the U200, CC §10.4):
//
//     burst generator --rq--> rd_req[ch] (LD)  or  wr_req[ch] (ST)        (independent of data)
//     LD: rd_rsp[ch] --push--> queue --pop--> read-out --> output reg --> spm_wr  (line = spm_line + i)
//     ST: spm_rd_req --SPM--> spm_rd_rsp --push--> queue --pop--> read-out --> output reg --> wr_dat[ch]
//
//   - Read-out: the queue RAM has read latency QRL = 2 (the BRAM read latch and its output
//     register DOB_REG), so the output register (a fabric flop: wr_dat_data goes to the channel
//     arbiters in other SLRs, rtl-core §1, and must not leave a BRAM latch) is fed from the RAM
//     output or from a 2-entry skid buffer. A pop (RAM read) is issued while the beats behind the
//     RAM (reads in flight + skid + output register) leave room after this cycle's output
//     transfer: QRL + 1 = 3 places, so a stream still moves one beat per cycle.
//   - Read id scheme: every LD burst uses local id 0 (memory id {DMA, 0}). The memory returns
//     the beats of one id in request order, so the response stream IS the SPM line order: no
//     reorder buffer, no tags; up to RD_OUTSTANDING bursts in flight (limit enforced by
//     tsr_mem_arb, shared with the other sources of the channel). The burst generator issues
//     back-to-back bursts as fast as the arbiter accepts them.
//   - LD flow control: burst reservation (CC §10.4, CC-13; R4 / M6). An LD read burst is
//     requested only when the beat queue has room for the whole burst: ld_free_q counts the queue
//     entries neither holding a beat nor claimed by a beat requested and not yet received
//     (QD - queue entries - beats in flight); a burst of blen beats needs blen <= ld_free_q and
//     takes them at its request, each pop gives one back. So every response beat finds a free
//     entry and rd_rsp_ready is constant 1: SPM contention (round-robin write ports shared with
//     QACT, the MVM gather, the VU, AT and the bridge) can never stall the channel's head-of-line
//     response path, so the MVM weight beats behind a DMA beat never wait (CC §10.4). QD beats
//     in flight cover 1.7-1.9 us of DRAM latency at one beat per cycle (*estimate*, CC §10.4),
//     below RD_OUTSTANDING x 64 beats: the DMA alone keeps at most QD / 64 full bursts in flight.
//     The queue drains into the SPM write port, which never waits for memory (CC §10.3 (i)). In
//     steady state (SPM ready) the LD streams one beat per cycle.
//   - ST flow control: an SPM read is requested only while queue entries + reads in flight
//     (requested, response not yet returned) < QD, so the responses (no backpressure) always
//     fit. QD >= SPM_RD_LAT + 4 covers the SPM read latency at one line per cycle.
//   - ST write requests are generated from the address/length alone and data beats are offered
//     as soon as they are in the output register (rule M5: neither waits for the other's
//     handshake). wr_dat_last is computed when a beat enters the output register: last beat of
//     the command, or DRAM address % 64 == 63 (same split as the request generator: D1).
//   - Completion: LD when the n-th SPM write is accepted; ST when every burst was requested,
//     every data beat accepted and every write response received (wr_pend == 0). `done` is
//     registered (>= 1 cycle after the accepting edge), cmd_ready returns with it.
//   - Every output is a flop (or a constant): per-channel valid flops, shared payload flops.
//     No combinational path from any input to any output (R5); the arbiters can sit in other
//     SLRs. rd_rsp_ready (reservation) and wr_rsp_ready (a response only decrements a counter) are
//     constant 1.
//   - Widths: line/beat counters NW = SPM_AW + 1 bits (n_lines <= SPM lines); the DRAM address
//     counter is ADDR_BEAT_W bits (dram_beat + n_lines may exceed 2^32 when ADDR_BEAT_W > 32:
//     the operand is a 32-bit start, the range check is 64-bit, see spec Range check).
//   - Simulation-only checks (`ifndef SYNTHESIS): command legality, read id/last consistency
//     with the D1 split, responses only on the active channel, queue overflow, the reservation
//     (queue entries + LD beats in flight <= QD), write responses.
// KVST (AT §5.1, §8.3; parameter HAS_KVST): a command with cmd_xc = 1 carries the
//   tsr_att_pkg::tsr_kvst_cmd_t payload in xcmd and runs on the KVST engine (tsr_dma_kvst.sv); the
//   LD / ST engine stays idle meanwhile (one command at a time, cmd_ready = both idle). While the
//   engine runs (kv_act_q, a flop) it drives the write source ports of the channels its
//   groups live in and the SPM read port: every such output is a 2:1 mux of two flops selected by
//   kv_act_q. Masked META writes do not issue explicit core RMW events (PERF KVST_RMW, AT §10).
//   HAS_KVST = 0: cmd_xc / xcmd are ignored and the unit is the LD / ST engine above.

module tsr_dma
  import tsr_pkg::*;
#(
  parameter  int unsigned NUM_CH      = TSR_DEF_NUM_CH,
  parameter  int unsigned ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter  int unsigned BEAT_BITS   = TSR_DEF_BEAT_BITS,
  parameter  int unsigned SPM_BYTES   = TSR_DEF_SPM_BYTES,
  parameter  int unsigned SPM_RD_LAT  = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned LDQ_BEATS   = 512,     // beat queue entries (power of two >= 64, CC-13)
  parameter  bit          HAS_KVST    = 1'b0,    // KVST engine (AT §8.3)
  parameter  bit          FAST_CLK    = 1'b0,    // the cycle-changing stages of the 250 MHz clock (KVST placer; tsr_core FAST_CLK)
  parameter  int unsigned XC_CMD_W    = 256,     // external-class payload width (tsr_issue)
  localparam int unsigned SPM_LINES   = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned SPM_AW      = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1
) (
  input  logic                                  clk,
  input  logic                                  rst_n,

  // ---- dispatch (the dispatcher, Y4) ----
  input  logic                                  cmd_valid,
  output logic                                  cmd_ready,
  input  tsr_dma_cmd_t                          cmd,
  input  logic                                  cmd_xc,       // KVST (HAS_KVST): payload in xcmd
  input  logic [XC_CMD_W-1:0]                   xcmd,
  output logic                                  done,

  // ---- memory source ports, one per channel (to tsr_mem_arb source TSR_SRC_DMA) ----
  output logic [NUM_CH-1:0]                     rd_req_valid,
  input  logic [NUM_CH-1:0]                     rd_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]    rd_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]      rd_req_len,
  output logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]   rd_req_id,
  input  logic [NUM_CH-1:0]                     rd_rsp_valid,
  output logic [NUM_CH-1:0]                     rd_rsp_ready,
  input  logic [NUM_CH-1:0][BEAT_BITS-1:0]      rd_rsp_data,
  input  logic [NUM_CH-1:0]                     rd_rsp_last,
  input  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]   rd_rsp_id,
  output logic [NUM_CH-1:0]                     wr_req_valid,
  input  logic [NUM_CH-1:0]                     wr_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]    wr_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]      wr_req_len,
  output logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]   wr_req_id,
  output logic [NUM_CH-1:0]                     wr_dat_valid,
  input  logic [NUM_CH-1:0]                     wr_dat_ready,
  output logic [NUM_CH-1:0][BEAT_BITS-1:0]      wr_dat_data,
  output logic [NUM_CH-1:0][BEAT_BITS/32-1:0]   wr_dat_mask,
  output logic [NUM_CH-1:0]                     wr_dat_last,
  input  logic [NUM_CH-1:0]                     wr_rsp_valid,
  output logic [NUM_CH-1:0]                     wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]   wr_rsp_id,

  // ---- SPM client (tsr_spm client TSR_SPM_CL_DMA) ----
  output logic                                  spm_rd_req_valid,
  input  logic                                  spm_rd_req_ready,
  output logic [SPM_AW-1:0]                     spm_rd_req_line,
  input  logic                                  spm_rd_rsp_valid,
  input  logic [TSR_LINE_BITS-1:0]              spm_rd_rsp_data,
  output logic                                  spm_wr_valid,
  input  logic                                  spm_wr_ready,
  output logic [SPM_AW-1:0]                     spm_wr_line,
  output logic [TSR_LINE_BITS-1:0]              spm_wr_data,
  output logic [TSR_SPM_BE_W-1:0]               spm_wr_be,

  output logic                                  perf_kvst_rmw
);

  if (BEAT_BITS != TSR_LINE_BITS) begin : g_chk_beat
    $fatal(1, "tsr_dma: BEAT_BITS must equal the SPM line width (512)");
  end
  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX) begin : g_chk_ch
    $fatal(1, "tsr_dma: NUM_CH must be in [1, 16]");
  end
  if (SPM_LINES < 1 || SPM_RD_LAT < 1) begin : g_chk_spm
    $fatal(1, "tsr_dma: SPM_BYTES must be >= 64 and SPM_RD_LAT >= 1");
  end
  if (LDQ_BEATS < 64 || (LDQ_BEATS & (LDQ_BEATS - 1)) != 0 || LDQ_BEATS < SPM_RD_LAT + 4) begin : g_chk_q
    $fatal(1, "tsr_dma: LDQ_BEATS must be a power of two >= max(64, SPM_RD_LAT + 4)");
  end

  localparam int unsigned NW  = SPM_AW + 1;                       // counts 0 .. SPM_LINES
  localparam int unsigned CW  = (NW > TSR_LEN_W) ? NW : TSR_LEN_W;
  localparam int unsigned QAW = $clog2(LDQ_BEATS);
  localparam int unsigned QD  = LDQ_BEATS;                        // queue entries
  localparam int unsigned QRL = 2;                                // queue RAM read latency
  localparam int unsigned LB  = TSR_LINE_BITS;

  // -------------------------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------------------------
  logic                    idle_q;          // = cmd_ready
  logic                    done_q;
  logic                    st_q;            // current command is ST
  logic [NUM_CH-1:0]       ch_oh_q;         // its channel, one-hot

  // burst generator (LD read bursts / ST write bursts)
  logic [ADDR_BEAT_W-1:0]  g_addr_q;        // next burst address
  logic [NW-1:0]           g_left_q;        // beats not yet requested
  logic [NUM_CH-1:0]       rq_rd_v_q;       // = rd_req_valid
  logic [NUM_CH-1:0]       rq_wr_v_q;       // = wr_req_valid
  logic [ADDR_BEAT_W-1:0]  rq_addr_q;
  logic [TSR_LEN_W-1:0]    rq_len_q;

  // ST: SPM read requests
  logic                    sr_v_q;          // = spm_rd_req_valid
  logic [SPM_AW-1:0]       sr_line_q;       // = spm_rd_req_line
  logic [SPM_AW-1:0]       sr_next_q;       // next line to request
  logic [NW-1:0]           sr_left_q;       // lines not yet requested
  logic [QAW:0]            sr_rsv_q;        // reads requested (incl. sr_v_q), response not in yet

  // beat queue
  logic [QAW-1:0]          q_wp_q, q_rp_q;
  logic [QAW:0]            q_cnt_q;
  logic [QAW:0]            ld_free_q;       // LD: entries neither filled nor claimed by beats in flight
  // read-out: RAM reads in flight, skid buffer
  logic [QRL-1:0]          rv_q;            // rv_q[i]: a RAM read issued i + 1 cycles ago
  logic [LB-1:0]           q_rdata;         // RAM output (valid in the cycle rv_q[QRL-1] = 1)
  logic [LB-1:0]           sk_data_q [2];
  logic                    sk_rp_q, sk_wp_q;
  logic [1:0]              sk_cnt_q;

  // output register: LD -> SPM write, ST -> DRAM write data
  logic                    o_spm_v_q;       // = spm_wr_valid
  logic [NUM_CH-1:0]       o_dat_v_q;       // = wr_dat_valid
  logic [LB-1:0]           o_data_q;
  logic                    o_last_q;
  logic [SPM_AW-1:0]       o_line_q;
  logic [SPM_AW-1:0]       p_line_q;        // LD: SPM line of the next beat entering o_*
  logic [5:0]              p_lo_q;          // ST: DRAM address bits [5:0] of the next beat
  logic [NW-1:0]           p_left_q;        // beats that have not entered o_* yet
  logic [NW-1:0]           o_left_q;        // beats not yet delivered (SPM write / wr_dat accepted)
  logic [NW-1:0]           wr_pend_q;       // ST bursts requested whose response is not in yet

  // -------------------------------------------------------------------------------------------
  // Command acceptance
  // -------------------------------------------------------------------------------------------
  logic                   kv_busy, kv_done, kv_act_q;
  wire                    is_kv    = HAS_KVST & cmd_xc;
  wire                    cmd_fire = cmd_valid & idle_q & ~kv_busy & ~is_kv;   // LD / ST
  wire                    kv_start = cmd_valid & idle_q & ~kv_busy & is_kv;
  wire [ADDR_BEAT_W-1:0]  cmd_addr = ADDR_BEAT_W'(cmd.dram_beat);  // zero-extends when W > 32
  wire [NW-1:0]           cmd_n    = NW'(cmd.n_lines);
  wire [SPM_AW-1:0]       cmd_line = SPM_AW'(cmd.spm_line);
  logic [NUM_CH-1:0]      cmd_oh;
  always_comb begin
    for (int unsigned c = 0; c < NUM_CH; c++) cmd_oh[c] = (cmd.ch == 4'(c));
  end

  // -------------------------------------------------------------------------------------------
  // Burst generator: len = min(left, 64 - addr % 64)
  // -------------------------------------------------------------------------------------------
  wire [TSR_LEN_W-1:0] to_blk_end = TSR_LEN_W'(7'd64 - {1'b0, 6'(g_addr_q)});   // 1..64
  wire [TSR_LEN_W-1:0] blen       = (CW'(g_left_q) < CW'(to_blk_end)) ? TSR_LEN_W'(g_left_q)
                                                                       : to_blk_end;
  wire rq_busy = (|rq_rd_v_q) | (|rq_wr_v_q);
  wire rq_fire = (|(rq_rd_v_q & rd_req_ready)) | (|(rq_wr_v_q & wr_req_ready));
  // LD: the whole burst must fit into the unclaimed queue entries (reservation, CC §10.4).
  wire ld_room = (QAW+1)'(blen) <= ld_free_q;
  wire rq_load = ~idle_q & (~rq_busy | rq_fire) & (g_left_q != '0) & (st_q | ld_room);

  // -------------------------------------------------------------------------------------------
  // Queue input: LD read data of the active channel or ST SPM read data
  // -------------------------------------------------------------------------------------------
  logic [LB-1:0] rsp_data;
  always_comb begin
    rsp_data = '0;
    for (int unsigned c = 0; c < NUM_CH; c++) rsp_data |= rd_rsp_data[c] & {LB{ch_oh_q[c]}};
  end
  wire           ld_push = ~idle_q & ~st_q & (|(rd_rsp_valid & ch_oh_q));   // rd_rsp_ready = 1 (idle_q during KVST)
  wire           st_push = ~idle_q & st_q & spm_rd_rsp_valid;
  wire           q_push  = ld_push | st_push;
  wire [LB-1:0]  q_wdata = st_q ? spm_rd_rsp_data : rsp_data;

  // ST SPM read requests: only while the responses are guaranteed to fit.
  wire [QAW+1:0] q_occ   = (QAW+2)'(q_cnt_q) + (QAW+2)'(sr_rsv_q);
  wire           sr_fire = sr_v_q & spm_rd_req_ready;
  wire           sr_load = ~idle_q & st_q & (~sr_v_q | sr_fire) & (sr_left_q != '0)
                         & (q_occ < (QAW+2)'(QD));

  // Output register, fed by the read-out (RAM output or skid).
  wire o_valid  = o_spm_v_q | (|o_dat_v_q);
  wire o_fire   = (o_spm_v_q & spm_wr_ready) | (|(o_dat_v_q & wr_dat_ready));
  wire q_arrive = rv_q[QRL-1];                                     // RAM data this cycle
  wire o_load   = (~o_valid | o_fire) & ((sk_cnt_q != '0) | q_arrive);
  wire sk_push  = q_arrive & ~(o_load & (sk_cnt_q == '0));         // not taken by o_* directly
  wire sk_pop   = o_load & (sk_cnt_q != '0);
  // Beats behind the RAM after this cycle: reads in flight + skid + output register - o_fire.
  wire [2:0] ro_occ = 3'(rv_q[0]) + 3'(rv_q[1]) + 3'(sk_cnt_q) + 3'(o_valid) - 3'(o_fire);
  wire q_pop    = (q_cnt_q != '0) & (ro_occ < 3'(QRL + 1));

  // ST write responses (wr_rsp_ready = 1).
  wire wr_rsp_fire = ~idle_q & st_q & (|(wr_rsp_valid & ch_oh_q));

  // Completion.
  wire ld_end  = ~st_q & o_fire & (o_left_q == NW'(1));
  wire st_end  = st_q & (g_left_q == '0) & ~rq_busy & (o_left_q == '0) & (wr_pend_q == '0);
  wire fin     = ~idle_q & (ld_end | st_end);

  wire [QAW:0] q_cnt_d = q_cnt_q + (QAW+1)'(q_push) - (QAW+1)'(q_pop);

  // -------------------------------------------------------------------------------------------
  // Control registers
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      idle_q    <= 1'b1;
      done_q    <= 1'b0;
      st_q      <= 1'b0;
      ch_oh_q   <= '0;
      g_addr_q  <= '0;
      g_left_q  <= '0;
      rq_rd_v_q <= '0;
      rq_wr_v_q <= '0;
      sr_v_q    <= 1'b0;
      sr_line_q <= '0;
      sr_next_q <= '0;
      sr_left_q <= '0;
      sr_rsv_q  <= '0;
      q_wp_q    <= '0;
      q_rp_q    <= '0;
      q_cnt_q   <= '0;
      ld_free_q <= (QAW+1)'(QD);
      rv_q      <= '0;
      sk_rp_q   <= 1'b0;
      sk_wp_q   <= 1'b0;
      sk_cnt_q  <= '0;
      o_spm_v_q <= 1'b0;
      o_dat_v_q <= '0;
      o_last_q  <= 1'b0;
      o_line_q  <= '0;
      p_line_q  <= '0;
      p_lo_q    <= '0;
      p_left_q  <= '0;
      o_left_q  <= '0;
      wr_pend_q <= '0;
    end else begin
      done_q <= fin | kv_done;

      if (cmd_fire) begin
        idle_q    <= 1'b0;
        st_q      <= cmd.is_st;
        ch_oh_q   <= cmd_oh;
        g_addr_q  <= cmd_addr;
        g_left_q  <= cmd_n;
        sr_next_q <= cmd_line;
        sr_left_q <= cmd.is_st ? cmd_n : '0;
        p_line_q  <= cmd_line;
        p_lo_q    <= cmd.dram_beat[5:0];
        p_left_q  <= cmd_n;
        o_left_q  <= cmd_n;
        wr_pend_q <= '0;
      end else if (fin) begin
        idle_q    <= 1'b1;
      end

      // burst generator
      if (rq_load) begin
        rq_rd_v_q <= st_q ? '0 : ch_oh_q;
        rq_wr_v_q <= st_q ? ch_oh_q : '0;
        rq_addr_q <= g_addr_q;
        rq_len_q  <= blen;
        g_addr_q  <= g_addr_q + ADDR_BEAT_W'(blen);
        g_left_q  <= g_left_q - NW'(blen);
      end else if (rq_fire) begin
        rq_rd_v_q <= '0;
        rq_wr_v_q <= '0;
      end
      if (st_q && !idle_q) begin
        wr_pend_q <= wr_pend_q + NW'(rq_fire) - NW'(wr_rsp_fire);
      end

      // ST SPM reads
      if (sr_load) begin
        sr_v_q    <= 1'b1;
        sr_line_q <= sr_next_q;
        sr_next_q <= sr_next_q + SPM_AW'(1);
        sr_left_q <= sr_left_q - NW'(1);
      end else if (sr_fire) begin
        sr_v_q    <= 1'b0;
      end
      sr_rsv_q <= sr_rsv_q + (QAW+1)'(sr_load) - (QAW+1)'(st_push);

      // queue
      if (q_push) q_wp_q <= q_wp_q + QAW'(1);
      if (q_pop)  q_rp_q <= q_rp_q + QAW'(1);
      q_cnt_q <= q_cnt_d;
      if (!st_q) begin
        ld_free_q <= ld_free_q - ((rq_load) ? (QAW+1)'(blen) : '0) + (QAW+1)'(q_pop);
      end

      // read-out
      rv_q <= {rv_q[QRL-2:0], q_pop};
      if (sk_push) sk_wp_q <= ~sk_wp_q;
      if (sk_pop)  sk_rp_q <= ~sk_rp_q;
      sk_cnt_q <= sk_cnt_q + 2'(sk_push) - 2'(sk_pop);

      // output register
      if (o_load) begin
        o_spm_v_q <= ~st_q;
        o_dat_v_q <= st_q ? ch_oh_q : '0;
        o_line_q  <= p_line_q;
        o_last_q  <= (p_lo_q == 6'h3F) || (p_left_q == NW'(1));
        p_line_q  <= p_line_q + SPM_AW'(1);
        p_lo_q    <= p_lo_q + 6'd1;
        p_left_q  <= p_left_q - NW'(1);
      end else if (o_fire) begin
        o_spm_v_q <= 1'b0;
        o_dat_v_q <= '0;
      end
      if (o_fire) o_left_q <= o_left_q - NW'(1);
    end
  end

  // Beat queue RAM: written at q_push, read at q_pop, the data QRL cycles later (the second
  // stage is free running: the read-out takes it in its arrival cycle). A read and a write never
  // meet at one address: a pop needs a beat in the queue and a push a free entry, so wp == rp in
  // a cycle with both means empty and full at once.
  tsr_sram_1r1w #(
    .DEPTH(QD), .WIDTH(LB), .RD_LAT(QRL)
  ) u_q (
    .clk   (clk),
    .we    (q_push),
    .waddr (q_wp_q),
    .wdata (q_wdata),
    .wbe   ({(LB/8){1'b1}}),
    .re    (q_pop),
    .raddr (q_rp_q),
    .rdata (q_rdata)
  );

  // Datapath registers of the read-out (no reset; qualified by sk_cnt_q / the valid flags).
  always_ff @(posedge clk) begin
    if (sk_push) sk_data_q[sk_wp_q] <= q_rdata;
    if (o_load)  o_data_q <= (sk_cnt_q != '0) ? sk_data_q[sk_rp_q] : q_rdata;
  end

  // -------------------------------------------------------------------------------------------
  // Outputs (flops or constants)
  // -------------------------------------------------------------------------------------------
  // KVST engine (HAS_KVST): its outputs replace the LD / ST engine's while kv_act_q (header)
  logic [NUM_CH-1:0]       kv_wr_v, kv_wd_v;
  logic [ADDR_BEAT_W-1:0]  kv_wr_addr;
  logic [TSR_LEN_W-1:0]    kv_wr_len;
  logic [LB-1:0]           kv_wd_data;
  logic [LB/32-1:0]        kv_wd_mask;
  logic                    kv_wd_last, kv_sr_v;
  logic [SPM_AW-1:0]       kv_sr_line;
  if (HAS_KVST) begin : g_kv
    tsr_dma_kvst #(
      .NUM_CH(NUM_CH), .ADDR_BEAT_W(ADDR_BEAT_W), .SPM_AW(SPM_AW), .SPM_RD_LAT(SPM_RD_LAT), .FAST_CLK(FAST_CLK)
    ) u_kv (
      .clk, .rst_n,
      .start(kv_start), .cmd(tsr_att_pkg::tsr_kvst_cmd_t'(xcmd[$bits(tsr_att_pkg::tsr_kvst_cmd_t)-1:0])),
      .busy(kv_busy), .done(kv_done),
      .wr_req_valid(kv_wr_v), .wr_req_ready, .wr_req_addr(kv_wr_addr), .wr_req_len(kv_wr_len),
      .wr_dat_valid(kv_wd_v), .wr_dat_ready, .wr_dat_data(kv_wd_data), .wr_dat_mask(kv_wd_mask), .wr_dat_last(kv_wd_last),
      .wr_rsp_valid(wr_rsp_valid & {NUM_CH{kv_act_q}}),
      .spm_rd_req_valid(kv_sr_v), .spm_rd_req_ready, .spm_rd_req_line(kv_sr_line),
      .spm_rd_rsp_valid(spm_rd_rsp_valid & kv_act_q), .spm_rd_rsp_data,
      .perf_rmw(perf_kvst_rmw)
    );
    always_ff @(posedge clk) begin
      if (!rst_n) kv_act_q <= 1'b0;
      else        kv_act_q <= kv_start | (kv_act_q & ~kv_done);
    end
  end else begin : g_nokv
    assign kv_busy       = 1'b0;
    assign kv_done       = 1'b0;
    assign kv_act_q      = 1'b0;
    assign kv_wr_v       = '0;
    assign kv_wd_v       = '0;
    assign kv_wr_addr    = '0;
    assign kv_wr_len     = '0;
    assign kv_wd_data    = '0;
    assign kv_wd_mask    = '1;
    assign kv_wd_last    = 1'b0;
    assign kv_sr_v       = 1'b0;
    assign kv_sr_line    = '0;
    assign perf_kvst_rmw = 1'b0;
    wire unused_kv = &{1'b0, cmd_xc, xcmd, kv_start};
  end

  assign cmd_ready    = idle_q & ~kv_busy;
  assign done         = done_q;
  assign rd_req_valid = rq_rd_v_q;
  assign wr_req_valid = kv_act_q ? kv_wr_v : rq_wr_v_q;
  assign rd_rsp_ready = '1;                       // LD burst reservation (CC §10.4)
  assign wr_dat_valid = kv_act_q ? kv_wd_v : o_dat_v_q;
  assign wr_rsp_ready = '1;
  for (genvar c = 0; c < NUM_CH; c++) begin : g_ch
    assign rd_req_addr[c] = rq_addr_q;
    assign rd_req_len[c]  = rq_len_q;
    assign rd_req_id[c]   = '0;
    assign wr_req_addr[c] = kv_act_q ? kv_wr_addr : rq_addr_q;
    assign wr_req_len[c]  = kv_act_q ? kv_wr_len : rq_len_q;
    assign wr_req_id[c]   = '0;
    assign wr_dat_data[c] = kv_act_q ? kv_wd_data : o_data_q;
    assign wr_dat_mask[c] = kv_act_q ? kv_wd_mask : '1;
    assign wr_dat_last[c] = kv_act_q ? kv_wd_last : o_last_q;
  end
  assign spm_rd_req_valid = kv_act_q ? kv_sr_v : sr_v_q;
  assign spm_rd_req_line  = kv_act_q ? kv_sr_line : sr_line_q;
  assign spm_wr_valid     = o_spm_v_q;
  assign spm_wr_line      = o_line_q;
  assign spm_wr_data      = o_data_q;
  assign spm_wr_be        = '1;

  // Read ids/last and write ids are checked in simulation only; upper operand bits are zero by
  // the dispatch range checks.
  wire unused_ok = &{1'b0, cmd, rd_rsp_last, rd_rsp_id, wr_rsp_id, xcmd};

  // -------------------------------------------------------------------------------------------
  // Simulation checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic [5:0]    a_lo_q;     // LD: DRAM address bits [5:0] of the next response beat
  logic [NW-1:0] a_left_q;   // LD: response beats still expected
  // Checks are armed once a reset has been seen: the time-0 state of a simulation (inputs and
  // rst_n included) may be random.
  logic          a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  function automatic logic chan_bit(logic [NUM_CH-1:0] v, logic [NUM_CH-1:0] oh);
    return |(v & oh);
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      a_lo_q   <= '0;
      a_left_q <= '0;
    end else if (a_on_q) begin
      if (cmd_fire) begin
        assert (32'(cmd.ch) < NUM_CH)
          else $error("%m: command channel %0d >= NUM_CH", cmd.ch);
        assert (cmd.n_lines != 0 && 64'(cmd.spm_line) + 64'(cmd.n_lines) <= 64'(SPM_LINES))
          else $error("%m: SPM operand [%0d, +%0d) illegal", cmd.spm_line, cmd.n_lines);
        assert (64'(cmd.dram_beat) + 64'(cmd.n_lines) <= (64'd1 << ADDR_BEAT_W))
          else $error("%m: DRAM operand [0x%0h, +%0d) beyond 2^ADDR_BEAT_W", cmd.dram_beat, cmd.n_lines);
        a_lo_q   <= cmd.dram_beat[5:0];
        a_left_q <= cmd_n;
      end
      if (ld_push) begin
        assert (chan_bit(rd_rsp_last, ch_oh_q) == (a_lo_q == 6'h3F || a_left_q == NW'(1)))
          else $error("%m: rd_rsp_last does not match the burst split");
        for (int unsigned c = 0; c < NUM_CH; c++) begin
          if (ch_oh_q[c]) assert (rd_rsp_id[c] == '0) else $error("%m: read response id %0d", rd_rsp_id[c]);
        end
        assert (a_left_q != '0) else $error("%m: more read beats than requested");
        a_lo_q   <= a_lo_q + 6'd1;
        a_left_q <= a_left_q - NW'(1);
      end
      assert (!(|(rd_rsp_valid & ~ch_oh_q)) && !(|rd_rsp_valid && (idle_q || st_q)))
        else $error("%m: read response on a channel without an LD in progress");
      if (spm_rd_rsp_valid && !kv_act_q) begin
        assert (!idle_q && st_q && sr_rsv_q != '0)
          else $error("%m: SPM read response without a request");
      end
      assert (!(q_push && !q_pop && q_cnt_q == (QAW+1)'(QD)))
        else $error("%m: beat queue overflow");
      assert (!(sk_push && !sk_pop && sk_cnt_q == 2'd2)) else $error("%m: read-out skid overflow");
      // Reservation: queue entries + LD beats in flight never exceed the queue (CC §10.4).
      assert (ld_free_q <= (QAW+1)'(QD) && (st_q || idle_q || (QAW+2)'(q_cnt_q) + (QAW+2)'(ld_free_q) <= (QAW+2)'(QD)))
        else $error("%m: LD burst reservation broken (free %0d, queue %0d)", ld_free_q, q_cnt_q);
      if (idle_q) begin
        assert (ld_free_q == (QAW+1)'(QD) && q_cnt_q == '0)
          else $error("%m: beat queue not empty while idle (free %0d, queue %0d)", ld_free_q, q_cnt_q);
      end
      assert (kv_act_q || (!(|(wr_rsp_valid & ~ch_oh_q)) && !(|wr_rsp_valid && (idle_q || !st_q))))
        else $error("%m: write response on a channel without an ST in progress");
      assert (!(kv_act_q && !idle_q)) else $error("%m: LD / ST engine busy during a KVST");
      if (wr_rsp_fire) begin
        assert (wr_pend_q != '0 || rq_fire) else $error("%m: write response without a request");
        for (int unsigned c = 0; c < NUM_CH; c++) begin
          if (ch_oh_q[c]) assert (wr_rsp_id[c] == '0) else $error("%m: write response id %0d", wr_rsp_id[c]);
        end
      end
      if (fin) begin
        assert (q_cnt_q == '0 && sr_rsv_q == '0 && !sr_v_q && !rq_busy && g_left_q == '0 && rv_q == '0
                && sk_cnt_q == '0)
          else $error("%m: command finished with traffic pending");
      end
    end
  end
`endif

endmodule
