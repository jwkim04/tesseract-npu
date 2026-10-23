// tsr_dma.sv — LD / ST engine: moves lines between the SPM and one DRAM channel.
//
// Owner: DMA agent (docs/rtl-m2.md §2, block (b); dispatch §3.2; memory port rules §4).
//
// Dispatch: cmd_valid/cmd_ready accept one tsr_dma_cmd_t (cmd_ready = idle). Operands are legal
//   (tsr_cmdproc checked them; n_lines >= 1). `done` pulses one cycle when the instruction's effects
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
// SPM side: client TSR_SPM_CL_DMA of tsr_spm (read responses arrive SPM_RD_LAT cycles after the
//   request is accepted, no backpressure).
//
// Microarchitecture
//   One engine, one command at a time. Three decoupled pipelines share a beat queue (QD entries,
//   LUTRAM-style array with a registered output stage):
//
//     burst generator --rq--> rd_req[ch] (LD)  or  wr_req[ch] (ST)        (independent of data)
//     LD: rd_rsp[ch] --push--> queue --pop--> output reg --> spm_wr        (line = spm_line + i)
//     ST: spm_rd_req --SPM--> spm_rd_rsp --push--> queue --pop--> output reg --> wr_dat[ch]
//
//   - Read id scheme: every LD burst uses local id 0 (memory id {DMA, 0}). The memory returns
//     the beats of one id in request order, so the response stream IS the SPM line order: no
//     reorder buffer, no tags; up to RD_OUTSTANDING bursts in flight (limit enforced by
//     tsr_mem_arb, shared with the other sources of the channel). The burst generator issues
//     back-to-back bursts as fast as the arbiter accepts them.
//   - LD flow control (D2/R4/M6): rd_rsp_ready is a flop = "queue not full after this edge".
//     Read bursts are NOT gated on reserved queue space (full reservation would need
//     RD_OUTSTANDING x 64 beats of buffer); rd_rsp_ready depends only on the DMA's own SPM write
//     port, into which the queue drains. tsr_spm grants write client 0 (the DMA, highest fixed
//     priority) the cycle after spm_wr_valid rises and keeps the grant while it stays high, so a
//     DMA SPM write stalls at most one cycle, and only after the queue ran empty: the queue holds
//     <= 2 beats (QD >= 8) and rd_rsp_ready stays 1 for the whole LD, even with another SPM
//     writer active. The SPM write port never waits for memory traffic, so the M6 no-deadlock
//     argument holds. This is narrower than the letter of docs/rtl-m2.md D2 ("reserve
//     SPM-write/FIFO room"); the D2 amendment is proposed in the M2 DMA report. Tested with the
//     real tsr_spm and a contending writer (tb/dma sys_rtl_spm_contend). Against an SPM that
//     stalls longer (the unit tests' SPM model) the queue fills and rd_rsp_ready drops: plain
//     backpressure, no data loss. In steady state (SPM ready) the LD streams one beat per cycle.
//   - ST flow control: an SPM read is requested only while queue entries + reads in flight
//     (requested, response not yet returned) < QD, so the responses (no backpressure) always
//     fit. QD = 2^ceil(log2(SPM_RD_LAT + 4)) covers the SPM read latency at one line per cycle.
//   - ST write requests are generated from the address/length alone and data beats are offered
//     as soon as they are in the output register (rule M5: neither waits for the other's
//     handshake). wr_dat_last is computed when a beat enters the output register: last beat of
//     the command, or DRAM address % 64 == 63 (same split as the request generator: D1).
//   - Completion: LD when the n-th SPM write is accepted; ST when every burst was requested,
//     every data beat accepted and every write response received (wr_pend == 0). `done` is
//     registered (>= 1 cycle after the accepting edge), cmd_ready returns with it.
//   - Every output is a flop (or a constant): per-channel valid flops, shared payload flops.
//     No combinational path from any input to any output (R5); the arbiters can sit in other
//     SLRs. wr_rsp_ready is constant 1 (a response only decrements a counter).
//   - Widths: line/beat counters NW = SPM_AW + 1 bits (n_lines <= SPM lines); the DRAM address
//     counter is ADDR_BEAT_W bits (dram_beat + n_lines may exceed 2^32 when ADDR_BEAT_W > 32:
//     the operand is a 32-bit start, the range check is 64-bit, see spec 범위 검사).
//   - Simulation-only checks (`ifndef SYNTHESIS): command legality, read id/last consistency
//     with the D1 split, responses only on the active channel, queue overflow, write responses.

module tsr_dma
  import tsr_pkg::*;
#(
  parameter  int unsigned NUM_CH      = TSR_DEF_NUM_CH,
  parameter  int unsigned ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter  int unsigned BEAT_BITS   = TSR_DEF_BEAT_BITS,
  parameter  int unsigned SPM_BYTES   = TSR_DEF_SPM_BYTES,
  parameter  int unsigned SPM_RD_LAT  = TSR_DEF_SPM_RD_LAT,
  localparam int unsigned SPM_LINES   = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned SPM_AW      = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1
) (
  input  logic                                  clk,
  input  logic                                  rst_n,

  // ---- dispatch from tsr_cmdproc ----
  input  logic                                  cmd_valid,
  output logic                                  cmd_ready,
  input  tsr_dma_cmd_t                          cmd,
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
  output logic [TSR_SPM_BE_W-1:0]               spm_wr_be
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

  localparam int unsigned NW  = SPM_AW + 1;                       // counts 0 .. SPM_LINES
  localparam int unsigned CW  = (NW > TSR_LEN_W) ? NW : TSR_LEN_W;
  localparam int unsigned QAW = $clog2(SPM_RD_LAT + 4);
  localparam int unsigned QD  = 1 << QAW;                         // queue entries
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
  logic [LB-1:0]           q_mem [QD];
  logic [QAW-1:0]          q_wp_q, q_rp_q;
  logic [QAW:0]            q_cnt_q;
  logic                    q_rdy_q;         // = rd_rsp_ready (queue not full)

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
  wire                    cmd_fire = cmd_valid & idle_q;
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
  wire rq_load = ~idle_q & (~rq_busy | rq_fire) & (g_left_q != '0);

  // -------------------------------------------------------------------------------------------
  // Queue input: LD read data of the active channel or ST SPM read data
  // -------------------------------------------------------------------------------------------
  logic [LB-1:0] rsp_data;
  always_comb begin
    rsp_data = '0;
    for (int unsigned c = 0; c < NUM_CH; c++) rsp_data |= rd_rsp_data[c] & {LB{ch_oh_q[c]}};
  end
  wire           ld_push = ~idle_q & ~st_q & q_rdy_q & (|(rd_rsp_valid & ch_oh_q));
  wire           st_push = ~idle_q & st_q & spm_rd_rsp_valid;
  wire           q_push  = ld_push | st_push;
  wire [LB-1:0]  q_wdata = st_q ? spm_rd_rsp_data : rsp_data;

  // ST SPM read requests: only while the responses are guaranteed to fit.
  wire [QAW+1:0] q_occ   = (QAW+2)'(q_cnt_q) + (QAW+2)'(sr_rsv_q);
  wire           sr_fire = sr_v_q & spm_rd_req_ready;
  wire           sr_load = ~idle_q & st_q & (~sr_v_q | sr_fire) & (sr_left_q != '0)
                         & (q_occ < (QAW+2)'(QD));

  // Output register.
  wire o_valid = o_spm_v_q | (|o_dat_v_q);
  wire o_fire  = (o_spm_v_q & spm_wr_ready) | (|(o_dat_v_q & wr_dat_ready));
  wire q_pop   = (q_cnt_q != '0) & (~o_valid | o_fire);

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
      q_rdy_q   <= 1'b1;
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
      done_q <= fin;

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
      q_rdy_q <= (q_cnt_d != (QAW+1)'(QD));

      // output register
      if (q_pop) begin
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

  // Datapath registers (no reset; qualified by the valid flags above).
  always_ff @(posedge clk) begin
    if (q_push) q_mem[q_wp_q] <= q_wdata;
    if (q_pop)  o_data_q      <= q_mem[q_rp_q];
  end

  // -------------------------------------------------------------------------------------------
  // Outputs (flops or constants)
  // -------------------------------------------------------------------------------------------
  assign cmd_ready    = idle_q;
  assign done         = done_q;
  assign rd_req_valid = rq_rd_v_q;
  assign wr_req_valid = rq_wr_v_q;
  assign rd_rsp_ready = {NUM_CH{q_rdy_q}};
  assign wr_dat_valid = o_dat_v_q;
  assign wr_rsp_ready = '1;
  for (genvar c = 0; c < NUM_CH; c++) begin : g_ch
    assign rd_req_addr[c] = rq_addr_q;
    assign rd_req_len[c]  = rq_len_q;
    assign rd_req_id[c]   = '0;
    assign wr_req_addr[c] = rq_addr_q;
    assign wr_req_len[c]  = rq_len_q;
    assign wr_req_id[c]   = '0;
    assign wr_dat_data[c] = o_data_q;
    assign wr_dat_last[c] = o_last_q;
  end
  assign spm_rd_req_valid = sr_v_q;
  assign spm_rd_req_line  = sr_line_q;
  assign spm_wr_valid     = o_spm_v_q;
  assign spm_wr_line      = o_line_q;
  assign spm_wr_data      = o_data_q;
  assign spm_wr_be        = '1;

  // Read ids/last and write ids are checked in simulation only; upper operand bits are zero by
  // the cmdproc range checks.
  wire unused_ok = &{1'b0, cmd, rd_rsp_last, rd_rsp_id, wr_rsp_id};

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
      if (spm_rd_rsp_valid) begin
        assert (!idle_q && st_q && sr_rsv_q != '0)
          else $error("%m: SPM read response without a request");
      end
      assert (!(q_push && !q_pop && q_cnt_q == (QAW+1)'(QD)))
        else $error("%m: beat queue overflow");
      assert (!(|(wr_rsp_valid & ~ch_oh_q)) && !(|wr_rsp_valid && (idle_q || !st_q)))
        else $error("%m: write response on a channel without an ST in progress");
      if (wr_rsp_fire) begin
        assert (wr_pend_q != '0 || rq_fire) else $error("%m: write response without a request");
        for (int unsigned c = 0; c < NUM_CH; c++) begin
          if (ch_oh_q[c]) assert (wr_rsp_id[c] == '0) else $error("%m: write response id %0d", wr_rsp_id[c]);
        end
      end
      if (fin) begin
        assert (q_cnt_q == '0 && sr_rsv_q == '0 && !sr_v_q && !rq_busy && g_left_q == '0)
          else $error("%m: command finished with traffic pending");
      end
    end
  end
`endif

endmodule
