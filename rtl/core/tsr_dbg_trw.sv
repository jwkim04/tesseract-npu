// tsr_dbg_trw.sv — DB's trace writer (docs/design/05-debug.md "DB" §8.3, DB-16): 32-byte records
// from the serializer (tsr_dbg_shq) into a staging buffer, then in bursts into the host-allocated
// DRAM ring {DBG_TRW_BASE, DBG_TRW_LINES} through a write source of the channel TRW_CH = 0
// arbiter (tsr_dbg_wrmux); DBG_TRW_WPTR, DBG_TRW_DROPS, the DROP records and PERF TRW_RECORDS /
// TRW_DROPS.
//
// Owner: DB (W4-B item 6). Instantiated by tsr_dbg. Oracle: the simulator's DBG model
// (sw/sim/tsr_dbg.c trw_stage / trw_flush / trw_put / trw_drop): tb/dbg compares the ring
// records in [RPTR, WPTR) (times excluded), WPTR and DROPS.
//
// Staging: 64 records (LUTRAM 64 x 256 b, two 1 KiB bursts). A record arriving with the staging
//   full is dropped. Before the first record staged after drops, a DROP record {KIND 4, the
//   records dropped since the previous DROP record, time} is staged (the input waits one cycle).
// Write triggers (the simulator flushes its 32-record staging at once at these points): enough
//   records to fill the ring's current 16-beat group (32 when the ring is aligned), DBG_CTRL
//   TRW_FLUSH, a LAUNCH_END marker (after it), TRW_EN falling, 65,536 cycles since the last
//   staged record. A trigger takes a batch (all staged records for a flush, else one group) and
//   writes or drops each record of it in order: a record takes ring position WPTR' mod capacity
//   (WPTR' = records issued; capacity = 2 x DBG_TRW_LINES) or is dropped when the ring is absent
//   (DBG_TRW_LINES = 0, a channel field other than TRW_CH, or a ring that does not fit in the
//   channel; the simulator checks the channel per record) or, with TRW_STOP_FULL, full (WPTR' -
//   RPTR = capacity). A dropped DROP record adds its count to the next one instead of counting.
// Bursts: the ring is 16-beat aligned and a multiple of 16 beats, so a burst never crosses a
//   16-beat group (hence never a 64-beat block or the ring end): beats [b, min(b + n, group end)).
//   The memory port writes whole beats (a full word mask, rtl-core §4 M4): a batch that ends with an
//   odd record writes a beat {record, PAD} and keeps the record (carry); the next burst starts by
//   rewriting that beat {record, next} (same AXI ID, so the second write lands last). The host
//   reads only [RPTR, WPTR), never the PAD half (the simulator writes records singly, byte-enabled,
//   so the ring contents in [RPTR, WPTR) are the same).
// WPTR = records whose write response returned (one response per burst, in order); PERF
//   TRW_RECORDS counts new records per beat sent, TRW_DROPS every dropped record (DROP records
//   excluded); DBG_TRW_DROPS saturates.

module tsr_dbg_trw
  import tsr_pkg::*;
#(
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter longint unsigned CH_BYTES    = TSR_DEF_CH_BYTES,
  parameter int unsigned     TRW_CH      = tsr_tables_pkg::TSR_TRW_CH,
  // FAST_CLK (docs/platform-u200.md §8.10): the DROPS counter adds the drops of the cycle before (a statistic: the counter
  // reads one cycle later; DROP records and the staging decisions use dpend_q as before)
  parameter bit              FAST_CLK    = 1'b0
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic [63:0]             mtime,

  // ---- configuration (DBG_CFG, DBG_TRW_*) ----
  input  logic                    cfg_en,
  input  logic                    cfg_stop_full,
  input  logic [31:0]             base,
  input  logic [31:0]             lines,
  input  logic [31:0]             rptr,
  input  logic                    flush,           // DBG_CTRL.TRW_FLUSH

  // ---- records (tsr_dbg_shq) ----
  input  logic                    in_v,
  input  logic [255:0]            in_rec,
  input  logic                    in_flush,
  output logic                    in_ready,
  input  logic                    in_lost,         // a record lost before the writer

  // ---- state ----
  output logic [31:0]             wptr,
  output logic [31:0]             drops,
  output logic                    drop_ev,         // DBG_STATUS.TRW_DROP
  output logic [1:0]              perf_rec,        // TRW_RECORDS increment
  output logic [1:0]              perf_drop,       // TRW_DROPS increment

  // ---- write source (tsr_dbg_wrmux) ----
  output logic                    wr_req_valid,
  input  logic                    wr_req_ready,
  output logic [ADDR_BEAT_W-1:0]  wr_req_addr,
  output logic [TSR_LEN_W-1:0]    wr_req_len,
  output logic                    wr_dat_valid,
  input  logic                    wr_dat_ready,
  output logic [511:0]            wr_dat_data,
  output logic                    wr_dat_last,
  input  logic                    wr_rsp_valid
);

  localparam logic [3:0] K_DROP = tsr_tables_pkg::TSR_TRW_W0_KIND_DROP;
  localparam logic [3:0] K_PAD  = tsr_tables_pkg::TSR_TRW_W0_KIND_PAD;
  localparam logic [255:0] PAD_REC = {252'd0, K_PAD};
  localparam logic [40:0] CH_BEATS = 41'(CH_BYTES / 64);
  localparam int unsigned SDEPTH = 64;

  // -------------------------------------------------------------------------------------------
  // Staging
  // -------------------------------------------------------------------------------------------
  logic [255:0] stg [SDEPTH];
  logic [6:0]   swr_q, srd_q;
  wire  [6:0]   ns = swr_q - srd_q;
  wire          sfull = (ns == 7'(SDEPTH));

  logic [31:0]  dpend_q;                 // records dropped since the last DROP record
  logic [16:0]  idle_q;                  // cycles since the last staged record (saturating at 2^16)
  logic         flush_q, en_q;

  wire          need_drop = (dpend_q != 32'd0);
  wire          put_drop  = in_v && need_drop && !sfull;            // stage a DROP record first
  wire          put_rec   = in_v && !need_drop && !sfull;
  wire          drop_in   = in_v && sfull;                          // dropped at the intake
  assign in_ready = !put_drop;

  // the writer's per-cycle drop of a staged record (below)
  logic         w_drop, w_drop_isd;
  logic [31:0]  w_drop_cnt;

  wire [1:0]  ndrop   = 2'(drop_in) + 2'(in_lost) + 2'(w_drop && !w_drop_isd);
  logic [31:0] drops_q;
  logic [1:0] ndrop_q;
  if (FAST_CLK) begin : g_nd
    always_ff @(posedge clk) begin
      if (!rst_n) ndrop_q <= '0;
      else        ndrop_q <= ndrop;
    end
  end else begin : g_nd0
    assign ndrop_q = ndrop;
  end
  wire [33:0] dsum    = 34'(drops_q) + 34'(ndrop_q);
  wire [31:0] dpadd   = 32'(ndrop) + ((w_drop && w_drop_isd) ? w_drop_cnt : 32'd0);

  always_ff @(posedge clk) begin
    if (put_drop)     stg[swr_q[5:0]] <= {96'd0, mtime[63:32], mtime[31:0], 32'd0, dpend_q, 28'd0, K_DROP};
    else if (put_rec) stg[swr_q[5:0]] <= in_rec;
  end

  // -------------------------------------------------------------------------------------------
  // Writer
  // -------------------------------------------------------------------------------------------
  typedef enum logic [1:0] { W_IDLE, W_DO, W_REQ, W_DAT } wstate_e;
  wstate_e      ws_q;
  logic [31:0]  wq_q, wptr_q;            // records issued / acknowledged
  logic [255:0] carry_q;                 // the low half of beat (wq - 1) / 2 while wq is odd
  logic [6:0]   batch_q;                 // records of the current trigger still to write or drop
  logic [5:0]   bn_q;                    // new records of the current burst still to place
  logic         bfirst_q;                // the next beat is the burst's first
  logic [ADDR_BEAT_W-1:0] baddr_q;
  logic [4:0]   blen_q;

  // ring
  wire [31:0] cap      = {lines[30:0], 1'b0};                       // records
  wire [31:0] pos      = wq_q & (cap - 32'd1);
  wire [27:0] b_beat   = {base[27:4], 4'd0};
  wire        ring_ok  = (lines != 32'd0) && (32'(base[31:28]) == TRW_CH) &&
                         (41'(b_beat) + 41'(lines) <= CH_BEATS);
  // registered ring state (timing): the writer acts in W_DO only after one settled cycle, so the
  // registers reflect the current WPTR'; a stale RPTR only underestimates the room (it only grows)
  logic        ring_ok_q, room_ok_q, settled_q;
  logic [31:0] room_q;
  wire  [31:0] used     = wq_q - rptr;
  always_ff @(posedge clk) begin
    ring_ok_q <= ring_ok;
    room_ok_q <= !cfg_stop_full || (used < cap);
    room_q    <= cfg_stop_full ? (cap - used) : 32'd64;
  end
  wire        room_ok  = room_ok_q;
  wire [31:0] room     = room_q;
  wire        odd      = pos[0];
  wire [4:0]  gleft    = 5'd16 - {1'b0, pos[4:1]};                  // beats left in the 16-beat group
  wire [5:0]  gmax     = {gleft, 1'b0} - 6'(odd);                   // new records that fit in it
  wire [6:0]  fill     = ring_ok_q ? {1'b0, gmax} : 7'd32;

  // a trigger: enough to fill the group, or a flush with something staged
  logic idle_flush;
  assign idle_flush = (ns != 7'd0) && idle_q[16];
  wire trig = (ws_q == W_IDLE) && (ns != 7'd0) && (flush_q || ns >= fill);

  // the burst at W_DO
  wire [6:0]  b_lim1 = (batch_q < {1'b0, gmax}) ? batch_q : {1'b0, gmax};
  wire [6:0]  b_n    = (32'(b_lim1) < room) ? b_lim1 : 7'(room);
  wire        do_drop = (ws_q == W_DO) && settled_q && (batch_q != 7'd0) && (!ring_ok_q || !room_ok);

  assign w_drop     = do_drop;
  assign w_drop_isd = (stg[srd_q[5:0]][3:0] == K_DROP);
  assign w_drop_cnt = stg[srd_q[5:0]][63:32];

  // beat assembly: two staging reads
  wire [255:0] r0 = stg[srd_q[5:0]];
  wire [255:0] r1 = stg[6'(srd_q[5:0] + 6'd1)];
  logic [255:0] lo_rec, hi_rec;
  logic [1:0]   bnew;                      // new records in this beat
  logic         hi_pad;                    // the high half is a PAD record
  always_comb begin
    if (bfirst_q && odd) begin
      lo_rec = carry_q;
      hi_pad = (bn_q == 6'd0);
      hi_rec = hi_pad ? PAD_REC : r0;
      bnew   = hi_pad ? 2'd0 : 2'd1;
    end else begin
      lo_rec = r0;
      hi_pad = (bn_q < 6'd2);
      hi_rec = hi_pad ? PAD_REC : r1;
      bnew   = hi_pad ? 2'd1 : 2'd2;
    end
  end
  wire beat_go = (ws_q == W_DAT) && wr_dat_ready;
  wire last_beat = (bn_q <= 6'(bnew));

  // outstanding bursts: their record counts, in order
  logic [5:0] ob_n [4];
  logic [1:0] ob_wr_q, ob_rd_q;
  logic [2:0] ob_cnt_q;
  wire        ob_full = (ob_cnt_q == 3'd4);

  wire [6:0] pop_n = beat_go ? 7'(bnew) : (do_drop ? 7'd1 : 7'd0);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      swr_q    <= '0;
      srd_q    <= '0;
      dpend_q  <= '0;
      drops_q  <= '0;
      idle_q   <= '0;
      flush_q  <= 1'b0;
      en_q     <= 1'b0;
      ws_q     <= W_IDLE;
      settled_q <= 1'b0;
      wq_q     <= '0;
      wptr_q   <= '0;
      batch_q  <= '0;
      bn_q     <= '0;
      bfirst_q <= 1'b0;
      ob_wr_q  <= '0;
      ob_rd_q  <= '0;
      ob_cnt_q <= '0;
    end else begin
      en_q <= cfg_en;
      // staging
      if (put_drop || put_rec) swr_q <= swr_q + 7'd1;
      srd_q <= srd_q + pop_n;
      dpend_q <= (put_drop ? 32'd0 : dpend_q) + dpadd;
      drops_q <= dsum[33:32] != 2'd0 ? 32'hFFFF_FFFF : dsum[31:0];
      if (put_drop || put_rec) idle_q <= '0;
      else if (ns != 7'd0 && !idle_q[16]) idle_q <= idle_q + 17'd1;
      // flush requests (one with nothing staged is void: the next record waits as usual)
      if (trig || (ws_q == W_IDLE && ns == 7'd0)) flush_q <= 1'b0;
      if (((flush || (en_q && !cfg_en)) && (ns != 7'd0 || put_rec || put_drop)) || (put_rec && in_flush) ||
          idle_flush)
        flush_q <= 1'b1;
      if (trig && idle_flush) idle_q <= '0;
      // writer
      settled_q <= (ws_q == W_DO) && !(settled_q && !do_drop && batch_q != 7'd0 && !ob_full);
      case (ws_q)
        W_IDLE: if (trig) begin
          batch_q <= flush_q ? ns : fill;
          ws_q    <= W_DO;
        end
        W_DO: begin
          if (batch_q == 7'd0) begin
            ws_q <= W_IDLE;
          end else if (!settled_q) begin
            // one cycle for the registered ring state
          end else if (do_drop) begin
            batch_q <= batch_q - 7'd1;
          end else if (!ob_full) begin
            bn_q     <= 6'(b_n);
            bfirst_q <= 1'b1;
            baddr_q  <= ADDR_BEAT_W'(41'(b_beat) + 41'(pos[31:1]));
            blen_q   <= 5'((7'(odd) + b_n + 7'd1) >> 1);
            batch_q  <= batch_q - b_n;
            ws_q     <= W_REQ;
          end
        end
        W_REQ: if (wr_req_ready) ws_q <= W_DAT;
        W_DAT: if (beat_go) begin
          bfirst_q <= 1'b0;
          bn_q     <= bn_q - 6'(bnew);
          wq_q     <= wq_q + 32'(bnew);
          if (last_beat) ws_q <= W_DO;
        end
        default: ws_q <= W_IDLE;
      endcase
      // outstanding bursts
      if (ws_q == W_REQ && wr_req_ready) begin
        ob_n[ob_wr_q] <= bn_q;
        ob_wr_q <= ob_wr_q + 2'd1;
      end
      if (wr_rsp_valid) begin
        wptr_q  <= wptr_q + 32'(ob_n[ob_rd_q]);
        ob_rd_q <= ob_rd_q + 2'd1;
      end
      ob_cnt_q <= ob_cnt_q + 3'(ws_q == W_REQ && wr_req_ready) - 3'(wr_rsp_valid);
    end
  end
  // the carry: the last new record of a beat written with a PAD half
  always_ff @(posedge clk) begin
    if (beat_go && hi_pad && !(bfirst_q && odd)) carry_q <= lo_rec;
  end

  assign wr_req_valid = (ws_q == W_REQ);
  assign wr_req_addr  = baddr_q;
  assign wr_req_len   = TSR_LEN_W'(blen_q);
  assign wr_dat_valid = (ws_q == W_DAT);
  assign wr_dat_data  = {hi_rec, lo_rec};
  assign wr_dat_last  = last_beat;

  assign wptr      = wptr_q;
  assign drops     = drops_q;
  assign drop_ev   = (ndrop != 2'd0);
  assign perf_rec  = beat_go ? bnew : 2'd0;
  assign perf_drop = ndrop;

  wire unused_t = &{1'b0, base[3:0], lines[31], pos[31:5], room[31:7]};

`ifndef SYNTHESIS
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      assert (!(wr_rsp_valid && ob_cnt_q == 3'd0)) else $error("%m: write response without a burst");
      assert (!(ws_q == W_DAT && bn_q == 6'd0 && !(bfirst_q && odd))) else $error("%m: empty beat");
    end
  end
`endif

endmodule
