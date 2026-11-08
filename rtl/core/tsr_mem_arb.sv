// tsr_mem_arb.sv — per-channel memory-port arbiter: TSR_NSRC read sources + one write source
// onto one channel memory port of the core boundary. Also defines tsr_rslice (below), the
// full-throughput valid/ready register slice used by the arbiter.
//
// Owner: DMA agent (docs/rtl-m2.md §2, block (b); memory port rules §4).
//
// Read sources (index = tsr_pkg TSR_SRC_*): 0 AUX (freed by the fetch removal, tied off by
// tsr_core: CC-26), 1 DMA, 2 WS (M3 weight stream; tied off in M2). Write source: DMA only.
// Contract:
//   - Read requests: round-robin between requesting sources, one burst per grant. A grant is
//     given only while fewer than RD_OUTSTANDING read requests are outstanding on this channel
//     (outstanding = memory-side request accepted, last beat not yet accepted), across all sources.
//     Memory-side id = {source[1:0], source-local id[1:0]} (TSR_MEM_ID_W = 4).
//   - Read responses: routed by m_rd_rsp_id[3:2] to s_rd_rsp_valid[source]; data/last/local id are
//     shared (s_rd_rsp_*), qualified by the per-source valid. m_rd_rsp_ready follows the ready of
//     the addressed source (through a register stage). Sources may deassert their ready only for
//     reasons local to that source (rule R4 in docs/rtl-m2.md: they reserve buffer space before
//     requesting), so routing never deadlocks across sources.
//   - Writes: s_wr_* pass to m_wr_* with id = {TSR_SRC_DMA, local}; request, data and response
//     channels are independent valid/ready streams (data beats follow request order, no
//     interleaving; the core never waits for wr_req_ready before offering wr_dat and vice versa).
//   - Requests are forwarded unchanged (sources already split bursts: 1..64 beats, inside one
//     4 KiB block, inside the channel limit). The arbiter asserts on violations in simulation.
//   - Every m_* output is driven from a register (timing: the memory port crosses SLRs).
//   - perf_rd_beat: 1 in each cycle with a memory-side read beat handshake.
//     perf_rd_wait: 1 in each cycle with >= 1 outstanding read and no read beat handshake.
//
// Microarchitecture (every stream crosses one tsr_rslice: registered valid/payload AND registered
// ready on both sides, full throughput; no combinational path from any input to any output, so
// the source side (core, possibly another SLR) and the memory side are both flop-to-flop):
//
//   s_rd_req[i] -> rslice[i] --+                                     +-> rslice -> m_rd_req
//                              +-> round robin + outstanding limit --+
//   m_rd_rsp -> rslice -> route by id[3:2] -> s_rd_rsp_valid[src], shared s_rd_rsp_data/last/id
//   s_wr_req -> rslice -> m_wr_req (id {DMA, local});  s_wr_dat -> rslice -> m_wr_dat
//   m_wr_rsp -> rslice -> s_wr_rsp (local id)
//
//   - Per-source input slices: s_rd_req_ready is a flop; the round robin works on the slice
//     heads (flops). The last granted source has the lowest priority next time.
//   - Outstanding limit: rd_cnt counts requests from the GRANT (entry into the m-side slice) to
//     the memory-side handshake of their last beat. It is >= the memory-side outstanding count
//     at all times (requests waiting in the m-side slice are included), so the memory never sees
//     more than RD_OUTSTANDING; a grant needs rd_cnt < RD_OUTSTANDING.
//   - Response routing: head-of-line. A response whose source is not ready waits (that source
//     drops ready only for local reasons, rule M6); the slice keeps the memory side at full rate.
//   - perf_rd_beat / perf_rd_wait are registered (one cycle late; counts are unchanged).
//     "Outstanding" for perf_rd_wait = rd_cnt != 0, i.e. it includes requests granted but still
//     waiting for m_rd_req_ready (the channel is waiting for the memory in both cases).
//   - Latency: request 2 cycles (input slice + output slice), read data 1 cycle, write data /
//     request 1 cycle, write response 1 cycle.
//   - M3: the WS source is a full peer of AUX/DMA (own slice, same round robin, same shared
//     outstanding budget); in M2 it is tied off (valid 0, ready 1) and its slice is constant.

// tsr_rslice — valid/ready register slice, 2 entries (main + skid), 100 % throughput.
// s_ready, m_valid and m_data are flops; m_valid/m_data hold while m_valid && !m_ready (R2).
// Reset clears both entries (s_ready = 1 after reset). m_data is don't-care while m_valid = 0.
// Clock enables (M3 part C timing, docs/platform-u200.md §8.2): the payload registers load on
// signals of this slice only, never on s_valid, which is the previous stage's main_v_q in a chain
// and would fan out to every payload bit's enable (the M0 link's worst kernel path, ~530 loads):
//   main_q loads whenever the main entry is free or handed over (m_free): with an item (skid
//   entry first, else the incoming one) exactly as before, else it takes don't-care data while
//   main_v_q becomes 0; with m_ready constant 1 the enable is constant;
//   skid_q loads whenever the skid entry is empty (rdy_q); it is read only while full (!rdy_q)
//   and holds exactly the item parked by the edge that cleared rdy_q.
// Transfers, their order and every valid / ready flop are unchanged (cycle for cycle); m_data
// differs only while m_valid = 0 (it then follows s_data one cycle late instead of holding the
// last item).
// FORM (W4-B at, CC §14.4 / CC-25): 0 = main + skid (above); 1 = two slots s0 / s1 with a write
//   pointer, a read pointer and an occupancy flop per slot: slot i loads s_data whenever it is free
//   (its enable is its own occupancy flop), m_data = slot[rp] (one 2:1 mux level after flops), and
//   m_ready / s_valid reach only the five control flops (rp, wp, the two occupancies, rdy_q), not the
//   W payload enables. Both forms are 2-entry FIFOs with registered s_ready = (count < 2) and
//   m_valid = (count > 0), so FORM 1 is cycle-identical at the ports: same transfers in the same
//   cycles, same data order; m_data differs only while m_valid = 0. FORM 1 is for slices whose
//   m_ready is a wide take (the gather's job-pipe heads, docs/rtl-m3.md §14.8: the k4 link's -0.285 ns
//   paths were m_ready -> the 577-bit main_q enable of g_s[1]).
// Waiver: this generic helper lives in tsr_mem_arb.sv only because rtl/filelist.f is owned by the
// lead (docs/rtl-m2.md §1 "Files"); proposal: move it to rtl/core/tsr_rslice.sv (filelist entry
// before tsr_mem_arb.sv) and drop the DECLFILENAME waiver.
/* verilator lint_off DECLFILENAME */
module tsr_rslice #(
  parameter int unsigned W    = 8,
  parameter int unsigned FORM = 0,     // 0 main + skid, 1 two slots with pointers (cycle-identical)
  // FORM 1: RPC copies of the read pointer and of the occupancy flops (`keep`), each serving one slice of the payload: the
  // output mux select and the slot enables then have W / RPC loads instead of W (cycle-identical; the 250 MHz work,
  // docs/platform-u200.md §8.10). 1 = the single flops.
  parameter int unsigned RPC  = 1
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         s_valid,
  output logic         s_ready,
  input  logic [W-1:0] s_data,
  output logic         m_valid,
  input  logic         m_ready,
  output logic [W-1:0] m_data
);

  if (FORM > 1) begin : g_chk_form
    $fatal(1, "tsr_rslice: FORM must be 0 or 1");
  end

  if (FORM == 0) begin : g_f0
    logic         main_v_q;      // main entry valid (= m_valid)
    logic         rdy_q;         // skid entry empty (= s_ready)
    logic [W-1:0] main_q, skid_q;

    wire m_free = ~main_v_q | m_ready;   // main entry free or handed over at this edge
    wire s_fire = s_valid & rdy_q;

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        main_v_q <= 1'b0;
        rdy_q    <= 1'b1;
      end else if (m_free) begin
        main_v_q <= ~rdy_q | s_fire;     // skid entry first, else the incoming item
        rdy_q    <= 1'b1;
      end else if (s_fire) begin
        rdy_q    <= 1'b0;                // main stalled: incoming item parks in the skid entry
      end
    end

    // Datapath: no reset (qualified by the valid flags); enables from this slice's flops (header).
    always_ff @(posedge clk) begin
      if (m_free) main_q <= rdy_q ? s_data : skid_q;
      if (rdy_q)  skid_q <= s_data;
    end

    assign s_ready = rdy_q;
    assign m_valid = main_v_q;
    assign m_data  = main_q;
  end else begin : g_f1
    logic         wp_q, rp_q;    // write / read slot
    logic [1:0]   occ_q;         // slot i holds an item
    logic         rdy_q;         // count < 2 (= s_ready)
    logic         mv_q;          // count > 0 (= m_valid)
    logic [W-1:0] slot_q [2];

    wire s_fire = s_valid & rdy_q;
    wire m_fire = mv_q & m_ready;

    logic [1:0] occ_n;
    always_comb begin
      occ_n = occ_q;
      if (m_fire) occ_n[rp_q] = 1'b0;
      if (s_fire) occ_n[wp_q] = 1'b1;
    end

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        wp_q  <= 1'b0;
        rp_q  <= 1'b0;
        occ_q <= 2'b00;
        rdy_q <= 1'b1;
        mv_q  <= 1'b0;
      end else begin
        if (s_fire) wp_q <= ~wp_q;
        if (m_fire) rp_q <= ~rp_q;
        occ_q <= occ_n;
        rdy_q <= ~&occ_n;
        mv_q  <= |occ_n;
      end
    end

    // A free slot loads every cycle (enable = its occupancy flop); the edge that fills it (s_fire
    // with wp_q at it) leaves exactly that item in it, held while occupied.
    if (RPC <= 1) begin : g_one
      always_ff @(posedge clk) begin
        if (!occ_q[0]) slot_q[0] <= s_data;
        if (!occ_q[1]) slot_q[1] <= s_data;
      end
      assign m_data = slot_q[rp_q];
    end else begin : g_cp
      localparam int unsigned CWD = (W + RPC - 1) / RPC;
      (* keep = "true" *) logic [RPC-1:0]      rpc_q;      // copies of rp_q
      (* keep = "true" *) logic [RPC-1:0][1:0] occc_q;     // copies of occ_q
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          rpc_q  <= '0;
          occc_q <= '0;
        end else begin
          if (m_fire) rpc_q <= {RPC{~rp_q}};
          occc_q <= {RPC{occ_n}};
        end
      end
      for (genvar j = 0; j < RPC; j++) begin : g_c
        localparam int unsigned LO = j * CWD;
        localparam int unsigned HI = (LO + CWD > W) ? W - 1 : LO + CWD - 1;
        if (LO <= HI) begin : g_w
          always_ff @(posedge clk) begin
            if (!occc_q[j][0]) slot_q[0][HI:LO] <= s_data[HI:LO];
            if (!occc_q[j][1]) slot_q[1][HI:LO] <= s_data[HI:LO];
          end
          assign m_data[HI:LO] = slot_q[rpc_q[j]][HI:LO];
        end
      end
    end

    assign s_ready = rdy_q;
    assign m_valid = mv_q;
  end

endmodule
/* verilator lint_on DECLFILENAME */

module tsr_mem_arb
  import tsr_pkg::*;
#(
  parameter int unsigned ADDR_BEAT_W    = TSR_DEF_ADDR_BEAT_W,
  parameter int unsigned BEAT_BITS      = TSR_DEF_BEAT_BITS,
  parameter int unsigned RD_OUTSTANDING = TSR_DEF_RD_OUTSTANDING
) (
  input  logic                                  clk,
  input  logic                                  rst_n,

  // ---- read sources ----
  input  logic [TSR_NSRC-1:0]                   s_rd_req_valid,
  output logic [TSR_NSRC-1:0]                   s_rd_req_ready,
  input  logic [TSR_NSRC-1:0][ADDR_BEAT_W-1:0]  s_rd_req_addr,
  input  logic [TSR_NSRC-1:0][TSR_LEN_W-1:0]    s_rd_req_len,
  input  logic [TSR_NSRC-1:0][TSR_SRC_ID_W-1:0] s_rd_req_id,
  output logic [TSR_NSRC-1:0]                   s_rd_rsp_valid,
  input  logic [TSR_NSRC-1:0]                   s_rd_rsp_ready,
  output logic [BEAT_BITS-1:0]                  s_rd_rsp_data,
  output logic                                  s_rd_rsp_last,
  output logic [TSR_SRC_ID_W-1:0]               s_rd_rsp_id,

  // ---- write source (DMA) ----
  input  logic                                  s_wr_req_valid,
  output logic                                  s_wr_req_ready,
  input  logic [ADDR_BEAT_W-1:0]                s_wr_req_addr,
  input  logic [TSR_LEN_W-1:0]                  s_wr_req_len,
  input  logic [TSR_SRC_ID_W-1:0]               s_wr_req_id,
  input  logic                                  s_wr_dat_valid,
  output logic                                  s_wr_dat_ready,
  input  logic [BEAT_BITS-1:0]                  s_wr_dat_data,
  input  logic [BEAT_BITS/32-1:0]               s_wr_dat_mask,
  input  logic                                  s_wr_dat_last,
  output logic                                  s_wr_rsp_valid,
  input  logic                                  s_wr_rsp_ready,
  output logic [TSR_SRC_ID_W-1:0]               s_wr_rsp_id,

  // ---- channel memory port (core boundary) ----
  output logic                                  m_rd_req_valid,
  input  logic                                  m_rd_req_ready,
  output logic [ADDR_BEAT_W-1:0]                m_rd_req_addr,
  output logic [TSR_LEN_W-1:0]                  m_rd_req_len,
  output logic [TSR_MEM_ID_W-1:0]               m_rd_req_id,
  input  logic                                  m_rd_rsp_valid,
  output logic                                  m_rd_rsp_ready,
  input  logic [BEAT_BITS-1:0]                  m_rd_rsp_data,
  input  logic [TSR_MEM_ID_W-1:0]               m_rd_rsp_id,
  input  logic                                  m_rd_rsp_last,
  output logic                                  m_wr_req_valid,
  input  logic                                  m_wr_req_ready,
  output logic [ADDR_BEAT_W-1:0]                m_wr_req_addr,
  output logic [TSR_LEN_W-1:0]                  m_wr_req_len,
  output logic [TSR_MEM_ID_W-1:0]               m_wr_req_id,
  output logic                                  m_wr_dat_valid,
  input  logic                                  m_wr_dat_ready,
  output logic [BEAT_BITS-1:0]                  m_wr_dat_data,
  output logic [BEAT_BITS/32-1:0]                  m_wr_dat_mask,
  output logic                                  m_wr_dat_last,
  input  logic                                  m_wr_rsp_valid,
  output logic                                  m_wr_rsp_ready,
  input  logic [TSR_MEM_ID_W-1:0]               m_wr_rsp_id,

  // ---- perf events ----
  output logic                                  perf_rd_beat,
  output logic                                  perf_rd_wait
);

  if (RD_OUTSTANDING < 1) begin : g_chk_rdo
    $fatal(1, "tsr_mem_arb: RD_OUTSTANDING must be >= 1");
  end
  if (TSR_NSRC > (1 << (TSR_MEM_ID_W - TSR_SRC_ID_W))) begin : g_chk_src
    $fatal(1, "tsr_mem_arb: the source index must fit id[3:2]");
  end

  localparam int unsigned SRC_W = TSR_MEM_ID_W - TSR_SRC_ID_W;            // 2
  localparam int unsigned RQ_W  = ADDR_BEAT_W + TSR_LEN_W + TSR_SRC_ID_W; // source request
  localparam int unsigned MQ_W  = ADDR_BEAT_W + TSR_LEN_W + TSR_MEM_ID_W; // memory request
  localparam int unsigned CNT_W = $clog2(RD_OUTSTANDING + 1);
  localparam logic [SRC_W-1:0] SRC_DMA = SRC_W'(TSR_SRC_DMA);

  // -------------------------------------------------------------------------------------------
  // Read requests: per-source input slices, round robin, outstanding limit, output slice
  // -------------------------------------------------------------------------------------------
  logic [TSR_NSRC-1:0]           h_valid, h_ready;
  logic [TSR_NSRC-1:0][RQ_W-1:0] h_data;

  for (genvar i = 0; i < TSR_NSRC; i++) begin : g_rq_in
    tsr_rslice #(.W(RQ_W)) u_rq_in (
      .clk, .rst_n,
      .s_valid(s_rd_req_valid[i]), .s_ready(s_rd_req_ready[i]),
      .s_data({s_rd_req_addr[i], s_rd_req_len[i], s_rd_req_id[i]}),
      .m_valid(h_valid[i]), .m_ready(h_ready[i]), .m_data(h_data[i])
    );
  end

  logic [SRC_W-1:0] rr_q;            // last granted source (lowest priority next)
  logic [SRC_W-1:0] gnt_idx;
  logic             gnt_any;
  logic [CNT_W-1:0] rd_cnt_q;        // granted read requests whose last beat has not returned
  logic             mq_ready;        // output slice can take a request
  logic [MQ_W-1:0]  mq_data;

  // Round robin: first valid source above rr_q, else first valid source from 0.
  always_comb begin
    logic             hi_found, lo_found;
    logic [SRC_W-1:0] hi_idx, lo_idx;
    hi_found = 1'b0;
    lo_found = 1'b0;
    hi_idx   = '0;
    lo_idx   = '0;
    for (int unsigned j = 0; j < TSR_NSRC; j++) begin
      if (h_valid[j] && !lo_found) begin
        lo_found = 1'b1;
        lo_idx   = SRC_W'(j);
      end
      if (h_valid[j] && !hi_found && SRC_W'(j) > rr_q) begin
        hi_found = 1'b1;
        hi_idx   = SRC_W'(j);
      end
    end
    gnt_any = lo_found;
    gnt_idx = hi_found ? hi_idx : lo_idx;
  end

  wire rd_cnt_ok = (rd_cnt_q < CNT_W'(RD_OUTSTANDING));
  wire grant     = gnt_any & rd_cnt_ok & mq_ready;

  always_comb begin
    logic [ADDR_BEAT_W-1:0]  a;
    logic [TSR_LEN_W-1:0]    l;
    logic [TSR_SRC_ID_W-1:0] id;
    {a, l, id} = h_data[gnt_idx];
    mq_data = {a, l, gnt_idx, id};
    for (int unsigned j = 0; j < TSR_NSRC; j++) begin
      h_ready[j] = grant && (gnt_idx == SRC_W'(j));
    end
  end

  tsr_rslice #(.W(MQ_W)) u_rq_out (
    .clk, .rst_n,
    .s_valid(grant), .s_ready(mq_ready), .s_data(mq_data),
    .m_valid(m_rd_req_valid), .m_ready(m_rd_req_ready),
    .m_data({m_rd_req_addr, m_rd_req_len, m_rd_req_id})
  );

  // A request leaves the outstanding budget when its last beat is accepted on the memory side.
  wire rd_beat = m_rd_rsp_valid & m_rd_rsp_ready;
  wire rd_done = rd_beat & m_rd_rsp_last;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rr_q         <= SRC_W'(TSR_NSRC - 1);   // source 0 has the highest priority after reset
      rd_cnt_q     <= '0;
      perf_rd_beat <= 1'b0;
      perf_rd_wait <= 1'b0;
    end else begin
      if (grant) rr_q <= gnt_idx;
      rd_cnt_q     <= rd_cnt_q + CNT_W'(grant) - CNT_W'(rd_done);
      perf_rd_beat <= rd_beat;
      perf_rd_wait <= (rd_cnt_q != '0) & ~rd_beat;
    end
  end

  // -------------------------------------------------------------------------------------------
  // Read responses: input slice, routing by source id
  // -------------------------------------------------------------------------------------------
  logic                    r_valid, r_ready, r_last;
  logic [BEAT_BITS-1:0]    r_data;
  logic [TSR_MEM_ID_W-1:0] r_id;

  tsr_rslice #(.W(BEAT_BITS + TSR_MEM_ID_W + 1)) u_rsp (
    .clk, .rst_n,
    .s_valid(m_rd_rsp_valid), .s_ready(m_rd_rsp_ready),
    .s_data({m_rd_rsp_data, m_rd_rsp_id, m_rd_rsp_last}),
    .m_valid(r_valid), .m_ready(r_ready), .m_data({r_data, r_id, r_last})
  );

  wire [SRC_W-1:0] r_src = r_id[TSR_MEM_ID_W-1:TSR_SRC_ID_W];

  always_comb begin
    r_ready = 1'b1;                    // reserved source ids are dropped (asserted below)
    for (int unsigned j = 0; j < TSR_NSRC; j++) begin
      s_rd_rsp_valid[j] = r_valid && (r_src == SRC_W'(j));
      if (r_src == SRC_W'(j)) r_ready = s_rd_rsp_ready[j];
    end
  end

  assign s_rd_rsp_data = r_data;
  assign s_rd_rsp_last = r_last;
  assign s_rd_rsp_id   = r_id[TSR_SRC_ID_W-1:0];

  // -------------------------------------------------------------------------------------------
  // Writes: three independent slices (request, data, response)
  // -------------------------------------------------------------------------------------------
  logic [TSR_SRC_ID_W-1:0] wq_id;
  logic [TSR_MEM_ID_W-1:0] ws_id;

  tsr_rslice #(.W(RQ_W)) u_wq (
    .clk, .rst_n,
    .s_valid(s_wr_req_valid), .s_ready(s_wr_req_ready),
    .s_data({s_wr_req_addr, s_wr_req_len, s_wr_req_id}),
    .m_valid(m_wr_req_valid), .m_ready(m_wr_req_ready),
    .m_data({m_wr_req_addr, m_wr_req_len, wq_id})
  );
  assign m_wr_req_id = {SRC_DMA, wq_id};

  tsr_rslice #(.W(BEAT_BITS + BEAT_BITS/32 + 1)) u_wd (
    .clk, .rst_n,
    .s_valid(s_wr_dat_valid), .s_ready(s_wr_dat_ready), .s_data({s_wr_dat_data, s_wr_dat_mask, s_wr_dat_last}),
    .m_valid(m_wr_dat_valid), .m_ready(m_wr_dat_ready), .m_data({m_wr_dat_data, m_wr_dat_mask, m_wr_dat_last})
  );

  tsr_rslice #(.W(TSR_MEM_ID_W)) u_ws (
    .clk, .rst_n,
    .s_valid(m_wr_rsp_valid), .s_ready(m_wr_rsp_ready), .s_data(m_wr_rsp_id),
    .m_valid(s_wr_rsp_valid), .m_ready(s_wr_rsp_ready), .m_data(ws_id)
  );
  assign s_wr_rsp_id = ws_id[TSR_SRC_ID_W-1:0];

  wire unused_ok = &{1'b0, ws_id[TSR_MEM_ID_W-1:TSR_SRC_ID_W]};

  // -------------------------------------------------------------------------------------------
  // Simulation checks (docs/rtl-m2.md §4 M1, M3; source rule D1)
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  // 1..64 beats inside one 64-beat block; addr_lo = address bits [5:0].
  function automatic logic burst_ok(logic [5:0] addr_lo, logic [TSR_LEN_W-1:0] len);
    return len != '0 && 32'(len) <= TSR_MAX_BURST_BEATS
           && 32'(addr_lo) + 32'(len) <= TSR_BURST_ALIGN;
  endfunction

  // Checks are armed once a reset has been seen: the time-0 state of a simulation (inputs and
  // rst_n included) may be random.
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      for (int unsigned j = 0; j < TSR_NSRC; j++) begin
        if (s_rd_req_valid[j] && s_rd_req_ready[j]) begin
          assert (burst_ok(6'(s_rd_req_addr[j]), s_rd_req_len[j]))
            else $error("%m: source %0d read burst addr 0x%0h len %0d: not 1..64 inside one 4 KiB block",
                        j, s_rd_req_addr[j], s_rd_req_len[j]);
        end
      end
      if (s_wr_req_valid && s_wr_req_ready) begin
        assert (burst_ok(6'(s_wr_req_addr), s_wr_req_len))
          else $error("%m: write burst addr 0x%0h len %0d: not 1..64 inside one 4 KiB block",
                      s_wr_req_addr, s_wr_req_len);
      end
      if (rd_done) begin
        assert (rd_cnt_q != '0 || grant)
          else $error("%m: read response 'last' with no read request outstanding");
      end
      if (r_valid) begin
        assert (32'(r_src) < TSR_NSRC)
          else $error("%m: read response with reserved source id %0d", r_id);
      end
      if (s_wr_rsp_valid) begin
        assert (ws_id[TSR_MEM_ID_W-1:TSR_SRC_ID_W] == SRC_DMA)
          else $error("%m: write response id %0d does not belong to the DMA source", ws_id);
      end
    end
  end
`endif

endmodule
