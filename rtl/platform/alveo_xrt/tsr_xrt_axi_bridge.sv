// tsr_xrt_axi_bridge.sv — one channel: the tsr_core memory port (docs/rtl-core.md §4) as an AXI4
// master (Vitis m_axi kernel port, 512-bit data, 64-bit address, 4-bit ID).
//
// Spec: docs/platform-u200.md §4.
//
// Mapping (kernel contract):
//   read / write request {addr, len, id}  ->  AR / AW: AxADDR = base + addr * 64, AxLEN = len - 1,
//                                             AxSIZE 6 (64 B), AxBURST INCR, AxID = id,
//                                             AxCACHE 4'b0011, AxPROT 0, AxLOCK 0, AxQOS 0
//   write data {data, word mask, last}    ->  W: WDATA, WSTRB, WLAST
//   R {RDATA, RID, RLAST, RRESP}          ->  read response {data, id, last}
//   B {BID, BRESP}                        ->  write response {id}
// The core already keeps every burst inside one 64-beat block of its channel (rtl-core §4 M1), so
// with a 4 KiB aligned base no AXI burst crosses a 4 KiB boundary and ARLEN / AWLEN <= 63. The
// control bridge refuses to start a run with a misaligned base (ADAPTER_STATUS bit3), so `base`
// is aligned whenever the core is running; it is registered here (base_r) next to the adders.
// Read data of different IDs may interleave at beat granularity and arrive in order per ID (the
// core allows exactly that), one B per AW: the bridge passes R and B through unchanged.
//
// Structure (every output is a flop; SLICES register slices tsr_rslice per AXI channel, the
// first / last of which give flop-driven AXI and core-side outputs):
//   core rd_req -> +base -> SLICES x rslice -> AR          core wr_req -> +base -> SLICES -> AW
//   core wr_dat -> SLICES x rslice -> W
//   R -> SLICES x rslice -> id filter -> rslice -> core rd_rsp
//   B -> SLICES x rslice -> id filter -> rslice -> core wr_rsp
// Response filter: per ID, the number of bursts whose AR (AW) handshake happened at the AXI
// boundary and whose last R beat (B) has not yet passed the filter. A response for an ID with
// none outstanding cannot belong to any request of this reset epoch (e.g. a response to a request
// issued before a kernel reset by an interconnect that was not reset with the kernel); it is
// consumed and dropped, never forwarded (tsr_mem_arb's outstanding counter would underflow), and
// reported as ev_unexpected (ADAPTER_STATUS bit2). A response can be presented at the earliest in
// the cycle after its request's handshake (edge t) and reaches the filter SLICES edges later; the
// counter of its ID (and the registered "count != 0" flag the filter reads) is updated at edge
// t + 1 from the registered handshake when SLICES >= 2 (the AXI READY inputs then only reach the
// slices and one flop), at edge t from the handshake itself when SLICES = 1: one cycle of margin
// either way, so a legal response always finds its count >= 1. Limit: a stale response that arrives after the core reissued a
// request with the same ID cannot be told apart; docs/platform-u200.md §5 explains why Vitis
// never produces one (the interconnect is reset together with the kernel).
// Error responses (RRESP / BRESP != OKAY) are forwarded like OKAY ones (the core has no error
// path for memory) and reported per beat / response as ev_rresp_err / ev_bresp_err
// (ADAPTER_STATUS bits 0 / 1 and its error count); the host checks ADAPTER_STATUS after a run.
// Events are registered one-cycle pulses.
// READ_ONLY = 1 (docs/platform-u200.md §4: the read-only port B of a channel, docs/rtl-core.md §4): the bridge has the
// read half only. The write request, write data and write response channels do not exist: the core side write ports
// never accept (c_wr_req_ready = c_wr_dat_ready = 0, c_wr_rsp_valid = 0), the AXI write outputs are 0 and the write
// inputs are ignored (the kernel top has no AW, W or B signals for it). The read path, its response filter and its
// events are those above, cycle for cycle.
// Reset: 4 + 2 x CNT_RST equal copies of the channel's reset (tsr_xrt_reset rst_ch_n[c][*], same
// stage, asserted and released in the same cycle; checked in simulation), one per part, so that
// each reset net stays local to its logic (timing, docs/platform-u200.md §5, §8.2: one
// copy for the whole bridge gave the k1 link's second-worst path, 0 levels and 3.07 ns of routing
// to the write counters; one copy for all write counters still routed 2.96 ns out of context):
//   rst_n         the core-side stages of the AR / AW / W slice chains
//   rst_axi_n     the AXI-side stage of every slice chain (the last of AR / AW / W, the first of
//                 R / B) and the registered AR / AW handshake (g_hs_q): the cells floorplan C' puts
//                 into the DDR's SLR for a crossing bridge (hw/u200/k4_floorplan.tcl)
//   rst_rd_n      the other R slice stages, the R output slice, ev_rresp_err / ev_unexpected
//   rst_wr_n      the other B slice stages, the B output slice, ev_bresp_err
//   rst_rdc_n[g]  the read counters and flags of IDs g * NID / CNT_RST .. (g + 1) * NID / CNT_RST - 1
//   rst_wrc_n[g]  the write counters and flags of the same IDs

module tsr_xrt_axi_bridge
  import tsr_pkg::*;
#(
  parameter int unsigned ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter int unsigned BEAT_BITS   = TSR_DEF_BEAT_BITS,
  parameter int unsigned SLICES      = 2,    // register slices per AXI channel, >= 1
  parameter int unsigned RD_CNT_W    = 8,    // outstanding read bursts per ID (core: <= RD_OUTSTANDING)
  parameter int unsigned WR_CNT_W    = 16,   // outstanding write bursts per ID (one ST: <= SPM lines)
  parameter int unsigned CNT_RST     = 4,    // reset copies per counter set (ID groups), divides 2^TSR_MEM_ID_W
  parameter bit          READ_ONLY   = 1'b0  // read half only (header)
) (
  input  logic                         clk,
  input  logic                         rst_n,          // request slices (header: equal reset copies)
  input  logic                         rst_axi_n,      // AXI-side slice stages, AR / AW handshake
  input  logic                         rst_rd_n,       // read response path
  input  logic                         rst_wr_n,       // write response path
  input  logic [CNT_RST-1:0]           rst_rdc_n,      // read counters, one copy per ID group
  input  logic [CNT_RST-1:0]           rst_wrc_n,      // write counters, one copy per ID group
  input  logic [63:0]                  base,           // device byte address of beat 0 (latched)

  // ---- core memory port (tsr_core m_*[c]) ----
  input  logic                         c_rd_req_valid,
  output logic                         c_rd_req_ready,
  input  logic [ADDR_BEAT_W-1:0]       c_rd_req_addr,
  input  logic [TSR_LEN_W-1:0]         c_rd_req_len,
  input  logic [TSR_MEM_ID_W-1:0]      c_rd_req_id,
  output logic                         c_rd_rsp_valid,
  input  logic                         c_rd_rsp_ready,
  output logic [BEAT_BITS-1:0]         c_rd_rsp_data,
  output logic [TSR_MEM_ID_W-1:0]      c_rd_rsp_id,
  output logic                         c_rd_rsp_last,
  input  logic                         c_wr_req_valid,
  output logic                         c_wr_req_ready,
  input  logic [ADDR_BEAT_W-1:0]       c_wr_req_addr,
  input  logic [TSR_LEN_W-1:0]         c_wr_req_len,
  input  logic [TSR_MEM_ID_W-1:0]      c_wr_req_id,
  input  logic                         c_wr_dat_valid,
  output logic                         c_wr_dat_ready,
  input  logic [BEAT_BITS-1:0]         c_wr_dat_data,
  input  logic [BEAT_BITS/32-1:0]      c_wr_dat_mask,
  input  logic                         c_wr_dat_last,
  output logic                         c_wr_rsp_valid,
  input  logic                         c_wr_rsp_ready,
  output logic [TSR_MEM_ID_W-1:0]      c_wr_rsp_id,

  // ---- AXI4 master ----
  output logic                         m_axi_awvalid,
  input  logic                         m_axi_awready,
  output logic [63:0]                  m_axi_awaddr,
  output logic [7:0]                   m_axi_awlen,
  output logic [2:0]                   m_axi_awsize,
  output logic [1:0]                   m_axi_awburst,
  output logic                         m_axi_awlock,
  output logic [3:0]                   m_axi_awcache,
  output logic [2:0]                   m_axi_awprot,
  output logic [3:0]                   m_axi_awqos,
  output logic [TSR_MEM_ID_W-1:0]      m_axi_awid,
  output logic                         m_axi_wvalid,
  input  logic                         m_axi_wready,
  output logic [BEAT_BITS-1:0]         m_axi_wdata,
  output logic [BEAT_BITS/8-1:0]       m_axi_wstrb,
  output logic                         m_axi_wlast,
  input  logic                         m_axi_bvalid,
  output logic                         m_axi_bready,
  input  logic [1:0]                   m_axi_bresp,
  input  logic [TSR_MEM_ID_W-1:0]      m_axi_bid,
  output logic                         m_axi_arvalid,
  input  logic                         m_axi_arready,
  output logic [63:0]                  m_axi_araddr,
  output logic [7:0]                   m_axi_arlen,
  output logic [2:0]                   m_axi_arsize,
  output logic [1:0]                   m_axi_arburst,
  output logic                         m_axi_arlock,
  output logic [3:0]                   m_axi_arcache,
  output logic [2:0]                   m_axi_arprot,
  output logic [3:0]                   m_axi_arqos,
  output logic [TSR_MEM_ID_W-1:0]      m_axi_arid,
  input  logic                         m_axi_rvalid,
  output logic                         m_axi_rready,
  input  logic [BEAT_BITS-1:0]         m_axi_rdata,
  input  logic [1:0]                   m_axi_rresp,
  input  logic                         m_axi_rlast,
  input  logic [TSR_MEM_ID_W-1:0]      m_axi_rid,

  // ---- events (registered pulses) ----
  output logic                         ev_rresp_err,    // a forwarded R beat with RRESP != OKAY
  output logic                         ev_bresp_err,    // a forwarded B with BRESP != OKAY
  output logic                         ev_unexpected    // an R beat or B dropped (no outstanding burst for its ID)
);

  if (SLICES < 1) begin : g_chk_slices
    $fatal(1, "tsr_xrt_axi_bridge: SLICES must be >= 1 (AXI outputs come from flops)");
  end
  if (ADDR_BEAT_W > 58) begin : g_chk_aw
    $fatal(1, "tsr_xrt_axi_bridge: ADDR_BEAT_W must be <= 58 (64-bit byte address)");
  end

  localparam int unsigned IDW  = TSR_MEM_ID_W;
  localparam int unsigned NID  = 1 << IDW;
  localparam int unsigned GSZ  = NID / CNT_RST;                // IDs per counter reset copy

  if (CNT_RST < 1 || CNT_RST > NID || NID % CNT_RST != 0) begin : g_chk_cnt_rst
    $fatal(1, "tsr_xrt_axi_bridge: CNT_RST must divide the number of IDs (%0d)", NID);
  end
  localparam int unsigned AQ_W = 64 + 8 + IDW;                 // {addr, len - 1, id}
  localparam int unsigned R_W  = BEAT_BITS + IDW + 1 + 2;      // {data, id, last, resp}
  localparam int unsigned W_W  = BEAT_BITS + BEAT_BITS/32 + 1;   // {data, word mask, last}
  localparam int unsigned B_W  = IDW + 2;                      // {id, resp}

  logic [63:0] base_r;
  always_ff @(posedge clk) base_r <= base;

  function automatic logic [AQ_W-1:0] areq(logic [63:0] b, logic [ADDR_BEAT_W-1:0] a,
                                           logic [TSR_LEN_W-1:0] l, logic [IDW-1:0] id);
    logic [63:0] off;
    off = {{(64 - ADDR_BEAT_W){1'b0}}, a} << 6;
    return {b + off, 8'(l - TSR_LEN_W'(1)), id};
  endfunction

  // Per-stage resets of the slice chains (header): the AXI-side stage from rst_axi_n, the others
  // from the copy of their path.
  logic [SLICES-1:0] rst_rq_st, rst_r_st, rst_b_st;
  for (genvar k = 0; k < SLICES; k++) begin : g_rst_st
    assign rst_rq_st[k] = (k == SLICES - 1) ? rst_axi_n : rst_n;     // AR / AW / W: last stage at AXI
    assign rst_r_st[k]  = (k == 0) ? rst_axi_n : rst_rd_n;           // R / B: first stage at AXI
    assign rst_b_st[k]  = (k == 0) ? rst_axi_n : rst_wr_n;
  end

  // -------------------------------------------------------------------------------------------
  // Request and write-data paths: SLICES slices each
  // -------------------------------------------------------------------------------------------
  tsr_xrt_slices #(.W(AQ_W), .N(SLICES)) u_ar (
    .clk, .rst_n(rst_rq_st),
    .s_valid(c_rd_req_valid), .s_ready(c_rd_req_ready),
    .s_data(areq(base_r, c_rd_req_addr, c_rd_req_len, c_rd_req_id)),
    .m_valid(m_axi_arvalid), .m_ready(m_axi_arready), .m_data({m_axi_araddr, m_axi_arlen, m_axi_arid})
  );
  logic [BEAT_BITS/32-1:0] w_word_mask;
  if (!READ_ONLY) begin : g_wreq
    tsr_xrt_slices #(.W(AQ_W), .N(SLICES)) u_aw (
      .clk, .rst_n(rst_rq_st),
      .s_valid(c_wr_req_valid), .s_ready(c_wr_req_ready),
      .s_data(areq(base_r, c_wr_req_addr, c_wr_req_len, c_wr_req_id)),
      .m_valid(m_axi_awvalid), .m_ready(m_axi_awready), .m_data({m_axi_awaddr, m_axi_awlen, m_axi_awid})
    );
    tsr_xrt_slices #(.W(W_W), .N(SLICES)) u_w (
      .clk, .rst_n(rst_rq_st),
      .s_valid(c_wr_dat_valid), .s_ready(c_wr_dat_ready), .s_data({c_wr_dat_data, c_wr_dat_mask, c_wr_dat_last}),
      .m_valid(m_axi_wvalid), .m_ready(m_axi_wready), .m_data({m_axi_wdata, w_word_mask, m_axi_wlast})
    );
  end else begin : g_nwreq
    assign c_wr_req_ready = 1'b0;
    assign m_axi_awvalid  = 1'b0;
    assign m_axi_awaddr   = '0;
    assign m_axi_awlen    = '0;
    assign m_axi_awid     = '0;
    assign c_wr_dat_ready = 1'b0;
    assign m_axi_wvalid   = 1'b0;
    assign m_axi_wdata    = '0;
    assign m_axi_wlast    = 1'b0;
    assign w_word_mask    = '0;
    wire unused_wreq = &{1'b0, c_wr_req_valid, c_wr_req_addr, c_wr_req_len, c_wr_req_id, c_wr_dat_valid, c_wr_dat_data,
                         c_wr_dat_mask, c_wr_dat_last, m_axi_awready, m_axi_wready};
  end

  assign m_axi_arsize  = 3'd6;          // 64 bytes per beat
  assign m_axi_arburst = 2'b01;         // INCR
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'b0011;       // normal non-cacheable bufferable
  assign m_axi_arprot  = 3'b000;
  assign m_axi_arqos   = 4'd0;
  assign m_axi_awsize  = READ_ONLY ? 3'd0 : 3'd6;
  assign m_axi_awburst = READ_ONLY ? 2'b00 : 2'b01;
  assign m_axi_awlock  = 1'b0;
  assign m_axi_awcache = READ_ONLY ? 4'b0000 : 4'b0011;
  assign m_axi_awprot  = 3'b000;
  assign m_axi_awqos   = 4'd0;
  for (genvar w = 0; w < BEAT_BITS/32; w++) begin : g_wstrb
    assign m_axi_wstrb[4*w +: 4] = {4{w_word_mask[w]}};
  end

  // -------------------------------------------------------------------------------------------
  // Read responses: input slices, ID filter, output slice
  // -------------------------------------------------------------------------------------------
  logic                rh_valid, rh_ready, rh_last;
  logic [BEAT_BITS-1:0] rh_data;
  logic [IDW-1:0]      rh_id;
  logic [1:0]          rh_resp;
  logic                ro_ready;
  logic [NID-1:0]      rd_nz_q;          // per ID: >= 1 read burst outstanding

  tsr_xrt_slices #(.W(R_W), .N(SLICES)) u_r (
    .clk, .rst_n(rst_r_st),
    .s_valid(m_axi_rvalid), .s_ready(m_axi_rready), .s_data({m_axi_rdata, m_axi_rid, m_axi_rlast, m_axi_rresp}),
    .m_valid(rh_valid), .m_ready(rh_ready), .m_data({rh_data, rh_id, rh_last, rh_resp})
  );

  wire r_exp  = rd_nz_q[rh_id];
  wire r_fwd  = rh_valid & r_exp;
  wire r_drop = rh_valid & ~r_exp;
  assign rh_ready = r_drop | ro_ready;

  tsr_rslice #(.W(BEAT_BITS + IDW + 1)) u_ro (
    .clk, .rst_n(rst_rd_n),
    .s_valid(r_fwd), .s_ready(ro_ready), .s_data({rh_data, rh_id, rh_last}),
    .m_valid(c_rd_rsp_valid), .m_ready(c_rd_rsp_ready), .m_data({c_rd_rsp_data, c_rd_rsp_id, c_rd_rsp_last})
  );

  wire r_take = r_fwd & ro_ready;                       // beat forwarded to the core side
  wire r_done = r_take & rh_last;                       // ... and it ends its burst

  // -------------------------------------------------------------------------------------------
  // Write responses: input slices, ID filter, output slice
  // -------------------------------------------------------------------------------------------
  logic [NID-1:0] wr_nz_q;               // per ID: >= 1 write burst outstanding
  logic           b_take, b_drop;
  logic [IDW-1:0] bh_id;
  logic [1:0]     bh_resp;

  if (!READ_ONLY) begin : g_wresp
    logic           bh_valid, bh_ready, bo_ready;
    tsr_xrt_slices #(.W(B_W), .N(SLICES)) u_b (
      .clk, .rst_n(rst_b_st),
      .s_valid(m_axi_bvalid), .s_ready(m_axi_bready), .s_data({m_axi_bid, m_axi_bresp}),
      .m_valid(bh_valid), .m_ready(bh_ready), .m_data({bh_id, bh_resp})
    );

    wire b_exp  = wr_nz_q[bh_id];
    wire b_fwd  = bh_valid & b_exp;
    assign b_drop = bh_valid & ~b_exp;
    assign bh_ready = b_drop | bo_ready;

    tsr_rslice #(.W(IDW)) u_bo (
      .clk, .rst_n(rst_wr_n),
      .s_valid(b_fwd), .s_ready(bo_ready), .s_data(bh_id),
      .m_valid(c_wr_rsp_valid), .m_ready(c_wr_rsp_ready), .m_data(c_wr_rsp_id)
    );

    assign b_take = b_fwd & bo_ready;
  end else begin : g_nwresp
    assign m_axi_bready   = 1'b0;
    assign c_wr_rsp_valid = 1'b0;
    assign c_wr_rsp_id    = '0;
    assign bh_id          = '0;
    assign bh_resp        = '0;
    assign b_take         = 1'b0;
    assign b_drop         = 1'b0;
    wire unused_wresp = &{1'b0, m_axi_bvalid, m_axi_bresp, m_axi_bid, c_wr_rsp_ready, rst_wr_n, rst_b_st, wr_nz_q};
  end

  // AR / AW handshakes for the counters: registered when SLICES >= 2 (header: timing margin)
  logic           ar_hs, aw_hs;
  logic [IDW-1:0] ar_id, aw_id;
  if (SLICES >= 2) begin : g_hs_q
    logic           ar_hs_q, aw_hs_q;
    logic [IDW-1:0] ar_id_q, aw_id_q;
    always_ff @(posedge clk) begin
      if (!rst_axi_n) begin
        ar_hs_q <= 1'b0;
        aw_hs_q <= 1'b0;
      end else begin
        ar_hs_q <= m_axi_arvalid & m_axi_arready;
        aw_hs_q <= m_axi_awvalid & m_axi_awready;
      end
      ar_id_q <= m_axi_arid;
      aw_id_q <= m_axi_awid;
    end
    assign ar_hs = ar_hs_q;
    assign aw_hs = aw_hs_q;
    assign ar_id = ar_id_q;
    assign aw_id = aw_id_q;
  end else begin : g_hs_d
    assign ar_hs = m_axi_arvalid & m_axi_arready;
    assign aw_hs = m_axi_awvalid & m_axi_awready;
    assign ar_id = m_axi_arid;
    assign aw_id = m_axi_awid;
  end

  // -------------------------------------------------------------------------------------------
  // Per-ID outstanding counters, events
  // -------------------------------------------------------------------------------------------
  logic [NID-1:0][RD_CNT_W-1:0] rd_cnt_q;
  logic [NID-1:0][WR_CNT_W-1:0] wr_cnt_q;

  // counters and flags of ID i: reset copy rst_rdc_n / rst_wrc_n[i / GSZ] (header)
  always_ff @(posedge clk) begin
    for (int unsigned i = 0; i < NID; i++) begin
      logic [RD_CNT_W-1:0] rn;
      logic [WR_CNT_W-1:0] wn;
      rn = rd_cnt_q[i] + RD_CNT_W'(ar_hs && ar_id == IDW'(i)) - RD_CNT_W'(r_done && rh_id == IDW'(i));
      wn = wr_cnt_q[i] + WR_CNT_W'(aw_hs && aw_id == IDW'(i)) - WR_CNT_W'(b_take && bh_id == IDW'(i));
      if (!rst_rdc_n[i / GSZ]) begin
        rd_cnt_q[i] <= '0;
        rd_nz_q[i]  <= 1'b0;
      end else begin
        rd_cnt_q[i] <= rn;
        rd_nz_q[i]  <= rn != '0;
      end
      if (!rst_wrc_n[i / GSZ]) begin
        wr_cnt_q[i] <= '0;
        wr_nz_q[i]  <= 1'b0;
      end else begin
        wr_cnt_q[i] <= wn;
        wr_nz_q[i]  <= wn != '0;
      end
    end
  end

  // events (reset copies rst_rd_n / rst_wr_n)
  always_ff @(posedge clk) begin
    if (!rst_rd_n) begin
      ev_rresp_err  <= 1'b0;
      ev_unexpected <= 1'b0;
    end else begin
      ev_rresp_err  <= r_take & (rh_resp != 2'b00);
      ev_unexpected <= r_drop | b_drop;      // a dropped item is consumed in the same cycle
    end
    if (!rst_wr_n) ev_bresp_err <= 1'b0;
    else           ev_bresp_err <= b_take & (bh_resp != 2'b00);
  end

  // -------------------------------------------------------------------------------------------
  // Simulation checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    // the four reset copies are one reset (header): a part released in another cycle than the
    // others could drop or duplicate an item between the parts
    if (a_on_q)
      assert (rst_axi_n == rst_n && rst_rd_n == rst_n && rst_wr_n == rst_n
              && rst_rdc_n == {CNT_RST{rst_n}} && rst_wrc_n == {CNT_RST{rst_n}})
        else $error("%m: reset copies differ (rst_n %0b, rst_axi_n %0b, rst_rd_n %0b, rst_wr_n %0b, rst_rdc_n %b, rst_wrc_n %b)",
                    rst_n, rst_axi_n, rst_rd_n, rst_wr_n, rst_rdc_n, rst_wrc_n);
    if (rst_n && a_on_q) begin
      for (int unsigned i = 0; i < NID; i++) begin
        if (ar_hs && ar_id == IDW'(i))
          assert (rd_cnt_q[i] != '1) else $error("%m: read burst counter of id %0d overflows", i);
        if (aw_hs && aw_id == IDW'(i))
          assert (wr_cnt_q[i] != '1) else $error("%m: write burst counter of id %0d overflows", i);
      end
      if (c_rd_req_valid && c_rd_req_ready)
        assert (base_r[11:0] == 12'd0) else $error("%m: read request with a base that is not 4 KiB aligned");
      if (c_wr_req_valid && c_wr_req_ready)
        assert (base_r[11:0] == 12'd0) else $error("%m: write request with a base that is not 4 KiB aligned");
    end
  end
`endif

endmodule
