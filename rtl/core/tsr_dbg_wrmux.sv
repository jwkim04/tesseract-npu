// tsr_dbg_wrmux.sv — the trace writer's write source on channel TRW_CH (docs/design/05-debug.md
// "DB" §8.3, DB-16; CC §13.2 "a second write source on TRW_CH = 0, burst-arbitrated with the
// DMA"): merges the DMA's write port of that channel and the trace writer's into the one write
// source of the channel's tsr_mem_arb, per burst.
//
// Owner: DB (W4-B item 6). tsr_core_dp places it between u_dma's channel-TRW_CH write port and
// u_arb's s_wr_* port of that channel (tsr_mem_arb itself stays unchanged).
//
// Protocol (docs/rtl-m2.md §4): request {addr, len, id}, data {data, word mask, last} and response {id}
// are independent valid / ready streams; the data beats of the bursts follow the request order
// without interleaving. A source may offer data before its request is taken (the DMA does).
// DMA mode (default): every DMA stream passes straight through, so with the trace writer idle
//   the DMA sees exactly the arbiter's handshakes (no cycle added: HAS_TRW 0 and 1 give the same
//   cycles, DB §10.3). When the trace writer requests a burst, the DMA's next request is held
//   (req_ready 0) unless its data already passed, and so is any DMA data beat that does not
//   belong to an accepted request; once
//   every accepted DMA burst has passed its last beat (bal = 0, no burst half passed), the mux
//   switches to TRW mode.
// TRW mode: the trace writer's request (id TRW_ID) and its data beats pass; after its last beat
//   the mux returns to DMA mode. The trace writer waits at most for the DMA bursts already
//   accepted; the DMA waits at most one trace burst (<= 16 beats).
// Responses: routed by id: TRW_ID to the trace writer (always ready), every other id to the DMA
//   (which uses local id 0, tsr_dma).

module tsr_dbg_wrmux
  import tsr_pkg::*;
#(
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter int unsigned     BEAT_BITS   = TSR_DEF_BEAT_BITS,
  parameter logic [TSR_SRC_ID_W-1:0] TRW_ID = TSR_SRC_ID_W'(1)
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // ---- DMA write port of channel TRW_CH ----
  input  logic                    d_wr_req_valid,
  output logic                    d_wr_req_ready,
  input  logic [ADDR_BEAT_W-1:0]  d_wr_req_addr,
  input  logic [TSR_LEN_W-1:0]    d_wr_req_len,
  input  logic [TSR_SRC_ID_W-1:0] d_wr_req_id,
  input  logic                    d_wr_dat_valid,
  output logic                    d_wr_dat_ready,
  input  logic [BEAT_BITS-1:0]    d_wr_dat_data,
  input  logic [BEAT_BITS/32-1:0] d_wr_dat_mask,
  input  logic                    d_wr_dat_last,
  output logic                    d_wr_rsp_valid,
  input  logic                    d_wr_rsp_ready,
  output logic [TSR_SRC_ID_W-1:0] d_wr_rsp_id,

  // ---- trace writer (tsr_dbg_trw) ----
  input  logic                    t_wr_req_valid,
  output logic                    t_wr_req_ready,
  input  logic [ADDR_BEAT_W-1:0]  t_wr_req_addr,
  input  logic [TSR_LEN_W-1:0]    t_wr_req_len,
  input  logic                    t_wr_dat_valid,
  output logic                    t_wr_dat_ready,
  input  logic [BEAT_BITS-1:0]    t_wr_dat_data,
  input  logic                    t_wr_dat_last,
  output logic                    t_wr_rsp_valid,

  // ---- write source of the channel's tsr_mem_arb ----
  output logic                    s_wr_req_valid,
  input  logic                    s_wr_req_ready,
  output logic [ADDR_BEAT_W-1:0]  s_wr_req_addr,
  output logic [TSR_LEN_W-1:0]    s_wr_req_len,
  output logic [TSR_SRC_ID_W-1:0] s_wr_req_id,
  output logic                    s_wr_dat_valid,
  input  logic                    s_wr_dat_ready,
  output logic [BEAT_BITS-1:0]    s_wr_dat_data,
  output logic [BEAT_BITS/32-1:0]    s_wr_dat_mask,
  output logic                    s_wr_dat_last,
  input  logic                    s_wr_rsp_valid,
  output logic                    s_wr_rsp_ready,
  input  logic [TSR_SRC_ID_W-1:0] s_wr_rsp_id
);

  logic              trw_q;              // TRW mode
  logic              treq_q;             // TRW mode: the trace writer's request was taken
  logic signed [6:0] bal_q;              // DMA requests taken - DMA bursts whose last beat passed
  logic              mid_q;              // a DMA burst has passed some beats, not its last

  wire hold_new = t_wr_req_valid;        // the trace writer waits: no new DMA burst
  wire d_dat_ok = !hold_new || (bal_q > 7'sd0) || mid_q;     // data of a taken request only
  wire d_req_ok = !hold_new || (bal_q < 7'sd0);              // a request whose data already passed

  always_comb begin
    if (!trw_q) begin
      s_wr_req_valid = d_wr_req_valid && d_req_ok;
      s_wr_req_addr  = d_wr_req_addr;
      s_wr_req_len   = d_wr_req_len;
      s_wr_req_id    = d_wr_req_id;
      s_wr_dat_valid = d_wr_dat_valid && d_dat_ok;
      s_wr_dat_data  = d_wr_dat_data;
      s_wr_dat_mask  = d_wr_dat_mask;
      s_wr_dat_last  = d_wr_dat_last;
      d_wr_req_ready = s_wr_req_ready && d_req_ok;
      d_wr_dat_ready = s_wr_dat_ready && d_dat_ok;
      t_wr_req_ready = 1'b0;
      t_wr_dat_ready = 1'b0;
    end else begin
      s_wr_req_valid = t_wr_req_valid && !treq_q;
      s_wr_req_addr  = t_wr_req_addr;
      s_wr_req_len   = t_wr_req_len;
      s_wr_req_id    = TRW_ID;
      s_wr_dat_valid = t_wr_dat_valid;
      s_wr_dat_data  = t_wr_dat_data;
      s_wr_dat_mask  = '1;
      s_wr_dat_last  = t_wr_dat_last;
      d_wr_req_ready = 1'b0;
      d_wr_dat_ready = 1'b0;
      t_wr_req_ready = s_wr_req_ready && !treq_q;
      t_wr_dat_ready = s_wr_dat_ready;
    end
  end

  wire d_req_go  = !trw_q && s_wr_req_valid && s_wr_req_ready;
  wire d_dat_go  = !trw_q && s_wr_dat_valid && s_wr_dat_ready;
  wire t_req_go  = trw_q && s_wr_req_valid && s_wr_req_ready;
  wire t_last_go = trw_q && s_wr_dat_valid && s_wr_dat_ready && t_wr_dat_last;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      trw_q  <= 1'b0;
      treq_q <= 1'b0;
      bal_q  <= '0;
      mid_q  <= 1'b0;
    end else begin
      bal_q <= bal_q + 7'(d_req_go) - 7'(d_dat_go && d_wr_dat_last);
      if (d_dat_go) mid_q <= !d_wr_dat_last;
      if (!trw_q) begin
        if (hold_new && bal_q == 7'sd0 && !mid_q) trw_q <= 1'b1;
      end else begin
        if (t_req_go) treq_q <= 1'b1;
        if (t_last_go) begin
          trw_q  <= 1'b0;
          treq_q <= 1'b0;
        end
      end
    end
  end

  // responses by id
  wire rsp_t = (s_wr_rsp_id == TRW_ID);
  assign t_wr_rsp_valid = s_wr_rsp_valid && rsp_t;
  assign d_wr_rsp_valid = s_wr_rsp_valid && !rsp_t;
  assign d_wr_rsp_id    = s_wr_rsp_id;
  assign s_wr_rsp_ready = rsp_t || d_wr_rsp_ready;

`ifndef SYNTHESIS
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      assert (!(d_wr_req_valid && !trw_q && d_wr_req_id == TRW_ID))
        else $error("%m: the DMA uses the trace writer's write id %0d", TRW_ID);
      assert (!(t_last_go && !(treq_q || t_req_go))) else $error("%m: trace data passed before its request");
    end
  end
`endif

endmodule
