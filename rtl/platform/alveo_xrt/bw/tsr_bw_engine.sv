// tsr_bw_engine.sv — one channel of the M0 DMA bandwidth kernel tsr_bw_k4 (docs/m0-bw.md): read
// and / or write traffic generator, read checksum, write pattern and counters, on the memory-port
// side of a tsr_xrt_axi_bridge (the adapter's AXI4 bridge: base add, register slices, ID filter,
// error events).
//
// Owner: m0bw agent (docs/m0-bw.md §2, §3).
//
// Configuration (latched at the start of a run, static from then on; ctrl word = docs/m0-bw.md §2):
//   ctrl[0] enable, ctrl[1] read, ctrl[2] write, ctrl[14:8] burst beats B (1..64), ctrl[22:16]
//   outstanding bursts per direction (1..64), ctrl[26:24] id_log2 (bursts use AXI IDs
//   0 .. 2^id_log2 - 1 round robin, 0..4); bytes (per direction), rd_off / wr_off (byte offsets of
//   the read / write region in the buffer), base (the buffer's device address), seed.
//   Invalid (an enabled channel only): mode 0, B or outstanding outside 1..64, id_log2 > 4, base
//   not 4 KiB aligned, bytes / an enabled region's offset not a multiple of 64, an enabled region
//   ending beyond 2^(ADDR_BEAT_W + 6) bytes -> status CFG_ERR, no traffic, the channel is done.
//   bytes = 0: both regions are empty (no traffic, status RAN), also with an offset at the limit.
// Burst sequence (per direction, the same rule for AR, AW and the W beats): starting at the
// region's first beat a, burst length = min(B, beats to the next 4 KiB boundary, beats left);
// a += length. No burst crosses 4 KiB; the first burst of a region that does not start on a
// 4 KiB boundary and every burst that meets a boundary are shorter than B; the region is covered
// exactly once. The generators track the 4 KiB block count to the region end instead of the beats
// left (no wide subtract in the one-burst-per-cycle loop).
// IDs: burst k of a direction uses ID k mod 2^id_log2.
// Outstanding: a burst counts from its generation (entry into the bridge) to its last R beat /
// its B, so at most `outstanding` bursts per direction are in flight at the AXI boundary.
// Write data: beats in burst order (W beats of burst k only after AW k was generated); 64-bit word
// i of the beat at byte offset o (from the buffer address) = mix64((o + 8i) ^ salt), salt =
// {seed, 24'h0, CH}, mix64 = one xorshift64 step (x ^= x << 13; x ^= x >> 7; x ^= x << 17),
// linear over GF(2): mix64((o + 8i) ^ salt) = mix64(o) ^ mix64(8i) ^ mix64(salt).
// Read checksum: sum (mod 2^64) over the R beats of sum_i rotl64(w_i, 8i): independent of the
// order in which beats arrive (R beats of different IDs may interleave), sensitive to every bit,
// to the word position in a beat and to duplicated or missing beats.
// Results (cleared at the start, final once the channel is done): status, R / B error-response
// counts, cycles from go (the first cycle a request may be generated) to the last R beat / the
// last B, bytes received / sent, checksum, cycles to the first R beat / the first B, completed
// read bursts / B responses. Timestamps are taken where the engine sees a response, i.e. after
// the bridge's response slices.
// Control: go_tgl / done_tgl (toggle handshake with tsr_bw_ctrl through register stages): a
// change of go_tgl while idle starts a run (the configuration inputs are then static);
// done_tgl := go_tgl when it is over.
// Result read port: res_sel (word index of the channel's 64-byte result block) -> res_rdata
// (registered, one cycle).

module tsr_bw_engine
  import tsr_pkg::*;
#(
  parameter int unsigned CH          = 0,                    // channel index (pattern salt)
  parameter int unsigned ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,  // beat address width (16 GiB: 28)
  parameter int unsigned CNT_W       = 48                    // cycle counters (10.8 days at 300 MHz)
) (
  input  logic                         clk,
  input  logic                         rst_n,

  // ---- control ----
  input  logic                         go_tgl,
  output logic                         done_tgl,
  input  logic [31:0]                  cfg_ctrl,
  input  logic [31:0]                  cfg_seed,
  input  logic [63:0]                  cfg_bytes,
  input  logic [63:0]                  cfg_rd_off,
  input  logic [63:0]                  cfg_wr_off,
  input  logic [63:0]                  cfg_base,
  output logic [63:0]                  base,             // latched buffer address (to the bridge)
  input  logic [3:0]                   res_sel,
  output logic [31:0]                  res_rdata,

  // ---- memory port (tsr_xrt_axi_bridge core side) ----
  output logic                         rd_req_valid,
  input  logic                         rd_req_ready,
  output logic [ADDR_BEAT_W-1:0]       rd_req_addr,
  output logic [TSR_LEN_W-1:0]         rd_req_len,
  output logic [TSR_MEM_ID_W-1:0]      rd_req_id,
  input  logic                         rd_rsp_valid,
  output logic                         rd_rsp_ready,
  input  logic [511:0]                 rd_rsp_data,
  input  logic [TSR_MEM_ID_W-1:0]      rd_rsp_id,
  input  logic                         rd_rsp_last,
  output logic                         wr_req_valid,
  input  logic                         wr_req_ready,
  output logic [ADDR_BEAT_W-1:0]       wr_req_addr,
  output logic [TSR_LEN_W-1:0]         wr_req_len,
  output logic [TSR_MEM_ID_W-1:0]      wr_req_id,
  output logic                         wr_dat_valid,
  input  logic                         wr_dat_ready,
  output logic [511:0]                 wr_dat_data,
  output logic                         wr_dat_last,
  input  logic                         wr_rsp_valid,
  output logic                         wr_rsp_ready,
  input  logic [TSR_MEM_ID_W-1:0]      wr_rsp_id,

  // ---- bridge events (registered pulses) ----
  input  logic                         ev_rresp_err,
  input  logic                         ev_bresp_err,
  input  logic                         ev_unexpected
);

  if (ADDR_BEAT_W < 7 || ADDR_BEAT_W > 52) begin : g_chk_aw
    $fatal(1, "tsr_bw_engine: ADDR_BEAT_W must be in [7, 52]");
  end
  if (CNT_W < 33 || CNT_W > 64) begin : g_chk_cnt
    $fatal(1, "tsr_bw_engine: CNT_W must be in [33, 64]");
  end

  localparam int unsigned AW  = ADDR_BEAT_W;
  localparam int unsigned HW  = AW - 6;            // 64-beat (4 KiB) block index width
  localparam int unsigned NBW = AW + 1;            // beat counts 0 .. 2^AW
  localparam int unsigned IDW = TSR_MEM_ID_W;
  localparam logic [64:0] LIM = 65'd1 << (AW + 6); // region end limit (bytes)

  // One xorshift64 step (linear over GF(2)); C++ reference: tsr_bw_regs.h mix64().
  function automatic logic [63:0] mix64(logic [63:0] x);
    logic [63:0] y, z;
    y = x ^ (x << 13);
    z = y ^ (y >> 7);
    return z ^ (z << 17);
  endfunction

  function automatic logic [63:0] rotl64(logic [63:0] x, int unsigned n);
    return (n == 0) ? x : ((x << n) | (x >> (64 - n)));
  endfunction

  // Burst length at beat [5:0] = lo: min(B, beats to the 4 KiB boundary, beats to the region end);
  // z = this is the end's 4 KiB block (then the end limits: end_lo > lo while beats are left).
  function automatic logic [6:0] burst_len(logic [5:0] lo, logic z, logic [5:0] end_lo, logic [6:0] b);
    logic [6:0] lim;
    lim = z ? ({1'b0, end_lo} - {1'b0, lo}) : (7'd64 - {1'b0, lo});
    return (b < lim) ? b : lim;
  endfunction

  // =============================================================================================
  // State
  // =============================================================================================
  typedef enum logic [2:0] {S_IDLE, S_CHK1, S_CHK2, S_CHK3, S_RUN, S_FIN} st_t;

  st_t          st;
  logic         go_seen_q;
  wire          go_new = (st == S_IDLE) & (go_tgl != go_seen_q);

  // configuration (latched at go_new) and its checks
  logic [31:0]  ctrl_q, seed_q;
  logic [63:0]  bytes_q, rd_off_q, wr_off_q, base_q;
  logic         bad_q, err_q;
  logic         nz_q;                     // bytes != 0 (bytes = 0: both regions empty)
  logic [64:0]  rd_end_q, wr_end_q;       // region ends (bytes)
  logic [63:0]  salt_mix_q;               // mix64(salt)

  wire          c_en  = ctrl_q[0];
  wire          c_rd  = ctrl_q[1];
  wire          c_wr  = ctrl_q[2];
  wire [6:0]    c_b   = ctrl_q[14:8];
  wire [6:0]    c_mo  = ctrl_q[22:16];
  wire [2:0]    c_idl = ctrl_q[26:24];
  wire [3:0]    id_mask = 4'((5'd1 << (c_idl > 3'd4 ? 3'd4 : c_idl)) - 5'd1);

  // generators (rg: AR, wg: AW, dg: W beats). Each walks its region's burst sequence:
  // lo = beat address [5:0], blk = [AW-1:6], nb = 4 KiB blocks to the end's block,
  // z = (nb == 0), elo = end address [5:0]; finished when z and lo == elo.
  logic [5:0]     rg_lo_q, wg_lo_q, dg_lo_q, rd_elo_q, wr_elo_q;
  logic [HW-1:0]  rg_blk_q, wg_blk_q, dg_blk_q;
  logic [HW:0]    rg_nb_q, wg_nb_q, dg_nb_q;
  logic           rg_z_q, wg_z_q, dg_z_q;
  logic [3:0]     rg_idc_q, wg_idc_q;
  logic [6:0]     rd_out_q, wr_out_q;     // bursts in flight per direction (0..64)
  logic [6:0]     dg_left_q;              // W beats of the current burst not yet sent
  logic [6:0]     ahead_q;                // AW bursts generated whose W beats have not started

  logic           ar_v_q, aw_v_q, dv_q, dlast_q;
  logic [AW-1:0]  ar_addr_q, aw_addr_q;
  logic [6:0]     ar_len_q, aw_len_q;
  logic [IDW-1:0] ar_id_q, aw_id_q;
  logic [63:0]    dmix_q;

  wire run    = (st == S_RUN);
  wire rg_fin = rg_z_q & (rg_lo_q == rd_elo_q);
  wire wg_fin = wg_z_q & (wg_lo_q == wr_elo_q);
  wire dg_fin = dg_z_q & (dg_lo_q == wr_elo_q);

  // AR: a new burst enters the output register
  wire       ar_ld  = run & (~ar_v_q | rd_req_ready) & ~rg_fin & (rd_out_q < c_mo);
  wire [6:0] rg_len = burst_len(rg_lo_q, rg_z_q, rd_elo_q, c_b);
  wire [6:0] rg_sum = {1'b0, rg_lo_q} + rg_len;           // [6]: the next burst is in the next block
  // AW
  wire       aw_ld  = run & (~aw_v_q | wr_req_ready) & ~wg_fin & (wr_out_q < c_mo);
  wire [6:0] wg_len = burst_len(wg_lo_q, wg_z_q, wr_elo_q, c_b);
  wire [6:0] wg_sum = {1'b0, wg_lo_q} + wg_len;
  // W beats: a new burst starts when the previous one is complete and its AW was generated
  wire       dg_new = (dg_left_q == 7'd0);
  wire       d_ld   = run & (~dv_q | wr_dat_ready) & ~dg_fin & (~dg_new | (ahead_q != 7'd0));
  wire [6:0] dg_len = dg_new ? burst_len(dg_lo_q, dg_z_q, wr_elo_q, c_b) : dg_left_q;

  // responses (rd_rsp_ready / wr_rsp_ready are constant 1)
  wire r_beat = rd_rsp_valid;
  wire r_end  = rd_rsp_valid & rd_rsp_last;
  wire b_resp = wr_rsp_valid;

  // results
  logic [CNT_W-1:0] cnt1_q;               // cycles since go, including the current one
  logic [CNT_W-1:0] rd_cyc_q, wr_cyc_q;
  logic [31:0]      rd_lat_q, wr_lat_q, rd_bursts_q, wr_bursts_q;
  logic [NBW-1:0]   rd_beats_q, wr_beats_q;
  logic [63:0]      csum_q;
  logic [15:0]      rd_err_q, wr_err_q;
  logic             ran_q, cfg_err_q, rresp_q, bresp_q, unexp_q, done_q, rd_first_q, wr_first_q;

  // checksum pipeline: 8 -> 4 -> 2 -> 1 words (registered), then the accumulator
  logic             cs_v1_q, cs_v2_q, cs_v3_q;
  logic [3:0][63:0] cs1_q;
  logic [1:0][63:0] cs2_q;
  logic [63:0]      cs3_q;
  wire              cs_busy = cs_v1_q | cs_v2_q | cs_v3_q;

  wire rd_done = rg_fin & (rd_out_q == 7'd0);
  wire wr_done = wg_fin & (wr_out_q == 7'd0);

  // CHK3: the generators' start state. A disabled (or refused) direction and bytes = 0 give an
  // empty region (end = start). With bytes = 0 an offset may lie exactly at the limit (a legal
  // empty region ending there): its beat index ra0 / wa0 then wraps to 0 while the end is 2^AW,
  // so the end must not be taken from rd_end_q / wr_end_q.
  logic [AW-1:0]  ra0, wa0;
  logic [NBW-1:0] rend, wend;
  always_comb begin
    ra0  = rd_off_q[AW+5:6];
    wa0  = wr_off_q[AW+5:6];
    rend = (c_en & c_rd & ~err_q & nz_q) ? rd_end_q[AW+6:6] : {1'b0, ra0};
    wend = (c_en & c_wr & ~err_q & nz_q) ? wr_end_q[AW+6:6] : {1'b0, wa0};
  end

  // =============================================================================================
  // Control FSM, counters, results (reset)
  // =============================================================================================
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st          <= S_IDLE;
      go_seen_q   <= 1'b0;
      done_tgl    <= 1'b0;
      ar_v_q      <= 1'b0;
      aw_v_q      <= 1'b0;
      dv_q        <= 1'b0;
      rd_out_q    <= '0;
      wr_out_q    <= '0;
      ahead_q     <= '0;
      dg_left_q   <= '0;
      cs_v1_q     <= 1'b0;
      cs_v2_q     <= 1'b0;
      cs_v3_q     <= 1'b0;
      ran_q       <= 1'b0;
      cfg_err_q   <= 1'b0;
      rresp_q     <= 1'b0;
      bresp_q     <= 1'b0;
      unexp_q     <= 1'b0;
      done_q      <= 1'b0;
      rd_first_q  <= 1'b0;
      wr_first_q  <= 1'b0;
      cnt1_q      <= '0;
      rd_cyc_q    <= '0;
      wr_cyc_q    <= '0;
      rd_lat_q    <= '0;
      wr_lat_q    <= '0;
      rd_bursts_q <= '0;
      wr_bursts_q <= '0;
      rd_beats_q  <= '0;
      wr_beats_q  <= '0;
      csum_q      <= '0;
      rd_err_q    <= '0;
      wr_err_q    <= '0;
    end else begin
      cs_v1_q <= r_beat;
      cs_v2_q <= cs_v1_q;
      cs_v3_q <= cs_v2_q;

      // ---------------- run control ----------------
      unique case (st)
        S_IDLE: begin
          if (go_new) begin
            go_seen_q   <= go_tgl;
            ran_q       <= 1'b0;
            cfg_err_q   <= 1'b0;
            rresp_q     <= 1'b0;
            bresp_q     <= 1'b0;
            unexp_q     <= 1'b0;
            done_q      <= 1'b0;
            rd_first_q  <= 1'b0;
            wr_first_q  <= 1'b0;
            rd_cyc_q    <= '0;
            wr_cyc_q    <= '0;
            rd_lat_q    <= '0;
            wr_lat_q    <= '0;
            rd_bursts_q <= '0;
            wr_bursts_q <= '0;
            rd_beats_q  <= '0;
            wr_beats_q  <= '0;
            csum_q      <= '0;
            rd_err_q    <= '0;
            wr_err_q    <= '0;
            st          <= S_CHK1;
          end
        end
        S_CHK1: st <= S_CHK2;
        S_CHK2: st <= S_CHK3;
        S_CHK3: begin
          // (rd_out, wr_out, ahead, dg_left are 0 between runs: reset, and checked at S_FIN)
          cfg_err_q <= err_q;
          ran_q     <= c_en & ~err_q;
          cnt1_q    <= CNT_W'(1);
          st        <= S_RUN;
        end
        S_RUN: begin
          cnt1_q <= cnt1_q + CNT_W'(1);
          if (rd_done && wr_done && !cs_busy && !dv_q) st <= S_FIN;
        end
        S_FIN: begin
          done_q   <= 1'b1;
          done_tgl <= go_seen_q;
          st       <= S_IDLE;
        end
        default: st <= S_IDLE;
      endcase

      // ---------------- request / data valids, in-flight counts ----------------
      if (ar_ld) ar_v_q <= 1'b1;
      else if (rd_req_ready) ar_v_q <= 1'b0;
      if (aw_ld) aw_v_q <= 1'b1;
      else if (wr_req_ready) aw_v_q <= 1'b0;
      if (d_ld) begin
        dv_q      <= 1'b1;
        dg_left_q <= dg_len - 7'd1;
      end else if (wr_dat_ready) begin
        dv_q <= 1'b0;
      end
      rd_out_q <= rd_out_q + 7'(ar_ld) - 7'(r_end);
      wr_out_q <= wr_out_q + 7'(aw_ld) - 7'(b_resp);
      ahead_q  <= ahead_q + 7'(aw_ld) - 7'(d_ld & dg_new);

      // ---------------- results ----------------
      if (r_beat) begin
        rd_beats_q <= rd_beats_q + NBW'(1);
        rd_cyc_q   <= cnt1_q;
        if (!rd_first_q) begin
          rd_first_q <= 1'b1;
          rd_lat_q   <= 32'(cnt1_q);
        end
      end
      if (r_end) rd_bursts_q <= rd_bursts_q + 32'd1;
      if (b_resp) begin
        wr_bursts_q <= wr_bursts_q + 32'd1;
        wr_cyc_q    <= cnt1_q;
        if (!wr_first_q) begin
          wr_first_q <= 1'b1;
          wr_lat_q   <= 32'(cnt1_q);
        end
      end
      if (dv_q && wr_dat_ready) wr_beats_q <= wr_beats_q + NBW'(1);
      if (cs_v3_q) csum_q <= csum_q + cs3_q;
      if (ev_rresp_err) begin
        rresp_q <= 1'b1;
        if (rd_err_q != 16'hFFFF) rd_err_q <= rd_err_q + 16'd1;
      end
      if (ev_bresp_err) begin
        bresp_q <= 1'b1;
        if (wr_err_q != 16'hFFFF) wr_err_q <= wr_err_q + 16'd1;
      end
      if (ev_unexpected) unexp_q <= 1'b1;
    end
  end

  // =============================================================================================
  // Configuration, generator state, payloads (no reset: used only in a run, initialised by the
  // check states)
  // =============================================================================================
  always_ff @(posedge clk) begin
    if (go_new) begin
      ctrl_q   <= cfg_ctrl;
      seed_q   <= cfg_seed;
      bytes_q  <= cfg_bytes;
      rd_off_q <= cfg_rd_off;
      wr_off_q <= cfg_wr_off;
      base_q   <= cfg_base;
    end
    if (st == S_CHK1) begin
      bad_q <= (~c_rd & ~c_wr) | (c_b == 7'd0) | (c_b > 7'd64) | (c_mo == 7'd0) | (c_mo > 7'd64)
             | (c_idl > 3'd4) | (base_q[11:0] != 12'd0) | (bytes_q[5:0] != 6'd0)
             | (c_rd & (rd_off_q[5:0] != 6'd0)) | (c_wr & (wr_off_q[5:0] != 6'd0));
      nz_q       <= (bytes_q != 64'd0);
      rd_end_q   <= {1'b0, rd_off_q} + {1'b0, bytes_q};
      wr_end_q   <= {1'b0, wr_off_q} + {1'b0, bytes_q};
      salt_mix_q <= mix64({seed_q, 24'd0, 8'(CH)});
    end
    if (st == S_CHK2) err_q <= c_en & (bad_q | (c_rd & (rd_end_q > LIM)) | (c_wr & (wr_end_q > LIM)));
    if (st == S_CHK3) begin
      rg_lo_q  <= ra0[5:0];
      rg_blk_q <= ra0[AW-1:6];
      rg_nb_q  <= rend[AW:6] - {1'b0, ra0[AW-1:6]};
      rg_z_q   <= (rend[AW:6] == {1'b0, ra0[AW-1:6]});
      rd_elo_q <= rend[5:0];
      rg_idc_q <= '0;
      wg_lo_q  <= wa0[5:0];
      wg_blk_q <= wa0[AW-1:6];
      wg_nb_q  <= wend[AW:6] - {1'b0, wa0[AW-1:6]};
      wg_z_q   <= (wend[AW:6] == {1'b0, wa0[AW-1:6]});
      wr_elo_q <= wend[5:0];
      wg_idc_q <= '0;
      dg_lo_q  <= wa0[5:0];
      dg_blk_q <= wa0[AW-1:6];
      dg_nb_q  <= wend[AW:6] - {1'b0, wa0[AW-1:6]};
      dg_z_q   <= (wend[AW:6] == {1'b0, wa0[AW-1:6]});
    end
    // AR generator
    if (ar_ld) begin
      ar_addr_q <= {rg_blk_q, rg_lo_q};
      ar_len_q  <= rg_len;
      ar_id_q   <= rg_idc_q & id_mask;
      rg_idc_q  <= rg_idc_q + 4'd1;
      rg_lo_q   <= rg_sum[5:0];
      if (rg_sum[6]) begin
        rg_blk_q <= rg_blk_q + HW'(1);
        rg_nb_q  <= rg_nb_q - (HW+1)'(1);
        rg_z_q   <= (rg_nb_q == (HW+1)'(1));
      end
    end
    // AW generator
    if (aw_ld) begin
      aw_addr_q <= {wg_blk_q, wg_lo_q};
      aw_len_q  <= wg_len;
      aw_id_q   <= wg_idc_q & id_mask;
      wg_idc_q  <= wg_idc_q + 4'd1;
      wg_lo_q   <= wg_sum[5:0];
      if (wg_sum[6]) begin
        wg_blk_q <= wg_blk_q + HW'(1);
        wg_nb_q  <= wg_nb_q - (HW+1)'(1);
        wg_z_q   <= (wg_nb_q == (HW+1)'(1));
      end
    end
    // W beats: pattern of the beat at byte offset {blk, lo, 6'b0}
    if (d_ld) begin
      dlast_q <= (dg_len == 7'd1);
      dmix_q  <= mix64({{(58 - AW){1'b0}}, dg_blk_q, dg_lo_q, 6'd0});
      dg_lo_q <= dg_lo_q + 6'd1;
      if (dg_lo_q == 6'd63) begin
        dg_blk_q <= dg_blk_q + HW'(1);
        dg_nb_q  <= dg_nb_q - (HW+1)'(1);
        dg_z_q   <= (dg_nb_q == (HW+1)'(1));
      end
    end
    // read checksum stages: rotl64(w_{2j}, 16j) + rotl64(w_{2j+1}, 16j + 8), then a tree
    for (int unsigned j = 0; j < 4; j++) begin
      cs1_q[j] <= rotl64(rd_rsp_data[128*j +: 64], 16 * j) + rotl64(rd_rsp_data[128*j + 64 +: 64], 16 * j + 8);
    end
    cs2_q[0] <= cs1_q[0] + cs1_q[1];
    cs2_q[1] <= cs1_q[2] + cs1_q[3];
    cs3_q    <= cs2_q[0] + cs2_q[1];
  end

  // =============================================================================================
  // Outputs
  // =============================================================================================
  assign base         = base_q;
  assign rd_req_valid = ar_v_q;
  assign rd_req_addr  = ar_addr_q;
  assign rd_req_len   = ar_len_q;
  assign rd_req_id    = ar_id_q;
  assign rd_rsp_ready = 1'b1;
  assign wr_req_valid = aw_v_q;
  assign wr_req_addr  = aw_addr_q;
  assign wr_req_len   = aw_len_q;
  assign wr_req_id    = aw_id_q;
  assign wr_dat_valid = dv_q;
  assign wr_dat_last  = dlast_q;
  assign wr_rsp_ready = 1'b1;
  for (genvar i = 0; i < 8; i++) begin : g_wd
    // word i = mix64(o) ^ mix64(8i) ^ mix64(salt)  (mix64(8i): a constant)
    assign wr_dat_data[64*i +: 64] = dmix_q ^ mix64(64'(8 * i)) ^ salt_mix_q;
  end

  // Result block (64 bytes; word index = byte offset [5:2]; docs/m0-bw.md §2)
  wire [63:0] rd_cyc64 = 64'(rd_cyc_q), wr_cyc64 = 64'(wr_cyc_q);
  wire [63:0] rd_byt64 = 64'(rd_beats_q) << 6, wr_byt64 = 64'(wr_beats_q) << 6;
  always_ff @(posedge clk) begin
    unique case (res_sel)
      4'd0:    res_rdata <= {26'd0, done_q, unexp_q, bresp_q, rresp_q, cfg_err_q, ran_q};
      4'd1:    res_rdata <= {wr_err_q, rd_err_q};
      4'd2:    res_rdata <= rd_cyc64[31:0];
      4'd3:    res_rdata <= rd_cyc64[63:32];
      4'd4:    res_rdata <= wr_cyc64[31:0];
      4'd5:    res_rdata <= wr_cyc64[63:32];
      4'd6:    res_rdata <= rd_byt64[31:0];
      4'd7:    res_rdata <= rd_byt64[63:32];
      4'd8:    res_rdata <= wr_byt64[31:0];
      4'd9:    res_rdata <= wr_byt64[63:32];
      4'd10:   res_rdata <= csum_q[31:0];
      4'd11:   res_rdata <= csum_q[63:32];
      4'd12:   res_rdata <= rd_lat_q;
      4'd13:   res_rdata <= wr_lat_q;
      4'd14:   res_rdata <= rd_bursts_q;
      default: res_rdata <= wr_bursts_q;
    endcase
  end

  wire unused_ok = &{1'b0, rd_rsp_id, wr_rsp_id, ctrl_q[31:27], ctrl_q[23], ctrl_q[15], ctrl_q[7:3]};

  // =============================================================================================
  // Simulation checks
  // =============================================================================================
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (r_end) assert (rd_out_q != 7'd0) else $error("%m: read burst completed with none outstanding");
      if (b_resp) assert (wr_out_q != 7'd0) else $error("%m: write response with no write outstanding");
      if (rd_rsp_valid) assert (run) else $error("%m: read beat outside a run");
      if (wr_rsp_valid) assert (run) else $error("%m: write response outside a run");
      assert (rd_out_q <= 7'd64 && wr_out_q <= 7'd64) else $error("%m: in-flight counter beyond 64");
      if (ar_ld) assert (rg_len >= 7'd1 && rg_len <= 7'd64 && 32'(rg_lo_q) + 32'(rg_len) <= 64)
        else $error("%m: read burst len %0d at beat[5:0] %0d (crosses 4 KiB)", rg_len, rg_lo_q);
      if (aw_ld) assert (wg_len >= 7'd1 && wg_len <= 7'd64 && 32'(wg_lo_q) + 32'(wg_len) <= 64)
        else $error("%m: write burst len %0d at beat[5:0] %0d (crosses 4 KiB)", wg_len, wg_lo_q);
      if (d_ld) assert (dg_len >= 7'd1 && dg_len <= 7'd64) else $error("%m: W burst length %0d", dg_len);
      if (st == S_FIN) assert (ahead_q == 7'd0 && dg_left_q == 7'd0 && dg_fin)
        else $error("%m: write data left at the end of the run");
    end
  end
`endif

endmodule
