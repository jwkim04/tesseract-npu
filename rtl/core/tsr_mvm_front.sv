// tsr_mvm_front.sv — MVM cluster front end (spec MVM cluster microarchitecture stages 1-4): weight streamer
// (tsr_mvm_ws), chunk parser, meta register with the Q4_K 6-bit scale unpack, weight unpack;
// produces the tagged beat stream tsr_mvm_beat_t for the back end.
//
// Owner: front agent (docs/rtl-mvm.md §2 block (front); contract §3.3, §4).
//
// Contract (normative text: docs/rtl-mvm.md §3.3; golden model tb/common/mvm_model.py front_beats):
//   - cmd (tsr_mvm_fcmd_t) valid/ready: one per MVM of a channel with rows >= 1; cmd_ready is a
//     register, 1 only when idle.
//   - the channel's rows (cmd.rows rows of cmd.row_beats beats from cmd.w_off) are streamed with
//     tsr_mvm_ws (exactly cmd.beats beats, docs/rtl-mvm.md §4) and parsed in order: per row, per
//     chunk (Q8_0: 1 meta + 16 quant beats, Q4_K: 1 + 8; the last chunk of a row may be partial:
//     Q8_0 rem blocks -> 1 + ceil(rem/2) beats, Q4_K rem superblocks -> 1 + 2*rem beats).
//     META beats are held in the meta register and not forwarded; every QUANT beat yields exactly
//     one tsr_mvm_beat_t, in stream order (tsr_ref.c block_offsets / tsr_dot_row walk):
//       Q8_0 quant beat j of chunk c: blk = 32c + 2j, two = (2j + 1 < blocks in chunk),
//         d_a / d_b = fp16_to_f32(meta bytes [4j, 4j+2) / [4j+2, 4j+4)) (d_b = +0 if !two),
//         w lanes 0..63 = the beat's bytes as int8 (lanes 32..63 = 0 if !two), lanes 64..127 = 0,
//         half = 0, sc = 0, m = 0.
//       Q4_K quant beat j of chunk c: s = 4c + j/2, half = j%2, blk = s,
//         meta = bytes [16(j/2), +16): d_a = fp16_to_f32(d), d_b = fp16_to_f32(dmin),
//         (sc_k, m_k) = get_scale_min_k4(k, scales) for k = 0..7: sc[g] = sc_(4*half+g), m = m_0..7,
//         w lane 32g' + l (g' = 0..3, l = 0..31) = nibble of beat byte 32*(g'/2) + l: low nibble
//         for even g', high nibble for odd g' (so lane L multiplies K position 256s + 128*half + L).
//       row_last = last quant beat of the row; last = row_last of row rows-1.
//   - beat_valid/beat_ready/beat follow R1/R2; one beat per cycle when the memory and the back end
//     allow (META beats cost a cycle of the weight stream but produce no output).
//   - idle: no command in progress, nothing buffered, no read outstanding.
// The front end never looks at activations; it does not know T.
// Attention streams (W4-B at, AT §8.1 / §8.2): with cmd_att = 1 the command is a KV plane stream in
//   "rows" of K = 1024 (one full Q8_0 chunk: a K plane group is E/64 chunks, a V plane slice of a
//   group is one chunk), so the tags above hold unchanged (blk = 2j, row_last at quant beat 15);
//   the back end / PV unit count cells, segments and slices themselves. cmd_rbeats / cmd_stride
//   pass to tsr_mvm_ws (ranges: the V plane's slices); the weight MVM sends cmd_rbeats = cmd.beats.
//   K = 1024 may exceed K_MAX in small configurations (attention needs only G E <= K_MAX).
//
// Microarchitecture (one weight beat per cycle; no combinational path from any input to any
// output; beat_ready only reaches the two output-side register stages):
//
//   tsr_mvm_ws ──out (OQ head)──► parser (counters: META / quant beat j, chunk, row, blk, half)
//        │                          │ META beat  -> meta_q (512 b)
//        │                          │ QUANT beat -> P = {tags, selected meta, raw beat}
//        │                          ▼
//        │              P: tsr_rslice (main + skid, registered s_ready, 656 b)
//        │                          ▼   unpack: fp16->f32 (d_a, d_b), get_scale_min_k4, lanes
//        └─ rd_* (WS source)    U: beat_q (tsr_mvm_beat_t, 1175 b) = the beat output
//
//   - setup: cmd fields are decoded into per-command registers at the accepting edge (chunks per
//     row, quant beats per full / last chunk, Q8_0 odd remainder) and the parser counters are
//     initialised from them one cycle later; tsr_mvm_ws starts requesting at the accepting edge
//     (its first beat cannot arrive before the parser runs, and the parser only takes beats while
//     it runs).
//   - parser: takes a beat from tsr_mvm_ws when it runs and (META beat or P can accept). The
//     ready towards tsr_mvm_ws is built from flops only (P's registered s_ready), so beat_ready
//     never reaches the counters, meta_q or the FIFO.
//   - meta select (registered index qj_q): Q8_0 meta_q[32j +: 32] (16:1), Q4_K
//     meta_q[128(j/2) +: 128] (4:1). A META beat may overwrite meta_q while quant beats of the
//     previous chunk wait in P: they carry their selected meta.
//   - U stage: fp16 -> f32 (tsr_fp_pkg::tsr_fp16_to_f32, exact) of d / d(b+1) or dmin, the 6-bit
//     scale unpack (bit slicing), the lane unpack (byte or nibble select, 1 LUT level).
//   Latency (cycles, ideal memory): FIFO write -> RAM read issue 1, RAM WS_RD_LAT, OQ 1, P 1, U 1.
//   Measured (tb/mvmfront, tsr_mem_arb + MemModel): 1.000 cycles per weight beat (META beats
//   included) for both formats across chunk, row and burst boundaries with an ideal memory and
//   beat_ready = 1 (FIFO 512 / 1024, any WS_RD_LAT 1..4); first tagged beat 16 cycles after the
//   command handshake with a 1-cycle memory (len 1, load 1, arbiter 2 + 1, memory, FIFO 1,
//   RAM 2, OQ 1, P 1, U 1); a fixed memory latency of 400 cycles still gives 1.000, 520 gives 1.14.
//   Logic depth: parser counters / meta select (16:1 or 4:1) / rslice input mux <= 4 LUT levels;
//   unpack: fp16 -> f32 (1-3 levels, tsr_fp_pkg) plus the d_b / lane zeroing; beat_ready drives
//   only ce_u (1 LUT) -> the CE pins of beat_q (1175) and of the P slice (2 x 656).

module tsr_mvm_front
  import tsr_pkg::*;
  import tsr_fp_pkg::*;
#(
  parameter int unsigned K_MAX         = TSR_DEF_K_MAX,
  parameter int unsigned ADDR_BEAT_W   = TSR_DEF_ADDR_BEAT_W,
  parameter int unsigned BEAT_BITS     = TSR_DEF_BEAT_BITS,
  parameter int unsigned WS_FIFO_BEATS = TSR_DEF_WS_FIFO_BEATS,
  parameter int unsigned WS_RD_LAT     = TSR_DEF_WS_RD_LAT        // weight FIFO RAM read latency
) (
  input  logic                        clk,
  input  logic                        rst_n,

  // ---- command (from the cluster wrapper) ----
  input  logic                        cmd_valid,
  output logic                        cmd_ready,
  input  tsr_mvm_fcmd_t               cmd,
  input  logic                        cmd_att,       // KV plane stream (K = 1024 rows)
  input  logic [TSR_MVM_BEATS_W-1:0]  cmd_rbeats,    // tsr_mvm_ws ranges (>= beats: one range)
  input  logic [31:0]                 cmd_stride,

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

  // ---- tagged beats to the back end ----
  output logic                        beat_valid,
  input  logic                        beat_ready,
  output tsr_mvm_beat_t               beat,

  output logic                        idle
);

  if (BEAT_BITS != 512) begin : g_chk_beat
    $fatal(1, "tsr_mvm_front: BEAT_BITS must be 512");
  end
  if (K_MAX < 256 || K_MAX > 65280 || (K_MAX % 256) != 0) begin : g_chk_k
    $fatal(1, "tsr_mvm_front: K_MAX must be a multiple of 256 in [256, 65280]");
  end

  // Chunks per row: ceil(K / 1024) for both formats (a chunk holds 1024 weights).
  localparam int unsigned NCH_MAX = (K_MAX + 1023) / 1024;
  localparam int unsigned NCH_W   = (NCH_MAX > 1) ? $clog2(NCH_MAX) : 1;   // holds NCH_MAX - 1

  // -------------------------------------------------------------------------------------------
  // Weight streamer
  // -------------------------------------------------------------------------------------------
  logic                 cmd_ready_q;
  logic                 ws_cmd_ready, ws_idle;
  logic                 ws_out_valid, ws_out_ready;
  logic [BEAT_BITS-1:0] ws_out_data;

  wire cmd_fire = cmd_valid & cmd_ready_q;

  tsr_mvm_ws #(
    .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .FIFO_BEATS(WS_FIFO_BEATS), .RD_LAT(WS_RD_LAT)
  ) u_ws (
    .clk, .rst_n,
    .cmd_valid(cmd_fire), .cmd_ready(ws_cmd_ready), .cmd_w_off(cmd.w_off), .cmd_beats(cmd.beats),
    .cmd_rbeats, .cmd_stride,
    .rd_req_valid, .rd_req_ready, .rd_req_addr, .rd_req_len, .rd_req_id,
    .rd_rsp_valid, .rd_rsp_ready, .rd_rsp_data, .rd_rsp_last, .rd_rsp_id,
    .out_valid(ws_out_valid), .out_ready(ws_out_ready), .out_data(ws_out_data),
    .idle(ws_idle)
  );

  // -------------------------------------------------------------------------------------------
  // Per-command configuration (decoded at the accepting edge)
  // -------------------------------------------------------------------------------------------
  logic             cfg_fmt_q;       // 0: Q8_0, 1: Q4_K
  logic             cfg_rem_odd_q;   // Q8_0: the row's last chunk has an odd block count
  logic [3:0]       cfg_cqf_m1_q;    // quant beats - 1 of a full chunk (15 / 7)
  logic [3:0]       cfg_cql_m1_q;    // quant beats - 1 of the row's last chunk
  logic [NCH_W-1:0] cfg_nch_m1_q;    // chunks per row - 1
  logic             cfg_one_q;       // one chunk per row (K <= 1024)
  logic [31:0]      cfg_rows_q;

  // Decode of the command (K is a multiple of the block size, 0 < K <= K_MAX):
  //   Q8_0: rem = K/32 mod 32 = k[9:5] blocks, last chunk ceil(rem/2) quant beats (rem = 0: full);
  //   Q4_K: rem = K/256 mod 4 = k[9:8] superblocks, last chunk 2*rem quant beats.
  logic [4:0] d_rem8;
  logic [1:0] d_rem4;
  logic [3:0] d_cql_m1;
  logic [6:0] d_nch;
  always_comb begin
    d_rem8 = cmd.k[9:5];
    d_rem4 = cmd.k[9:8];
    if (!cmd.fmt) d_cql_m1 = (d_rem8 == 5'd0) ? 4'd15 : 4'((d_rem8 - 5'd1) >> 1);
    else          d_cql_m1 = (d_rem4 == 2'd0) ? 4'd7  : {1'b0, d_rem4 - 2'd1, 1'b1};
    d_nch = {1'b0, cmd.k[15:10]} + 7'(cmd.k[9:0] != 10'd0);
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cfg_fmt_q     <= 1'b0;
      cfg_rem_odd_q <= 1'b0;
      cfg_cqf_m1_q  <= '0;
      cfg_cql_m1_q  <= '0;
      cfg_nch_m1_q  <= '0;
      cfg_one_q     <= 1'b0;
      cfg_rows_q    <= '0;
    end else if (cmd_fire) begin
      cfg_fmt_q     <= cmd.fmt;
      cfg_rem_odd_q <= ~cmd.fmt & cmd.k[5];
      cfg_cqf_m1_q  <= cmd.fmt ? 4'd7 : 4'd15;
      cfg_cql_m1_q  <= d_cql_m1;
      cfg_nch_m1_q  <= NCH_W'(d_nch - 7'd1);
      cfg_one_q     <= (cmd.k <= 16'd1024);
      cfg_rows_q    <= cmd.rows;
    end
  end

  // -------------------------------------------------------------------------------------------
  // Chunk parser (stage 2) and meta register (stage 3)
  // -------------------------------------------------------------------------------------------
  // P stage payload: one QUANT beat with its tags and its selected meta entry.
  typedef struct packed {
    logic                     fmt;
    logic                     last;
    logic                     row_last;
    logic [TSR_MVM_BLK_W-1:0] blk;
    logic                     two;
    logic                     half;
    logic [127:0]             sel;   // [15:0] d (Q8_0: d of b), [31:16] Q8_0 d of b+1 / Q4_K dmin,
                                     // [127:32] Q4_K scales[12] (Q8_0: 0)
    logic [BEAT_BITS-1:0]     raw;   // the quant beat
  } pbeat_t;

  logic                     setup_q;     // cycle after the accept: counters start from cfg_*
  logic                     run_q;       // parsing (until the take of the last beat)
  logic                     meta_nx_q;   // the next stream beat is a META beat
  logic [3:0]               qj_q;        // quant beat index j inside the chunk
  logic [3:0]               cq_m1_q;     // quant beats - 1 of the current chunk
  logic [NCH_W-1:0]         chl_q;       // chunks left in the row after the current one
  logic                     lastch_q;    // the current chunk is the row's last
  logic [31:0]              rowl_q;      // rows left after the current one
  logic                     lastrow_q;   // the current row is the last
  logic [TSR_MVM_BLK_W-1:0] blk_q;       // Q8_0: first block b of the next quant beat; Q4_K: s
  logic                     half_q;      // Q4_K: second beat of superblock s
  logic [BEAT_BITS-1:0]     meta_q;      // META beat of the current chunk

  logic   p_s_ready, p_m_valid, ce_u;
  pbeat_t p_in, p_m;

  assign ws_out_ready = run_q & (meta_nx_q | p_s_ready);
  wire take      = ws_out_valid & ws_out_ready;
  wire take_q    = take & ~meta_nx_q;              // a QUANT beat enters P
  wire chunk_end = (qj_q == cq_m1_q);
  wire row_end   = chunk_end & lastch_q;
  wire mvm_end   = row_end & lastrow_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      setup_q   <= 1'b0;
      run_q     <= 1'b0;
      meta_nx_q <= 1'b1;
      qj_q      <= '0;
      cq_m1_q   <= '0;
      chl_q     <= '0;
      lastch_q  <= 1'b0;
      rowl_q    <= '0;
      lastrow_q <= 1'b0;
      blk_q     <= '0;
      half_q    <= 1'b0;
    end else begin
      setup_q <= cmd_fire;
      if (setup_q) begin
        run_q     <= (cfg_rows_q != 32'd0);
        meta_nx_q <= 1'b1;
        qj_q      <= '0;
        cq_m1_q   <= cfg_one_q ? cfg_cql_m1_q : cfg_cqf_m1_q;
        chl_q     <= cfg_nch_m1_q;
        lastch_q  <= cfg_one_q;
        rowl_q    <= cfg_rows_q - 32'd1;
        lastrow_q <= (cfg_rows_q == 32'd1);
        blk_q     <= '0;
        half_q    <= 1'b0;
      end else if (take) begin
        if (meta_nx_q) begin
          meta_nx_q <= 1'b0;
        end else begin
          // quant beat: advance j, blk / half
          qj_q <= qj_q + 4'd1;
          if (cfg_fmt_q) begin
            half_q <= ~half_q;
            if (half_q) blk_q <= blk_q + 1'b1;
          end else begin
            blk_q <= blk_q + TSR_MVM_BLK_W'(2);
          end
          if (chunk_end) begin
            meta_nx_q <= 1'b1;
            qj_q      <= '0;
            if (lastch_q) begin            // row end: next row starts at chunk 0, block 0
              blk_q     <= '0;
              half_q    <= 1'b0;
              chl_q     <= cfg_nch_m1_q;
              lastch_q  <= cfg_one_q;
              cq_m1_q   <= cfg_one_q ? cfg_cql_m1_q : cfg_cqf_m1_q;
              rowl_q    <= rowl_q - 32'd1;
              lastrow_q <= (rowl_q == 32'd1);
              if (lastrow_q) run_q <= 1'b0;
            end else begin
              chl_q     <= chl_q - 1'b1;
              lastch_q  <= (chl_q == NCH_W'(1));
              cq_m1_q   <= (chl_q == NCH_W'(1)) ? cfg_cql_m1_q : cfg_cqf_m1_q;
            end
          end
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (take && meta_nx_q) meta_q <= ws_out_data;
  end

  // Tags and meta selection of the quant beat being taken.
  always_comb begin
    p_in          = '0;
    p_in.fmt      = cfg_fmt_q;
    p_in.last     = mvm_end;
    p_in.row_last = row_end;
    p_in.blk      = blk_q;
    p_in.two      = ~cfg_fmt_q & ~(row_end & cfg_rem_odd_q);
    p_in.half     = cfg_fmt_q & half_q;
    p_in.sel      = cfg_fmt_q ? meta_q[128 * qj_q[2:1] +: 128] : {96'd0, meta_q[32 * qj_q +: 32]};
    p_in.raw      = ws_out_data;
  end

  tsr_rslice #(.W($bits(pbeat_t))) u_p (
    .clk, .rst_n,
    .s_valid(take_q), .s_ready(p_s_ready), .s_data(p_in),
    .m_valid(p_m_valid), .m_ready(ce_u), .m_data(p_m)
  );

  // -------------------------------------------------------------------------------------------
  // Unpack (stage 4) into the output register
  // -------------------------------------------------------------------------------------------
  function automatic tsr_mvm_beat_t unpack(pbeat_t p);
    tsr_mvm_beat_t   b;
    logic [11:0][7:0] q;       // Q4_K scales[12]
    logic [7:0][5:0]  sc, mn;  // get_scale_min_k4(j, q), j = 0..7
    b          = '0;
    b.last     = p.last;
    b.row_last = p.row_last;
    b.blk      = p.blk;
    b.two      = p.two;
    b.half     = p.half;
    b.d_a      = tsr_fp16_to_f32(p.sel[15:0]);
    b.d_b      = (p.fmt | p.two) ? tsr_fp16_to_f32(p.sel[31:16]) : 32'd0;
    // ggml get_scale_min_k4 (tsr_ref.c 359-367):
    //   j < 4: sc = q[j] & 63,                        m = q[j+4] & 63
    //   j >= 4: sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
    q = p.sel[127:32];
    for (int j = 0; j < 4; j++) begin
      sc[j]     = q[j][5:0];
      mn[j]     = q[j+4][5:0];
      sc[j + 4] = {q[j][7:6], q[j+8][3:0]};
      mn[j + 4] = {q[j+4][7:6], q[j+8][7:4]};
    end
    if (p.fmt) begin
      for (int g = 0; g < TSR_MVM_GROUPS; g++) b.sc[g] = p.half ? sc[g + 4] : sc[g];
      b.m = mn;
      // lane 32g + l <- nibble of byte 32(g/2) + l: low for even g, high for odd g
      for (int g = 0; g < TSR_MVM_GROUPS; g++) begin
        for (int l = 0; l < TSR_MVM_GROUP_LANES; l++) begin
          b.w[32*g + l] = {4'd0, p.raw[8*(32*(g/2) + l) + 4*(g%2) +: 4]};
        end
      end
    end else begin
      for (int l = 0; l < 32; l++) b.w[l]      = p.raw[8*l +: 8];
      for (int l = 32; l < 64; l++) b.w[l]     = p.two ? p.raw[8*l +: 8] : 8'd0;
    end
    return b;
  endfunction

  logic          u_valid_q;
  tsr_mvm_beat_t beat_q;

  assign ce_u = ~u_valid_q | beat_ready;

  always_ff @(posedge clk) begin
    if (!rst_n) u_valid_q <= 1'b0;
    else if (ce_u) u_valid_q <= p_m_valid;
  end

  always_ff @(posedge clk) begin
    if (ce_u && p_m_valid) beat_q <= unpack(p_m);
  end

  assign beat_valid = u_valid_q;
  assign beat       = beat_q;

  // -------------------------------------------------------------------------------------------
  // Command handshake / idle
  // -------------------------------------------------------------------------------------------
  wire fe_idle = ~setup_q & ~run_q & ws_idle & ~p_m_valid & ~u_valid_q;

  always_ff @(posedge clk) begin
    if (!rst_n) cmd_ready_q <= 1'b1;
    else        cmd_ready_q <= cmd_fire ? 1'b0 : (cmd_ready_q | fe_idle);
  end

  assign cmd_ready = cmd_ready_q;
  assign idle      = cmd_ready_q;

  // cmd.row_beats is implied by fmt and k (checked below); chl_q is the counter behind lastch_q.
  wire unused_ok = &{1'b0, cmd.row_beats, ws_cmd_ready};

  // -------------------------------------------------------------------------------------------
  // Simulation checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (cmd_fire) begin
        assert (ws_cmd_ready) else $error("%m: command accepted while the weight streamer is busy");
        assert (cmd.k != 16'd0 && (32'(cmd.k) <= K_MAX || (cmd_att && cmd.k == 16'd1024 && !cmd.fmt))
                && (cmd.fmt ? cmd.k[7:0] == 8'd0 : cmd.k[4:0] == 5'd0))
          else $error("%m: K = %0d is not a legal multiple of the block size <= K_MAX", cmd.k);
        assert (cmd.rows != 32'd0) else $error("%m: command with rows = 0");
        assert (32'(cmd.row_beats) == tsr_row_beats(cmd.fmt, 32'(cmd.k)))
          else $error("%m: row_beats %0d != ROW_BEATS(%0d, %0d)", cmd.row_beats, cmd.fmt, cmd.k);
        assert (64'(cmd.beats) == 64'(cmd.rows) * 64'(cmd.row_beats))
          else $error("%m: beats %0d != rows %0d * row_beats %0d", cmd.beats, cmd.rows, cmd.row_beats);
      end
      // every weight beat delivered belongs to a parsed row
      assert (!(ws_out_valid && !run_q && !setup_q))
        else $error("%m: weight beat beyond the command's rows");
      assert (!(take && !meta_nx_q && qj_q > cq_m1_q)) else $error("%m: quant beat index overflow");
    end
  end
`endif

endmodule
