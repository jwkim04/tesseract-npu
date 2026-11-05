// tsr_vu.sv — the vector unit (unit 3): VOP, RED, ROPE, SOP and AT's vector ops ASMAX, AMASK, AMERGE, ANORM
// (TOPK and DQROW with HAS_TOPK / HAS_DQROW), with the s registers s0-s15.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §4 programming model, §5 encodings, §6 numerics, §7
// microarchitecture; AT §5.4-§5.7; W4-B task "vu").
// Contract (VU §4.5, §7.8, CC §9.4 I3):
//   cmd_valid / cmd_ready / cmd: the eight words of a checked op (the VU checker class's payload,
//   tsr_xc_pkg); cmd_ready while fewer than TSR_VU_ACCEPT_AHEAD (2) ops are held. done pulses once per op in
//   accept order when every SPM write of the op has been accepted by tsr_spm and its s write is done.
//   No error output: every check is at dispatch (tsr_vu_chk).
//   Results equal the simulator's (sw/sim/tsr_vu.c, tsr_at.c on tsr_ref_vu.c / tsr_ref_at.c) bit for bit.
//   The s bank is +0.0 after a hardware reset (rst_n); nothing else changes it (VU §4.4, VU-12).
// Accept-ahead and hazards (VU §7.8): the second held op starts its reads while the first drains unless
//   the first op's write extents overlap the second's read or write extents, the first writes an s
//   register the second reads or writes, the second is an SOP SQRT / RSQRT / DIV (scalar path) or the
//   first one is, or both are ROPE (one angle unit). Write-after-read needs no check: the injector takes
//   slots in order, so every read of the first op has returned before the second op's first slot runs.
// Blocks: tsr_vu_gen (slots) -> tsr_vu_rd (SPM reads) -> tsr_vu_inj (operands) -> FRONT_LANES x tsr_vu_lane ->
//   tsr_vu_wr (assembler, output FIFO, SPM writes); tsr_vu_red (partial drain, trees, post, scalar path);
//   tsr_vu_angle (ROPE); the local memory (512 lines x 512 bits, 1024 with NUM_CH > 8; two read ports) for
//   AT's multi-pass ops (ANORM 1/l; AMERGE a, b, m' vectors and the channel-tree scratch).
// Knobs (VU §7.10, results unchanged): EXP_LANES 16 / 8, BACK_LANES 8 / 4.
// LANE_PIPE (CC §14.1 remedy 3, results unchanged): register stages on each side of the lane array
//   (tsr_vu_lanes), 0 = none. With p stages the lanes can sit in another SLR than the SPM-facing part:
//   every lane exit, and a partial drain read, takes 2p cycles longer (LANE_X below).

module tsr_vu
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
  import tsr_xc_pkg::*;
  import tsr_att_pkg::*;
#(
  parameter  int unsigned SPM_BYTES  = TSR_DEF_SPM_BYTES,
  parameter  int unsigned SPM_RD_LAT = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned NUM_CH     = TSR_DEF_NUM_CH,
  parameter  int unsigned FRONT_LANES = 16,
  parameter  int unsigned EXP_LANES  = 16,
  parameter  int unsigned BACK_LANES = 8,
  parameter  int unsigned LANE_PIPE  = 0,             // registered pipe stages on each side of the lanes
  parameter  bit          LOCAL_SCALE = 1'b0,
  parameter  bit          LOCAL_INTERMEDIATES = 1'b0,
  parameter  bit          HAS_TOPK   = 1'b0,          // TOPK (CAPS bit3, VU §5.10: RTL from M8)
  parameter  bit          HAS_DQROW  = 1'b0,          // DQROW (CAPS bit6, VU §5.6: RTL from M8)
  parameter  int unsigned OFD        = 64,
  localparam int unsigned LINES      = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW         = (LINES > 1) ? $clog2(LINES) : 1
) (
  input  logic                           clk,
  input  logic                           rst_n,
  // unit command (tsr_issue: vu_cmd_valid / vu_cmd_ready / vu_cmd / vu_done)
  input  logic                           cmd_valid,
  output logic                           cmd_ready,
  input  logic [TSR_XC_CMD_W-1:0]        cmd,
  output logic                           done,
  input logic                             scale_request,
  output logic                            scale_grant,
  input logic                             scale_done,
  input tsr_vu_local_req_t [1:0]           scale_req,
  output tsr_vu_local_rsp_t [1:0]          scale_rsp,
  output tsr_local_score_req_t [NUM_CH-1:0] local_score_req,
  input  logic [NUM_CH-1:0][TSR_LINE_BITS-1:0] local_score_data,
  // SPM client 4 (both bank ports)
  output logic [1:0]                     rd_req_valid,
  input  logic [1:0]                     rd_req_ready,
  output logic [1:0][AW-1:0]             rd_req_line,
  input  logic [1:0]                     rd_rsp_valid,
  input  logic [1:0][TSR_LINE_BITS-1:0]  rd_rsp_data,
  output logic [1:0]                     wr_valid,
  input  logic [1:0]                     wr_ready,
  output logic [1:0][AW-1:0]             wr_line,
  output logic [1:0][TSR_LINE_BITS-1:0]  wr_data,
  output logic [1:0][TSR_SPM_BE_W-1:0]   wr_be,
  // s registers (SREG0-15, EX §5.1; the core window)
  output logic [15:0][31:0]              sreg,
  // PERF (VU §5.10): VU_SPM_WAIT, VU_HAZARD (levels), VU_LINES (event)
  output logic                           perf_spm_wait,
  output logic                           perf_hazard,
  output logic                           perf_line,
  // debug state (VU §5.10)
  output logic [1:0]                     dbg_held,
  output logic [31:0]                    dbg_w0,
  output logic [2:0]                     dbg_flags,   // {waiting on a hazard, waiting for SPM, reduction draining}
  output logic                           progress     // a handshake of the unit this cycle (watchdog)
);
  import tsr_tables_pkg::*;

  localparam int unsigned LANE_X   = 2 * LANE_PIPE;   // cycles the lane pipes add to an exit / a drain read
  localparam int unsigned LM_X     = (NUM_CH > 8) ? 2 : 1;   // AMERGE scratch for up to 16 channels
  localparam int unsigned LM_LINES = 512 * LM_X;
  localparam int unsigned LMAW     = $clog2(LM_LINES);
  localparam int unsigned EXW  = TSR_VU_LW + 1;        // extent bounds (lines; checked < 2^24)

  // block interconnect (declared before the held-op logic, which uses some of it)
  tsr_vu_opk_t [1:0] opk;
  logic              slot_valid, slot_ready, slot_take, slot_rc, slot_lmw, slot_tag, slot_take_tag;
  tsr_vu_slot_t      slot;
  logic [1:0]        slot_oc, slot_cnt;
  logic [7:0]        oc_free;
  logic [3:0]        rc_free;
  logic              lm_idle;
  logic              rope_start, rope_neox, rope_ff, rope_fs, rope_atab;
  logic [31:0]       rope_m, rope_ts, rope_fsv;
  logic [9:0]        rope_na;
  logic [15:0]       rope_t;
  logic                   q_valid, q_pop;
  tsr_vu_slot_t           q_slot;
  logic [1:0][1:0]        f_valid, f_pop;
  logic [1:0][1:0][TSR_LINE_BITS-1:0] f_data;
  logic                   rd_wait, wr_wait;
  logic                     lm_re_a, lm_re_b, lm_we;
  logic [LMAW-1:0]          lm_ra_a, lm_ra_b, lm_wa;
  logic [TSR_LINE_BITS-1:0] lm_rd_a, lm_rd_b, lm_wd;
  logic [1:0]        ang_ready;
  logic              ang_rel, ang_rel_h, ang_rh;
  logic [4:0]        ang_re;
  logic [15:0][63:0] ang_cs;
  logic              tbl_we, pos_we;
  logic [4:0]        tbl_line;
  logic [15:0]       pos_idx;
  logic [TSR_LINE_BITS-1:0] pa_data;
  logic              ang_idle;
  logic [4:0]  ring_free_q, ring_rel;
  logic [3:0]  ring_take;
  logic        row_v, red_idle, sopx_v, sopx_tag, sopx_idle;
  tsr_vu_row_t row_tok;
  logic [3:0]  dr_addr;
  logic [15:0][31:0] dr_data;
  logic [3:0][31:0] dr_l1;
  logic dr_sel, dr_empty, dr_mx;
  localparam bit LOCAL_L1 = (FRONT_LANES == 8);
  logic        r_v, r_tag;
  logic [TSR_VU_LW-1:0] r_line;
  logic [TSR_LINE_BITS-1:0] r_data;
  logic [TSR_SPM_BE_W-1:0]  r_be;
  logic [15:0]       l_v, l_neg, l_lv;
  tsr_vu_uop_e       l_uop;
  logic [15:0][31:0] l_a, l_b;
  logic [(FRONT_LANES == 8 ? 1 : 16)-1:0][31:0] l_c1, l_c2;
  tsr_vu_bank_coeff_t l_coeff;
  tsr_vu_bank_angle_t [1:0] bank_angle;
  logic              l_fresh;
  logic [3:0]        l_pa;
  logic              d_v, d_back, inj_busy;
  tsr_vu_desc_t      d_desc;
  logic              topk_v;
  tsr_vu_slot_t      topk_slot;
  logic [15:0][31:0] lane_y;
  logic [1:0] ev_w;
  logic       ev_line, wr_idle;
  logic        s_we, s2_we, s_wtag, s2_wtag;
  logic [3:0]  s_wi, s2_wi;
  logic [31:0] s_wd, s2_wd;
  logic [1:0]  s_ev, s2_ev;

  if (SPM_RD_LAT < 1 || NUM_CH < 1 || NUM_CH > 16) begin : g_chk
    $fatal(1, "tsr_vu: SPM_RD_LAT >= 1, NUM_CH in [1, 16]");
  end

  // -------------------------------------------------------------------------------------------
  // Held ops
  // -------------------------------------------------------------------------------------------
  typedef struct packed {
    logic [1:0][EXW-1:0] wlo, whi;      // write extents ([lo, hi), empty when lo = hi)
    logic [3:0][EXW-1:0] rlo, rhi;      // read extents
    logic [15:0]         sr, sw;        // s registers read / written
    logic                sopx;          // SOP SQRT / RSQRT / DIV
    logic                rope;
  } hz_t;

  logic [1:0]       ov_q, os_q, og_q;    // [tag] held, started, generated
  logic [1:0]       odec_q;              // extents decoded
  tsr_insn_t [1:0]  ow_q;
  hz_t [1:0]        ohz_q;
  logic [1:0][15:0] ocnt_q;              // writes of the op still to complete
  logic             wtag_q;              // tag of the next accepted op
  logic             old_q;               // tag of the oldest held op
  logic             ct_v_q, ct_tag_q;    // the previous cycle's slot load: its write count, tag
  logic [1:0]       ct_cnt_q;

  tsr_insn_t incoming;
  tsr_vu_local_req_t [1:0] local_req, bank_req;
  tsr_vu_local_rsp_t [1:0] bank_rsp;
  tsr_vu_local_rsp_t [1:0] local_rsp;
  logic [1:0] generic_rd_valid;
  logic [1:0][AW-1:0] generic_rd_line;
  logic generic_done, local_busy, local_done;
  assign incoming = tsr_insn_t'(cmd);
  wire unused_incoming = &{1'b0, incoming[1], incoming[2]};
  wire local_cmd = LOCAL_INTERMEDIATES && incoming[0][7:0] == TSR_OP_ASMAX && incoming[7][1:0] != 0;
  logic scale_locked;
  if (LOCAL_SCALE) begin : g_scale_lock
    logic held_q;
    assign scale_locked = held_q;
    // A request reserves both banks before MVM issue. Earlier VU work retires
    // first; newly offered VU commands wait while the reservation is pending.
    assign scale_grant = scale_request && !held_q && !local_busy && ov_q == 0;
    always_ff @(posedge clk) begin
      if (!rst_n) held_q <= 1'b0;
      else if (scale_grant) held_q <= 1'b1;
      else if (scale_done) held_q <= 1'b0;
    end
  end else begin : g_no_scale_lock
    assign scale_locked = 1'b0;
    assign scale_grant = 1'b0;
    wire unused_scale_lock = &{1'b0, scale_done};
  end
  assign bank_req = scale_locked ? scale_req : local_req;
  assign scale_rsp = scale_locked ? bank_rsp : '0;
  assign local_rsp = scale_locked ? '0 : bank_rsp;
  assign cmd_ready = !local_busy && !scale_locked && !(LOCAL_SCALE && scale_request) &&
                     (local_cmd ? ov_q == 0 : !(ov_q[0] && ov_q[1]));
  wire acc = cmd_valid && cmd_ready && !local_cmd;
  assign done = generic_done || local_done;

  if (LOCAL_INTERMEDIATES) begin : g_local_att
    if (NUM_CH != 4 || FRONT_LANES != 8 || EXP_LANES != 8 || BACK_LANES != 4) begin : g_chk
      $fatal(1, "tsr_vu: local attention needs K4 VU 8/8/4");
    end
    typedef enum logic [2:0] {IDLE, REQUEST, RESPONSE, MAXIMA, START, DRAIN} state_t;
    state_t state_q;
    logic [1:0] finished_q, pair_done;
    logic [7:0] row_q;
    logic [AW-1:0] mk_addr_q;
    logic [TSR_LINE_BITS-1:0] mk_line_q;
    tsr_local_asmax_cmd_t local_asmax;
    logic [1:0] slot_q, first_owner_q;
    logic [7:0] groups_q, rows_q;
    logic [31:0] scale_q;
    wire bank = mk_addr_q[0];

    assign local_busy = state_q != IDLE;
    always_comb begin
      local_asmax = '{start: state_q == START, slot: slot_q, first_owner: first_owner_q,
                      groups: groups_q, max_valid: state_q == MAXIMA,
                      row: state_q == START ? rows_q : row_q,
                      value: state_q == START ? scale_q : mk_line_q[32*row_q[3:0] +: 32]};
      rd_req_valid = generic_rd_valid;
      rd_req_line = generic_rd_line;
      if (local_busy) begin
        rd_req_valid = '0;
        rd_req_valid[bank] = state_q == REQUEST;
        rd_req_line = {2{mk_addr_q}};
      end
    end
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        state_q <= IDLE;
        local_done <= 1'b0;
      end else begin
        local_done <= 1'b0;
        case (state_q)
          IDLE: if (cmd_valid && cmd_ready && local_cmd) begin
            slot_q <= incoming[7][1:0];
            first_owner_q <= incoming[7][3:2];
            groups_q <= incoming[4][7:0];
            rows_q <= incoming[5][7:0];
            scale_q <= incoming[6];
            mk_addr_q <= AW'(incoming[3]);
            row_q <= '0;
            state_q <= REQUEST;
          end
          REQUEST: if (rd_req_ready[bank]) state_q <= RESPONSE;
          RESPONSE: if (rd_rsp_valid[bank]) begin
            mk_line_q <= rd_rsp_data[bank];
            state_q <= MAXIMA;
          end
          MAXIMA: begin
            row_q <= row_q + 8'd1;
            if (row_q + 8'd1 == rows_q) state_q <= START;
            else if (row_q[3:0] == 4'hf) begin
              mk_addr_q <= mk_addr_q + 1'b1;
              state_q <= REQUEST;
            end
          end
          START: begin
            finished_q <= '0;
            state_q <= DRAIN;
          end
          DRAIN: begin
            finished_q <= finished_q | pair_done;
            if (&(finished_q | pair_done)) begin
              local_done <= 1'b1;
              state_q <= IDLE;
            end
          end
          default: state_q <= IDLE;
        endcase
      end
    end
    for (genvar pair = 0; pair < 2; pair++) begin : g_pair
      (* keep_hierarchy = "yes" *) tsr_local_asmax #(.PAIR(pair)) u_local_asmax (
        .clk, .rst_n, .cmd(local_asmax), .done(pair_done[pair]),
        .mem_req({local_score_req[pair+2], local_score_req[pair]}),
        .mem_data({local_score_data[pair+2], local_score_data[pair]}),
        .lane_req(local_req[pair]), .lane_rsp(local_rsp[pair])
      );
    end
  end else begin : g_no_local_att
    assign local_busy = 1'b0;
    assign local_done = 1'b0;
    assign local_req = '0;
    assign local_score_req = '0;
    assign rd_req_valid = generic_rd_valid;
    assign rd_req_line = generic_rd_line;
    wire unused_local = &{1'b0, local_score_data, local_rsp, incoming[6:3]};
  end

  // extent decode (4 stages after accept: the words are registered first, dc0)
  logic       dc0_v, dc1_v, dc2_v;
  logic       dc0_t, dc1_t, dc2_t;
  tsr_insn_t  dc0_w;
  logic [5:0][EXW-1:0] dc1_lo, dc1_len;
  logic [5:0][15:0]    dc1_nm1;
  logic [5:0][23:0]    dc1_s;
  logic [5:0]          dc1_nz;
  logic [5:0][EXW+16-1:0] dc2_p;
  logic [5:0][EXW-1:0]    dc2_lo, dc2_len;
  logic [5:0]             dc2_nz;
  logic [15:0]  dc1_sr, dc1_sw, dc2_sr, dc2_sw;
  logic         dc1_sopx, dc1_rope, dc2_sopx, dc2_rope;

  function automatic logic [EXW-1:0] c16(input logic [31:0] x);   // ceil(x / 16)
    return EXW'((33'(x) + 33'd15) >> 4);
  endfunction
  function automatic logic [EXW-1:0] c32(input logic [31:0] x);   // ceil(x / 32)
    return EXW'((33'(x) + 33'd31) >> 5);
  endfunction

  // extents of the words w: slots 0, 1 write, 2..5 read (VU §5.9 / AT §5.8 operands)
  always_ff @(posedge clk) begin
    tsr_insn_t   w;
    logic [7:0]  op;
    logic [3:0]  sub;
    logic [15:0] x16, rows, tp;
    logic [23:0] m, lm;
    logic [1:0]  rd2;
    logic [23:0] H, dsp, ssp;
    w   = dc0_w;
    op  = w[0][7:0];
    sub = w[0][11:8];
    x16 = w[0][31:16];
    rows = w[6][15:0];
    m   = w[5][23:0];
    lm  = 24'(c16(32'(m)));
    rd2 = 2'(tsr_vop_func_reads(x16[7:0]));
    dc0_w <= tsr_insn_t'(cmd);                          // every cycle (no enable on the 256 bits)
    if (acc) dc0_t <= wtag_q;
    if (dc0_v) begin
      dc1_t <= dc0_t;
      for (int k = 0; k < 6; k++) begin
        dc1_lo[k]  <= '0;
        dc1_len[k] <= '0;
        dc1_nm1[k] <= '0;
        dc1_s[k]   <= '0;
        dc1_nz[k]  <= 1'b0;
      end
      dc1_sr   <= '0;
      dc1_sw   <= '0;
      dc1_sopx <= 1'b0;
      dc1_rope <= 1'b0;
      case (op)
        tsr_tables_pkg::TSR_OP_VOP: begin
          // E(rows, stride, L(m)) for d (W), a, b (R) when read; c: ceil(rows/16)
          dc1_lo[0] <= EXW'(w[1][23:0]); dc1_nm1[0] <= rows - 16'd1; dc1_s[0] <= 24'(w[6][31:16]);
          dc1_len[0] <= EXW'(lm); dc1_nz[0] <= rows != 16'd0 && m != 24'd0;
          dc1_lo[2] <= EXW'(w[2][23:0]); dc1_nm1[2] <= rows - 16'd1; dc1_s[2] <= 24'(w[7][15:0]);
          dc1_len[2] <= EXW'(lm); dc1_nz[2] <= rd2[0] && rows != 16'd0 && m != 24'd0;
          dc1_lo[3] <= EXW'(w[3][23:0]); dc1_nm1[3] <= rows - 16'd1; dc1_s[3] <= 24'(w[7][31:16]);
          dc1_len[3] <= EXW'(lm); dc1_nz[3] <= rd2[1] && rows != 16'd0 && m != 24'd0;
          dc1_lo[4] <= EXW'(w[4][23:0]); dc1_len[4] <= c16(32'(rows));
          dc1_nz[4] <= (sub == TSR_VOP_CMODE_VEC) && rows != 16'd0 && m != 24'd0;
          if (sub == TSR_VOP_CMODE_SREG && rows != 16'd0 && m != 24'd0) dc1_sr[x16[11:8]] <= 1'b1;
        end
        tsr_tables_pkg::TSR_OP_RED: begin
          dc1_lo[0] <= EXW'(w[1][23:0]); dc1_len[0] <= c16(32'(rows));
          dc1_nz[0] <= !w[0][12] && rows != 16'd0 && m != 24'd0;
          dc1_lo[2] <= EXW'(w[2][23:0]); dc1_nm1[2] <= rows - 16'd1; dc1_s[2] <= 24'(w[7][15:0]);
          dc1_len[2] <= EXW'(lm); dc1_nz[2] <= rows != 16'd0 && m != 24'd0;
          if (w[0][12]) dc1_sw[x16[3:0]] <= 1'b1;
          if (x16[5:4] == TSR_RED_POST_RMS) begin
            dc1_sr[x16[11:8]]  <= 1'b1;
            dc1_sr[x16[15:12]] <= 1'b1;
          end
        end
        tsr_tables_pkg::TSR_OP_ROPE: begin
          H   = 24'((32'(w[6][15:0]) * 32'(w[5][15:0])) >> 4);
          dsp = (w[7][31:16] != 16'd0) ? 24'(w[7][31:16]) : H;
          ssp = (w[7][15:0] != 16'd0) ? 24'(w[7][15:0]) : H;
          tp  = (w[6][15:0] != 16'd0) ? w[6][31:16] : 16'd0;
          dc1_rope <= 1'b1;
          dc1_lo[0] <= EXW'(w[1][23:0]); dc1_nm1[0] <= tp - 16'd1; dc1_s[0] <= 24'(dsp);
          dc1_len[0] <= EXW'(H); dc1_nz[0] <= tp != 16'd0;
          dc1_lo[2] <= EXW'(w[2][23:0]); dc1_nm1[2] <= tp - 16'd1; dc1_s[2] <= 24'(ssp);
          dc1_len[2] <= EXW'(H); dc1_nz[2] <= tp != 16'd0;
          dc1_lo[3] <= EXW'(w[3][23:0]); dc1_len[3] <= c16(32'(tp)); dc1_nz[3] <= tp != 16'd0;
          dc1_lo[4] <= EXW'(w[4][23:0]); dc1_len[4] <= c32(32'(w[5][31:16]));
          dc1_nz[4] <= tp != 16'd0 && (w[0][12] || w[0][14]);
          if (tp != 16'd0) begin
            dc1_sr[x16[3:0]] <= 1'b1;
            if (!w[0][14]) dc1_sr[x16[7:4]] <= 1'b1;
            if (w[0][13]) dc1_sr[x16[11:8]] <= 1'b1;
          end
        end
        tsr_tables_pkg::TSR_OP_SOP: begin
          logic [2:0] u;
          u = 3'(tsr_sop_uses(sub));                      // SD, SA, SB
          if (u[0]) dc1_sw[w[1][3:0]] <= 1'b1;
          if (u[1]) dc1_sr[w[1][7:4]] <= 1'b1;
          if (u[2]) dc1_sr[w[1][11:8]] <= 1'b1;
          dc1_sopx <= (sub == TSR_SOP_SUB_SQRT || sub == TSR_SOP_SUB_RSQRT || sub == TSR_SOP_SUB_DIV);
          if (sub == TSR_SOP_SUB_STS) begin
            dc1_lo[0] <= EXW'(w[3][27:4]); dc1_len[0] <= EXW'(1); dc1_nz[0] <= 1'b1;
          end
          if (sub == TSR_SOP_SUB_LDS) begin
            dc1_lo[2] <= EXW'(w[3][27:4]); dc1_len[2] <= EXW'(1); dc1_nz[2] <= 1'b1;
          end
        end
        TSR_OP_ASMAX, TSR_OP_AMASK: begin
          logic [15:0] st;
          st = (w[4][31:16] != 16'd0) ? w[4][31:16] : w[4][15:0];
          dc1_lo[0] <= EXW'(w[1]); dc1_nm1[0] <= w[5][15:0] - 16'd1; dc1_s[0] <= 24'(st);
          dc1_len[0] <= EXW'(w[4][15:0]); dc1_nz[0] <= 1'b1;
          dc1_lo[2] <= EXW'(w[2]); dc1_nm1[2] <= w[5][15:0] - 16'd1; dc1_s[2] <= 24'(st);
          dc1_len[2] <= EXW'(w[4][15:0]); dc1_nz[2] <= 1'b1;
          if (op == TSR_OP_ASMAX) begin
            dc1_lo[3] <= EXW'(w[3]); dc1_len[3] <= c16(32'(w[5][15:0])); dc1_nz[3] <= 1'b1;
          end else begin
            // mask rows: (ceil(R/rpq) - 1)·mstride + n_lines (over-approximated by R - 1 rows when rpq
            // is 1; the hazard check may only be conservative)
            dc1_lo[3] <= EXW'(w[3]); dc1_nm1[3] <= w[5][15:0] - 16'd1; dc1_s[3] <= 24'(w[6]);
            dc1_len[3] <= EXW'(w[4][15:0]); dc1_nz[3] <= 1'b1;
            dc1_lo[1] <= EXW'(w[7]); dc1_len[1] <= c16(32'(w[5][15:0])); dc1_nz[1] <= 1'b1;
          end
        end
        TSR_OP_ANORM: begin
          dc1_lo[0] <= EXW'(w[1]); dc1_len[0] <= EXW'(32'(w[4][15:0]) << (2 + w[4][17:16])); dc1_nz[0] <= 1'b1;
          dc1_lo[2] <= EXW'(w[2]); dc1_len[2] <= EXW'(32'(w[4][15:0]) << (2 + w[4][17:16])); dc1_nz[2] <= 1'b1;
          dc1_lo[3] <= EXW'(w[3]); dc1_len[3] <= c16(32'(w[4][15:0])); dc1_nz[3] <= 1'b1;
        end
        tsr_tables_pkg::TSR_OP_TOPK: begin
          // d E(rows, d_stride', ceil(k/8)) W; a E(rows, a_stride, L(m)) R (VU §5.9)
          logic [15:0] kl;
          kl = (16'(x16[4:0]) + 16'd7) >> 3;
          dc1_lo[0] <= EXW'(w[1][23:0]); dc1_nm1[0] <= rows - 16'd1;
          dc1_s[0]  <= 24'((w[6][31:16] != 16'd0) ? w[6][31:16] : kl);
          dc1_len[0] <= EXW'(kl); dc1_nz[0] <= rows != 16'd0 && m != 24'd0;
          dc1_lo[2] <= EXW'(w[2][23:0]); dc1_nm1[2] <= rows - 16'd1; dc1_s[2] <= 24'(w[7][15:0]);
          dc1_len[2] <= EXW'(lm); dc1_nz[2] <= rows != 16'd0 && m != 24'd0;
        end
        tsr_tables_pkg::TSR_OP_DQROW: begin
          // d E(rows, d_stride', L(K)) W; src E(rows, src_stride', ROW_BEATS(fmt, K)) R (VU §5.9)
          logic [15:0] lk, rb;
          lk = 16'(w[5] >> 4);
          rb = 16'(tsr_row_beats(sub[0], w[5]));
          dc1_lo[0] <= EXW'(w[1][23:0]); dc1_nm1[0] <= rows - 16'd1;
          dc1_s[0]  <= 24'((w[7][31:16] != 16'd0) ? w[7][31:16] : lk);
          dc1_len[0] <= EXW'(lk); dc1_nz[0] <= rows != 16'd0;
          dc1_lo[2] <= EXW'(w[2][23:0]); dc1_nm1[2] <= rows - 16'd1;
          dc1_s[2]  <= 24'((w[7][15:0] != 16'd0) ? w[7][15:0] : rb);
          dc1_len[2] <= EXW'(rb); dc1_nz[2] <= rows != 16'd0;
        end
        TSR_OP_AMERGE: begin
          logic [31:0] rl, cs;
          rl = 32'(w[5][15:0]) << (2 + w[5][17:16]);
          cs = (w[7] != 32'd0) ? w[7] : rl;
          dc1_lo[0] <= EXW'(w[1]); dc1_len[0] <= EXW'(rl); dc1_nz[0] <= 1'b1;               // Orun
          dc1_lo[1] <= EXW'(w[4]); dc1_len[1] <= EXW'(2 * c16(32'(w[5][15:0]))); dc1_nz[1] <= 1'b1;
          dc1_lo[2] <= EXW'(w[1]); dc1_len[2] <= EXW'(rl); dc1_nz[2] <= 1'b1;
          dc1_lo[3] <= EXW'(w[2]);
          dc1_len[3] <= EXW'(32'(NUM_CH) * cs + 32'(NUM_CH) * 32'(c16(32'(w[5][15:0])))); dc1_nz[3] <= 1'b1;
          dc1_lo[4] <= EXW'(w[3]); dc1_len[4] <= c16(32'(w[5][15:0])); dc1_nz[4] <= 1'b1;
          dc1_lo[5] <= EXW'(w[4]); dc1_len[5] <= EXW'(2 * c16(32'(w[5][15:0]))); dc1_nz[5] <= 1'b1;
        end
        default: ;
      endcase
    end
    // stage 2: products
    dc2_t   <= dc1_t;
    dc2_lo  <= dc1_lo;
    dc2_len <= dc1_len;
    dc2_nz  <= dc1_nz;
    dc2_sr  <= dc1_sr;
    dc2_sw  <= dc1_sw;
    dc2_sopx <= dc1_sopx;
    dc2_rope <= dc1_rope;
    for (int k = 0; k < 6; k++) dc2_p[k] <= (EXW+16)'(dc1_nm1[k]) * (EXW+16)'(dc1_s[k]);
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      dc0_v <= 1'b0;
      dc1_v <= 1'b0;
      dc2_v <= 1'b0;
    end else begin
      dc0_v <= acc;
      dc1_v <= dc0_v;
      dc2_v <= dc1_v;
    end
  end

  // stage 3: [lo, hi) into the entry
  hz_t dc3;
  always_comb begin
    dc3 = '0;
    for (int k = 0; k < 6; k++) begin
      logic [EXW-1:0] lo, hi;
      lo = dc2_lo[k];
      hi = dc2_nz[k] && dc2_len[k] != '0 ? dc2_lo[k] + EXW'(dc2_p[k]) + dc2_len[k] : dc2_lo[k];
      if (k < 2) begin
        dc3.wlo[k] = lo;
        dc3.whi[k] = hi;
      end else begin
        dc3.rlo[k-2] = lo;
        dc3.rhi[k-2] = hi;
      end
    end
    dc3.sr   = dc2_sr;
    dc3.sw   = dc2_sw;
    dc3.sopx = dc2_sopx;
    dc3.rope = dc2_rope;
  end

  // -------------------------------------------------------------------------------------------
  // Start: the oldest held op not started, when the generator is idle and no hazard holds it
  // -------------------------------------------------------------------------------------------
  function automatic logic ovl(input logic [EXW-1:0] alo, ahi, blo, bhi);
    return (alo != ahi) && (blo != bhi) && (alo < bhi) && (blo < ahi);
  endfunction
  // The older op's read extents and s reads take no part (write-after-read needs no check, see the
  // header), hence the targeted waiver.
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic logic hazard(input hz_t a, input hz_t b);   // a older, b newer
    logic h;
    h = ((a.sw & (b.sr | b.sw)) != 16'd0) || a.sopx || b.sopx || (a.rope && b.rope);
    for (int i = 0; i < 2; i++) begin
      for (int j = 0; j < 4; j++) h |= ovl(a.wlo[i], a.whi[i], b.rlo[j], b.rhi[j]);
      for (int j = 0; j < 2; j++) h |= ovl(a.wlo[i], a.whi[i], b.wlo[j], b.whi[j]);
    end
    return h;
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  logic gen_busy, gen_done, gen_done_tag;
  // candidate: the oldest not started
  wire  c_old  = ov_q[old_q] && !os_q[old_q];
  wire  c_new  = !c_old && ov_q[!old_q] && !os_q[!old_q];
  wire  c_tag  = c_old ? old_q : !old_q;
  // the newer candidate is blocked: both orders registered, selected by old_q
  logic hz01_q, hz10_q;
  always_ff @(posedge clk) begin
    hz01_q <= hazard(ohz_q[0], ohz_q[1]);
    hz10_q <= hazard(ohz_q[1], ohz_q[0]);
  end
  wire  hz_q = old_q ? hz10_q : hz01_q;
  logic hz_ok_q;                                        // hz_q is fresh for the current pair
  logic start_q;
  logic c_tag_q;                                        // the started op's tag (generator input)
  wire  can_start = !gen_busy && !start_q && (c_old || (c_new && hz_ok_q && !hz_q)) && odec_q[c_tag];
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ov_q    <= '0;
      os_q    <= '0;
      og_q    <= '0;
      odec_q  <= '0;
      wtag_q  <= 1'b0;
      old_q   <= 1'b0;
      generic_done    <= 1'b0;
      start_q <= 1'b0;
      hz_ok_q <= 1'b0;
      ct_v_q  <= 1'b0;
      for (int t = 0; t < 2; t++) ocnt_q[t] <= '0;
    end else begin
      generic_done    <= 1'b0;
      start_q <= can_start;
      hz_ok_q <= odec_q[0] && odec_q[1] && !(acc || dc2_v);
      if (acc) begin
        ov_q[wtag_q]   <= 1'b1;
        os_q[wtag_q]   <= 1'b0;
        og_q[wtag_q]   <= 1'b0;
        odec_q[wtag_q] <= 1'b0;
        wtag_q         <= !wtag_q;
        hz_ok_q        <= 1'b0;
      end
      if (dc0_v) ow_q[dc0_t] <= dc0_w;                    // the held op's words (from the decode register)
      if (dc2_v) begin
        odec_q[dc2_t] <= 1'b1;
        ohz_q[dc2_t]  <= dc3;
      end
      if (can_start) begin
        os_q[c_tag] <= 1'b1;
      end
      if (gen_done) og_q[gen_done_tag] <= 1'b1;
      // a slot's writes are counted one cycle after its load (ct_*: off the generator's load path)
      ct_v_q   <= slot_take && slot_cnt != 2'd0;
      ct_tag_q <= slot_tag;
      ct_cnt_q <= slot_cnt;
      for (int t = 0; t < 2; t++) begin
        ocnt_q[t] <= ocnt_q[t] + ((ct_v_q && ct_tag_q == 1'(t)) ? 16'(ct_cnt_q) : 16'd0)
                     - 16'(ev_w[t]) - 16'(s_ev[t]) - 16'(s2_ev[t]);
      end
      // done: the oldest op, generated, all writes complete
      // (og_q: no slot of the op is still to come; ct_v_q: the last one's count is being added)
      if (ov_q[old_q] && og_q[old_q] && ocnt_q[old_q] == 16'd0 && !(ct_v_q && ct_tag_q == old_q) &&
          !generic_done) begin
        generic_done        <= 1'b1;
        ov_q[old_q] <= 1'b0;
        os_q[old_q] <= 1'b0;
        og_q[old_q] <= 1'b0;
        old_q       <= !old_q;
        hz_ok_q     <= 1'b0;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // s registers: +0.0 after a hardware reset; written by RED (TO_S) and SOP
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sreg <= '0;
    end else begin
      if (s_we)  sreg[s_wi]  <= s_wd;
      if (s2_we) sreg[s2_wi] <= s2_wd;
    end
  end
  always_comb begin
    s_ev  = '0;
    s2_ev = '0;
    if (s_we)  s_ev[s_wtag]   = 1'b1;
    if (s2_we) s2_ev[s2_wtag] = 1'b1;
  end

  // -------------------------------------------------------------------------------------------
  // Blocks
  // -------------------------------------------------------------------------------------------
  assign slot_tag = slot_take_tag;                     // the slot being loaded (its writes, its LM line)

  tsr_vu_gen #(.FRONT_LANES(FRONT_LANES), .EXP_LANES(EXP_LANES), .BACK_LANES(BACK_LANES), .NUM_CH(NUM_CH), .LM_X(LM_X), .HAS_TOPK(HAS_TOPK),
               .HAS_DQROW(HAS_DQROW)) u_gen (
    .clk, .rst_n, .start(start_q), .start_tag(c_tag_q), .start_w(ow_q[c_tag_q]), .sreg,
    .busy(gen_busy), .gen_done, .gen_done_tag, .opk,
    .rope_start, .rope_neox, .rope_ff, .rope_fs, .rope_atab, .rope_m, .rope_ts, .rope_fsv, .rope_na, .rope_t,
    .slot_valid, .slot_ready, .slot, .slot_take, .slot_take_tag, .slot_oc, .slot_rc, .slot_cnt, .slot_lmw,
    .oc_free, .rc_free, .lm_idle
  );
  always_ff @(posedge clk) if (can_start) c_tag_q <= c_tag;

  tsr_vu_rd #(.SPM_BYTES(SPM_BYTES), .RD_LAT(SPM_RD_LAT)) u_rd (
    .clk, .rst_n, .s_valid(slot_valid), .s_ready(slot_ready), .s_slot(slot),
    .rd_req_valid(generic_rd_valid), .rd_req_ready(rd_req_ready & {2{!local_busy}}),
    .rd_req_line(generic_rd_line), .rd_rsp_valid(rd_rsp_valid & {2{!local_busy}}), .rd_rsp_data,
    .q_valid, .q_slot, .q_pop, .f_valid, .f_data, .f_pop, .rd_wait
  );

  // local memory: two copies (read ports A and B), written together
  tsr_sram_1r1w #(.DEPTH(LM_LINES), .WIDTH(TSR_LINE_BITS), .RD_LAT(1)) u_lma (
    .clk, .we(lm_we), .waddr(lm_wa), .wdata(lm_wd), .wbe('1), .re(lm_re_a), .raddr(lm_ra_a), .rdata(lm_rd_a)
  );
  tsr_sram_1r1w #(.DEPTH(LM_LINES), .WIDTH(TSR_LINE_BITS), .RD_LAT(1)) u_lmb (
    .clk, .we(lm_we), .waddr(lm_wa), .wdata(lm_wd), .wbe('1), .re(lm_re_b), .raddr(lm_ra_b), .rdata(lm_rd_b)
  );
  logic [9:0] lm_pend_q;                                 // LM writes generated, not yet done
  logic       lm_inc_q;                                  // one generated in the previous cycle
  logic       lm_z_q;                                    // lm_pend_q == 0
  always_ff @(posedge clk) begin
    logic [9:0] nx;
    nx = lm_pend_q + 10'(lm_inc_q) - 10'(lm_we);
    if (!rst_n) begin
      lm_pend_q <= '0;
      lm_inc_q  <= 1'b0;
      lm_z_q    <= 1'b1;
    end else begin
      lm_inc_q  <= slot_take && slot_lmw;
      lm_pend_q <= nx;
      lm_z_q    <= (nx == '0);
    end
  end
  assign lm_idle = lm_z_q && !lm_inc_q;

  // angle unit
  logic angle_release, angle_release_half;
  if (FRONT_LANES == 8 && LANE_PIPE > 0) begin : g_angle_release
    // The last read has moved through the input pipe into the lane bank.
    logic [LANE_PIPE-1:0][1:0] rel_q;
    always_ff @(posedge clk) begin
      if (!rst_n) rel_q <= '0;
      else begin
        rel_q[0] <= {ang_rel, ang_rel_h};
        for (int i = 1; i < LANE_PIPE; i++) rel_q[i] <= rel_q[i-1];
      end
    end
    assign {angle_release, angle_release_half} = rel_q[LANE_PIPE-1];
  end else begin : g_angle_release_direct
    assign {angle_release, angle_release_half} = {ang_rel, ang_rel_h};
  end
  tsr_vu_angle #(.LOCAL_ANGLE(FRONT_LANES == 8)) u_ang (
    .clk, .rst_n, .cfg_start(rope_start), .cfg_neox(rope_neox), .cfg_ff(rope_ff), .cfg_fs(rope_fs), .cfg_atab(rope_atab),
    .cfg_m(rope_m), .cfg_ts(rope_ts), .cfg_fsv(rope_fsv), .cfg_na(rope_na), .cfg_t(rope_t),
    .tbl_we, .tbl_line, .pos_we, .pos_idx, .pa_data, .ang_ready, .ang_rel(angle_release), .ang_rel_h(angle_release_half),
    .rd_h(ang_rh), .rd_e(ang_re), .rd_cs(ang_cs), .bank_angle, .idle(ang_idle)
  );

  // reduction unit
  always_ff @(posedge clk) begin
    if (!rst_n) ring_free_q <= 5'd16;
    else        ring_free_q <= ring_free_q - 5'(ring_take) + ring_rel;
  end
  tsr_vu_red #(.DRL(LANE_X), .LOCAL_L1(LOCAL_L1)) u_red (
    .clk, .rst_n, .opk, .row_v, .row_tok, .dr_addr, .dr_sel, .dr_empty, .dr_mx, .dr_data, .dr_l1, .ring_rel, .sopx_v, .sopx_tag, .sopx_idle,
    .r_v, .r_tag, .r_line, .r_data, .r_be, .s_we(s2_we), .s_wi(s2_wi), .s_wd(s2_wd), .s_wtag(s2_wtag),
    .idle(red_idle)
  );

  // injector
  logic l_ctx;
  tsr_vu_inj #(.FRONT_LANES(FRONT_LANES), .LMAW(LMAW), .HAS_DQROW(HAS_DQROW), .LANE_X(LANE_X)) u_inj (
    .clk, .rst_n, .q_valid, .q_slot, .q_pop, .f_valid, .f_data, .f_pop, .opk,
    .lm_re_a, .lm_ra_a, .lm_rd_a, .lm_re_b, .lm_ra_b, .lm_rd_b,
    .ang_ready, .ang_rel, .ang_rel_h, .ang_rh, .ang_re, .ang_cs, .tbl_we, .tbl_line, .pos_we, .pos_idx, .pa_data,
    .ring_free(ring_free_q), .ring_take, .row_v, .row_tok, .red_idle, .sopx_v, .sopx_tag, .sopx_idle,
    .topk_ready, .topk_v, .topk_slot,
    .l_v, .l_uop, .l_a, .l_b, .l_c1, .l_c2, .l_coeff, .l_neg, .l_lv, .l_fresh, .l_pa, .l_ctx, .d_v, .d_back, .d_desc,
    .busy(inj_busy)
  );
  // TOPK (HAS_TOPK)
  logic                     topk_ready, tk_v, tk_ready, tk_tag, tk_idle;
  logic [TSR_VU_LW-1:0]     tk_line;
  logic [TSR_LINE_BITS-1:0] tk_data;
  if (HAS_TOPK) begin : g_topk
    tsr_vu_topk u_topk (
      .clk, .rst_n, .in_v(topk_v), .in_ready(topk_ready), .in_line(pa_data), .in_slot(topk_slot),
      .in_k(opk[topk_slot.tag].k), .o_v(tk_v), .o_ready(tk_ready), .o_tag(tk_tag), .o_line(tk_line),
      .o_data(tk_data), .idle(tk_idle)
    );
  end else begin : g_notopk
    assign topk_ready = 1'b1;
    assign tk_v       = 1'b0;
    assign tk_tag     = 1'b0;
    assign tk_line    = '0;
    assign tk_data    = '0;
    assign tk_idle    = 1'b1;
    wire unused_topk = &{1'b0, topk_v, topk_slot, tk_ready};
  end

  // lanes (behind LANE_PIPE register stages on each side)
  tsr_vu_lanes #(.FRONT_LANES(FRONT_LANES), .EXP_LANES(EXP_LANES), .BACK_LANES(BACK_LANES), .PIPE(LANE_PIPE), .HAS_LOCAL(LOCAL_INTERMEDIATES || LOCAL_SCALE)) u_lanes (
    .clk, .rst_n, .l_v, .l_uop, .l_a, .l_b, .l_c1, .l_c2, .l_coeff, .l_neg, .l_lv, .l_fresh, .l_pa, .l_ctx, .bank_angle,
    .back_valid(d_v && d_back), .dr_addr, .dr_sel, .dr_empty, .dr_mx, .dr_data, .dr_l1, .lane_y, .local_req(bank_req), .local_rsp(bank_rsp)
  );

  // write side
  tsr_vu_wr #(.SPM_BYTES(SPM_BYTES), .OFD(OFD), .LMAW(LMAW), .LANE_X(LANE_X)) u_wr (
    .clk, .rst_n, .d_v, .d_back, .d_desc, .lane_y,
    .oc_take(slot_take ? slot_oc : 2'd0), .oc_free, .rc_take(slot_take && slot_rc), .rc_free,
    .r_v, .r_tag, .r_line, .r_data, .r_be,
    .t_v(tk_v), .t_ready(tk_ready), .t_tag(tk_tag), .t_line(tk_line), .t_data(tk_data),
    .lm_we, .lm_wa, .lm_wd, .s_we, .s_wi, .s_wd, .s_wtag,
    .wr_valid, .wr_ready, .wr_line, .wr_data, .wr_be, .ev_w, .ev_line, .wr_wait, .idle(wr_idle)
  );

  // -------------------------------------------------------------------------------------------
  // PERF, debug, progress
  // -------------------------------------------------------------------------------------------
  // registered (one cycle late: counters, debug state and the watchdog only)
  always_ff @(posedge clk) begin
    logic hz_c, wt_c;
    hz_c = !gen_busy && c_new && odec_q[!old_q] && (!hz_ok_q || hz_q);
    wt_c = rd_wait || wr_wait;
    if (!rst_n) begin
      perf_spm_wait <= 1'b0;
      perf_hazard   <= 1'b0;
      perf_line     <= 1'b0;
      progress      <= 1'b0;
    end else begin
      perf_spm_wait <= wt_c;
      perf_hazard   <= hz_c;
      perf_line     <= ev_line;
      progress      <= done || slot_take || ev_line || (|(rd_req_valid & rd_req_ready)) || d_v || (|local_rsp[0].valid) || (|local_rsp[1].valid);
    end
    dbg_held  <= 2'(ov_q[0]) + 2'(ov_q[1]);
    dbg_w0    <= ow_q[old_q][0];
    dbg_flags <= {hz_c, wt_c, !red_idle};
  end

  wire unused_ok = &{1'b0, inj_busy, wr_idle, ang_idle, tk_idle};

`ifndef SYNTHESIS
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      for (int t = 0; t < 2; t++) begin
        assert (!(ov_q[t] == 1'b0 && ocnt_q[t] != 16'd0))
          else $error("tsr_vu: op slot %0d free with %0d writes outstanding", t, ocnt_q[t]);
      end
      assert (ring_free_q <= 5'd16) else $error("tsr_vu: partial ring overflow");
    end
  end
`endif

endmodule
