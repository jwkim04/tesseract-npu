// tsr_mvm.sv — the MVM unit (spec MVM 클러스터 마이크로아키텍처, ISA MVM; M3): dispatch from
// tsr_cmdproc, activation broadcast to the NUM_CH clusters, cluster command, result gather and
// ACC into the SPM, done, perf_mvm_busy.
//
// Owner: top agent (docs/rtl-m3.md §2 block (top); contract §3.0, §3.1, §6).
// tsr_core instantiates it when HAS_MVM = 1 (integration agent).
//
// Contract (docs/rtl-m2.md §3.0 dispatch; semantics = tsr_sim.c ex_mvm):
//   - cmd_valid/cmd_ready/cmd (tsr_mvm_cmd_t): every check done by tsr_cmdproc (K, T, SPM and
//     DRAM ranges, dst_stride resolved, rows_per_ch / row_beats / y_lines / qa_lines precomputed).
//     cmd_ready is a register, 1 only while idle. done: registered 1-cycle pulse when every
//     architectural effect is complete and visible (all y lines accepted by tsr_spm) and the
//     memory ports are quiescent (every requested weight beat received); never in the accepting
//     cycle.
//   - N = 0: no SPM access, no memory traffic, done one or more cycles after the accept.
//   - N >= 1: the clusters get one tsr_mvm_ccmd_t (broadcast pipe), then the T QA images
//     (x_line .. x_line + T*qa_lines - 1, one SPM read per cycle on client TSR_SPM_CL_MVM) as
//     tsr_mvm_bcline_t items (§3.1); results come back per cluster in row order and are written by
//     tsr_mvm_gather (§6.3). Channel c streams exactly rows_c * ROW_BEATS weight beats from w_off
//     (§4). done after the broadcast and every expected cluster's last result are complete and
//     the gather's last SPM write was accepted.
//   - SPM client 2: read port shared by the broadcast (first) and the gather's ACC reads (later;
//     they never overlap: a cluster produces results only after the broadcast completed). Reads
//     have no backpressure (R4); writes only from the gather.
//   - perf_mvm_busy: registered, 1 in every cycle from the cycle after the accepting edge up to
//     and including the cycle of done (cycles the unit holds an instruction; PERF_MVM_CYCLES).
// Implementation parameters (results identical for every value): WS_FIFO_BEATS (per-cluster
// weight FIFO), BC_PIPE (broadcast pipe stages per cluster, tsr_vpipe), RES_PIPE (result pipe
// stages per cluster, tsr_rpipe), ACT_RD_LAT, MAC_LAT, MUL_LAT, ADD_LAT, WS_RD_LAT
// (docs/rtl-m3.md §1).
//
// Interface addition (additive, top agent): parameter WS_RD_LAT (weight FIFO RAM read latency,
//   docs/rtl-m3.md §1, default TSR_DEF_WS_RD_LAT), passed to every cluster's front end.
// Interface addition (additive, M3 part C timing, docs/rtl-m3.md §14): parameter RST_LOCAL (0..2).
//   Every part (top logic, gather, cluster c with its pipes) takes rst_n through its own chain of
//   RST_LOCAL registers (Xilinx `keep`: the copies are not merged), so the reset reaches every
//   flop RST_LOCAL cycles after rst_n, from registers next to that part (the clusters sit in
//   different SLRs on the U200). tsr_core sets it from RST_PIPE and feeds rst_n that many stages
//   earlier, so the whole core sees the same reset delay. RST_LOCAL = 0: rst_n everywhere.
//
// Microarchitecture:
//
//   cmd ─► c_q ─► su (1 cycle) ─► bc_cmd ──────────────┐   one source register bc_d_q (527 b):
//                 │                                      ├─► holds the ccmd (low 137 bits) or a
//                 └─► broadcast reader: SPM reads x_line │   line; per cluster tsr_vpipe(BC_PIPE)
//                     .. + T*qa_lines - 1, one per cycle │   carries {cmd_valid, line_valid} and
//                     response -> {last, tok, dbs, addr, ┘   the shared payload (the two valids are
//                     data} (counters on the response side)   never 1 together, docs §3.1)
//                 └─► expect_ch: c * R < N, one channel per cycle (NUM_CH cycles), then the
//                     gather's start pulse (results cannot come earlier; if they did they would
//                     wait in the result pipes, res_ready = 0 while the gather is idle)
//   clusters ─► tsr_rpipe(RES_PIPE) ─► tsr_mvm_gather ─► SPM write port; ACC reads on the read port
//     (the gather's per-cluster assemblers take the rows; their jobs reach its central part
//     through JOB_PIPE = RES_PIPE slices)
//   cluster c weight stream ◄─► arbiter c: requests through tsr_rpipe(RES_PIPE) (u_wqp),
//     responses through a kept tsr_vpipe(BC_PIPE) (u_wrp; no backpressure: the streamer's
//     rd_rsp_ready is constant 1, asserted), so a cluster may sit in another SLR than its arbiter
//
//   - SPM read port: the broadcast owns it from the accept until its last response returned
//     (bc_ph_q); the gather's ACC reads are only presented after that (assertion), so the
//     request mux and the response routing are selected by bc_ph_q (a flop).
//   - done_q := 1 when the broadcast is complete and the gather's done pulse was seen; the next
//     cycle cmd_ready returns (the unit is idle from the cycle after done).
//   - Timing: every output except the SPM read request (one OR / 2:1 mux of flops) and the
//     per-cluster pipes comes from a flop; the multi-cycle arithmetic (c * R, d_lines, ceil(K/64))
//     is spread over registered steps.
//   Latency (ideal SPM, SPM_RD_LAT L): accept -> bc_cmd at the pipe input 2 cycles; first SPM
//   read request 2 cycles after the accept, first bc_line 1 + L cycles later; N = 0: done in the
//   cycle after the accept.
//   Measured (tb/mvm/test_mvm.py, real tsr_mvm_back / tsr_mvm_actbuf, ideal memory, defaults
//   4 ch, T_MAX 1): 1.044 cycles per weight beat for Q8_0 K 896 (15-beat rows, 1920 beats per
//   channel), 1.064 for Q8_0 K 4096; overhead accept -> done = T x QA_LINES broadcast + ~70
//   cycles (front-end start, back-end pipeline and tree, gather); N = 1, K = 32: 85 cycles.
//   Steady state (two-point, overhead removed): 1.000 cycles per weight beat for rows >= RB_MIN;
//   tb/mvm mvm_perf asserts the rate and the fixed overhead in every test configuration.
//   Cost (RTL structure, no synthesis run here): command 294 FF + broadcast register 527 FF +
//   ~140 FF control; per cluster BC_PIPE x 529 FF (tsr_vpipe; the ccmd is overlaid on the
//   line payload, saving 137 FF per stage and cluster) and RES_PIPE x 2 x (33 + 32 T_MAX) FF;
//   weight-stream pipes per cluster BC_PIPE x 516 FF + RES_PIPE x 2 x 38 FF; with RST_LOCAL
//   RST_LOCAL x (NUM_CH + 2) reset flops. Measured sizes: docs/rtl-m3.md §14.

module tsr_mvm
  import tsr_pkg::*;
#(
  parameter  int unsigned NUM_CH        = TSR_DEF_NUM_CH,
  parameter  int unsigned T_MAX         = TSR_DEF_T_MAX,
  parameter  int unsigned K_MAX         = TSR_DEF_K_MAX,
  parameter  int unsigned SPM_BYTES     = TSR_DEF_SPM_BYTES,
  parameter  int unsigned SPM_RD_LAT    = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned ADDR_BEAT_W   = TSR_DEF_ADDR_BEAT_W,
  parameter  int unsigned BEAT_BITS     = TSR_DEF_BEAT_BITS,
  parameter  int unsigned WS_FIFO_BEATS = TSR_DEF_WS_FIFO_BEATS,
  parameter  int unsigned BC_PIPE       = TSR_DEF_MVM_BC_PIPE,
  parameter  int unsigned RES_PIPE      = TSR_DEF_MVM_RES_PIPE,
  parameter  int unsigned ACT_RD_LAT    = TSR_DEF_ACT_RD_LAT,
  parameter  int unsigned MAC_LAT       = TSR_DEF_MAC_LAT,
  parameter  int unsigned MUL_LAT       = TSR_DEF_FMUL_LAT,
  parameter  int unsigned ADD_LAT       = TSR_DEF_FADD_LAT,
  parameter  int unsigned WS_RD_LAT     = TSR_DEF_WS_RD_LAT,
  parameter  int unsigned RST_LOCAL     = 0,      // reset register stages per cluster / top / gather (0..2)
  localparam int unsigned SPM_LINES     = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned SPM_AW        = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1
) (
  input  logic                                   clk,
  input  logic                                   rst_n,

  // ---- dispatch (tsr_cmdproc) ----
  input  logic                                   cmd_valid,
  output logic                                   cmd_ready,
  input  tsr_mvm_cmd_t                           cmd,
  output logic                                   done,

  // ---- SPM client TSR_SPM_CL_MVM ----
  output logic                                   spm_rd_req_valid,
  input  logic                                   spm_rd_req_ready,
  output logic [SPM_AW-1:0]                      spm_rd_req_line,
  input  logic                                   spm_rd_rsp_valid,
  input  logic [TSR_LINE_BITS-1:0]               spm_rd_rsp_data,
  output logic                                   spm_wr_valid,
  input  logic                                   spm_wr_ready,
  output logic [SPM_AW-1:0]                      spm_wr_line,
  output logic [TSR_LINE_BITS-1:0]               spm_wr_data,
  output logic [TSR_SPM_BE_W-1:0]                spm_wr_be,

  // ---- weight-stream read sources, one per channel (tsr_mem_arb source TSR_SRC_WS) ----
  output logic [NUM_CH-1:0]                      ws_rd_req_valid,
  input  logic [NUM_CH-1:0]                      ws_rd_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     ws_rd_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       ws_rd_req_len,
  output logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]    ws_rd_req_id,
  input  logic [NUM_CH-1:0]                      ws_rd_rsp_valid,
  output logic [NUM_CH-1:0]                      ws_rd_rsp_ready,
  input  logic [NUM_CH-1:0][BEAT_BITS-1:0]       ws_rd_rsp_data,
  input  logic [NUM_CH-1:0]                      ws_rd_rsp_last,
  input  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]    ws_rd_rsp_id,

  // ---- perf ----
  output logic                                   perf_mvm_busy
);

  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX) begin : g_chk_ch
    $fatal(1, "tsr_mvm: NUM_CH must be in [1, 16]");
  end
  if (T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH) begin : g_chk_t
    $fatal(1, "tsr_mvm: T_MAX must be in [1, 8]");
  end
  if (BEAT_BITS != 512) begin : g_chk_beat
    $fatal(1, "tsr_mvm: BEAT_BITS must be 512");
  end

  // Reset distribution (RST_LOCAL, header): rst_t_n top logic, rst_g_n gather, rst_c_n[c]
  // cluster c with its pipes.
  logic              rst_t_n, rst_g_n;
  logic [NUM_CH-1:0] rst_c_n;
  if (RST_LOCAL > 0) begin : g_rst
    (* keep = "true" *) logic [RST_LOCAL-1:0]             rt_q, rg_q;
    (* keep = "true" *) logic [RST_LOCAL-1:0][NUM_CH-1:0] rc_q;
    initial begin            // FPGA INIT / simulation start: in reset until rst_n is seen high
      rt_q = '0;
      rg_q = '0;
      rc_q = '0;
    end
    always @(posedge clk) begin   // not always_ff: the initial block writes them too (tsr_core)
      rt_q[0] <= rst_n;
      rg_q[0] <= rst_n;
      rc_q[0] <= {NUM_CH{rst_n}};
      for (int unsigned k = 1; k < RST_LOCAL; k++) begin
        rt_q[k] <= rt_q[k-1];
        rg_q[k] <= rg_q[k-1];
        rc_q[k] <= rc_q[k-1];
      end
    end
    assign rst_t_n = rt_q[RST_LOCAL-1];
    assign rst_g_n = rg_q[RST_LOCAL-1];
    assign rst_c_n = rc_q[RST_LOCAL-1];
  end else begin : g_rst0
    assign rst_t_n = rst_n;
    assign rst_g_n = rst_n;
    assign rst_c_n = {NUM_CH{rst_n}};
  end

  localparam int unsigned BCW = $bits(tsr_mvm_bcline_t);   // 527: broadcast payload
  localparam int unsigned CCW = $bits(tsr_mvm_ccmd_t);     // 137: overlaid on its low bits
  localparam int unsigned RW  = 33 + 32 * T_MAX;           // {row, last, y}
  localparam int unsigned CW  = (NUM_CH > 1) ? $clog2(NUM_CH) : 1;

  if (CCW > BCW) begin : g_chk_overlay
    $fatal(1, "tsr_mvm: tsr_mvm_ccmd_t must fit in the broadcast payload");
  end

  // -------------------------------------------------------------------------------------------
  // Command / control
  // -------------------------------------------------------------------------------------------
  typedef enum logic [1:0] { S_IDLE, S_RUN, S_DONE } st_t;

  st_t          st_q;
  logic         cmd_ready_q, done_q, busy_q;
  tsr_mvm_cmd_t c_q;
  logic         su_q;          // first cycle after an accept with N >= 1
  logic         gd_q;          // the gather's done pulse was seen

  wire fire   = cmd_valid & cmd_ready_q;
  wire n_zero = (cmd.n == 32'd0);

  // gather
  logic                                 g_start_q, g_done;
  logic [NUM_CH-1:0]                    ex_q;
  logic [CW-1:0]                        ex_c_q;
  logic                                 ex_run_q;
  logic [35:0]                          ex_acc_q;
  logic                                 g_rd_v;
  logic [SPM_AW-1:0]                    g_rd_line;

  // broadcast
  logic                                 bc_ph_q, bc_done_q;
  logic                                 rq_v_q;
  logic [SPM_AW-1:0]                    rq_line_q;
  logic [15:0]                          rq_l_q, rs_l_q;
  logic [2:0]                           rq_tok_q, rs_tok_q;
  logic                                 rs_dbs_q;
  logic [9:0]                           rs_addr_q;
  logic [15:0]                          qam1_q, qsm1_q;
  logic [2:0]                           tm1_q;
  logic [1:0]                           bc_v_q;          // {cmd, line}
  logic [BCW-1:0]                       bc_d_q;

  wire rq_fire  = rq_v_q & spm_rd_req_ready;
  wire rq_l_end = (rq_l_q == qam1_q);
  wire rq_t_end = (rq_tok_q == tm1_q);
  wire bc_rsp   = spm_rd_rsp_valid & bc_ph_q;
  wire rs_l_end = (rs_l_q == qam1_q);
  wire rs_t_end = (rs_tok_q == tm1_q);
  wire rs_q_end = (rs_l_q == qsm1_q);

  // cluster command of this MVM (from c_q, assembled in the su cycle)
  tsr_mvm_ccmd_t ccmd;
  logic [15:0]   nblk;
  always_comb begin
    nblk         = c_q.fmt ? (c_q.k >> 8) : (c_q.k >> 5);
    ccmd.fmt     = c_q.fmt;
    ccmd.k       = c_q.k;
    ccmd.n       = c_q.n;
    ccmd.rows_per_ch = c_q.rows_per_ch;
    ccmd.row_beats   = c_q.row_beats[15:0];
    ccmd.w_off   = c_q.w_off;
    ccmd.d_lines = 8'((nblk + 16'd15) >> 4);             // tsr_qa_d_lines(fmt, k) <= 128
  end

  wire run_end = (st_q == S_RUN) & bc_done_q & (gd_q | g_done);

  always_ff @(posedge clk) begin
    if (!rst_t_n) begin
      st_q        <= S_IDLE;
      cmd_ready_q <= 1'b1;
      done_q      <= 1'b0;
      busy_q      <= 1'b0;
      su_q        <= 1'b0;
      gd_q        <= 1'b0;
      g_start_q   <= 1'b0;
      ex_run_q    <= 1'b0;
      ex_c_q      <= '0;
      ex_acc_q    <= '0;
      ex_q        <= '0;
      bc_ph_q     <= 1'b0;
      bc_done_q   <= 1'b0;
      rq_v_q      <= 1'b0;
      rq_l_q      <= '0;
      rq_tok_q    <= '0;
      rs_l_q      <= '0;
      rs_tok_q    <= '0;
      rs_dbs_q    <= 1'b0;
      rs_addr_q   <= '0;
      bc_v_q      <= '0;
    end else begin
      // ---- dispatch / done ----
      done_q    <= 1'b0;
      su_q      <= fire & ~n_zero;
      g_start_q <= 1'b0;
      if (g_done) gd_q <= 1'b1;
      unique case (st_q)
        S_IDLE: if (fire) begin
          cmd_ready_q <= 1'b0;
          busy_q      <= 1'b1;
          gd_q        <= 1'b0;
          if (n_zero) begin
            done_q <= 1'b1;
            st_q   <= S_DONE;
          end else begin
            st_q      <= S_RUN;
            bc_ph_q   <= 1'b1;
            bc_done_q <= 1'b0;
            ex_run_q  <= 1'b1;
            ex_c_q    <= '0;
            ex_acc_q  <= '0;
          end
        end
        S_RUN: if (run_end) begin
          done_q <= 1'b1;
          st_q   <= S_DONE;
        end
        S_DONE: begin                           // the cycle of done
          busy_q      <= 1'b0;
          cmd_ready_q <= 1'b1;
          st_q        <= S_IDLE;
        end
        default: st_q <= S_IDLE;
      endcase

      // ---- expect_ch: channel c has rows iff c * R < N (one channel per cycle) ----
      if (ex_run_q) begin
        ex_q[ex_c_q] <= (ex_acc_q < 36'(c_q.n));
        ex_acc_q     <= ex_acc_q + 36'(c_q.rows_per_ch);
        ex_c_q       <= ex_c_q + 1'b1;
        if (ex_c_q == CW'(NUM_CH - 1)) begin
          ex_run_q  <= 1'b0;
          g_start_q <= 1'b1;
        end
      end

      // ---- broadcast: setup, requests, responses ----
      bc_v_q <= {su_q, bc_rsp};
      if (su_q) begin
        tm1_q    <= 3'(c_q.t - 4'd1);
        qam1_q   <= c_q.qa_lines - 16'd1;
        qsm1_q   <= ((c_q.k + 16'd63) >> 6) - 16'd1;  // ceil(K/64) - 1
        rq_v_q   <= 1'b1;
        rq_l_q   <= '0;
        rq_tok_q <= '0;
        rs_l_q   <= '0;
        rs_tok_q <= '0;
        rs_dbs_q <= 1'b0;
        rs_addr_q <= '0;
      end
      if (rq_fire) begin
        if (rq_l_end) begin
          rq_l_q   <= '0;
          rq_tok_q <= rq_tok_q + 3'd1;
          if (rq_t_end) rq_v_q <= 1'b0;
        end else begin
          rq_l_q <= rq_l_q + 16'd1;
        end
      end
      if (bc_rsp) begin
        if (rs_l_end) begin
          rs_l_q    <= '0;
          rs_tok_q  <= rs_tok_q + 3'd1;
          rs_dbs_q  <= 1'b0;
          rs_addr_q <= '0;
          if (rs_t_end) begin
            bc_ph_q   <= 1'b0;
            bc_done_q <= 1'b1;
          end
        end else begin
          rs_l_q <= rs_l_q + 16'd1;
          if (rs_q_end) begin
            rs_dbs_q  <= 1'b1;
            rs_addr_q <= '0;
          end else begin
            rs_addr_q <= rs_addr_q + 10'd1;
          end
        end
      end
    end
  end

  // Datapath (no reset): command, request line, broadcast payload.
  always_ff @(posedge clk) begin
    if (fire) c_q <= cmd;
    if (su_q) rq_line_q <= c_q.x_line[SPM_AW-1:0];
    else if (rq_fire) rq_line_q <= rq_line_q + 1'b1;
    if (bc_rsp) begin
      bc_d_q <= {rs_l_end & rs_t_end, rs_tok_q, rs_dbs_q, rs_addr_q, spm_rd_rsp_data};
    end else if (su_q) begin
      bc_d_q[CCW-1:0] <= ccmd;
    end
  end

  assign cmd_ready     = cmd_ready_q;
  assign done          = done_q;
  assign perf_mvm_busy = busy_q;

  // ---- SPM read port: broadcast first, then the gather's ACC reads ----
  logic g_rd_ready, g_rsp_valid;
  assign spm_rd_req_valid = rq_v_q | g_rd_v;
  assign spm_rd_req_line  = bc_ph_q ? rq_line_q : g_rd_line;
  assign g_rd_ready       = spm_rd_req_ready & ~bc_ph_q;
  assign g_rsp_valid      = spm_rd_rsp_valid & ~bc_ph_q;

  // -------------------------------------------------------------------------------------------
  // Clusters with their broadcast and result pipes
  // -------------------------------------------------------------------------------------------
  logic [NUM_CH-1:0]                g_res_v, g_res_r, g_res_last, cl_idle;
  logic [NUM_CH-1:0][31:0]          g_res_row;
  logic [NUM_CH-1:0][T_MAX-1:0][31:0] g_res_y;

  for (genvar c = 0; c < NUM_CH; c++) begin : g_cl
    logic [1:0]          pv;
    logic [BCW-1:0]      pd;
    tsr_mvm_ccmd_t       p_cmd;
    tsr_mvm_bcline_t     p_line;
    logic                cr_v, cr_r, cr_last;
    logic [31:0]         cr_row;
    logic [T_MAX-1:0][31:0] cr_y;
    logic                wq_v, wq_r, wr_v, wr_last, wr_rdy;
    logic [ADDR_BEAT_W-1:0]  wq_addr;
    logic [TSR_LEN_W-1:0]    wq_len;
    logic [TSR_SRC_ID_W-1:0] wq_id, wr_id;
    logic [BEAT_BITS-1:0]    wr_data;

    tsr_vpipe #(.VW(2), .W(BCW), .DEPTH(BC_PIPE), .KEEP(1'b1)) u_bcp (
      .clk, .rst_n(rst_c_n[c]), .in_valid(bc_v_q), .in_data(bc_d_q), .out_valid(pv), .out_data(pd)
    );
    assign p_cmd  = pd[CCW-1:0];
    assign p_line = pd;

    tsr_mvm_cluster #(
      .CH(c), .K_MAX(K_MAX), .T_MAX(T_MAX), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS),
      .WS_FIFO_BEATS(WS_FIFO_BEATS), .ACT_RD_LAT(ACT_RD_LAT), .MAC_LAT(MAC_LAT), .MUL_LAT(MUL_LAT),
      .ADD_LAT(ADD_LAT), .WS_RD_LAT(WS_RD_LAT)
    ) u_cl (
      .clk, .rst_n(rst_c_n[c]),
      .bc_cmd_valid(pv[1]), .bc_cmd(p_cmd), .bc_line_valid(pv[0]), .bc_line(p_line),
      .rd_req_valid(wq_v), .rd_req_ready(wq_r), .rd_req_addr(wq_addr), .rd_req_len(wq_len),
      .rd_req_id(wq_id),
      .rd_rsp_valid(wr_v), .rd_rsp_ready(wr_rdy), .rd_rsp_data(wr_data), .rd_rsp_last(wr_last),
      .rd_rsp_id(wr_id),
      .res_valid(cr_v), .res_ready(cr_r), .res_row(cr_row), .res_last(cr_last), .res_y(cr_y),
      .idle(cl_idle[c])
    );

    // Weight stream of cluster c <-> arbiter c (M3 part C, docs/rtl-m3.md §14): requests through
    // RES_PIPE register slices (towards the centre, like the results), responses through BC_PIPE
    // registers (towards the cluster, like the broadcast; no backpressure: the streamer's
    // rd_rsp_ready is constant 1, §4), so the arbiters can sit with the DMA while a cluster sits
    // in another SLR.
    tsr_rpipe #(.W(ADDR_BEAT_W + TSR_LEN_W + TSR_SRC_ID_W), .DEPTH(RES_PIPE)) u_wqp (
      .clk, .rst_n(rst_c_n[c]),
      .s_valid(wq_v), .s_ready(wq_r), .s_data({wq_addr, wq_len, wq_id}),
      .m_valid(ws_rd_req_valid[c]), .m_ready(ws_rd_req_ready[c]),
      .m_data({ws_rd_req_addr[c], ws_rd_req_len[c], ws_rd_req_id[c]})
    );
    tsr_vpipe #(.VW(1), .W(BEAT_BITS + 1 + TSR_SRC_ID_W), .DEPTH(BC_PIPE), .KEEP(1'b1)) u_wrp (
      .clk, .rst_n(rst_c_n[c]), .in_valid(ws_rd_rsp_valid[c]),
      .in_data({ws_rd_rsp_data[c], ws_rd_rsp_last[c], ws_rd_rsp_id[c]}),
      .out_valid(wr_v), .out_data({wr_data, wr_last, wr_id})
    );
    assign ws_rd_rsp_ready[c] = wr_rdy;
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
      if (rst_c_n[c]) assert (wr_rdy) else $error("%m: weight-stream rd_rsp_ready low (the response pipe has no backpressure)");
    end
`endif

    tsr_rpipe #(.W(RW), .DEPTH(RES_PIPE)) u_rp (
      .clk, .rst_n(rst_c_n[c]),
      .s_valid(cr_v), .s_ready(cr_r), .s_data({cr_row, cr_last, cr_y}),
      .m_valid(g_res_v[c]), .m_ready(g_res_r[c]), .m_data({g_res_row[c], g_res_last[c], g_res_y[c]})
    );
  end

  tsr_mvm_gather #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT), .ADD_LAT(ADD_LAT),
    .JOB_PIPE(RES_PIPE)
  ) u_gather (
    .clk, .rst_n(rst_g_n),
    .start(g_start_q), .acc(c_q.acc), .t(c_q.t), .n(c_q.n), .dst_line(c_q.dst_line[SPM_AW-1:0]),
    .dst_stride(c_q.dst_stride), .expect_ch(ex_q), .done(g_done),
    .res_valid(g_res_v), .res_ready(g_res_r), .res_row(g_res_row), .res_last(g_res_last),
    .res_y(g_res_y),
    .spm_rd_req_valid(g_rd_v), .spm_rd_req_ready(g_rd_ready), .spm_rd_req_line(g_rd_line),
    .spm_rd_rsp_valid(g_rsp_valid), .spm_rd_rsp_data,
    .spm_wr_valid, .spm_wr_ready, .spm_wr_line, .spm_wr_data, .spm_wr_be
  );

  wire unused_ok = &{1'b0, c_q.y_lines, c_q.dst_line[31:SPM_AW], c_q.x_line[31:SPM_AW],
                     c_q.row_beats[31:16]};

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_t_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_t_n && a_on_q) begin
      if (fire) begin
        assert (cmd.fmt ? (cmd.k != 16'd0 && cmd.k[7:0] == 8'd0) : (cmd.k != 16'd0 && cmd.k[4:0] == 5'd0))
          else $error("%m: K = %0d not a multiple of the block size", cmd.k);
        assert (32'(cmd.k) <= K_MAX) else $error("%m: K = %0d > K_MAX", cmd.k);
        assert (cmd.t >= 4'd1 && 32'(cmd.t) <= T_MAX) else $error("%m: T = %0d", cmd.t);
        assert (cmd.row_beats == tsr_row_beats(cmd.fmt, 32'(cmd.k))) else $error("%m: row_beats");
        assert (32'(cmd.qa_lines) == tsr_qa_lines(cmd.fmt, 32'(cmd.k))) else $error("%m: qa_lines");
        assert (cmd.y_lines == (cmd.n + 32'd15) >> 4 || cmd.n > 32'h7FFF_FFF0) else $error("%m: y_lines");
        assert (cmd.n == 32'd0 || cmd.dst_stride != 32'd0) else $error("%m: dst_stride not resolved");
        assert (64'(cmd.rows_per_ch) == (64'(cmd.n) + 64'(NUM_CH - 1)) / 64'(NUM_CH))
          else $error("%m: rows_per_ch != ceil(N / NUM_CH)");
        assert (64'(cmd.x_line) + 64'(cmd.t) * 64'(cmd.qa_lines) <= 64'(SPM_LINES) || cmd.n == 32'd0)
          else $error("%m: x range");
      end
      assert (!(g_rd_v && bc_ph_q)) else $error("%m: gather read request during the broadcast");
      assert (!(rq_v_q && !bc_ph_q)) else $error("%m: broadcast read outside the broadcast phase");
      assert (!(bc_v_q[1] && bc_v_q[0])) else $error("%m: bc_cmd and bc_line in one cycle");
      if (st_q == S_IDLE) begin
        assert (!spm_rd_req_valid && !spm_wr_valid && ws_rd_req_valid == '0)
          else $error("%m: SPM / weight traffic while idle");
      end
      if (done_q) begin
        assert (!spm_wr_valid && !spm_rd_req_valid && ws_rd_req_valid == '0 && &cl_idle)
          else $error("%m: done while not quiescent (spm wr %0b rd %0b ws %b idle %b)",
                      spm_wr_valid, spm_rd_req_valid, ws_rd_req_valid, cl_idle);
      end
    end
  end
`endif

endmodule
