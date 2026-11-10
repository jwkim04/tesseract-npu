// tsr_mvm.sv — the MVM unit (spec MVM cluster microarchitecture, ISA MVM): dispatch from
// the dispatcher, activation broadcast to the NUM_CH clusters, cluster command, result gather and
// ACC into the SPM, done, perf_mvm_busy.
//
// Spec: docs/rtl-mvm.md §2 block (top); contract §3.0, §3.1, §6.
// tsr_core instantiates it when HAS_MVM = 1.
//
// Contract (docs/rtl-core.md §3.0 dispatch; semantics = tsr_sim.c ex_mvm):
//   - cmd_valid/cmd_ready/cmd (tsr_mvm_cmd_t): every check done at dispatch (K, T, SPM and
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
// weight FIFO), BC_PIPE (total broadcast delay, tsr_vpipe), RES_PIPE (result pipe
// stages per cluster, tsr_rpipe), ACT_RD_LAT, MAC_LAT, MUL_LAT, ADD_LAT, WS_RD_LAT
// (docs/rtl-mvm.md §1).
//
// Interface addition (additive): parameter WS_RD_LAT (weight FIFO RAM read latency,
//   docs/rtl-mvm.md §1, default TSR_DEF_WS_RD_LAT), passed to every cluster's front end.
// Interface addition (additive; timing, docs/rtl-mvm.md §14): parameter RST_LOCAL (0..2).
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
//                 └─► broadcast reader: SPM reads x_line │   line. With BC_PIPE >= 2, one central
//                     .. + T*qa_lines - 1, one per cycle │   stage feeds separate local tails of
//                     response -> {last, tok, dbs, addr, ┘   BC_PIPE-1 stages. The three valids
//                     data} (counters on the response side)   {acmd, cmd, line} are exclusive.
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
//
// Attention (AT §5.2, §5.3, §8.1, §8.2; parameter HAS_ATT): a command with cmd_xc = 1 carries
//   the ATTQK / ATTPV payload (tsr_att_pkg::tsr_att_cmd_t) in xcmd. tsr_mvm derives the shape (E/D,
//   n_head, R, T_q GS, pass rows, ceil(R/16), u0 / C, (u0 + n) / C by tsr_att_divc) over 5 registered
//   steps, sends every cluster one tsr_att_ccmd_t (the broadcast pipe's third valid, overlaid on the
//   line payload), then the lines:
//     ATTQK: the T_q QA(A32, G E) query vectors at x (tok = query), as the MVM's activations;
//     ATTPV: the p rows in the order (g < E/D, u - u0 < n, slot t < T_q GS): line p + r(q(t), g,
//       i(t)) pstride + (u - u0) (tok = t; each cluster keeps its own groups' lines).
//   The clusters' attention jobs come back through one tsr_rpipe (RES_PIPE) each into the gather's
//   job pipes (tsr_mvm_gather HAS_ATT); expect_ch = the clusters with groups (ATTQK) or all (ATTPV:
//   every channel writes its partials). done when the broadcast is complete and the gather is done
//   (with MAXOUT after the row maxima). perf_att_qk / perf_att_pv: the unit holds an ATTQK /
//   ATTPV (PERF ATT_QK_CYCLES / ATT_PV_CYCLES, subsets of perf_mvm_busy); perf_att_stall: a cluster's
//   attention stream waits for its line buffer or p line (ATT_PATH_STALL, registered OR; XSLR_REG: each cluster's
//   strobe passes two register stages first, 2 cycles later). HAS_ATT = 0:
//   cmd_xc / xcmd are ignored, the unit above.
//   Cost (RTL structure, no synthesis run here): command 294 FF + broadcast register 527 FF +
//   ~140 FF control; broadcast 530 FF per stage (527 payload + 3 valid), with one shared
//   first stage when BC_PIPE >= 2 and all remaining stages per cluster. Commands overlay
//   the line payload. Per cluster RES_PIPE x 2 x (33 + 32 T_MAX) FF;
//   weight-stream pipes per cluster BC_PIPE x 516 FF + RES_PIPE x 2 x 38 FF; with RST_LOCAL
//   RST_LOCAL x (NUM_CH + 2) reset flops. Measured sizes: docs/rtl-mvm.md §14.

module tsr_mvm
  import tsr_pkg::*;
  import tsr_att_pkg::*;
  import tsr_vu_pkg::*;
#(
  parameter  bit          LOCAL_SCALE = 1'b0,
  parameter  bit          LOCAL_INTERMEDIATES = 1'b0,
  parameter  bit          HAS_ATT       = 1'b0,   // ATTQK / ATTPV
  parameter  int unsigned XC_CMD_W      = 256,
  parameter  int unsigned NUM_CH        = TSR_DEF_NUM_CH,
  parameter  int unsigned T_MAX         = TSR_DEF_T_MAX,
  parameter  int unsigned K_MAX         = TSR_DEF_K_MAX,
  parameter  int unsigned SPM_BYTES     = TSR_DEF_SPM_BYTES,
  parameter  int unsigned SPM_RD_LAT    = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned ADDR_BEAT_W   = TSR_DEF_ADDR_BEAT_W,
  parameter  int unsigned BEAT_BITS     = TSR_DEF_BEAT_BITS,
  parameter  int unsigned WS_FIFO_BEATS = TSR_DEF_WS_FIFO_BEATS,
  parameter  int unsigned BC_PIPE       = TSR_DEF_MVM_BC_PIPE,
  parameter  bit          GATHER_PAIR   = 1'b0,
  parameter  int unsigned GATHER_FLITS  = 1,
  parameter  int unsigned RES_PIPE      = TSR_DEF_MVM_RES_PIPE,
  parameter  int unsigned ACT_RD_LAT    = TSR_DEF_ACT_RD_LAT,
  parameter  int unsigned MAC_LAT       = TSR_DEF_MAC_LAT,
  parameter  int unsigned MUL_LAT       = TSR_DEF_FMUL_LAT,
  parameter  int unsigned ADD_LAT       = TSR_DEF_FADD_LAT,
  parameter  int unsigned WS_RD_LAT     = TSR_DEF_WS_RD_LAT,
  parameter  int unsigned RST_LOCAL     = 0,      // reset register stages per cluster / top / gather (0..2)
  parameter  bit          XSLR_REG      = 1'b0,
  parameter  bit          FAST_CLK      = 1'b0,   // the cycle-changing stages of the 250 MHz clock (tsr_core FAST_CLK)   // cross-SLR stages: att_stall (2 per cluster), pair links of the gather (1)
  localparam int unsigned SPM_LINES     = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned SPM_AW        = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1
) (
  input  logic                                   clk,
  input  logic                                   rst_n,

  // ---- dispatch (the dispatcher, Y4) ----
  input  logic                                   cmd_valid,
  output logic                                   cmd_ready,
  input  tsr_mvm_cmd_t                           cmd,
  input  logic                                   cmd_xc,       // ATTQK / ATTPV (HAS_ATT): payload in xcmd
  input  logic [XC_CMD_W-1:0]                    xcmd,
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
  output logic                                   perf_mvm_busy,
  output logic                                   perf_att_qk,
  output logic                                   perf_att_pv,
  output tsr_vu_local_req_t [1:0]                scale_req,
  input  tsr_vu_local_rsp_t [1:0]                 scale_rsp,
  input  tsr_local_score_req_t [NUM_CH-1:0]       local_score_req,
  output logic [NUM_CH-1:0][TSR_LINE_BITS-1:0]    local_score_data,
  output logic                                   perf_att_stall
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
  // rst_ce_n: the clusters take the reset one stage early (stage RST_LOCAL - 1) and register the last
  // stage in copies of their own (tsr_mvm_cluster RST_COPY): the same delay as rst_c_n, without one
  // register driving a whole cluster. rst_c_n stays for the pipes around the clusters.
  logic              rst_t_n, rst_g_n;
  logic [NUM_CH-1:0] rst_c_n, rst_ce_n;
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
    if (RST_LOCAL > 1) begin : g_ce
      assign rst_ce_n = rc_q[RST_LOCAL-2];
    end else begin : g_ce1
      assign rst_ce_n = {NUM_CH{rst_n}};
    end
  end else begin : g_rst0
    assign rst_t_n  = rst_n;
    assign rst_g_n  = rst_n;
    assign rst_c_n  = {NUM_CH{rst_n}};
    assign rst_ce_n = {NUM_CH{rst_n}};
  end

  localparam int unsigned BCW = $bits(tsr_mvm_bcline_t);   // 527: broadcast payload
  localparam int unsigned CCW = $bits(tsr_mvm_ccmd_t);     // 137: overlaid on its low bits
  localparam int unsigned ACW = $bits(tsr_att_ccmd_t);     // 347: the attention command, overlaid
  localparam int unsigned RW  = 33 + 32 * T_MAX;           // {row, last, y}
  localparam int unsigned CW  = (NUM_CH > 1) ? $clog2(NUM_CH) : 1;
  localparam int unsigned AJW = 1 + SPM_AW + 16 + 16 + TSR_LINE_BITS;
  localparam int unsigned TW  = (T_MAX > 1) ? $clog2(T_MAX) : 1;

  if (CCW > BCW || ACW > BCW) begin : g_chk_overlay
    $fatal(1, "tsr_mvm: tsr_mvm_ccmd_t / tsr_att_ccmd_t must fit in the broadcast payload");
  end
  if (HAS_ATT && XC_CMD_W < $bits(tsr_att_cmd_t)) begin : g_chk_xc
    $fatal(1, "tsr_mvm: XC_CMD_W too small for tsr_att_cmd_t");
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

  // attention (HAS_ATT)
  logic         at_q, apv_q;   // the op is ATTQK / ATTPV (at_q), ATTPV (apv_q)
  tsr_att_cmd_t a_q;
  wire tsr_att_cmd_t dispatch_att = tsr_att_cmd_t'(xcmd[$bits(tsr_att_cmd_t)-1:0]);
  logic [5:0]   as_q;          // derivation steps
  logic [6:0]   a_g;
  logic [3:0]   a_gs, a_tq, a_tact;
  logic [8:0]   a_nseg;
  logic [14:0]  a_nh;
  logic [17:0]  a_r;
  logic [7:0]   a_prows;
  logic [13:0]  a_rl;
  logic [20:0]  a_x0, a_x1;    // u0, u0 + n
  logic [16:0]  a_q0, a_qn;
  logic [3:0]   a_m0, a_mn;
  logic [15:0]  a_qs, a_qax;   // ATTQK: QS lines G E/64, QA lines of a query vector
  tsr_att_ccmd_t accmd;
  always_comb begin
    accmd = '{c: a_q, nseg: a_nseg, n_head: a_nh, r: a_r, tact: a_tact, prows: a_prows,
              q0: 16'(a_q0), m0: a_m0, qn: a_qn, mn: a_mn, rl: a_rl};
  end

  wire fire   = cmd_valid & cmd_ready_q;
  wire is_att = HAS_ATT & cmd_xc;
  wire n_zero = (cmd.n == 32'd0) & ~is_att;

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
  logic [2:0]                           bc_v_q;          // {acmd, cmd, line}
  logic [BCW-1:0]                       bc_d_q;
  // ATTPV reader: request (g, uo, t), rb0[t] = p + r(q(t), 0, i(t)) pstride; response counters
  logic                                 pv_rd_q;         // the reader walks the p rows
  logic [8:0]                           pr_g_q;
  logic [11:0]                          pr_uo_q;
  logic [3:0]                           pr_t_q;
  logic [SPM_AW-1:0]                    pr_gofs_q, pr_gstep_q;
  logic [T_MAX-1:0][SPM_AW-1:0]         pr_rb0_q;
  logic                                 pb_act_q;        // rb0 build
  logic [3:0]                           pb_t_q, pb_i_q;
  logic [17:0]                          pb_r_q;          // q nh + hg0 + i of slot pb_t
  logic [3:0]                           ps_t_q;
  logic [31:0]                          ps_left_q;       // p lines whose response is still to come

  wire local_pv = LOCAL_INTERMEDIATES && apv_q && a_q.local_slot != 0;
  wire rq_fire  = rq_v_q & (local_pv | spm_rd_req_ready);
  wire rq_l_end = (rq_l_q == qam1_q);
  wire rq_t_end = (rq_tok_q == tm1_q);
  wire bc_rsp   = (spm_rd_rsp_valid & bc_ph_q) | (rq_fire & local_pv);
  wire rs_l_end = (rs_l_q == qam1_q);
  wire rs_t_end = (rs_tok_q == tm1_q);
  wire rs_q_end = (rs_l_q == qsm1_q);
  wire ps_end   = (ps_left_q == 32'd1);
  wire pr_t_end = (pr_t_q == a_tact - 4'd1);
  wire pr_u_end = (pr_uo_q == a_q.n - 12'd1);
  wire pr_g_end = (pr_g_q == a_nseg - 9'd1);

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

  // ATT: channel c has groups: l_end(c) > l_first(c) (AT §4.2; ATTPV: every channel)
  wire [16:0] ex_lf = (32'(ex_c_q) >= 32'(a_m0)) ? a_q0 : a_q0 + 17'd1;
  wire [16:0] ex_le = (32'(ex_c_q) <  32'(a_mn)) ? a_qn + 17'd1 : a_qn;

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
      at_q        <= 1'b0;
      apv_q       <= 1'b0;
      as_q        <= '0;
      pv_rd_q     <= 1'b0;
      pb_act_q    <= 1'b0;
      ps_left_q   <= '0;
    end else begin
      // ---- dispatch / done ----
      done_q    <= 1'b0;
      su_q      <= fire & ~n_zero & ~is_att;
      as_q      <= {as_q[4:0], fire & is_att};
      g_start_q <= 1'b0;
      if (g_done) gd_q <= 1'b1;
      unique case (st_q)
        S_IDLE: if (fire) begin
          cmd_ready_q <= 1'b0;
          busy_q      <= 1'b1;
          gd_q        <= 1'b0;
          at_q        <= is_att;
          apv_q       <= is_att & dispatch_att.pv;
          if (n_zero) begin
            done_q <= 1'b1;
            st_q   <= S_DONE;
          end else begin
            st_q      <= S_RUN;
            bc_ph_q   <= 1'b1;
            bc_done_q <= 1'b0;
            ex_run_q  <= ~is_att;
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

      // ---- expect_ch: MVM c * R < N; ATT the channels with groups (one channel per cycle) ----
      if (as_q[3]) ex_run_q <= 1'b1;
      if (ex_run_q) begin
        ex_q[ex_c_q] <= at_q ? (apv_q | (ex_le > ex_lf)) : (ex_acc_q < 36'(c_q.n));
        ex_acc_q     <= ex_acc_q + 36'(c_q.rows_per_ch);
        ex_c_q       <= ex_c_q + 1'b1;
        if (ex_c_q == CW'(NUM_CH - 1)) begin
          ex_run_q  <= 1'b0;
          g_start_q <= 1'b1;
        end
      end

      // ---- broadcast: setup, requests, responses ----
      bc_v_q <= {as_q[4], su_q, bc_rsp};
      if (su_q || (as_q[4] && !a_q.pv)) begin            // MVM / ATTQK: T QA images
        tm1_q    <= su_q ? 3'(c_q.t - 4'd1) : 3'(a_tq - 4'd1);
        qam1_q   <= su_q ? c_q.qa_lines - 16'd1 : a_qax - 16'd1;
        qsm1_q   <= su_q ? ((c_q.k + 16'd63) >> 6) - 16'd1 : a_qs - 16'd1;   // ceil(K/64) - 1
        rq_v_q   <= 1'b1;
        rq_l_q   <= '0;
        rq_tok_q <= '0;
        rs_l_q   <= '0;
        rs_tok_q <= '0;
        rs_dbs_q <= 1'b0;
        rs_addr_q <= '0;
      end
      if (as_q[4] && a_q.pv) begin                        // ATTPV: rb0 first, then the reader
        pb_act_q  <= 1'b1;
        pb_t_q    <= '0;
        pb_i_q    <= '0;
        ps_left_q <= 32'(a_prows) * 32'(a_q.n);                      // (E/D) T_q GS n p lines
        ps_t_q    <= '0;
      end
      if (pb_act_q) begin
        pb_t_q <= pb_t_q + 4'd1;
        pb_i_q <= (pb_i_q == a_gs - 4'd1) ? 4'd0 : pb_i_q + 4'd1;
        if (pb_t_q == a_tact - 4'd1) begin
          pb_act_q <= 1'b0;
          pv_rd_q  <= 1'b1;
          rq_v_q   <= 1'b1;
          pr_g_q   <= '0;
          pr_uo_q  <= '0;
          pr_t_q   <= '0;
        end
      end
      if (rq_fire && !pv_rd_q) begin
        if (rq_l_end) begin
          rq_l_q   <= '0;
          rq_tok_q <= rq_tok_q + 3'd1;
          if (rq_t_end) rq_v_q <= 1'b0;
        end else begin
          rq_l_q <= rq_l_q + 16'd1;
        end
      end
      if (rq_fire && pv_rd_q) begin
        if (pr_t_end) begin
          pr_t_q <= '0;
          if (pr_u_end) begin
            pr_uo_q <= '0;
            pr_g_q  <= pr_g_q + 9'd1;
            if (pr_g_end) begin
              rq_v_q  <= 1'b0;
              pv_rd_q <= 1'b0;
            end
          end else begin
            pr_uo_q <= pr_uo_q + 12'd1;
          end
        end else begin
          pr_t_q <= pr_t_q + 4'd1;
        end
      end
      if (bc_rsp && apv_q) begin                          // ATTPV responses
        ps_left_q <= ps_left_q - 32'd1;
        ps_t_q    <= (ps_t_q == a_tact - 4'd1) ? 4'd0 : ps_t_q + 4'd1;
        if (ps_end) begin
          bc_ph_q   <= 1'b0;
          bc_done_q <= 1'b1;
        end
      end else if (bc_rsp) begin
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

  // Datapath (no reset): command, attention derivation, request line, broadcast payload.
  always_ff @(posedge clk) begin
    if (fire) c_q <= cmd;
    if (fire && is_att) a_q <= dispatch_att;
    // derivation steps (header)
    if (as_q[0]) begin
      a_g    <= 7'(a_q.g_m1) + 7'd1;
      a_gs   <= 4'(a_q.gs_m1) + 4'd1;
      a_tq   <= 4'(a_q.tq_m1) + 4'd1;
      a_nseg <= a_q.e64 >> a_q.dcode;
      a_tact <= 4'((4'(a_q.tq_m1) + 4'd1) * (4'(a_q.gs_m1) + 4'd1));
      a_x0   <= 21'(a_q.u0);
      a_x1   <= 21'(a_q.u0) + 21'(a_q.n);
    end
    if (as_q[1]) begin
      a_nh    <= 15'(16'(a_g) * 16'(a_nseg));
      a_prows <= 8'(16'(a_nseg) * 16'(a_tact));
      a_q0    <= 17'(tsr_att_divc(a_x0, NUM_CH));
      a_qn    <= 17'(tsr_att_divc(a_x1, NUM_CH));
      a_qs    <= 16'(a_g) * 16'(a_q.e64);
    end
    if (as_q[2]) begin
      a_r        <= 18'(a_tq) * 18'(a_nh);
      a_m0       <= 4'(a_x0 - 21'(a_q0) * 21'(NUM_CH));
      a_mn       <= 4'(a_x1 - 21'(a_qn) * 21'(NUM_CH));
      a_qax      <= a_qs + ((a_qs + 16'd7) >> 3);
      pr_gstep_q <= SPM_AW'(32'(a_g) * 32'(a_q.rs));
    end
    if (as_q[3]) a_rl <= 14'((a_r + 18'd15) >> 4);
    // ATTPV rb0[t] = p + (q(t) nh + hg0 + i(t)) pstride, t = 0 .. tact - 1 (one per cycle)
    if (as_q[4]) pb_r_q <= 18'(a_q.hg0);
    if (pb_act_q) begin
      pr_rb0_q[TW'(pb_t_q)] <= SPM_AW'(a_q.x) + SPM_AW'(32'(pb_r_q) * 32'(a_q.rs));
      pb_r_q <= (pb_i_q == a_gs - 4'd1) ? pb_r_q - 18'(a_gs - 4'd1) + 18'(a_nh) : pb_r_q + 18'd1;
    end
    // request line: MVM / ATTQK consecutive from x; ATTPV rb0[t] + g G pstride + uo
    if (su_q) rq_line_q <= c_q.x_line[SPM_AW-1:0];
    else if (as_q[4] && !a_q.pv) rq_line_q <= a_q.x[SPM_AW-1:0];
    else if (pb_act_q && pb_t_q == a_tact - 4'd1) begin
      rq_line_q <= SPM_AW'(a_q.x) + SPM_AW'(32'(18'(a_q.hg0)) * 32'(a_q.rs));     // rb0[0]
      pr_gofs_q <= '0;
    end else if (rq_fire && !pv_rd_q) rq_line_q <= rq_line_q + 1'b1;
    else if (rq_fire && pv_rd_q) begin
      if (pr_t_end) begin
        if (pr_u_end) begin
          pr_gofs_q <= pr_gofs_q + pr_gstep_q;
          rq_line_q <= pr_rb0_q[0] + pr_gofs_q + pr_gstep_q;
        end else begin
          rq_line_q <= pr_rb0_q[0] + pr_gofs_q + SPM_AW'(pr_uo_q) + 1'b1;
        end
      end else begin
        rq_line_q <= pr_rb0_q[TW'(pr_t_q + 4'd1)] + pr_gofs_q + SPM_AW'(pr_uo_q);
      end
    end
    if (bc_rsp) begin
      if (apv_q) bc_d_q <= {ps_end, 3'(ps_t_q), 1'b0, 10'd0,
                             local_pv ? TSR_LINE_BITS'(rq_line_q - SPM_AW'(a_q.x)) : spm_rd_rsp_data};
      else       bc_d_q <= {rs_l_end & rs_t_end, rs_tok_q, rs_dbs_q, rs_addr_q, spm_rd_rsp_data};
    end else if (su_q) begin
      bc_d_q[CCW-1:0] <= ccmd;
    end else if (as_q[4]) begin
      bc_d_q[ACW-1:0] <= accmd;
    end
  end

  assign cmd_ready      = cmd_ready_q;
  assign done           = done_q;
  assign perf_mvm_busy  = busy_q;
  logic att_stall_q;
  assign perf_att_qk    = busy_q & at_q & ~apv_q;
  assign perf_att_pv    = busy_q & apv_q;
  assign perf_att_stall = att_stall_q;

  // ---- SPM read port: broadcast first, then the gather's ACC reads ----
  logic g_rd_ready, g_rsp_valid;
  assign spm_rd_req_valid = (rq_v_q & ~local_pv) | g_rd_v;
  assign spm_rd_req_line  = bc_ph_q ? rq_line_q : g_rd_line;
  assign g_rd_ready       = spm_rd_req_ready & ~bc_ph_q;
  assign g_rsp_valid      = spm_rd_rsp_valid & ~bc_ph_q;

  // -------------------------------------------------------------------------------------------
  // Clusters with their broadcast and result pipes
  // -------------------------------------------------------------------------------------------
  logic [NUM_CH-1:0]                g_res_v, g_res_r, g_res_last, cl_idle;
  logic [NUM_CH-1:0][31:0]          g_res_row;
  logic [NUM_CH-1:0][T_MAX-1:0][31:0] g_res_y;
  logic [NUM_CH-1:0]                g_aj_v, g_aj_r, cl_stall;
  logic [NUM_CH-1:0][AJW-1:0]       g_aj;

  // Share only the source-side stage. Each destination keeps its own registers so
  // synthesis cannot turn a local cluster's activation/decode registers into a
  // driver for clusters in another SLR (docs/rtl-mvm.md §14.4). The tail is local
  // to its cluster; a common source net can cross once into each destination SLR.
  // rt_q and every rc_q have the same INIT and shift recurrence, so this stage's
  // reset is cycle-identical to the original per-cluster stage, including flush.
  localparam int unsigned BC_HEAD = (BC_PIPE >= 2) ? 1 : 0;
  logic [2:0]     bc_head_v;
  logic [BCW-1:0] bc_head_d;
  tsr_vpipe #(.VW(3), .W(BCW), .DEPTH(BC_HEAD), .KEEP_REGS(1'b1)) u_bc_head (
    .clk, .rst_n(rst_t_n), .in_valid(bc_v_q), .in_data(bc_d_q),
    .out_valid(bc_head_v), .out_data(bc_head_d)
  );

  for (genvar c = 0; c < NUM_CH; c++) begin : g_cl
    logic [2:0]          pv;
    logic [BCW-1:0]      pd;
    tsr_mvm_ccmd_t       p_cmd;
    tsr_att_ccmd_t       p_acmd;
    tsr_mvm_bcline_t     p_line;
    logic                cr_v, cr_r, cr_last;
    logic [31:0]         cr_row;
    logic [T_MAX-1:0][31:0] cr_y;
    logic                wq_v, wq_r, wr_v, wr_last, wr_rdy;
    logic [ADDR_BEAT_W-1:0]  wq_addr;
    logic [TSR_LEN_W-1:0]    wq_len;
    logic [TSR_SRC_ID_W-1:0] wq_id, wr_id;
    logic [BEAT_BITS-1:0]    wr_data;
    logic                ca_v, ca_r;
    logic [AJW-1:0]      ca_j;

    tsr_vpipe #(.VW(3), .W(BCW), .DEPTH(BC_PIPE-BC_HEAD), .KEEP_REGS(1'b1)) u_bcp (
      .clk, .rst_n(rst_c_n[c]), .in_valid(bc_head_v), .in_data(bc_head_d), .out_valid(pv), .out_data(pd)
    );
    assign p_cmd  = pd[CCW-1:0];
    assign p_acmd = pd[ACW-1:0];
    assign p_line = pd;

    tsr_mvm_cluster #(
      .CH(c), .NUM_CH(NUM_CH), .HAS_ATT(HAS_ATT), .LOCAL_INTERMEDIATES(LOCAL_INTERMEDIATES), .K_MAX(K_MAX), .T_MAX(T_MAX), .ADDR_BEAT_W(ADDR_BEAT_W),
      .BEAT_BITS(BEAT_BITS), .WS_FIFO_BEATS(WS_FIFO_BEATS), .ACT_RD_LAT(ACT_RD_LAT), .MAC_LAT(MAC_LAT),
      .MUL_LAT(MUL_LAT), .ADD_LAT(ADD_LAT), .WS_RD_LAT(WS_RD_LAT), .FAST_CLK(FAST_CLK), .RST_COPY(RST_LOCAL > 0), .SPM_AW(SPM_AW)
    ) u_cl (
      .clk, .rst_n(rst_ce_n[c]),
      .bc_cmd_valid(pv[1]), .bc_cmd(p_cmd), .bc_acmd_valid(pv[2]), .bc_acmd(p_acmd),
      .bc_line_valid(pv[0]), .bc_line(p_line),
      .rd_req_valid(wq_v), .rd_req_ready(wq_r), .rd_req_addr(wq_addr), .rd_req_len(wq_len),
      .rd_req_id(wq_id),
      .rd_rsp_valid(wr_v), .rd_rsp_ready(wr_rdy), .rd_rsp_data(wr_data), .rd_rsp_last(wr_last),
      .rd_rsp_id(wr_id),
      .res_valid(cr_v), .res_ready(cr_r), .res_row(cr_row), .res_last(cr_last), .res_y(cr_y),
      .aj_valid(ca_v), .aj_ready(ca_r), .aj(ca_j), .att_stall(cl_stall[c]),
      .local_score_req(local_score_req[c]), .local_score_data(local_score_data[c]),
      .idle(cl_idle[c])
    );

    // Weight stream of cluster c <-> arbiter c (docs/rtl-mvm.md §14): requests through
    // RES_PIPE register slices (toward the center, like the results), responses through BC_PIPE
    // registers (toward the cluster, like the broadcast; no backpressure: the streamer's
    // rd_rsp_ready is constant 1, §4), so the arbiters can sit with the DMA while a cluster sits
    // in another SLR.
    tsr_rpipe #(.W(ADDR_BEAT_W + TSR_LEN_W + TSR_SRC_ID_W), .DEPTH(RES_PIPE)) u_wqp (
      .clk, .rst_n(rst_c_n[c]),
      .s_valid(wq_v), .s_ready(wq_r), .s_data({wq_addr, wq_len, wq_id}),
      .m_valid(ws_rd_req_valid[c]), .m_ready(ws_rd_req_ready[c]),
      .m_data({ws_rd_req_addr[c], ws_rd_req_len[c], ws_rd_req_id[c]})
    );
    tsr_vpipe #(.VW(1), .W(BEAT_BITS + 1 + TSR_SRC_ID_W), .DEPTH(BC_PIPE), .KEEP_REGS(1'b1)) u_wrp (
      .clk, .rst_n(rst_c_n[c]), .in_valid(ws_rd_rsp_valid[c]),
      .in_data({ws_rd_rsp_data[c], ws_rd_rsp_last[c], ws_rd_rsp_id[c]}),
      .out_valid(wr_v), .out_data({wr_data, wr_last, wr_id})
    );
    assign ws_rd_rsp_ready[c] = wr_rdy;
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
      if (rst_c_n[c]) assert (wr_rdy) else $error("%m: weight-stream rd_rsp_ready low (the response pipe has no backpressure)");
      // the cluster's own reset copies (RST_COPY) equal the last stage rst_c_n in every cycle
      assert (u_cl.rst_c_n == rst_c_n[c] && u_cl.rst_f_n == rst_c_n[c] && u_cl.rst_t_n == {T_MAX{rst_c_n[c]}})
        else $error("%m: the cluster's reset copies differ from rst_c_n");
    end
`endif

    tsr_rpipe #(.W(RW), .DEPTH(RES_PIPE)) u_rp (
      .clk, .rst_n(rst_c_n[c]),
      .s_valid(cr_v), .s_ready(cr_r), .s_data({cr_row, cr_last, cr_y}),
      .m_valid(g_res_v[c]), .m_ready(g_res_r[c]), .m_data({g_res_row[c], g_res_last[c], g_res_y[c]})
    );
    // attention jobs of cluster c -> its job pipe in the gather (HAS_ATT)
    if (HAS_ATT) begin : g_ajp
      tsr_rpipe #(.W(AJW), .DEPTH(RES_PIPE)) u_ajp (
        .clk, .rst_n(rst_c_n[c]),
        .s_valid(ca_v), .s_ready(ca_r), .s_data(ca_j),
        .m_valid(g_aj_v[c]), .m_ready(g_aj_r[c]), .m_data(g_aj[c])
      );
    end else begin : g_noajp
      assign g_aj_v[c] = 1'b0;
      assign g_aj[c]   = '0;
      assign ca_r      = 1'b0;
      wire unused_aj = &{1'b0, ca_v, ca_j, g_aj_r[c]};
    end
  end

  // XSLR_REG: in the K4 floorplan clusters 0 / 2 sit in SLR0 and 1 / 3 in SLR2, and att_stall_q lands in
  // one of them. perf_att_stall only feeds a PERF counter (ATT_PATH_STALL), so each cluster's strobe passes two
  // register stages before the OR: the strobe reaches the counter 2 cycles later. The stages have no
  // reset (like the XSLR_REG stages in tsr_core_dp): they start at 0, and the cluster's reset clears
  // the strobe at their input, so they are clear after 2 cycles of reset (att_stall_q has its own reset).
  logic [NUM_CH-1:0] cl_stall_o;
  if (XSLR_REG) begin : g_stx
    (* shreg_extract = "no" *) logic [1:0][NUM_CH-1:0] stx_q;
    initial stx_q = '0;
    always @(posedge clk) begin           // not always_ff: the initial block writes it too
      stx_q[0] <= cl_stall;
      stx_q[1] <= stx_q[0];
    end
    assign cl_stall_o = stx_q[1];
  end else begin : g_stx0
    assign cl_stall_o = cl_stall;
  end

  always_ff @(posedge clk) begin
    if (!rst_t_n) att_stall_q <= 1'b0;
    else          att_stall_q <= |cl_stall_o;
  end

  tsr_mvm_gather #(
    .HAS_ATT(HAS_ATT), .LOCAL_SCALE(LOCAL_SCALE), .NUM_CH(NUM_CH), .T_MAX(T_MAX), .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT),
    .ADD_LAT(ADD_LAT), .JOB_PIPE(RES_PIPE), .GATHER_PAIR(GATHER_PAIR), .GATHER_FLITS(GATHER_FLITS),
    .XSLR_REG(XSLR_REG), .FAST_CLK(FAST_CLK)
  ) u_gather (
    .clk, .rst_n(rst_g_n),
    .start(g_start_q), .acc(c_q.acc & ~at_q), .t(c_q.t), .n(at_q ? 32'd1 : c_q.n),
    .dst_line(c_q.dst_line[SPM_AW-1:0]), .dst_stride(at_q ? 32'd1 : c_q.dst_stride), .expect_ch(ex_q),
    .done(g_done),
    .scale_en((c_q.local_scale || c_q.local_softcap) && !at_q), .scale(c_q.scale),
    .post_scale(c_q.post_scale), .softcap(c_q.local_softcap), .scale_req, .scale_rsp,
    .att(at_q), .att_maxout(at_q & ~apv_q & a_q.flags[TSR_ATT_FL_MAXOUT]), .att_tq(a_tq), .att_gs(a_gs),
    .att_tact(a_tact), .att_nseg(a_nseg), .att_g(a_g), .att_hg0(a_q.hg0), .att_nhead(a_nh),
    .att_mdst(a_q.w7[SPM_AW-1:0]),
    .aj_valid(g_aj_v), .aj_ready(g_aj_r), .aj(g_aj),
    .res_valid(g_res_v), .res_ready(g_res_r), .res_row(g_res_row), .res_last(g_res_last),
    .res_y(g_res_y),
    .spm_rd_req_valid(g_rd_v), .spm_rd_req_ready(g_rd_ready), .spm_rd_req_line(g_rd_line),
    .spm_rd_rsp_valid(g_rsp_valid), .spm_rd_rsp_data,
    .spm_wr_valid, .spm_wr_ready, .spm_wr_line, .spm_wr_data, .spm_wr_be
  );

  wire unused_ok = &{1'b0, c_q.y_lines, c_q.dst_line[31:SPM_AW], c_q.x_line[31:SPM_AW],
                     c_q.row_beats[31:16], xcmd, a_q.base, a_q.dst, a_q.aux0, a_q.aux1, a_q.x, a_q.w7,
                     a_x0[20:17], a_x1[20:17], as_q[5], a_q.e64};

  // -------------------------------------------------------------------------------------------
  // Simulation-only checks
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_t_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_t_n && a_on_q) begin
      if (fire && !is_att) begin
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
      assert ($onehot0(bc_v_q)) else $error("%m: bc_cmd / bc_acmd / bc_line in one cycle");
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

  // att_stall_q is the OR of the clusters' strobes of 1 cycle before (XSLR_REG: 3 cycles before)
  logic [2:0] stl_h_q;
  logic [2:0] stl_n_q;                       // cycles out of reset, saturating at 3
  always_ff @(posedge clk) begin
    stl_h_q <= {stl_h_q[1:0], |cl_stall};
    stl_n_q <= rst_t_n ? {stl_n_q[1:0], 1'b1} : 3'b000;
    if (rst_t_n && &stl_n_q && a_on_q)
      assert (att_stall_q == (XSLR_REG ? stl_h_q[2] : stl_h_q[0]))
        else $error("%m: att_stall_q is not the cluster strobes delayed by %0d cycles", XSLR_REG ? 3 : 1);
  end
`endif

endmodule
