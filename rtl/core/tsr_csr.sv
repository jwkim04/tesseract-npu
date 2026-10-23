// tsr_csr.sv — CSR block: CSR port, host-visible registers, STATUS / ERR_* / PERF_*, done output.
//
// Owner: cmdproc agent (docs/rtl-m2.md §2, block (a); CSR rules §6).
//
// CSR port (core boundary): the master drives csr_valid with csr_addr/csr_we/csr_wdata and holds
// them until csr_ready. csr_ready is a registered one-cycle strobe with csr_rdata (reads; 0 for
// writes); the transfer completes at the edge where csr_valid && csr_ready. Unmapped or unaligned
// (addr[1:0] != 0) offsets read 0, writes are ignored.
// Access pipeline (every stage starts from registers; the port is latency-insensitive):
//   C  edge with csr_valid && !csr_ready && no access in progress: capture addr / we / wdata
//   S1 next edge: write side effects; read: decode, select the 64-bit counter or the register
//      value (counter LO and HI sampled together)
//   S2 next edge: final read mux (HI snapshot), csr_ready := 1 for one cycle
//   so a read or write takes 4 cycles including the handshake edge.
// Registers (spec CSR map + 구현 명확화; tsr_sim.c tsr_sim_csr_read/_write is the reference):
//   ID, VERSION, CFG0..2, CAPS (constants from parameters), CTRL (write-only, reads 0),
//   STATUS (BUSY, DONE, ERR, WAITING=0; W1C on DONE/ERR only), CMD_ADDR_LO/HI, CMD_COUNT, ARG0..7
//   (RW), ERR_PC/ERR_CODE/ERR_INFO (R), PC (R, = cur_pc), PERF_* (R, 64-bit, LO/HI).
// CTRL write, bits handled in the order SOFT_RESET, START, ABORT (same as the simulator):
//   SOFT_RESET, not BUSY: STATUS, ERR_*, PERF_* := 0, abort flag := 0, pulse run_soft_reset
//     (cmdproc clears R1..R15 and PC); CMD_ADDR/CMD_COUNT/ARG* kept.
//   SOFT_RESET while BUSY: becomes an abort; the reset is applied when the run ends (run_finish),
//     instead of recording DONE/ERR (docs/rtl-m2.md §6, spec proposal P3). START in the same write
//     is ignored (BUSY).
//   START, not BUSY (after the reset above, if any): STATUS := BUSY, ERR_* := 0, abort flag := 0,
//     pulse run_start with run_cmd_addr/run_cmd_count/run_args (valid in that cycle only).
//   START while BUSY: ignored.
//   ABORT: abort flag := 1 (run_abort level). Cleared by an accepted START and by SOFT_RESET, so
//     START|ABORT in one write aborts the new run before its first instruction (as the simulator).
// run_finish (from cmdproc, core quiescent): STATUS := DONE | (code != 0 ? ERR : 0), BUSY := 0,
//   ERR_CODE/ERR_PC/ERR_INFO := run_err_*. done = STATUS.DONE (registered level).
//   A CSR write performed at the same edge as run_finish still sees BUSY = 1 (START ignored,
//   SOFT_RESET deferred and applied at once by the finishing run, W1C acts on DONE/ERR = 0).
// PERF (cleared by reset and SOFT_RESET only, never by START):
//   CYCLES += BUSY; RETIRED += perf_retire; MVM_CYCLES += perf_mvm_busy; VU, WAITM = 0;
//   CH_RD_BEATS[c] += perf_ch_rd_beat[c]; CH_WAIT[c] += perf_ch_wait[c].
//   A read of counter i's LO half returns LO and snapshots HI (both sampled in S1); the next read
//   of i's HI half returns the snapshot and consumes it (tear-free 64-bit reads: read LO first,
//   then HI). A HI read without a pending snapshot of that counter returns the live HI half.

module tsr_csr
  import tsr_pkg::*;
#(
  parameter int unsigned NUM_CH    = TSR_DEF_NUM_CH,
  parameter int unsigned LANES     = TSR_DEF_LANES,
  parameter int unsigned T_MAX     = TSR_DEF_T_MAX,
  parameter int unsigned K_MAX     = TSR_DEF_K_MAX,
  parameter int unsigned SPM_BYTES = TSR_DEF_SPM_BYTES,
  parameter int unsigned VL        = TSR_DEF_VL,
  parameter int unsigned NSREG     = TSR_DEF_NSREG,
  parameter logic [31:0] CAPS      = 32'd0
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // ---- CSR port (core boundary) ----
  input  logic                    csr_valid,
  output logic                    csr_ready,
  input  logic                    csr_we,
  input  logic [TSR_CSR_AW-1:0]   csr_addr,
  input  logic [31:0]             csr_wdata,
  output logic [31:0]             csr_rdata,
  output logic                    done,

  // ---- run control to tsr_cmdproc ----
  output logic                    run_start,
  output logic [63:0]             run_cmd_addr,
  output logic [31:0]             run_cmd_count,
  output logic [7:0][31:0]        run_args,
  output logic                    run_abort,
  output logic                    run_soft_reset,

  // ---- run status from tsr_cmdproc ----
  input  logic                    run_finish,
  input  logic [TSR_ERR_W-1:0]    run_err_code,
  input  logic [31:0]             run_err_pc,
  input  logic [31:0]             run_err_info,
  input  logic [31:0]             cur_pc,

  // ---- perf events ----
  input  logic                    perf_retire,
  input  logic                    perf_mvm_busy,
  input  logic [NUM_CH-1:0]       perf_ch_rd_beat,
  input  logic [NUM_CH-1:0]       perf_ch_wait
);

  // -------------------------------------------------------------------------------------------
  // Constants
  // -------------------------------------------------------------------------------------------
  localparam int unsigned NPERF  = TSR_PERF_CH_RD_BEATS + 2 * NUM_CH;  // tsr_perf_count(NUM_CH)
  localparam int unsigned PSEL_W = $clog2(NPERF);                      // counter select bits
  localparam int unsigned NPERF2 = 1 << PSEL_W;                        // padded to a power of 2
  localparam int unsigned PIDX_W = TSR_CSR_AW - 3;                     // counter index = addr[11:3] - 0x20
  localparam logic [PIDX_W-1:0] PERF_IDX0 = PIDX_W'(TSR_CSR_PERF_BASE >> 3);

  localparam logic [31:0] VERSION_VAL = {16'd0, TSR_ISA_MAJOR, TSR_ISA_MINOR};
  localparam logic [31:0] CFG0_VAL    = {8'd0, 4'(T_MAX), 12'(LANES), 8'(NUM_CH)};
  localparam logic [31:0] CFG1_VAL    = 32'(SPM_BYTES / TSR_LINE_BYTES);
  localparam logic [31:0] CFG2_VAL    = {16'(K_MAX / 256), 8'(NSREG), 8'(VL)};

  if (NPERF > (1 << PIDX_W) - (int'(TSR_CSR_PERF_BASE) >> 3)) begin : g_chk_perf
    $fatal(1, "tsr_csr: PERF_* counters do not fit the CSR address space");
  end

  // -------------------------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------------------------
  logic                 busy_q, done_q, err_q;
  logic [31:0]          cmd_addr_lo_q, cmd_addr_hi_q, cmd_count_q;
  logic [7:0][31:0]     args_q;
  logic [TSR_ERR_W-1:0] err_code_q;
  logic [31:0]          err_pc_q, err_info_q;
  logic                 abort_q;       // CTRL.ABORT seen (run_abort)
  logic                 srst_pend_q;   // SOFT_RESET written while BUSY: apply at run end
  logic                 run_start_q, run_srst_q;
  logic [63:0]          perf_q [NPERF2];
  logic [31:0]          snap_hi_q;     // HI half captured by the last LO read
  logic                 snap_v_q;
  logic [PSEL_W-1:0]    snap_idx_q;

  // access pipeline
  logic                 req_q;         // an access is in progress (C .. S2)
  logic                 s1_q;          // S1 done: S2 at the next edge
  logic [TSR_CSR_AW-1:0] a_addr_q;
  logic                 a_we_q;
  logic [31:0]          a_wdata_q;
  logic                 r_perf_q, r_hi_q;
  logic [PSEL_W-1:0]    r_idx_q;
  logic [63:0]          r_perf64_q;
  logic [31:0]          r_reg_q;
  logic                 csr_ready_q;
  logic [31:0]          csr_rdata_q;

  // -------------------------------------------------------------------------------------------
  // Access decode (from the captured request)
  // -------------------------------------------------------------------------------------------
  wire cap     = csr_valid && !req_q && !csr_ready_q;
  wire st1     = req_q && !s1_q;
  wire aligned = (a_addr_q[1:0] == 2'b00);
  wire wr      = st1 && a_we_q && aligned;
  wire rd      = st1 && !a_we_q && aligned;

  wire is_ctrl    = (a_addr_q == TSR_CSR_CTRL);
  wire is_status  = (a_addr_q == TSR_CSR_STATUS);
  wire is_addr_lo = (a_addr_q == TSR_CSR_CMD_ADDR_LO);
  wire is_addr_hi = (a_addr_q == TSR_CSR_CMD_ADDR_HI);
  wire is_count   = (a_addr_q == TSR_CSR_CMD_COUNT);
  wire is_arg     = (a_addr_q[TSR_CSR_AW-1:5] == TSR_CSR_ARG0[TSR_CSR_AW-1:5]);   // 0x060..0x07C

  // PERF window: addr >= 0x100 and counter index < NPERF; HI half when addr[2].
  wire [PIDX_W-1:0] perf_idx = a_addr_q[TSR_CSR_AW-1:3] - PERF_IDX0;
  wire              perf_hit = (a_addr_q >= TSR_CSR_PERF_BASE) && (perf_idx < PIDX_W'(NPERF));

  wire srst_req  = wr && is_ctrl && a_wdata_q[TSR_CTRL_SOFT_RESET_BIT];
  wire start_req = wr && is_ctrl && a_wdata_q[TSR_CTRL_START_BIT];
  wire abort_req = wr && is_ctrl && a_wdata_q[TSR_CTRL_ABORT_BIT];
  wire w1c_req   = wr && is_status;

  // Soft reset now: SOFT_RESET written while idle, or a run ends with a reset pending (or with
  // one written at the very edge it ends).
  wire do_reset  = busy_q ? (run_finish && (srst_pend_q || srst_req)) : srst_req;
  wire do_start  = !busy_q && start_req;
  wire do_finish = busy_q && run_finish && !do_reset;

  // S1 register read value (everything but PERF).
  logic [31:0] reg_val;
  always_comb begin
    reg_val = 32'd0;
    case (a_addr_q)
      TSR_CSR_ID:          reg_val = TSR_ID_MAGIC;
      TSR_CSR_VERSION:     reg_val = VERSION_VAL;
      TSR_CSR_CFG0:        reg_val = CFG0_VAL;
      TSR_CSR_CFG1:        reg_val = CFG1_VAL;
      TSR_CSR_CFG2:        reg_val = CFG2_VAL;
      TSR_CSR_CAPS:        reg_val = CAPS;
      TSR_CSR_STATUS:      reg_val = {28'd0, 1'b0, err_q, done_q, busy_q};   // WAITING = 0
      TSR_CSR_CMD_ADDR_LO: reg_val = cmd_addr_lo_q;
      TSR_CSR_CMD_ADDR_HI: reg_val = cmd_addr_hi_q;
      TSR_CSR_CMD_COUNT:   reg_val = cmd_count_q;
      TSR_CSR_ERR_PC:      reg_val = err_pc_q;
      TSR_CSR_ERR_CODE:    reg_val = 32'(err_code_q);
      TSR_CSR_ERR_INFO:    reg_val = err_info_q;
      TSR_CSR_PC:          reg_val = cur_pc;
      default:             if (is_arg) reg_val = args_q[a_addr_q[4:2]];
    endcase
  end

  wire snap_match = snap_v_q && (snap_idx_q == r_idx_q);

  // -------------------------------------------------------------------------------------------
  // CSR port, access pipeline, host registers
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      req_q         <= 1'b0;
      s1_q          <= 1'b0;
      csr_ready_q   <= 1'b0;
      csr_rdata_q   <= 32'd0;
      cmd_addr_lo_q <= 32'd0;
      cmd_addr_hi_q <= 32'd0;
      cmd_count_q   <= 32'd0;
      args_q        <= '0;
      snap_v_q      <= 1'b0;
    end else begin
      // C: capture
      if (cap) begin
        req_q     <= 1'b1;
        a_addr_q  <= csr_addr;
        a_we_q    <= csr_we;
        a_wdata_q <= csr_wdata;
      end
      // S1: writes; read selection
      if (st1) begin
        s1_q       <= 1'b1;
        r_perf_q   <= rd && perf_hit;
        r_hi_q     <= a_addr_q[2];
        r_idx_q    <= perf_idx[PSEL_W-1:0];
        r_perf64_q <= perf_q[perf_idx[PSEL_W-1:0]];
        r_reg_q    <= rd ? reg_val : 32'd0;
      end
      if (wr && is_addr_lo) cmd_addr_lo_q <= a_wdata_q;
      if (wr && is_addr_hi) cmd_addr_hi_q <= a_wdata_q;
      if (wr && is_count)   cmd_count_q   <= a_wdata_q;
      if (wr && is_arg)     args_q[a_addr_q[4:2]] <= a_wdata_q;
      // S2: read data, ready strobe, HI snapshot
      csr_ready_q <= s1_q;
      if (s1_q) begin
        req_q <= 1'b0;
        s1_q  <= 1'b0;
        if (!r_perf_q)    csr_rdata_q <= r_reg_q;
        else if (!r_hi_q) csr_rdata_q <= r_perf64_q[31:0];
        else              csr_rdata_q <= snap_match ? snap_hi_q : r_perf64_q[63:32];
      end
      if (do_reset) begin
        snap_v_q <= 1'b0;
      end else if (s1_q && r_perf_q && !r_hi_q) begin
        snap_v_q   <= 1'b1;
        snap_idx_q <= r_idx_q;
        snap_hi_q  <= r_perf64_q[63:32];
      end else if (s1_q && r_perf_q && r_hi_q && snap_match) begin
        snap_v_q <= 1'b0;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Run status and control
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      busy_q      <= 1'b0;
      done_q      <= 1'b0;
      err_q       <= 1'b0;
      err_code_q  <= '0;
      err_pc_q    <= 32'd0;
      err_info_q  <= 32'd0;
      abort_q     <= 1'b0;
      srst_pend_q <= 1'b0;
      run_start_q <= 1'b0;
      run_srst_q  <= 1'b0;
    end else begin
      run_start_q <= do_start;
      run_srst_q  <= do_reset;
      // STATUS W1C (DONE, ERR only); a run ending at this edge sets them afterwards.
      if (w1c_req) begin
        if (a_wdata_q[TSR_STATUS_DONE_BIT]) done_q <= 1'b0;
        if (a_wdata_q[TSR_STATUS_ERR_BIT])  err_q  <= 1'b0;
      end
      if (busy_q) begin
        if (srst_req) begin               // SOFT_RESET while BUSY: abort now, reset at the end
          srst_pend_q <= 1'b1;
          abort_q     <= 1'b1;
        end
        if (abort_req) abort_q <= 1'b1;
        if (run_finish) busy_q <= 1'b0;
        if (do_finish) begin
          done_q     <= 1'b1;
          err_q      <= (run_err_code != TSR_ERR_NONE);
          err_code_q <= run_err_code;
          err_pc_q   <= run_err_pc;
          err_info_q <= run_err_info;
        end
      end else begin
        if (abort_req) abort_q <= 1'b1;
      end
      if (do_reset) begin
        busy_q      <= 1'b0;
        done_q      <= 1'b0;
        err_q       <= 1'b0;
        err_code_q  <= '0;
        err_pc_q    <= 32'd0;
        err_info_q  <= 32'd0;
        abort_q     <= 1'b0;
        srst_pend_q <= 1'b0;
      end
      if (do_start) begin                 // after the reset of the same write, if any
        busy_q     <= 1'b1;
        done_q     <= 1'b0;
        err_q      <= 1'b0;
        err_code_q <= '0;
        err_pc_q   <= 32'd0;
        err_info_q <= 32'd0;
        abort_q    <= abort_req;          // START clears the flag; ABORT of the same write sets it
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // PERF_* counters (slots >= NPERF only pad the read mux to a power of 2)
  // -------------------------------------------------------------------------------------------
  logic [NPERF-1:0] perf_inc;
  always_comb begin
    perf_inc = '0;
    perf_inc[TSR_PERF_CYCLES]     = busy_q;
    perf_inc[TSR_PERF_RETIRED]    = perf_retire;
    perf_inc[TSR_PERF_MVM_CYCLES] = perf_mvm_busy;
    for (int unsigned c = 0; c < NUM_CH; c++) begin
      perf_inc[TSR_PERF_CH_RD_BEATS + c]          = perf_ch_rd_beat[c];
      perf_inc[TSR_PERF_CH_RD_BEATS + NUM_CH + c] = perf_ch_wait[c];
    end
  end

  for (genvar i = 0; i < NPERF2; i++) begin : g_perf
    if (i >= NPERF || i == TSR_PERF_VU_CYCLES || i == TSR_PERF_WAITM_CYCLES) begin : g_zero
      assign perf_q[i] = 64'd0;           // no VU / WAITM in this core; padding
    end else begin : g_cnt
      always_ff @(posedge clk) begin
        if (!rst_n || do_reset) perf_q[i] <= 64'd0;
        else if (perf_inc[i])   perf_q[i] <= perf_q[i] + 64'd1;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Outputs
  // -------------------------------------------------------------------------------------------
  assign csr_ready      = csr_ready_q;
  assign csr_rdata      = csr_rdata_q;
  assign done           = done_q;
  assign run_start      = run_start_q;
  assign run_cmd_addr   = {cmd_addr_hi_q, cmd_addr_lo_q};
  assign run_cmd_count  = cmd_count_q;
  assign run_args       = args_q;
  assign run_abort      = abort_q;
  assign run_soft_reset = run_srst_q;

  // perf_inc bits of the VU / WAITM slots are constant 0 (no counter behind them).
  wire unused_ok = &{1'b0, perf_inc[TSR_PERF_VU_CYCLES], perf_inc[TSR_PERF_WAITM_CYCLES]};

endmodule
