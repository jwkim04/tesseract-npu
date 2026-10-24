// tsr_xrt_ctrl.sv — control bridge of the Alveo XRT platform adapter: the kernel's s_axi_control
// (AXI4-Lite slave, 32-bit data, 13-bit address) with UG1393 ap_ctrl_hs semantics, the kernel
// arguments, the run sequencer, ADAPTER_STATUS and the window onto the tsr_core CSR port.
//
// Owner: adapter agent (docs/platform-u200.md §2, §3; register map = the lead's kernel contract).
//
// Register map (byte offsets; reads of unmapped offsets return 0, writes to them are ignored;
// offsets are decoded on bits [12:2]; every response is OKAY):
//   0x000 AP_CTRL   bit0 ap_start (RW: a write of 1 sets it; cleared when the adapter accepts
//                   the start), bit1 ap_done (R, clear on read), bit2 ap_idle (R), bit3 ap_ready
//                   (R, clear on read), bit7 auto_restart (not supported: reads 0, writes ignored)
//   0x004 GIE       bit0 global interrupt enable (RW)
//   0x008 IER       bit0 ap_done interrupt enable (RW; bit1 ap_ready interrupt not supported)
//   0x00C ISR       bit0 ap_done interrupt status: set when a run completes while IER[0] = 1,
//                   toggled by a write of 1 (TOW)
//   0x010 / 0x014   cmd_addr lo / hi (ch0 beat address -> core CMD_ADDR_LO / HI)
//   0x018           cmd_count (-> CMD_COUNT); 0x01C reserved
//   0x020 + 4i      arg<i>, i < 8 (-> ARG0..7)
//   0x040 + 8c      ch<c>_base lo / hi, c < NUM_CH: device byte address of the buffer bound to
//                   m_axi_ch<c> (must be 4 KiB aligned); latched at the start of a run
//   0x0F0           ADAPTER_STATUS (R / W1C, sticky): bit0 RRESP error, bit1 BRESP error, bit2
//                   response with no outstanding request for its id, bit3 base not 4 KiB aligned
//                   at start; bits [31:16] saturating count of error responses (R beats and B
//                   with RRESP / BRESP != OKAY). W1C: a 1 in bit k (k < 4) clears bit k, a 1 in
//                   any of bits [31:16] clears the count. A new event in the same cycle wins.
//   0x0F4           ADAPTER_VERSION (R) = {16'h0001, 8'h00, NUM_CH}
//   0x1000 + off    tsr_core CSR port, off < 0x1000 (spec CSR map), at any time, also while a run
//                   is in progress. Writes need WSTRB = 4'hF (others are dropped: the CSR port
//                   has no byte enables).
// Writes use the byte strobes (RW registers); AP_CTRL / ISR / ADAPTER_STATUS act on the bytes
// whose strobe is set.
//
// ap_ctrl_hs (UG1393; XRT polls AP_CTRL or takes the interrupt):
//   - ap_start while idle: the adapter accepts the start — ap_start := 0 and ap_idle := 0 in the
//     same edge (XRT treats "ap_start = 0 and (ap_done or ap_idle)" as completion, so the two
//     never show start = 0 with idle = 1 during a run) — latches the channel bases and runs the
//     start sequence on the core CSR port: CMD_ADDR_LO, CMD_ADDR_HI, CMD_COUNT, ARG0..7 (from the
//     registers above), STATUS := DONE | ERR (W1C), CTRL := START. A misaligned base (any
//     ch<c>_base[11:0] != 0) instead sets ADAPTER_STATUS bit3 and completes the handshake at once
//     without touching the core (no AXI burst may cross a 4 KiB boundary).
//   - completion: the first cycle after the START write in which the core's `done` (=
//     STATUS.DONE, also with ERR) is 1: ap_done := 1, ap_ready := 1, ap_idle := 1, ISR[0] := 1 if
//     IER[0]. If the host wrote CTRL.SOFT_RESET through the window during the run (the run then
//     ends without DONE, spec 구현 명확화), the adapter also polls STATUS every 16 cycles and
//     completes when DONE = 1 or BUSY = 0.
//   - ap_start written while a run is in progress stays pending (reads 1) and is accepted when
//     the adapter is idle again: a queued run with the register values of that moment. XRT does
//     not do this for ap_ctrl_hs kernels (one command per compute unit at a time).
//   - interrupt = GIE & IER[0] & ISR[0] (registered).
// The adapter's own CSR transfers and the host window accesses share the core CSR port through a
// round-robin arbiter (one transfer at a time, docs/rtl-m2.md §6 protocol, one idle cycle between
// transfers). The AXI4-Lite slave handles one transaction at a time (reads and writes alternate
// when both are pending); AWREADY / WREADY are raised together once both AWVALID and WVALID are
// seen. Every output is a flop. Reset: everything to its reset value, ap_idle = 1.

module tsr_xrt_ctrl
  import tsr_pkg::*;
#(
  parameter int unsigned NUM_CH  = TSR_DEF_NUM_CH,
  parameter int unsigned AXIL_AW = 13
) (
  input  logic                       clk,
  input  logic                       rst_n,

  // ---- s_axi_control (AXI4-Lite slave) ----
  input  logic                       s_awvalid,
  output logic                       s_awready,
  input  logic [AXIL_AW-1:0]         s_awaddr,
  input  logic                       s_wvalid,
  output logic                       s_wready,
  input  logic [31:0]                s_wdata,
  input  logic [3:0]                 s_wstrb,
  output logic                       s_bvalid,
  input  logic                       s_bready,
  output logic [1:0]                 s_bresp,
  input  logic                       s_arvalid,
  output logic                       s_arready,
  input  logic [AXIL_AW-1:0]         s_araddr,
  output logic                       s_rvalid,
  input  logic                       s_rready,
  output logic [31:0]                s_rdata,
  output logic [1:0]                 s_rresp,
  output logic                       irq,              // kernel `interrupt`

  // ---- tsr_core CSR port (master side) and completion ----
  output logic                       csr_valid,
  input  logic                       csr_ready,
  output logic                       csr_we,
  output logic [TSR_CSR_AW-1:0]      csr_addr,
  output logic [31:0]                csr_wdata,
  input  logic [31:0]                csr_rdata,
  input  logic                       core_done,

  // ---- channel bridges ----
  output logic [NUM_CH-1:0][63:0]    ch_base,          // latched at the start of a run
  input  logic [NUM_CH-1:0]          ev_rresp_err,
  input  logic [NUM_CH-1:0]          ev_bresp_err,
  input  logic [NUM_CH-1:0]          ev_unexpected
);

  if (AXIL_AW != 13) begin : g_chk_aw
    $fatal(1, "tsr_xrt_ctrl: the control range is 0x2000 (13 address bits)");
  end
  if (NUM_CH < 1 || NUM_CH > 16) begin : g_chk_ch
    $fatal(1, "tsr_xrt_ctrl: NUM_CH must be in [1, 16]");
  end

  localparam logic [31:0] ADAPTER_VERSION = {16'h0001, 8'h00, 8'(NUM_CH)};

  // Register offsets (byte address bits [11:0] of the adapter half, bit 12 = 0)
  localparam logic [11:0] A_AP_CTRL  = 12'h000;
  localparam logic [11:0] A_GIE      = 12'h004;
  localparam logic [11:0] A_IER      = 12'h008;
  localparam logic [11:0] A_ISR      = 12'h00C;
  localparam logic [11:0] A_CMD_LO   = 12'h010;
  localparam logic [11:0] A_CMD_HI   = 12'h014;
  localparam logic [11:0] A_CMD_CNT  = 12'h018;
  localparam logic [11:0] A_ARG0     = 12'h020;
  localparam logic [11:0] A_BASE0    = 12'h040;
  localparam logic [11:0] A_ASTATUS  = 12'h0F0;
  localparam logic [11:0] A_AVERSION = 12'h0F4;

  // core STATUS bits (spec CSR map)
  localparam logic [31:0] ST_BUSY = 32'h1, ST_DONE = 32'h2, ST_ERR = 32'h4;
  localparam int unsigned SEQ_LAST = 12;      // steps 0..12 (see seq_step below)

  function automatic logic [31:0] bmask(logic [31:0] old, logic [31:0] nw, logic [3:0] be);
    logic [31:0] r;
    for (int unsigned b = 0; b < 4; b++) r[8*b +: 8] = be[b] ? nw[8*b +: 8] : old[8*b +: 8];
    return r;
  endfunction

  // =============================================================================================
  // AXI4-Lite slave: one transaction at a time
  // =============================================================================================
  typedef enum logic [2:0] {AX_IDLE, AX_ACC, AX_EXEC, AX_CSR, AX_RESP} ax_st_t;

  ax_st_t             ax_st;
  logic               ax_wr_q;            // current transaction is a write
  logic               ax_rd_pref_q;       // both pending: the read goes first
  logic [AXIL_AW-1:0] ax_addr_q;
  logic [31:0]        ax_wdata_q;
  logic [3:0]         ax_wstrb_q;
  logic               awready_q, wready_q, arready_q, bvalid_q, rvalid_q;
  logic [31:0]        rdata_q;

  // host request to the CSR port arbiter
  logic                  host_req_q, host_we_q;
  logic [TSR_CSR_AW-1:0] host_addr_q;
  logic [31:0]           host_wdata_q;
  logic                  host_cpl, seq_cpl;    // transfer of that requester ends at this edge

  wire        ax_win   = ax_addr_q[AXIL_AW-1];
  wire [11:0] ax_off   = {ax_addr_q[11:2], 2'b00};
  wire        ax_exec  = (ax_st == AX_EXEC);
  wire        reg_wr   = ax_exec & ax_wr_q & ~ax_win;     // adapter register write performed now
  wire        reg_rd   = ax_exec & ~ax_wr_q & ~ax_win;    // adapter register read performed now
  wire        ctrl_rd  = reg_rd & (ax_off == A_AP_CTRL);  // clear-on-read of ap_done / ap_ready

  logic [31:0] reg_rdata;                                  // adapter register read mux (below)

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ax_st          <= AX_IDLE;
      ax_wr_q        <= 1'b0;
      ax_rd_pref_q   <= 1'b0;
      awready_q      <= 1'b0;
      wready_q       <= 1'b0;
      arready_q      <= 1'b0;
      bvalid_q       <= 1'b0;
      rvalid_q       <= 1'b0;
      host_req_q     <= 1'b0;
    end else begin
      unique case (ax_st)
        AX_IDLE: begin
          if (s_awvalid && s_wvalid && !(s_arvalid && ax_rd_pref_q)) begin
            awready_q <= 1'b1;
            wready_q  <= 1'b1;
            ax_wr_q   <= 1'b1;
            ax_st     <= AX_ACC;
          end else if (s_arvalid) begin
            arready_q <= 1'b1;
            ax_wr_q   <= 1'b0;
            ax_st     <= AX_ACC;
          end
        end
        AX_ACC: begin                   // the handshake(s) happen at this edge (VALID is held)
          awready_q    <= 1'b0;
          wready_q     <= 1'b0;
          arready_q    <= 1'b0;
          ax_rd_pref_q <= ax_wr_q;      // alternate when both kinds are pending
          ax_st        <= AX_EXEC;
        end
        AX_EXEC: begin
          if (ax_win && (!ax_wr_q || ax_wstrb_q == 4'hF)) begin
            host_req_q <= 1'b1;
            ax_st      <= AX_CSR;
          end else begin
            bvalid_q <= ax_wr_q;
            rvalid_q <= ~ax_wr_q;
            ax_st    <= AX_RESP;
          end
        end
        AX_CSR: begin
          if (host_cpl) begin
            host_req_q <= 1'b0;
            bvalid_q   <= ax_wr_q;
            rvalid_q   <= ~ax_wr_q;
            ax_st      <= AX_RESP;
          end
        end
        AX_RESP: begin
          if (bvalid_q && s_bready) begin
            bvalid_q <= 1'b0;
            ax_st    <= AX_IDLE;
          end
          if (rvalid_q && s_rready) begin
            rvalid_q <= 1'b0;
            ax_st    <= AX_IDLE;
          end
        end
        default: ax_st <= AX_IDLE;
      endcase
    end
  end

  // Datapath registers of the slave (no reset: qualified by the state).
  always_ff @(posedge clk) begin
    if (ax_st == AX_ACC) begin
      ax_addr_q  <= ax_wr_q ? s_awaddr : s_araddr;
      ax_wdata_q <= s_wdata;
      ax_wstrb_q <= ax_wr_q ? s_wstrb : 4'h0;
    end
    if (ax_exec) begin
      host_we_q    <= ax_wr_q;
      host_addr_q  <= ax_addr_q[TSR_CSR_AW-1:0];
      host_wdata_q <= ax_wdata_q;
    end
    if (reg_rd) rdata_q <= reg_rdata;
    else if (host_cpl) rdata_q <= csr_rdata;        // window read: the core's registered rdata
  end

  assign s_awready = awready_q;
  assign s_wready  = wready_q;
  assign s_arready = arready_q;
  assign s_bvalid  = bvalid_q;
  assign s_bresp   = 2'b00;
  assign s_rvalid  = rvalid_q;
  assign s_rdata   = rdata_q;
  assign s_rresp   = 2'b00;

  // =============================================================================================
  // Core CSR port: round-robin arbiter between the host window and the sequencer
  // =============================================================================================
  logic                  seq_req_q, seq_we_q;
  logic [TSR_CSR_AW-1:0] seq_addr_q;
  logic [31:0]           seq_wdata_q;

  logic csr_valid_q, csr_we_q, csr_own_seq_q, csr_last_seq_q, csr_gap_q;
  logic [TSR_CSR_AW-1:0] csr_addr_q;
  logic [31:0]           csr_wdata_q;

  wire csr_end   = csr_valid_q & csr_ready;
  assign host_cpl = csr_end & ~csr_own_seq_q;
  assign seq_cpl  = csr_end & csr_own_seq_q;
  // Grant when the port is free: alternate when both request (host_req_q / seq_req_q stay set
  // until their transfer ends, so a granted request is never granted twice).
  wire csr_free  = ~csr_valid_q & ~csr_gap_q;
  wire pick_seq  = seq_req_q & (~host_req_q | ~csr_last_seq_q);
  wire csr_grant = csr_free & (host_req_q | seq_req_q);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      csr_valid_q    <= 1'b0;
      csr_own_seq_q  <= 1'b0;
      csr_last_seq_q <= 1'b0;
      csr_gap_q      <= 1'b0;
    end else begin
      csr_gap_q <= csr_end;          // one idle cycle between transfers
      if (csr_grant) begin
        csr_valid_q    <= 1'b1;
        csr_own_seq_q  <= pick_seq;
        csr_last_seq_q <= pick_seq;
      end else if (csr_end) begin
        csr_valid_q <= 1'b0;   // the requester drops its request at this edge
      end
    end
  end

  always_ff @(posedge clk) begin
    if (csr_grant) begin
      csr_we_q    <= pick_seq ? seq_we_q : host_we_q;
      csr_addr_q  <= pick_seq ? seq_addr_q : host_addr_q;
      csr_wdata_q <= pick_seq ? seq_wdata_q : host_wdata_q;
    end
  end

  assign csr_valid = csr_valid_q;
  assign csr_we    = csr_we_q;
  assign csr_addr  = csr_addr_q;
  assign csr_wdata = csr_wdata_q;

  // =============================================================================================
  // Kernel argument registers, GIE / IER, ADAPTER_STATUS
  // =============================================================================================
  logic [63:0]                cmd_addr_q;
  logic [31:0]                cmd_count_q;
  logic [7:0][31:0]           arg_q;
  logic [NUM_CH-1:0][63:0]    base_reg_q;
  logic                       gie_q, ier_q;
  logic [3:0]                 ast_q;           // ADAPTER_STATUS sticky bits
  logic [15:0]                acnt_q;          // ADAPTER_STATUS error count
  logic                       misal_ev;        // run refused: misaligned base (from the sequencer)

  // Bridge events: registered here (they come from other SLRs), counted one cycle later.
  logic [NUM_CH-1:0]          evr_q, evb_q, evu_q;
  logic [5:0]                 nerr_q;          // error responses in the cycle before
  logic                       anyr_q, anyb_q, anyu_q;
  logic [3:0]                 ast_d;
  logic [15:0]                acnt_d;

  // ADAPTER_STATUS next value: W1C first, then this cycle's events (an event wins)
  always_comb begin
    logic [16:0] sum;
    logic [15:0] n;
    ast_d = ast_q;
    n     = acnt_q;
    if (reg_wr && ax_off == A_ASTATUS) begin
      if (ax_wstrb_q[0]) ast_d = ast_d & ~ax_wdata_q[3:0];
      if ((ax_wstrb_q[2] && ax_wdata_q[23:16] != 8'd0) || (ax_wstrb_q[3] && ax_wdata_q[31:24] != 8'd0))
        n = '0;
    end
    ast_d  = ast_d | {misal_ev, anyu_q, anyb_q, anyr_q};
    sum    = 17'(n) + 17'(nerr_q);
    acnt_d = sum[16] ? 16'hFFFF : sum[15:0];
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cmd_addr_q  <= '0;
      cmd_count_q <= '0;
      arg_q       <= '0;
      base_reg_q  <= '0;
      gie_q       <= 1'b0;
      ier_q       <= 1'b0;
      ast_q       <= '0;
      acnt_q      <= '0;
      evr_q       <= '0;
      evb_q       <= '0;
      evu_q       <= '0;
      nerr_q      <= '0;
      anyr_q      <= 1'b0;
      anyb_q      <= 1'b0;
      anyu_q      <= 1'b0;
    end else begin
      evr_q  <= ev_rresp_err;
      evb_q  <= ev_bresp_err;
      evu_q  <= ev_unexpected;
      anyr_q <= |evr_q;
      anyb_q <= |evb_q;
      anyu_q <= |evu_q;
      nerr_q <= 6'($countones(evr_q)) + 6'($countones(evb_q));
      if (reg_wr) begin
        unique case (ax_off)
          A_GIE:     if (ax_wstrb_q[0]) gie_q <= ax_wdata_q[0];
          A_IER:     if (ax_wstrb_q[0]) ier_q <= ax_wdata_q[0];
          A_CMD_LO:  cmd_addr_q[31:0]  <= bmask(cmd_addr_q[31:0], ax_wdata_q, ax_wstrb_q);
          A_CMD_HI:  cmd_addr_q[63:32] <= bmask(cmd_addr_q[63:32], ax_wdata_q, ax_wstrb_q);
          A_CMD_CNT: cmd_count_q       <= bmask(cmd_count_q, ax_wdata_q, ax_wstrb_q);
          default: ;
        endcase
        for (int unsigned i = 0; i < 8; i++) begin
          if (ax_off == A_ARG0 + 12'(4 * i)) arg_q[i] <= bmask(arg_q[i], ax_wdata_q, ax_wstrb_q);
        end
        for (int unsigned c = 0; c < NUM_CH; c++) begin
          if (ax_off == A_BASE0 + 12'(8 * c))
            base_reg_q[c][31:0] <= bmask(base_reg_q[c][31:0], ax_wdata_q, ax_wstrb_q);
          if (ax_off == A_BASE0 + 12'(8 * c + 4))
            base_reg_q[c][63:32] <= bmask(base_reg_q[c][63:32], ax_wdata_q, ax_wstrb_q);
        end
      end
      ast_q  <= ast_d;
      acnt_q <= acnt_d;
    end
  end

  // =============================================================================================
  // Run sequencer and ap_ctrl_hs state
  // =============================================================================================
  typedef enum logic [1:0] {RS_IDLE, RS_SEQ, RS_RUN, RS_FIN} rs_t;

  rs_t                     rs;
  logic [3:0]              step_q;          // start-sequence step, 0..SEQ_LAST
  logic                    poll_q;          // the outstanding sequencer request is a STATUS poll
  logic [3:0]              poll_wait_q;
  logic                    srst_seen_q;     // host wrote CTRL.SOFT_RESET during this run
  logic                    done_q;          // core_done, registered
  logic                    ap_start_q, ap_done_q, ap_idle_q, ap_ready_q, isr_q, irq_q;
  logic [NUM_CH-1:0][63:0] base_q;

  logic misaligned;
  always_comb begin
    misaligned = 1'b0;
    for (int unsigned c = 0; c < NUM_CH; c++) misaligned |= (base_reg_q[c][11:0] != 12'd0);
  end

  wire start_wr = reg_wr & (ax_off == A_AP_CTRL) & ax_wstrb_q[0] & ax_wdata_q[0];
  wire isr_tow  = reg_wr & (ax_off == A_ISR) & ax_wstrb_q[0] & ax_wdata_q[0];
  wire accept   = (rs == RS_IDLE) & ap_start_q & ~seq_req_q;
  wire srst_wr  = host_cpl & host_we_q & (host_addr_q == TSR_CSR_CTRL) & host_wdata_q[TSR_CTRL_SOFT_RESET_BIT];
  assign misal_ev = accept & misaligned;

  // the core CSR write of start-sequence step k
  function automatic logic [TSR_CSR_AW + 31:0] seq_step(logic [3:0] k);
    logic [TSR_CSR_AW-1:0] a;
    logic [31:0]           d;
    unique case (k)
      4'd0:    begin a = TSR_CSR_CMD_ADDR_LO; d = cmd_addr_q[31:0];  end
      4'd1:    begin a = TSR_CSR_CMD_ADDR_HI; d = cmd_addr_q[63:32]; end
      4'd2:    begin a = TSR_CSR_CMD_COUNT;   d = cmd_count_q;       end
      4'd11:   begin a = TSR_CSR_STATUS;      d = ST_DONE | ST_ERR;  end
      4'd12:   begin a = TSR_CSR_CTRL;        d = 32'(1) << TSR_CTRL_START_BIT; end
      default: begin a = TSR_CSR_ARG0 + TSR_CSR_AW'({k - 4'd3, 2'b00}); d = arg_q[3'(k - 4'd3)]; end
    endcase
    return {a, d};
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rs          <= RS_IDLE;
      step_q      <= '0;
      poll_q      <= 1'b0;
      poll_wait_q <= '0;
      srst_seen_q <= 1'b0;
      done_q      <= 1'b0;
      seq_req_q   <= 1'b0;
      ap_start_q  <= 1'b0;
      ap_done_q   <= 1'b0;
      ap_idle_q   <= 1'b1;
      ap_ready_q  <= 1'b0;
      isr_q       <= 1'b0;
      irq_q       <= 1'b0;
    end else begin
      done_q <= core_done;
      irq_q  <= gie_q & ier_q & isr_q;
      // AP_CTRL: a start write sets ap_start; clear-on-read of ap_done / ap_ready (a completion
      // in the same edge wins: the read returned the old value, the next read sees the new one)
      ap_start_q <= (ap_start_q & ~accept) | start_wr;
      if (ctrl_rd) begin
        ap_done_q  <= 1'b0;
        ap_ready_q <= 1'b0;
      end
      if (isr_tow) isr_q <= ~isr_q;
      if (rs != RS_IDLE && srst_wr) srst_seen_q <= 1'b1;
      if (seq_cpl) begin
        seq_req_q <= 1'b0;
        poll_q    <= 1'b0;
      end
      unique case (rs)
        RS_IDLE: begin
          if (accept) begin
            ap_idle_q <= 1'b0;
            step_q    <= '0;
            rs        <= misaligned ? RS_FIN : RS_SEQ;
          end
        end
        RS_SEQ: begin
          if (seq_cpl) begin
            step_q <= step_q + 4'd1;
            if (step_q == 4'(SEQ_LAST)) rs <= RS_RUN;
          end else if (!seq_req_q) begin
            seq_req_q <= 1'b1;
          end
        end
        RS_RUN: begin
          if (seq_cpl && poll_q && ((csr_rdata & ST_DONE) != 0 || (csr_rdata & ST_BUSY) == 0)) begin
            rs <= RS_FIN;
          end else if (done_q) begin
            rs <= RS_FIN;
          end else if (srst_seen_q && !seq_req_q) begin
            poll_wait_q <= poll_wait_q + 4'd1;
            if (poll_wait_q == 4'hF) begin
              seq_req_q <= 1'b1;
              poll_q    <= 1'b1;
            end
          end
        end
        RS_FIN: begin
          ap_done_q   <= 1'b1;
          ap_ready_q  <= 1'b1;
          ap_idle_q   <= 1'b1;
          srst_seen_q <= 1'b0;
          poll_wait_q <= '0;
          if (ier_q) isr_q <= 1'b1;
          rs <= RS_IDLE;
        end
        default: rs <= RS_IDLE;
      endcase
    end
  end

  // Sequencer request payload and latched bases (no reset: qualified by seq_req_q / the run).
  always_ff @(posedge clk) begin
    if (rs == RS_SEQ && !seq_req_q && !seq_cpl) begin
      seq_we_q <= 1'b1;
      {seq_addr_q, seq_wdata_q} <= seq_step(step_q);
    end else if (rs == RS_RUN && !seq_req_q) begin
      seq_we_q    <= 1'b0;
      seq_addr_q  <= TSR_CSR_STATUS;
      seq_wdata_q <= '0;
    end
    if (accept) base_q <= base_reg_q;
  end

  assign ch_base   = base_q;
  assign irq       = irq_q;

  // =============================================================================================
  // Adapter register read mux
  // =============================================================================================
  always_comb begin
    reg_rdata = '0;
    unique case (ax_off)
      A_AP_CTRL:  reg_rdata = {28'd0, ap_ready_q, ap_idle_q, ap_done_q, ap_start_q};
      A_GIE:      reg_rdata = {31'd0, gie_q};
      A_IER:      reg_rdata = {31'd0, ier_q};
      A_ISR:      reg_rdata = {31'd0, isr_q};
      A_CMD_LO:   reg_rdata = cmd_addr_q[31:0];
      A_CMD_HI:   reg_rdata = cmd_addr_q[63:32];
      A_CMD_CNT:  reg_rdata = cmd_count_q;
      A_ASTATUS:  reg_rdata = {acnt_q, 12'd0, ast_q};
      A_AVERSION: reg_rdata = ADAPTER_VERSION;
      default: ;
    endcase
    for (int unsigned i = 0; i < 8; i++) begin
      if (ax_off == A_ARG0 + 12'(4 * i)) reg_rdata = arg_q[i];
    end
    for (int unsigned c = 0; c < NUM_CH; c++) begin
      if (ax_off == A_BASE0 + 12'(8 * c))     reg_rdata = base_reg_q[c][31:0];
      if (ax_off == A_BASE0 + 12'(8 * c + 4)) reg_rdata = base_reg_q[c][63:32];
    end
  end

  // =============================================================================================
  // Simulation checks
  // =============================================================================================
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (ax_st == AX_ACC) begin
        assert (ax_wr_q ? (s_awvalid && s_wvalid) : s_arvalid)
          else $error("%m: AXI4-Lite VALID dropped before the handshake");
      end
      assert (!(csr_valid_q && csr_gap_q)) else $error("%m: CSR port: no idle cycle between transfers");
      assert (!(ap_idle_q && rs != RS_IDLE && rs != RS_FIN))
        else $error("%m: ap_idle while a run is in progress");
    end
  end
`endif

endmodule
