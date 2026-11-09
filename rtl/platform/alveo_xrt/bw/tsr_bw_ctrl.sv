// tsr_bw_ctrl.sv — control block of the M0 bandwidth kernel tsr_bw_k4: s_axi_control (AXI4-Lite
// slave, 32-bit data, 12-bit address) with UG1393 ap_ctrl_hs semantics, the kernel arguments, the
// run handshake with the channel engines and the result read path.
//
// Owner: m0bw agent (docs/bandwidth-kernel.md §2; the ap_ctrl_hs part follows the adapter's tsr_xrt_ctrl,
// docs/platform-u200.md §3).
//
// Register map (byte offsets, decoded on bits [11:2]; unmapped offsets read 0 and ignore writes;
// every response is OKAY; RW registers honour WSTRB):
//   0x000 AP_CTRL   bit0 ap_start (RW: a write of 1 sets it; cleared when the start is accepted),
//                   bit1 ap_done (R, clear on read), bit2 ap_idle (R), bit3 ap_ready (R, clear on
//                   read), bit7 auto_restart (not supported: reads 0, writes ignored)
//   0x004 GIE       bit0 global interrupt enable
//   0x008 IER       bit0 ap_done interrupt enable
//   0x00C ISR       bit0: set when a run completes while IER[0] = 1, toggled by a write of 1
//   0x010           seed (pattern seed, all channels); 0x014 .. 0x01C reserved
//   0x020 + 0x20c   channel c (c < NUM_CH): +0x00 ctrl, +0x04 reserved, +0x08 / +0x0C bytes lo / hi,
//                   +0x10 / +0x14 rd_off lo / hi, +0x18 / +0x1C wr_off lo / hi
//   0x0A0 + 8c      ch<c> lo / hi: device address of the buffer bound to m_axi_ch<c>
//   0x0F0           BW_VERSION (R) = {16'h4257 ("BW"), 8'h01, 8'(NUM_CH)}
//   0x0F4           BW_INFO (R) = {16'd0, 8'(ADDR_BEAT_W), 8'(AXI_SLICES)}
//   0x0F8 / 0x0FC   TOTAL_CYCLES lo / hi (R): cycles from the accepted start to ap_done
//   0x100 + 0x40c   channel c result block (R, 16 words; tsr_bw_engine), read through the
//                   engines' registered result ports (3 more cycles than the other registers)
// ap_ctrl_hs (as tsr_xrt_ctrl): ap_start while idle is accepted in the next cycle — ap_start := 0
// and ap_idle := 0 in the same edge — and every argument is latched (a snapshot: writes during a
// run do not change it); go_tgl toggles. The run is complete when every engine has returned the
// toggle (done_tgl[c] == go_tgl): ap_done := 1, ap_ready := 1, ap_idle := 1, ISR[0] := 1 if
// IER[0]; interrupt = GIE & IER[0] & ISR[0] (registered). ap_start written during a run stays
// pending and is accepted when idle. The AXI4-Lite slave handles one transaction at a time
// (reads and writes alternate when both are pending); every output is a flop.

module tsr_bw_ctrl #(
  parameter int unsigned NUM_CH      = 4,
  parameter int unsigned AXIL_AW     = 12,
  parameter int unsigned ADDR_BEAT_W = 28,   // reported in BW_INFO
  parameter int unsigned AXI_SLICES  = 2     // reported in BW_INFO
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // ---- s_axi_control (AXI4-Lite slave) ----
  input  logic                     s_awvalid,
  output logic                     s_awready,
  input  logic [AXIL_AW-1:0]       s_awaddr,
  input  logic                     s_wvalid,
  output logic                     s_wready,
  input  logic [31:0]              s_wdata,
  input  logic [3:0]               s_wstrb,
  output logic                     s_bvalid,
  input  logic                     s_bready,
  output logic [1:0]               s_bresp,
  input  logic                     s_arvalid,
  output logic                     s_arready,
  input  logic [AXIL_AW-1:0]       s_araddr,
  output logic                     s_rvalid,
  input  logic                     s_rready,
  output logic [31:0]              s_rdata,
  output logic [1:0]               s_rresp,
  output logic                     irq,

  // ---- channel engines ----
  output logic                     go_tgl,
  input  logic [NUM_CH-1:0]        done_tgl,
  output logic [31:0]              seed,
  output logic [NUM_CH-1:0][31:0]  ch_ctrl,
  output logic [NUM_CH-1:0][63:0]  ch_bytes,
  output logic [NUM_CH-1:0][63:0]  ch_rd_off,
  output logic [NUM_CH-1:0][63:0]  ch_wr_off,
  output logic [NUM_CH-1:0][63:0]  ch_base,
  output logic [3:0]               res_sel,
  input  logic [NUM_CH-1:0][31:0]  res_rdata
);

  if (AXIL_AW != 12) begin : g_chk_aw
    $fatal(1, "tsr_bw_ctrl: the control range is 0x1000 (12 address bits)");
  end
  if (NUM_CH < 1 || NUM_CH > 4) begin : g_chk_ch
    $fatal(1, "tsr_bw_ctrl: NUM_CH must be in [1, 4] (register map)");
  end

  localparam logic [31:0] BW_VERSION = {16'h4257, 8'h01, 8'(NUM_CH)};
  localparam logic [31:0] BW_INFO    = {16'd0, 8'(ADDR_BEAT_W), 8'(AXI_SLICES)};

  localparam logic [11:0] A_AP_CTRL = 12'h000;
  localparam logic [11:0] A_GIE     = 12'h004;
  localparam logic [11:0] A_IER     = 12'h008;
  localparam logic [11:0] A_ISR     = 12'h00C;
  localparam logic [11:0] A_SEED    = 12'h010;
  localparam logic [11:0] A_CH0     = 12'h020;   // + 0x20 c
  localparam logic [11:0] A_BUF0    = 12'h0A0;   // + 8 c
  localparam logic [11:0] A_VERSION = 12'h0F0;
  localparam logic [11:0] A_INFO    = 12'h0F4;
  localparam logic [11:0] A_TOT_LO  = 12'h0F8;
  localparam logic [11:0] A_TOT_HI  = 12'h0FC;
  localparam logic [11:0] A_RES0    = 12'h100;   // + 0x40 c

  function automatic logic [31:0] bmask(logic [31:0] old, logic [31:0] nw, logic [3:0] be);
    logic [31:0] r;
    for (int unsigned b = 0; b < 4; b++) r[8*b +: 8] = be[b] ? nw[8*b +: 8] : old[8*b +: 8];
    return r;
  endfunction

  // =============================================================================================
  // AXI4-Lite slave: one transaction at a time
  // =============================================================================================
  typedef enum logic [2:0] {AX_IDLE, AX_ACC, AX_EXEC, AX_R1, AX_R2, AX_R3, AX_RESP} ax_st_t;

  ax_st_t             ax_st;
  logic               ax_wr_q, ax_rd_pref_q;
  logic [AXIL_AW-1:0] ax_addr_q;
  logic [31:0]        ax_wdata_q;
  logic [3:0]         ax_wstrb_q;
  logic               awready_q, wready_q, arready_q, bvalid_q, rvalid_q;
  logic [31:0]        rdata_q;
  logic [1:0]         rch_q;                                 // result read: channel

  wire [11:0] ax_off  = {ax_addr_q[11:2], 2'b00};
  wire        ax_exec = (ax_st == AX_EXEC);
  wire        reg_wr  = ax_exec & ax_wr_q;
  wire        reg_rd  = ax_exec & ~ax_wr_q;
  wire        ctrl_rd = reg_rd & (ax_off == A_AP_CTRL);     // clear-on-read of ap_done / ap_ready
  // a result word of an existing channel: 0x100 .. 0x100 + 0x40 NUM_CH - 4
  wire        is_res  = (ax_off >= A_RES0) && (ax_off < A_RES0 + 12'(64 * NUM_CH));

  logic [31:0]              reg_rdata;
  logic [NUM_CH-1:0][31:0]  res_in_q;                         // engines' result words, registered
  logic [3:0][31:0]         res_pad;                          // res_in_q, padded to 4 channels
  assign res_pad = 128'(res_in_q);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ax_st        <= AX_IDLE;
      ax_wr_q      <= 1'b0;
      ax_rd_pref_q <= 1'b0;
      awready_q    <= 1'b0;
      wready_q     <= 1'b0;
      arready_q    <= 1'b0;
      bvalid_q     <= 1'b0;
      rvalid_q     <= 1'b0;
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
          if (!ax_wr_q && is_res) begin
            ax_st <= AX_R1;
          end else begin
            bvalid_q <= ax_wr_q;
            rvalid_q <= ~ax_wr_q;
            ax_st    <= AX_RESP;
          end
        end
        AX_R1: ax_st <= AX_R2;          // engine result mux -> engine register
        AX_R2: ax_st <= AX_R3;          // -> res_in_q
        AX_R3: begin                    // -> rdata_q
          rvalid_q <= 1'b1;
          ax_st    <= AX_RESP;
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
    if (reg_rd) begin
      rdata_q <= reg_rdata;
      res_sel <= ax_off[5:2];
      rch_q   <= 2'(ax_off[7:6]);
    end else if (ax_st == AX_R3) begin
      rdata_q <= res_pad[rch_q];
    end
    res_in_q <= res_rdata;
  end

  wire unused_ok = &{1'b0, ax_addr_q[1:0]};

  assign s_awready = awready_q;
  assign s_wready  = wready_q;
  assign s_arready = arready_q;
  assign s_bvalid  = bvalid_q;
  assign s_bresp   = 2'b00;
  assign s_rvalid  = rvalid_q;
  assign s_rdata   = rdata_q;
  assign s_rresp   = 2'b00;

  // =============================================================================================
  // Argument registers (as written by the host) and GIE / IER
  // =============================================================================================
  logic [31:0]              seed_q;
  logic [NUM_CH-1:0][31:0]  ctl_q;
  logic [NUM_CH-1:0][63:0]  bytes_q, rdo_q, wro_q, buf_q;
  logic                     gie_q, ier_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      seed_q  <= '0;
      ctl_q   <= '0;
      bytes_q <= '0;
      rdo_q   <= '0;
      wro_q   <= '0;
      buf_q   <= '0;
      gie_q   <= 1'b0;
      ier_q   <= 1'b0;
    end else if (reg_wr) begin
      if (ax_off == A_GIE && ax_wstrb_q[0]) gie_q <= ax_wdata_q[0];
      if (ax_off == A_IER && ax_wstrb_q[0]) ier_q <= ax_wdata_q[0];
      if (ax_off == A_SEED) seed_q <= bmask(seed_q, ax_wdata_q, ax_wstrb_q);
      for (int unsigned c = 0; c < NUM_CH; c++) begin
        if (ax_off == A_CH0 + 12'(32 * c))        ctl_q[c]          <= bmask(ctl_q[c], ax_wdata_q, ax_wstrb_q);
        if (ax_off == A_CH0 + 12'(32 * c + 8))    bytes_q[c][31:0]  <= bmask(bytes_q[c][31:0], ax_wdata_q, ax_wstrb_q);
        if (ax_off == A_CH0 + 12'(32 * c + 12))   bytes_q[c][63:32] <= bmask(bytes_q[c][63:32], ax_wdata_q, ax_wstrb_q);
        if (ax_off == A_CH0 + 12'(32 * c + 16))   rdo_q[c][31:0]    <= bmask(rdo_q[c][31:0], ax_wdata_q, ax_wstrb_q);
        if (ax_off == A_CH0 + 12'(32 * c + 20))   rdo_q[c][63:32]   <= bmask(rdo_q[c][63:32], ax_wdata_q, ax_wstrb_q);
        if (ax_off == A_CH0 + 12'(32 * c + 24))   wro_q[c][31:0]    <= bmask(wro_q[c][31:0], ax_wdata_q, ax_wstrb_q);
        if (ax_off == A_CH0 + 12'(32 * c + 28))   wro_q[c][63:32]   <= bmask(wro_q[c][63:32], ax_wdata_q, ax_wstrb_q);
        if (ax_off == A_BUF0 + 12'(8 * c))        buf_q[c][31:0]    <= bmask(buf_q[c][31:0], ax_wdata_q, ax_wstrb_q);
        if (ax_off == A_BUF0 + 12'(8 * c + 4))    buf_q[c][63:32]   <= bmask(buf_q[c][63:32], ax_wdata_q, ax_wstrb_q);
      end
    end
  end

  // =============================================================================================
  // ap_ctrl_hs, run handshake, total cycles
  // =============================================================================================
  typedef enum logic [1:0] {RS_IDLE, RS_RUN, RS_FIN} rs_t;

  rs_t               rs;
  logic              ap_start_q, ap_done_q, ap_idle_q, ap_ready_q, isr_q, irq_q, go_q;
  logic [NUM_CH-1:0] done_in_q;
  logic [63:0]       tot_q;

  wire start_wr = reg_wr & (ax_off == A_AP_CTRL) & ax_wstrb_q[0] & ax_wdata_q[0];
  wire isr_tow  = reg_wr & (ax_off == A_ISR) & ax_wstrb_q[0] & ax_wdata_q[0];
  wire accept   = (rs == RS_IDLE) & ap_start_q;
  wire all_done = (done_in_q == {NUM_CH{go_q}});

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rs         <= RS_IDLE;
      ap_start_q <= 1'b0;
      ap_done_q  <= 1'b0;
      ap_idle_q  <= 1'b1;
      ap_ready_q <= 1'b0;
      isr_q      <= 1'b0;
      irq_q      <= 1'b0;
      go_q       <= 1'b0;
      done_in_q  <= '0;
      tot_q      <= '0;
    end else begin
      done_in_q  <= done_tgl;
      irq_q      <= gie_q & ier_q & isr_q;
      // a completion in the same edge as a clear-on-read wins (the read returned the old value)
      ap_start_q <= (ap_start_q & ~accept) | start_wr;
      if (ctrl_rd) begin
        ap_done_q  <= 1'b0;
        ap_ready_q <= 1'b0;
      end
      if (isr_tow) isr_q <= ~isr_q;
      unique case (rs)
        RS_IDLE: begin
          if (accept) begin
            ap_idle_q <= 1'b0;
            go_q      <= ~go_q;
            tot_q     <= '0;
            rs        <= RS_RUN;
          end
        end
        RS_RUN: begin
          tot_q <= tot_q + 64'd1;
          if (all_done) rs <= RS_FIN;
        end
        RS_FIN: begin
          ap_done_q  <= 1'b1;
          ap_ready_q <= 1'b1;
          ap_idle_q  <= 1'b1;
          if (ier_q) isr_q <= 1'b1;
          rs <= RS_IDLE;
        end
        default: rs <= RS_IDLE;
      endcase
    end
  end

  // Argument snapshot at the accepted start (no reset: the engines latch it after go_tgl).
  logic [31:0]              seed_s;
  logic [NUM_CH-1:0][31:0]  ctl_s;
  logic [NUM_CH-1:0][63:0]  bytes_s, rdo_s, wro_s, buf_s;
  always_ff @(posedge clk) begin
    if (accept) begin
      seed_s  <= seed_q;
      ctl_s   <= ctl_q;
      bytes_s <= bytes_q;
      rdo_s   <= rdo_q;
      wro_s   <= wro_q;
      buf_s   <= buf_q;
    end
  end

  assign go_tgl    = go_q;
  assign seed      = seed_s;
  assign ch_ctrl   = ctl_s;
  assign ch_bytes  = bytes_s;
  assign ch_rd_off = rdo_s;
  assign ch_wr_off = wro_s;
  assign ch_base   = buf_s;
  assign irq       = irq_q;

  // =============================================================================================
  // Register read mux (results: through res_in_q, see above)
  // =============================================================================================
  always_comb begin
    reg_rdata = '0;
    unique case (ax_off)
      A_AP_CTRL: reg_rdata = {28'd0, ap_ready_q, ap_idle_q, ap_done_q, ap_start_q};
      A_GIE:     reg_rdata = {31'd0, gie_q};
      A_IER:     reg_rdata = {31'd0, ier_q};
      A_ISR:     reg_rdata = {31'd0, isr_q};
      A_SEED:    reg_rdata = seed_q;
      A_VERSION: reg_rdata = BW_VERSION;
      A_INFO:    reg_rdata = BW_INFO;
      A_TOT_LO:  reg_rdata = tot_q[31:0];
      A_TOT_HI:  reg_rdata = tot_q[63:32];
      default: ;
    endcase
    for (int unsigned c = 0; c < NUM_CH; c++) begin
      if (ax_off == A_CH0 + 12'(32 * c))      reg_rdata = ctl_q[c];
      if (ax_off == A_CH0 + 12'(32 * c + 8))  reg_rdata = bytes_q[c][31:0];
      if (ax_off == A_CH0 + 12'(32 * c + 12)) reg_rdata = bytes_q[c][63:32];
      if (ax_off == A_CH0 + 12'(32 * c + 16)) reg_rdata = rdo_q[c][31:0];
      if (ax_off == A_CH0 + 12'(32 * c + 20)) reg_rdata = rdo_q[c][63:32];
      if (ax_off == A_CH0 + 12'(32 * c + 24)) reg_rdata = wro_q[c][31:0];
      if (ax_off == A_CH0 + 12'(32 * c + 28)) reg_rdata = wro_q[c][63:32];
      if (ax_off == A_BUF0 + 12'(8 * c))      reg_rdata = buf_q[c][31:0];
      if (ax_off == A_BUF0 + 12'(8 * c + 4))  reg_rdata = buf_q[c][63:32];
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
      assert (!(ap_idle_q && rs == RS_RUN)) else $error("%m: ap_idle while a run is in progress");
    end
  end
`endif

endmodule
