// tsr_xrt_ctrl.sv — control bridge of the Alveo XRT platform adapter: the kernel's s_axi_control
// (AXI4-Lite slave, 32-bit data, 13-bit address) with the channel bases, ADAPTER_CTRL.MEM_EN,
// ADAPTER_STATUS, ADAPTER_VERSION and the window onto the tsr_core CSR port.
//
// Owner: adapter agent (docs/platform-u200.md §2, §3; docs/design EX §9). The host only uses register
// access (EX §9.1, `ap_ctrl_none`); there is no start sequencer, interrupt or kernel argument
// register. Oracle of the register map: the mock adapter model (sw/transport/tsr_kdev_mock.cpp,
// TSR_DEVICE=mock), compared in tb/platform (test_adapter_regs).
//
// Register map (EX §9.2; byte offsets; reads of unmapped offsets return 0, writes to them are
// ignored; offsets are decoded on bits [12:2]; every response is OKAY):
//   0x000..0x03C    reserved, read 0 (no AP_CTRL, GIE, IER, ISR or kernel arguments)
//   0x040 + 8c      ch<c>_base lo / hi, c < NUM_CH: device byte address of the buffer bound to
//                   m_axi_ch<c> (4 KiB aligned by the host). Written while MEM_EN = 0; a write while
//                   MEM_EN = 1 is ignored and sets BASE_LOCKED. Reads return the register.
//   0x0E0           ADAPTER_CTRL (RW): bit0 MEM_EN. A write of 1 while MEM_EN = 0: if every base is
//                   4 KiB aligned (lo[11:0] = 0), the bases are latched into the bridges (ch_base)
//                   and MEM_EN := 1; otherwise MEM_EN stays 0 and BASE_MISALIGNED is set. A write
//                   of 0 clears MEM_EN (the host does so only with the core idle; outstanding
//                   transactions complete either way). While MEM_EN = 0 the kernel holds the core's
//                   memory requests (tsr_xrt_kernel) and reports them (ev_req_held -> REQ_HELD).
//   0x0F0           ADAPTER_STATUS (R / W1C, sticky): bit0 RRESP error, bit1 BRESP error, bit2
//                   response with no outstanding request for its id, bit3 BASE_MISALIGNED (a base
//                   not 4 KiB aligned at a MEM_EN write), bit4 REQ_HELD (a core memory request held
//                   while MEM_EN = 0), bit5 BASE_LOCKED (a base write while MEM_EN = 1, ignored);
//                   bits [31:16] saturating count of error responses (R beats and B with RRESP /
//                   BRESP != OKAY). W1C: a 1 in bit k (k < 6) clears bit k, a 1 in any of bits
//                   [31:16] clears the count. A new event in the same cycle wins.
//   0x0F4           ADAPTER_VERSION (R) = {16'(TSR_ADAPTER_MAJOR), 8'h00, NUM_CH} (EX §9.2: major 1)
//   0x1000 + off    tsr_core CSR port, off < 0x1000 (EX §5), at any time. Writes need WSTRB = 4'hF
//                   (others are dropped: the CSR port has no byte enables).
// plat_err (EX §9.3) = ADAPTER_STATUS bits 0..2 or a count != 0, registered: tsr_core's plat_err_i
//   (STATUS.PLAT_ERR, LQ_FLAGS).
// Writes use the byte strobes (RW registers); ADAPTER_STATUS acts on the bytes whose strobe is set.
// The AXI4-Lite slave handles one transaction at a time (reads and writes alternate when both are
// pending); AWREADY / WREADY are raised together once both AWVALID and WVALID are seen. A window
// access is one transfer on the core CSR port (docs/rtl-m2.md §6 protocol); the next one starts
// only after the AXI4-Lite response, so transfers are always separated by idle cycles. Every
// output is a flop. Reset: everything to its reset value.

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

  // ---- tsr_core CSR port (master side) ----
  output logic                       csr_valid,
  input  logic                       csr_ready,
  output logic                       csr_we,
  output logic [TSR_CSR_AW-1:0]      csr_addr,
  output logic [31:0]                csr_wdata,
  input  logic [31:0]                csr_rdata,

  // ---- channel bridges, memory-request gate, platform error ----
  output logic [NUM_CH-1:0][63:0]    ch_base,        // latched at MEM_EN
  output logic                       mem_en,
  input  logic                       ev_req_held,    // a core memory request offered while MEM_EN = 0
  output logic                       plat_err,
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

  // The adapter major is the table's (EX §9.2), the host's TSR_ADAPTER_MAJOR: one source.
  localparam logic [31:0] ADAPTER_VERSION = {16'(tsr_tables_pkg::TSR_ADAPTER_MAJOR), 8'h00, 8'(NUM_CH)};

  // Register offsets (byte address bits [11:0] of the adapter half, bit 12 = 0)
  localparam logic [11:0] A_BASE0    = 12'h040;
  localparam logic [11:0] A_ACTRL    = 12'h0E0;
  localparam logic [11:0] A_ASTATUS  = 12'h0F0;
  localparam logic [11:0] A_AVERSION = 12'h0F4;

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

  // host window transfer on the core CSR port
  logic                  host_req_q, host_we_q;
  logic [TSR_CSR_AW-1:0] host_addr_q;
  logic [31:0]           host_wdata_q;
  logic                  host_cpl;            // the transfer ends at this edge

  wire        ax_win   = ax_addr_q[AXIL_AW-1];
  wire [11:0] ax_off   = {ax_addr_q[11:2], 2'b00};
  wire        ax_exec  = (ax_st == AX_EXEC);
  wire        reg_wr   = ax_exec & ax_wr_q & ~ax_win;     // adapter register write performed now
  wire        reg_rd   = ax_exec & ~ax_wr_q & ~ax_win;    // adapter register read performed now

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
  // Core CSR port: one window transfer per AXI4-Lite transaction (valid from the cycle after the
  // request until the handshake; the payload registers hold since AX_EXEC)
  // =============================================================================================
  logic csr_valid_q;

  assign host_cpl = csr_valid_q & csr_ready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      csr_valid_q <= 1'b0;
    end else if (host_cpl) begin
      csr_valid_q <= 1'b0;     // host_req_q drops at this edge as well
    end else if (host_req_q) begin
      csr_valid_q <= 1'b1;
    end
  end

  assign csr_valid = csr_valid_q;
  assign csr_we    = host_we_q;
  assign csr_addr  = host_addr_q;
  assign csr_wdata = host_wdata_q;

  // =============================================================================================
  // Channel bases, ADAPTER_STATUS
  // =============================================================================================
  logic [NUM_CH-1:0][63:0]    base_reg_q;      // the registers as written
  logic [NUM_CH-1:0][63:0]    base_lat_q;      // latched at MEM_EN (the bridges' bases)
  logic                       mem_en_q;
  logic [5:0]                 ast_q;           // ADAPTER_STATUS sticky bits 0..5
  logic [15:0]                acnt_q;          // ADAPTER_STATUS error count
  logic                       plat_err_q;
  logic                       held_q;          // ev_req_held, registered

  // Bridge events: registered here (they come from other SLRs), counted one cycle later.
  logic [NUM_CH-1:0]          evr_q, evb_q, evu_q;
  logic [5:0]                 nerr_q;          // error responses in the cycle before
  logic                       anyr_q, anyb_q, anyu_q;
  logic [5:0]                 ast_d;
  logic [15:0]                acnt_d;

  // writes of this access: a base, ADAPTER_CTRL
  logic base_hit;
  always_comb begin
    base_hit = 1'b0;
    for (int unsigned c = 0; c < 2 * NUM_CH; c++)
      if (ax_off == A_BASE0 + 12'(4 * c)) base_hit = 1'b1;
  end
  wire base_wr  = reg_wr && base_hit;
  wire actrl_wr = reg_wr && ax_off == A_ACTRL && ax_wstrb_q[0];
  logic all_aligned;
  always_comb begin
    all_aligned = 1'b1;
    for (int unsigned c = 0; c < NUM_CH; c++)
      if (base_reg_q[c][11:0] != 12'd0) all_aligned = 1'b0;
  end
  wire en_rise = actrl_wr && ax_wdata_q[0] && !mem_en_q;

  // ADAPTER_STATUS next value: W1C first, then this cycle's events (an event wins)
  always_comb begin
    logic [16:0] sum;
    logic [15:0] n;
    ast_d = ast_q;
    n     = acnt_q;
    if (reg_wr && ax_off == A_ASTATUS) begin
      if (ax_wstrb_q[0]) ast_d = ast_d & ~ax_wdata_q[5:0];
      if ((ax_wstrb_q[2] && ax_wdata_q[23:16] != 8'd0) || (ax_wstrb_q[3] && ax_wdata_q[31:24] != 8'd0))
        n = '0;
    end
    ast_d  = ast_d | {base_wr && mem_en_q, held_q, en_rise && !all_aligned, anyu_q, anyb_q, anyr_q};
    sum    = 17'(n) + 17'(nerr_q);
    acnt_d = sum[16] ? 16'hFFFF : sum[15:0];
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      base_reg_q  <= '0;
      base_lat_q  <= '0;
      mem_en_q    <= 1'b0;
      plat_err_q  <= 1'b0;
      held_q      <= 1'b0;
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
      held_q <= ev_req_held;
      if (actrl_wr) begin
        if (!ax_wdata_q[0]) begin
          mem_en_q <= 1'b0;
        end else if (en_rise && all_aligned) begin
          mem_en_q   <= 1'b1;
          base_lat_q <= base_reg_q;
        end
      end
      if (reg_wr && !mem_en_q) begin
        for (int unsigned c = 0; c < NUM_CH; c++) begin
          if (ax_off == A_BASE0 + 12'(8 * c))
            base_reg_q[c][31:0] <= bmask(base_reg_q[c][31:0], ax_wdata_q, ax_wstrb_q);
          if (ax_off == A_BASE0 + 12'(8 * c + 4))
            base_reg_q[c][63:32] <= bmask(base_reg_q[c][63:32], ax_wdata_q, ax_wstrb_q);
        end
      end
      ast_q  <= ast_d;
      acnt_q <= acnt_d;
      plat_err_q <= (ast_d[2:0] != 3'd0) || (acnt_d != 16'd0);
    end
  end

  assign ch_base  = base_lat_q;
  assign mem_en   = mem_en_q;
  assign plat_err = plat_err_q;

  // =============================================================================================
  // Adapter register read mux
  // =============================================================================================
  always_comb begin
    reg_rdata = '0;
    unique case (ax_off)
      A_ACTRL:    reg_rdata = {31'd0, mem_en_q};
      A_ASTATUS:  reg_rdata = {acnt_q, 10'd0, ast_q};
      A_AVERSION: reg_rdata = ADAPTER_VERSION;
      default: ;
    endcase
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
      assert (!(csr_valid_q && !host_req_q)) else $error("%m: CSR port transfer without a window access");
    end
  end
`endif

endmodule
