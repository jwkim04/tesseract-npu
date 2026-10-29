// tsr_csr.sv — CSR block: CSR port, the identification / configuration registers and PERF_*.
//
// Owner: core (docs/rtl-m2.md §2, block (a); CSR rules §6). X0 (docs/v2 EX §12): the v1 run
// registers (CTRL START / SOFT_RESET / ABORT, STATUS, the command-buffer address and count,
// ARG0..7, ERR_PC, ERR_CODE, ERR_INFO, PC) and the `done` output are deleted; the v2 map (EX §5:
// LQ / LCQ / log rings, SR sequencer, loader with VERIFY, Q_*, PERF v2) replaces this block in Y4.
//
// CSR port (core boundary; protocol unchanged in v2): the master drives csr_valid with
// csr_addr/csr_we/csr_wdata and holds them until csr_ready. csr_ready is a registered one-cycle
// strobe with csr_rdata (reads; 0 for writes); the transfer completes at the edge where
// csr_valid && csr_ready. Unmapped or unaligned (addr[1:0] != 0) offsets read 0, writes are ignored.
// Access pipeline (every stage starts from registers; the port is latency-insensitive):
//   C  edge with csr_valid && !csr_ready && no access in progress: capture addr / we
//   S1 next edge: read: decode, select the 64-bit counter or the register value (counter LO and
//      HI sampled together)
//   S2 next edge: final read mux (HI snapshot), csr_ready := 1 for one cycle
//   so a read or write takes 4 cycles including the handshake edge.
// Registers: ID, VERSION, CFG0..2, CAPS (constants from parameters), PERF_* (R, 64-bit, LO/HI).
//   No register is writable: every write completes and is ignored; every other offset (CTRL and
//   STATUS included) reads 0.
// PERF (cleared by reset only):
//   MVM_CYCLES += perf_mvm_busy; CH_RD_BEATS[c] += perf_ch_rd_beat[c]; CH_WAIT[c] += perf_ch_wait[c].
//   CYCLES and RETIRED read 0 until their v2 sources exist (the ACTIVE gate and the queue
//   completions, EX §6, Y4), as VU_CYCLES and slot 4 (reserved).
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

  // ---- perf events ----
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
  logic [63:0]          perf_q [NPERF2];
  logic [31:0]          snap_hi_q;     // HI half captured by the last LO read
  logic                 snap_v_q;
  logic [PSEL_W-1:0]    snap_idx_q;

  // access pipeline
  logic                 req_q;         // an access is in progress (C .. S2)
  logic                 s1_q;          // S1 done: S2 at the next edge
  logic [TSR_CSR_AW-1:0] a_addr_q;
  logic                 a_we_q;
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
  wire rd      = st1 && !a_we_q && aligned;

  // PERF window: addr >= 0x100 and counter index < NPERF; HI half when addr[2].
  wire [PIDX_W-1:0] perf_idx = a_addr_q[TSR_CSR_AW-1:3] - PERF_IDX0;
  wire              perf_hit = (a_addr_q >= TSR_CSR_PERF_BASE) && (perf_idx < PIDX_W'(NPERF));

  // S1 register read value (everything but PERF).
  logic [31:0] reg_val;
  always_comb begin
    reg_val = 32'd0;
    case (a_addr_q)
      TSR_CSR_ID:      reg_val = TSR_ID_MAGIC;
      TSR_CSR_VERSION: reg_val = VERSION_VAL;
      TSR_CSR_CFG0:    reg_val = CFG0_VAL;
      TSR_CSR_CFG1:    reg_val = CFG1_VAL;
      TSR_CSR_CFG2:    reg_val = CFG2_VAL;
      TSR_CSR_CAPS:    reg_val = CAPS;
      default:         ;
    endcase
  end

  wire snap_match = snap_v_q && (snap_idx_q == r_idx_q);

  // -------------------------------------------------------------------------------------------
  // CSR port, access pipeline
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      req_q       <= 1'b0;
      s1_q        <= 1'b0;
      csr_ready_q <= 1'b0;
      csr_rdata_q <= 32'd0;
      snap_v_q    <= 1'b0;
    end else begin
      // C: capture
      if (cap) begin
        req_q    <= 1'b1;
        a_addr_q <= csr_addr;
        a_we_q   <= csr_we;
      end
      // S1: read selection
      if (st1) begin
        s1_q       <= 1'b1;
        r_perf_q   <= rd && perf_hit;
        r_hi_q     <= a_addr_q[2];
        r_idx_q    <= perf_idx[PSEL_W-1:0];
        r_perf64_q <= perf_q[perf_idx[PSEL_W-1:0]];
        r_reg_q    <= rd ? reg_val : 32'd0;
      end
      // S2: read data, ready strobe, HI snapshot
      csr_ready_q <= s1_q;
      if (s1_q) begin
        req_q <= 1'b0;
        s1_q  <= 1'b0;
        if (!r_perf_q)    csr_rdata_q <= r_reg_q;
        else if (!r_hi_q) csr_rdata_q <= r_perf64_q[31:0];
        else              csr_rdata_q <= snap_match ? snap_hi_q : r_perf64_q[63:32];
      end
      if (s1_q && r_perf_q && !r_hi_q) begin
        snap_v_q   <= 1'b1;
        snap_idx_q <= r_idx_q;
        snap_hi_q  <= r_perf64_q[63:32];
      end else if (s1_q && r_perf_q && r_hi_q && snap_match) begin
        snap_v_q <= 1'b0;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // PERF_* counters (slots >= NPERF only pad the read mux to a power of 2)
  // -------------------------------------------------------------------------------------------
  logic [NPERF-1:0] perf_inc;
  always_comb begin
    perf_inc = '0;
    perf_inc[TSR_PERF_MVM_CYCLES] = perf_mvm_busy;
    for (int unsigned c = 0; c < NUM_CH; c++) begin
      perf_inc[TSR_PERF_CH_RD_BEATS + c]          = perf_ch_rd_beat[c];
      perf_inc[TSR_PERF_CH_RD_BEATS + NUM_CH + c] = perf_ch_wait[c];
    end
  end

  for (genvar i = 0; i < NPERF2; i++) begin : g_perf
    if (i >= NPERF || i == TSR_PERF_CYCLES || i == TSR_PERF_RETIRED || i == TSR_PERF_VU_CYCLES ||
        i == TSR_PERF_RSVD4) begin : g_zero
      assign perf_q[i] = 64'd0;           // no source in this core (header); padding
    end else begin : g_cnt
      always_ff @(posedge clk) begin
        if (!rst_n)           perf_q[i] <= 64'd0;
        else if (perf_inc[i]) perf_q[i] <= perf_q[i] + 64'd1;
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // Outputs
  // -------------------------------------------------------------------------------------------
  assign csr_ready = csr_ready_q;
  assign csr_rdata = csr_rdata_q;

  // perf_inc bits of the slots without a counter are constant 0; no register takes write data.
  wire unused_ok = &{1'b0, perf_inc[TSR_PERF_CYCLES], perf_inc[TSR_PERF_RETIRED],
                     perf_inc[TSR_PERF_VU_CYCLES], perf_inc[TSR_PERF_RSVD4], csr_wdata};

endmodule
