// tsr_core.sv — Tesseract core datapath without a command source: CSR port, NUM_CH channel
// memory ports.
//
// Owner: core agent (docs/rtl-m2.md §2, block (f); docs/design CC §13). D14 / D16: the core has no
// host command path (no instruction fetch, command processor, run registers or `done` output).
// It is the datapath the dispatcher will drive: the CSR block (identification and PERF), the DMA, QACT and MVM units, the SPM and the
// per-channel memory arbiters, with the reset distribution and the unit-local integration checks.
// The units' command ports are idle until Y4, where tsr_core_dp (CC §13.2: tsr_issue with
// tsr_chk, the full tsr_csr, these units) takes over this netlist and tsr_core gets the final boundary
// (clk_rv, rst_rv_n, plat_err_i, host_irq_o; EX §9.3). Until then the core elaborates and lints in
// every configuration but does no work: no unit is ever offered a command, so no memory request
// leaves it.
//
// Core boundary (spec Core boundary interface, without `done`; EX §9.3):
//   - CSR port: csr_valid/csr_ready/csr_we/csr_addr/csr_wdata/csr_rdata (tsr_csr contract).
//   - memory port per channel c (packed [NUM_CH], index c): read request {addr_beat, len_beats, id},
//     read response {data, id, last}, write request {addr_beat, len_beats, id}, write data
//     {data, last}, write response {id}; all valid/ready (docs/rtl-m2.md §4). Every output is
//     driven from a register inside tsr_mem_arb.
// Clock/reset: one clock `clk`; `rst_n` synchronous, active low, already synchronised to clk.
// Units: DMA (LD / ST), QACT and, with HAS_MVM = 1 (default: the U200 configuration), MVM.
//   HAS_MVM = 1: tsr_mvm (docs/rtl-m3.md) on its command port (mvm_cmd_* / mvm_done), SPM client
//     TSR_SPM_CL_MVM (broadcast reads, gather writes, ACC reads), the weight-stream source
//     TSR_SRC_WS of every channel's tsr_mem_arb (cluster c <-> arbiter c) and
//     perf_mvm_busy -> PERF_MVM_CYCLES (cycles the MVM unit holds an op). LANES must be 128 (one
//     Q4_K beat; CFG0 reports it).
//   HAS_MVM = 0: no MVM unit; the weight-stream sources of the arbiters and SPM client
//     TSR_SPM_CL_MVM are tied off.
//   Memory source TSR_SRC_AUX (0) of every arbiter is tied off (CC-26).
// Implementation parameters (results identical for every legal value; tests vary them):
//   SPM_RD_LAT (SPM read latency seen by every SPM client; 1..4, U200 timing wants 4 at 4 MiB,
//   docs/rtl-m2.md §12), and for HAS_MVM = 1 (docs/rtl-m3.md §1): MVM_WS_FIFO (weight FIFO beats
//   per cluster, power of two >= 64), MVM_BC_PIPE / MVM_RES_PIPE (broadcast / result pipe stages
//   per cluster; U200: >= 2 for clusters in another SLR), MVM_ACT_RD_LAT, MVM_MAC_LAT,
//   MVM_MUL_LAT, MVM_ADD_LAT (1..8), MVM_WS_RD_LAT (1..4), RST_PIPE (0..8, below).
// Reset distribution (RST_PIPE, M3 part C timing, docs/rtl-m3.md §14): with RST_PIPE = P >= 1
//   every flop of the core sees rst_n exactly P cycles late (assertion and deassertion), through
//   P - min(P, 2) shared stages and min(P, 2) stages per block (CSR, DMA, QACT, SPM, the arbiter
//   of every channel, and inside tsr_mvm (RST_LOCAL) its top, gather and every cluster), each copy
//   kept separate (Xilinx `keep`) so it can sit next to its block: on the U200 the arbiters and
//   clusters of the four channels sit in three SLRs, and one rst_n net to all of them misses
//   300 MHz. The whole core is a P-cycle-delayed copy of the RST_PIPE = 0 core (results and cycle
//   behaviour after the reset identical); the registers start in reset (configuration INIT /
//   simulation initial value 0). P = 0: no extra flops.
// Verification: rtl/lint.sh (every configuration); the unit benches (tb/dma, tb/qact, tb/mvm, ...)
// test the blocks; the core-level replay benches come with the dispatcher (docs/design SW §10.3:
// tb/core, tb/fastreplay). Simulation-only integration checks I1, I2, I4 at the end of this file.

module tsr_core
  import tsr_pkg::*;
#(
  parameter int unsigned     NUM_CH         = TSR_DEF_NUM_CH,
  parameter int unsigned     BEAT_BITS      = TSR_DEF_BEAT_BITS,
  parameter int unsigned     LANES          = TSR_DEF_LANES,
  parameter int unsigned     T_MAX          = TSR_DEF_T_MAX,
  parameter int unsigned     K_MAX          = TSR_DEF_K_MAX,
  parameter int unsigned     SPM_BYTES      = TSR_DEF_SPM_BYTES,
  parameter int unsigned     VL             = TSR_DEF_VL,
  parameter int unsigned     NSREG          = TSR_DEF_NSREG,
  parameter int unsigned     RD_OUTSTANDING = TSR_DEF_RD_OUTSTANDING,
  parameter longint unsigned CH_BYTES       = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W    = TSR_DEF_ADDR_BEAT_W,
  parameter bit              HAS_MVM        = 1'b1,
  parameter int unsigned     SPM_RD_LAT     = TSR_DEF_SPM_RD_LAT,
  // MVM unit implementation parameters (HAS_MVM = 1; docs/rtl-m3.md §1)
  parameter int unsigned     MVM_WS_FIFO    = TSR_DEF_WS_FIFO_BEATS,
  parameter int unsigned     MVM_BC_PIPE    = TSR_DEF_MVM_BC_PIPE,
  parameter int unsigned     MVM_RES_PIPE   = TSR_DEF_MVM_RES_PIPE,
  parameter int unsigned     MVM_ACT_RD_LAT = TSR_DEF_ACT_RD_LAT,
  parameter int unsigned     MVM_MAC_LAT    = TSR_DEF_MAC_LAT,
  parameter int unsigned     MVM_MUL_LAT    = TSR_DEF_FMUL_LAT,
  parameter int unsigned     MVM_ADD_LAT    = TSR_DEF_FADD_LAT,
  parameter int unsigned     MVM_WS_RD_LAT  = TSR_DEF_WS_RD_LAT,
  // Reset register stages inside the core (implementation; docs/rtl-m3.md §14): 0 = rst_n
  // straight to every block (M2 / M3 test builds); U200: 3 (the measured OOC closure, §14.2)
  parameter int unsigned     RST_PIPE       = 0
) (
  input  logic                                   clk,
  input  logic                                   rst_n,

  // ---- CSR port ----
  input  logic                                   csr_valid,
  output logic                                   csr_ready,
  input  logic                                   csr_we,
  input  logic [TSR_CSR_AW-1:0]                  csr_addr,
  input  logic [31:0]                            csr_wdata,
  output logic [31:0]                            csr_rdata,

  // ---- memory ports, one per channel ----
  output logic [NUM_CH-1:0]                      m_rd_req_valid,
  input  logic [NUM_CH-1:0]                      m_rd_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_rd_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_rd_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_rd_req_id,
  input  logic [NUM_CH-1:0]                      m_rd_rsp_valid,
  output logic [NUM_CH-1:0]                      m_rd_rsp_ready,
  input  logic [NUM_CH-1:0][BEAT_BITS-1:0]       m_rd_rsp_data,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_rd_rsp_id,
  input  logic [NUM_CH-1:0]                      m_rd_rsp_last,
  output logic [NUM_CH-1:0]                      m_wr_req_valid,
  input  logic [NUM_CH-1:0]                      m_wr_req_ready,
  output logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]     m_wr_req_addr,
  output logic [NUM_CH-1:0][TSR_LEN_W-1:0]       m_wr_req_len,
  output logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_req_id,
  output logic [NUM_CH-1:0]                      m_wr_dat_valid,
  input  logic [NUM_CH-1:0]                      m_wr_dat_ready,
  output logic [NUM_CH-1:0][BEAT_BITS-1:0]       m_wr_dat_data,
  output logic [NUM_CH-1:0]                      m_wr_dat_last,
  input  logic [NUM_CH-1:0]                      m_wr_rsp_valid,
  output logic [NUM_CH-1:0]                      m_wr_rsp_ready,
  input  logic [NUM_CH-1:0][TSR_MEM_ID_W-1:0]    m_wr_rsp_id
);

  // -------------------------------------------------------------------------------------------
  // Elaboration checks (same limits as tsr_sim_create's cfg_problem(), plus M2 restrictions)
  // -------------------------------------------------------------------------------------------
  if (BEAT_BITS != 512) begin : g_chk_beat
    $fatal(1, "tsr_core: BEAT_BITS must be 512 (a beat is one 64-byte SPM line)");
  end
  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX) begin : g_chk_ch
    $fatal(1, "tsr_core: NUM_CH must be in [1, 16] (4-bit LD/ST channel field)");
  end
  if (T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH) begin : g_chk_t
    $fatal(1, "tsr_core: T_MAX must be in [1, 8]");
  end
  if (K_MAX < 256 || K_MAX > 65280 || (K_MAX % 256) != 0) begin : g_chk_k
    $fatal(1, "tsr_core: K_MAX must be a multiple of 256 in [256, 65280]");
  end
  if (SPM_BYTES < TSR_LINE_BYTES || (SPM_BYTES % TSR_LINE_BYTES) != 0) begin : g_chk_spm
    $fatal(1, "tsr_core: SPM_BYTES must be a positive multiple of 64");
  end
  if (NSREG < 1 || NSREG > 16 || VL < 1 || VL > 255 || LANES < 1 || LANES > 4095) begin : g_chk_cfg
    $fatal(1, "tsr_core: NSREG in [1, 16], VL in [1, 255], LANES in [1, 4095] (CFG field widths)");
  end
  if (CH_BYTES < 64 || (CH_BYTES % 64) != 0 || ADDR_BEAT_W < 1 || ADDR_BEAT_W > 40) begin : g_chk_dram
    $fatal(1, "tsr_core: CH_BYTES must be a positive multiple of 64, ADDR_BEAT_W in [1, 40]");
  end
  if (RD_OUTSTANDING < 1) begin : g_chk_rdo
    $fatal(1, "tsr_core: RD_OUTSTANDING must be >= 1");
  end
  if (HAS_MVM && LANES != TSR_MVM_LANES) begin : g_chk_lanes
    $fatal(1, "tsr_core: HAS_MVM = 1 needs LANES == 128 (= BEAT_BITS / 4, one Q4_K beat; docs/rtl-m3.md §1)");
  end
  if (HAS_MVM && (MVM_WS_FIFO < 64 || (MVM_WS_FIFO & (MVM_WS_FIFO - 1)) != 0)) begin : g_chk_wsfifo
    $fatal(1, "tsr_core: MVM_WS_FIFO must be a power of two >= 64 (docs/rtl-m3.md §4)");
  end
  if (RST_PIPE > 8) begin : g_chk_rst
    $fatal(1, "tsr_core: RST_PIPE must be in [0, 8]");
  end
  if (HAS_MVM && (MVM_ADD_LAT < 1 || MVM_ADD_LAT > 8 || MVM_WS_RD_LAT < 1 || MVM_WS_RD_LAT > 4 ||
                  MVM_ACT_RD_LAT < 1 || MVM_MAC_LAT < 1 || MVM_MUL_LAT < 1)) begin : g_chk_mvmlat
    $fatal(1, "tsr_core: MVM latencies: ADD 1..8, WS_RD 1..4, ACT_RD / MAC / MUL >= 1 (docs/rtl-m3.md §1)");
  end

  localparam logic [31:0] CAPS = 32'd0;   // no optional feature (VU, TOPK, ATT, DQROW; EX §7.5)
  localparam int unsigned SPM_LINES = SPM_BYTES / TSR_LINE_BYTES;
  localparam int unsigned SPM_AW    = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1;

  // -------------------------------------------------------------------------------------------
  // Reset distribution (RST_PIPE, header): rst_pre_n after the RST_SH shared stages, then
  // RST_BLK (<= 2) stages per block: rst_b_n[i]; tsr_mvm makes its own RST_BLK stages per part
  // (RST_LOCAL). Two per-block stages let the first copy sit at the SLR boundary and the second
  // (replicated by phys_opt) next to the loads. The stage registers start at 0 (initial block:
  // INIT 0) and load in a plain always: IEEE 1800 9.2.2.4 forbids a second writer of an always_ff
  // variable (XSim rejects the initial block then; Verilator and Vivado synthesis accept it).
  // -------------------------------------------------------------------------------------------
  localparam int unsigned RB_CSR = 0, RB_DMA = 1, RB_QACT = 2, RB_SPM = 3, RB_CHK = 4,
                          RB_ARB = 5;                                    // RB_ARB + c: arbiter c
  localparam int unsigned NRB     = RB_ARB + NUM_CH;
  localparam int unsigned RST_BLK = (RST_PIPE > 2) ? 2 : RST_PIPE;
  localparam int unsigned RST_SH  = RST_PIPE - RST_BLK;
  logic [NRB-1:0] rst_b_n;
  logic           rst_pre_n;

  if (RST_SH == 0) begin : g_sh0
    assign rst_pre_n = rst_n;
  end else begin : g_sh
    (* shreg_extract = "no" *) logic [RST_SH-1:0] sh_q;      // shared stages 1 .. RST_SH
    initial sh_q = '0;
    always @(posedge clk) begin
      sh_q[0] <= rst_n;
      for (int unsigned k = 1; k < RST_SH; k++) sh_q[k] <= sh_q[k-1];
    end
    assign rst_pre_n = sh_q[RST_SH-1];
  end
  if (RST_BLK == 0) begin : g_rst0
    assign rst_b_n = {NRB{rst_pre_n}};
  end else begin : g_rst
    (* keep = "true" *) logic [RST_BLK-1:0][NRB-1:0] rb_q;   // per block: stages RST_SH+1 ..
    initial rb_q = '0;
    always @(posedge clk) begin
      rb_q[0] <= {NRB{rst_pre_n}};
      for (int unsigned k = 1; k < RST_BLK; k++) rb_q[k] <= rb_q[k-1];
    end
    assign rst_b_n = rb_q[RST_BLK-1];
  end

  // -------------------------------------------------------------------------------------------
  // CSR block
  // -------------------------------------------------------------------------------------------
  logic                 perf_mvm_busy;
  logic [NUM_CH-1:0]    perf_ch_rd_beat, perf_ch_wait;

  tsr_csr #(
    .NUM_CH(NUM_CH), .LANES(LANES), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES),
    .VL(VL), .NSREG(NSREG), .CAPS(CAPS)
  ) u_csr (
    .clk, .rst_n(rst_b_n[RB_CSR]),
    .csr_valid, .csr_ready, .csr_we, .csr_addr, .csr_wdata, .csr_rdata,
    .perf_mvm_busy, .perf_ch_rd_beat, .perf_ch_wait
  );

  // -------------------------------------------------------------------------------------------
  // Unit command ports: idle until the dispatcher (tsr_chk + tsr_issue, CC §13, Y4) drives
  // them (header).
  // -------------------------------------------------------------------------------------------
  logic            dma_cmd_valid, dma_cmd_ready, dma_done;
  tsr_dma_cmd_t    dma_cmd;
  logic            qact_cmd_valid, qact_cmd_ready, qact_done;
  tsr_qact_cmd_t   qact_cmd;
  logic            mvm_cmd_valid, mvm_cmd_ready, mvm_done;
  tsr_mvm_cmd_t    mvm_cmd;

  assign dma_cmd_valid  = 1'b0;
  assign dma_cmd        = '0;
  assign qact_cmd_valid = 1'b0;
  assign qact_cmd       = '0;
  assign mvm_cmd_valid  = 1'b0;
  assign mvm_cmd        = '0;

  // Read-source ports of the per-channel arbiters: [channel][source].
  logic [NUM_CH-1:0][TSR_NSRC-1:0]                   s_rd_req_valid, s_rd_req_ready;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][ADDR_BEAT_W-1:0]  s_rd_req_addr;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][TSR_LEN_W-1:0]    s_rd_req_len;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][TSR_SRC_ID_W-1:0] s_rd_req_id;
  logic [NUM_CH-1:0][TSR_NSRC-1:0]                   s_rd_rsp_valid, s_rd_rsp_ready;
  logic [NUM_CH-1:0][BEAT_BITS-1:0]                  s_rd_rsp_data;
  logic [NUM_CH-1:0]                                 s_rd_rsp_last;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]               s_rd_rsp_id;

  // DMA source ports, one per channel.
  logic [NUM_CH-1:0]                   d_rd_req_valid, d_rd_rsp_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  d_rd_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    d_rd_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_rd_req_id;
  logic [NUM_CH-1:0]                   d_rd_req_ready, d_rd_rsp_valid;
  logic [NUM_CH-1:0]                   d_wr_req_valid, d_wr_req_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  d_wr_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    d_wr_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_wr_req_id;
  logic [NUM_CH-1:0]                   d_wr_dat_valid, d_wr_dat_ready, d_wr_dat_last;
  logic [NUM_CH-1:0][BEAT_BITS-1:0]    d_wr_dat_data;
  logic [NUM_CH-1:0]                   d_wr_rsp_valid, d_wr_rsp_ready;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] d_wr_rsp_id;

  // SPM client ports [TSR_SPM_NCL]: 0 DMA, 1 QACT, 2 MVM (tied off when HAS_MVM = 0).
  logic [TSR_SPM_NCL-1:0]                         spm_rd_req_valid, spm_rd_req_ready, spm_rd_rsp_valid;
  logic [TSR_SPM_NCL-1:0][SPM_AW-1:0]             spm_rd_req_line;
  logic [TSR_LINE_BITS-1:0]                       spm_rd_rsp_data;
  logic [TSR_SPM_NCL-1:0]                         spm_wr_valid, spm_wr_ready;
  logic [TSR_SPM_NCL-1:0][SPM_AW-1:0]             spm_wr_line;
  logic [TSR_SPM_NCL-1:0][TSR_LINE_BITS-1:0]      spm_wr_data;
  logic [TSR_SPM_NCL-1:0][TSR_SPM_BE_W-1:0]       spm_wr_be;

  tsr_dma #(
    .NUM_CH(NUM_CH), .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .SPM_BYTES(SPM_BYTES),
    .SPM_RD_LAT(SPM_RD_LAT)
  ) u_dma (
    .clk, .rst_n(rst_b_n[RB_DMA]),
    .cmd_valid(dma_cmd_valid), .cmd_ready(dma_cmd_ready), .cmd(dma_cmd), .done(dma_done),
    .rd_req_valid(d_rd_req_valid), .rd_req_ready(d_rd_req_ready), .rd_req_addr(d_rd_req_addr),
    .rd_req_len(d_rd_req_len), .rd_req_id(d_rd_req_id),
    .rd_rsp_valid(d_rd_rsp_valid), .rd_rsp_ready(d_rd_rsp_ready), .rd_rsp_data(s_rd_rsp_data),
    .rd_rsp_last(s_rd_rsp_last), .rd_rsp_id(s_rd_rsp_id),
    .wr_req_valid(d_wr_req_valid), .wr_req_ready(d_wr_req_ready), .wr_req_addr(d_wr_req_addr),
    .wr_req_len(d_wr_req_len), .wr_req_id(d_wr_req_id),
    .wr_dat_valid(d_wr_dat_valid), .wr_dat_ready(d_wr_dat_ready), .wr_dat_data(d_wr_dat_data),
    .wr_dat_last(d_wr_dat_last),
    .wr_rsp_valid(d_wr_rsp_valid), .wr_rsp_ready(d_wr_rsp_ready), .wr_rsp_id(d_wr_rsp_id),
    .spm_rd_req_valid(spm_rd_req_valid[TSR_SPM_CL_DMA]), .spm_rd_req_ready(spm_rd_req_ready[TSR_SPM_CL_DMA]),
    .spm_rd_req_line(spm_rd_req_line[TSR_SPM_CL_DMA]), .spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_DMA]),
    .spm_rd_rsp_data(spm_rd_rsp_data),
    .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_DMA]), .spm_wr_ready(spm_wr_ready[TSR_SPM_CL_DMA]),
    .spm_wr_line(spm_wr_line[TSR_SPM_CL_DMA]), .spm_wr_data(spm_wr_data[TSR_SPM_CL_DMA]),
    .spm_wr_be(spm_wr_be[TSR_SPM_CL_DMA])
  );

  tsr_qact #(
    .K_MAX(K_MAX), .T_MAX(T_MAX), .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT)
  ) u_qact (
    .clk, .rst_n(rst_b_n[RB_QACT]),
    .cmd_valid(qact_cmd_valid), .cmd_ready(qact_cmd_ready), .cmd(qact_cmd), .done(qact_done),
    .spm_rd_req_valid(spm_rd_req_valid[TSR_SPM_CL_QACT]), .spm_rd_req_ready(spm_rd_req_ready[TSR_SPM_CL_QACT]),
    .spm_rd_req_line(spm_rd_req_line[TSR_SPM_CL_QACT]), .spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_QACT]),
    .spm_rd_rsp_data(spm_rd_rsp_data),
    .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_QACT]), .spm_wr_ready(spm_wr_ready[TSR_SPM_CL_QACT]),
    .spm_wr_line(spm_wr_line[TSR_SPM_CL_QACT]), .spm_wr_data(spm_wr_data[TSR_SPM_CL_QACT]),
    .spm_wr_be(spm_wr_be[TSR_SPM_CL_QACT])
  );

  // -------------------------------------------------------------------------------------------
  // MVM unit (M3): SPM client TSR_SPM_CL_MVM, weight-stream source TSR_SRC_WS of every channel
  // -------------------------------------------------------------------------------------------
  // Weight-stream source ports (cluster c <-> arbiter c), [channel].
  logic [NUM_CH-1:0]                   w_rd_req_valid, w_rd_req_ready, w_rd_rsp_valid, w_rd_rsp_ready;
  logic [NUM_CH-1:0][ADDR_BEAT_W-1:0]  w_rd_req_addr;
  logic [NUM_CH-1:0][TSR_LEN_W-1:0]    w_rd_req_len;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0] w_rd_req_id;

  if (HAS_MVM) begin : g_mvm
    tsr_mvm #(
      .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .SPM_RD_LAT(SPM_RD_LAT),
      .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .WS_FIFO_BEATS(MVM_WS_FIFO),
      .BC_PIPE(MVM_BC_PIPE), .RES_PIPE(MVM_RES_PIPE), .ACT_RD_LAT(MVM_ACT_RD_LAT),
      .MAC_LAT(MVM_MAC_LAT), .MUL_LAT(MVM_MUL_LAT), .ADD_LAT(MVM_ADD_LAT), .WS_RD_LAT(MVM_WS_RD_LAT),
      .RST_LOCAL(RST_BLK)
    ) u_mvm (
      .clk, .rst_n(rst_pre_n),
      .cmd_valid(mvm_cmd_valid), .cmd_ready(mvm_cmd_ready), .cmd(mvm_cmd), .done(mvm_done),
      .spm_rd_req_valid(spm_rd_req_valid[TSR_SPM_CL_MVM]), .spm_rd_req_ready(spm_rd_req_ready[TSR_SPM_CL_MVM]),
      .spm_rd_req_line(spm_rd_req_line[TSR_SPM_CL_MVM]), .spm_rd_rsp_valid(spm_rd_rsp_valid[TSR_SPM_CL_MVM]),
      .spm_rd_rsp_data(spm_rd_rsp_data),
      .spm_wr_valid(spm_wr_valid[TSR_SPM_CL_MVM]), .spm_wr_ready(spm_wr_ready[TSR_SPM_CL_MVM]),
      .spm_wr_line(spm_wr_line[TSR_SPM_CL_MVM]), .spm_wr_data(spm_wr_data[TSR_SPM_CL_MVM]),
      .spm_wr_be(spm_wr_be[TSR_SPM_CL_MVM]),
      .ws_rd_req_valid(w_rd_req_valid), .ws_rd_req_ready(w_rd_req_ready), .ws_rd_req_addr(w_rd_req_addr),
      .ws_rd_req_len(w_rd_req_len), .ws_rd_req_id(w_rd_req_id),
      .ws_rd_rsp_valid(w_rd_rsp_valid), .ws_rd_rsp_ready(w_rd_rsp_ready), .ws_rd_rsp_data(s_rd_rsp_data),
      .ws_rd_rsp_last(s_rd_rsp_last), .ws_rd_rsp_id(s_rd_rsp_id),
      .perf_mvm_busy
    );
  end else begin : g_nomvm
    // HAS_MVM = 0: no MVM unit (the dispatch refuses MVM; no mvm_cmd is ever offered).
    assign mvm_cmd_ready  = 1'b0;
    assign mvm_done       = 1'b0;
    assign perf_mvm_busy  = 1'b0;
    assign w_rd_req_valid = '0;
    assign w_rd_req_addr  = '0;
    assign w_rd_req_len   = '0;
    assign w_rd_req_id    = '0;
    assign w_rd_rsp_ready = '1;
    assign spm_rd_req_valid[TSR_SPM_CL_MVM] = 1'b0;
    assign spm_rd_req_line[TSR_SPM_CL_MVM]  = '0;
    assign spm_wr_valid[TSR_SPM_CL_MVM]     = 1'b0;
    assign spm_wr_line[TSR_SPM_CL_MVM]      = '0;
    assign spm_wr_data[TSR_SPM_CL_MVM]      = '0;
    assign spm_wr_be[TSR_SPM_CL_MVM]        = '0;
    wire unused_mvm = &{1'b0, rst_pre_n, w_rd_req_ready, w_rd_rsp_valid, spm_rd_req_ready[TSR_SPM_CL_MVM],
                        spm_rd_rsp_valid[TSR_SPM_CL_MVM], spm_wr_ready[TSR_SPM_CL_MVM], mvm_cmd_valid,
                        mvm_cmd};
  end

  tsr_spm #(
    .SPM_BYTES(SPM_BYTES), .NCL(TSR_SPM_NCL), .RD_LAT(SPM_RD_LAT)
  ) u_spm (
    .clk, .rst_n(rst_b_n[RB_SPM]),
    .rd_req_valid(spm_rd_req_valid), .rd_req_ready(spm_rd_req_ready), .rd_req_line(spm_rd_req_line),
    .rd_rsp_valid(spm_rd_rsp_valid), .rd_rsp_data(spm_rd_rsp_data),
    .wr_valid(spm_wr_valid), .wr_ready(spm_wr_ready), .wr_line(spm_wr_line), .wr_data(spm_wr_data),
    .wr_be(spm_wr_be)
  );

  // -------------------------------------------------------------------------------------------
  // Per-channel memory arbiters
  // -------------------------------------------------------------------------------------------
  for (genvar c = 0; c < NUM_CH; c++) begin : g_ch
    // Source AUX: tied off on every channel (CC-26).
    assign s_rd_req_valid[c][TSR_SRC_AUX] = 1'b0;
    assign s_rd_req_addr[c][TSR_SRC_AUX]  = '0;
    assign s_rd_req_len[c][TSR_SRC_AUX]   = '0;
    assign s_rd_req_id[c][TSR_SRC_AUX]    = '0;
    assign s_rd_rsp_ready[c][TSR_SRC_AUX] = 1'b1;
    // Source DMA.
    assign s_rd_req_valid[c][TSR_SRC_DMA] = d_rd_req_valid[c];
    assign s_rd_req_addr[c][TSR_SRC_DMA]  = d_rd_req_addr[c];
    assign s_rd_req_len[c][TSR_SRC_DMA]   = d_rd_req_len[c];
    assign s_rd_req_id[c][TSR_SRC_DMA]    = d_rd_req_id[c];
    assign s_rd_rsp_ready[c][TSR_SRC_DMA] = d_rd_rsp_ready[c];
    assign d_rd_req_ready[c]              = s_rd_req_ready[c][TSR_SRC_DMA];
    assign d_rd_rsp_valid[c]              = s_rd_rsp_valid[c][TSR_SRC_DMA];
    // Source WS (M3 weight stream of cluster c; tied off when HAS_MVM = 0).
    assign s_rd_req_valid[c][TSR_SRC_WS]  = w_rd_req_valid[c];
    assign s_rd_req_addr[c][TSR_SRC_WS]   = w_rd_req_addr[c];
    assign s_rd_req_len[c][TSR_SRC_WS]    = w_rd_req_len[c];
    assign s_rd_req_id[c][TSR_SRC_WS]     = w_rd_req_id[c];
    assign s_rd_rsp_ready[c][TSR_SRC_WS]  = w_rd_rsp_ready[c];
    assign w_rd_req_ready[c]              = s_rd_req_ready[c][TSR_SRC_WS];
    assign w_rd_rsp_valid[c]              = s_rd_rsp_valid[c][TSR_SRC_WS];

    tsr_mem_arb #(
      .ADDR_BEAT_W(ADDR_BEAT_W), .BEAT_BITS(BEAT_BITS), .RD_OUTSTANDING(RD_OUTSTANDING)
    ) u_arb (
      .clk, .rst_n(rst_b_n[RB_ARB + c]),
      .s_rd_req_valid(s_rd_req_valid[c]), .s_rd_req_ready(s_rd_req_ready[c]),
      .s_rd_req_addr(s_rd_req_addr[c]), .s_rd_req_len(s_rd_req_len[c]), .s_rd_req_id(s_rd_req_id[c]),
      .s_rd_rsp_valid(s_rd_rsp_valid[c]), .s_rd_rsp_ready(s_rd_rsp_ready[c]),
      .s_rd_rsp_data(s_rd_rsp_data[c]), .s_rd_rsp_last(s_rd_rsp_last[c]), .s_rd_rsp_id(s_rd_rsp_id[c]),
      .s_wr_req_valid(d_wr_req_valid[c]), .s_wr_req_ready(d_wr_req_ready[c]),
      .s_wr_req_addr(d_wr_req_addr[c]), .s_wr_req_len(d_wr_req_len[c]), .s_wr_req_id(d_wr_req_id[c]),
      .s_wr_dat_valid(d_wr_dat_valid[c]), .s_wr_dat_ready(d_wr_dat_ready[c]),
      .s_wr_dat_data(d_wr_dat_data[c]), .s_wr_dat_last(d_wr_dat_last[c]),
      .s_wr_rsp_valid(d_wr_rsp_valid[c]), .s_wr_rsp_ready(d_wr_rsp_ready[c]), .s_wr_rsp_id(d_wr_rsp_id[c]),
      .m_rd_req_valid(m_rd_req_valid[c]), .m_rd_req_ready(m_rd_req_ready[c]),
      .m_rd_req_addr(m_rd_req_addr[c]), .m_rd_req_len(m_rd_req_len[c]), .m_rd_req_id(m_rd_req_id[c]),
      .m_rd_rsp_valid(m_rd_rsp_valid[c]), .m_rd_rsp_ready(m_rd_rsp_ready[c]),
      .m_rd_rsp_data(m_rd_rsp_data[c]), .m_rd_rsp_id(m_rd_rsp_id[c]), .m_rd_rsp_last(m_rd_rsp_last[c]),
      .m_wr_req_valid(m_wr_req_valid[c]), .m_wr_req_ready(m_wr_req_ready[c]),
      .m_wr_req_addr(m_wr_req_addr[c]), .m_wr_req_len(m_wr_req_len[c]), .m_wr_req_id(m_wr_req_id[c]),
      .m_wr_dat_valid(m_wr_dat_valid[c]), .m_wr_dat_ready(m_wr_dat_ready[c]),
      .m_wr_dat_data(m_wr_dat_data[c]), .m_wr_dat_last(m_wr_dat_last[c]),
      .m_wr_rsp_valid(m_wr_rsp_valid[c]), .m_wr_rsp_ready(m_wr_rsp_ready[c]), .m_wr_rsp_id(m_wr_rsp_id[c]),
      .perf_rd_beat(perf_ch_rd_beat[c]), .perf_rd_wait(perf_ch_wait[c])
    );
  end

  // Arbiter outputs without a consumer: AUX ready/valid on every channel.
  logic unused_aux;
  always_comb begin
    unused_aux = 1'b0;
    for (int c = 0; c < NUM_CH; c++) begin
      unused_aux |= s_rd_req_ready[c][TSR_SRC_AUX] | s_rd_rsp_valid[c][TSR_SRC_AUX];
    end
  end

  // -------------------------------------------------------------------------------------------
  // Integration checks (simulation only; docs/rtl-m2.md §3.0, §5; docs/rtl-m3.md §3.0, §6.4,
  // §10.4). Armed by the first clock edge with rst_n low (before it, Verilator's X randomisation
  // makes state arbitrary); with RST_PIPE they see the blocks' delayed reset. Only the
  // unit-local rules are checked here: the units run concurrently, and CC §9.4 defines the
  // integration checks of the dispatcher.
  //   I1 a unit pulses done only while it holds an op (never in its accepting cycle); no MVM
  //      command with HAS_MVM = 0. MVM: cmd_ready is low while it holds the op, and perf_mvm_busy
  //      equals "holds an op" in every cycle (from the cycle after the accepting edge up to and
  //      including the cycle of done = PERF_MVM_CYCLES).
  //   I2 a unit touches the SPM only while it holds an op (its done means its SPM effects are
  //      complete): DMA / QACT (request in the accepting cycle allowed, as in M2), MVM strictly
  //      after its accept; an MVM SPM read response only while it holds the op. Weight-stream
  //      requests and responses (source WS of every arbiter) only while the MVM unit holds an op.
  //   I4 MVM done => quiescent and exact (docs/rtl-m3.md §3.0, §4, §6.4): in the cycle of
  //      mvm_done no WS request is offered on any channel, every WS beat requested has been
  //      received (counted at the arbiter's source port), no MVM SPM request or write is offered
  //      and no MVM SPM read is outstanding;
  //      channel c requested exactly rows_c x ROW_BEATS beats (rows_c = max(0, min(R, N - c*R)))
  //      and never more at any point; WS requests follow the §4 burst rule (len =
  //      min(64 - addr % 64, beats left), contiguous from w_off).
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic dma_busy_q, qact_busy_q, mvm_busy_q;
  wire  dma_take  = dma_cmd_valid && dma_cmd_ready;
  wire  qact_take = qact_cmd_valid && qact_cmd_ready;
  wire  mvm_take  = mvm_cmd_valid && mvm_cmd_ready;
  wire  dma_spm   = spm_rd_req_valid[TSR_SPM_CL_DMA] || spm_wr_valid[TSR_SPM_CL_DMA];
  wire  qact_spm  = spm_rd_req_valid[TSR_SPM_CL_QACT] || spm_wr_valid[TSR_SPM_CL_QACT];
  wire  mvm_spm   = spm_rd_req_valid[TSR_SPM_CL_MVM] || spm_wr_valid[TSR_SPM_CL_MVM];

  // Same arming scheme as tsr_spm / tsr_mem_arb: the initial block runs before Verilator's first
  // evaluation, which may already see a clock edge with X-randomised rst_n and state.
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_b_n[RB_CHK]) chk_on_q <= 1'b1;

  // I4 bookkeeping: per channel WS beats expected / requested / received in the current MVM,
  // next expected request address; MVM SPM reads outstanding.
  logic [NUM_CH-1:0][47:0] ws_exp_q, ws_req_q, ws_rsp_q;
  logic [NUM_CH-1:0][63:0] ws_addr_q;
  logic [7:0]              mvm_rd_out_q;
  wire                     mvm_rd_take = spm_rd_req_valid[TSR_SPM_CL_MVM] && spm_rd_req_ready[TSR_SPM_CL_MVM];

  function automatic logic [47:0] ws_beats_exp(logic [31:0] n, logic [31:0] r, logic [31:0] row_beats,
                                               int unsigned ch);
    logic [31:0] rows;
    rows = tsr_mvm_rows_c(n, r, 64'(ch) * 64'(r));
    return 48'(rows) * 48'(row_beats);
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_b_n[RB_CHK]) begin
      dma_busy_q   <= 1'b0;
      qact_busy_q  <= 1'b0;
      mvm_busy_q   <= 1'b0;
      mvm_rd_out_q <= '0;
      ws_exp_q     <= '0;
      ws_req_q     <= '0;
      ws_rsp_q     <= '0;
      ws_addr_q    <= '0;
    end else if (chk_on_q) begin
      if (dma_take)   dma_busy_q  <= 1'b1;
      if (dma_done)   dma_busy_q  <= 1'b0;
      if (qact_take)  qact_busy_q <= 1'b1;
      if (qact_done)  qact_busy_q <= 1'b0;
      if (mvm_take)   mvm_busy_q  <= 1'b1;
      if (mvm_done)   mvm_busy_q  <= 1'b0;
      // I1
      assert (!(mvm_cmd_valid && !HAS_MVM))
        else $error("tsr_core: MVM command offered with HAS_MVM = 0");
      assert (!(dma_done && !dma_busy_q) && !(qact_done && !qact_busy_q) && !(mvm_done && !mvm_busy_q))
        else $error("tsr_core: unit done without an op in flight (dma %0b qact %0b mvm %0b)",
                    dma_done, qact_done, mvm_done);
      assert (!(mvm_cmd_ready && mvm_busy_q))
        else $error("tsr_core: MVM unit ready while it holds an op");
      assert (perf_mvm_busy == mvm_busy_q)
        else $error("tsr_core: perf_mvm_busy %0b while the MVM unit %s an op", perf_mvm_busy,
                    mvm_busy_q ? "holds" : "does not hold");
      // I2
      assert (!(dma_spm && !dma_busy_q && !dma_take) && !(qact_spm && !qact_busy_q && !qact_take) &&
              !(mvm_spm && !mvm_busy_q))
        else $error("tsr_core: SPM access by an idle unit (dma %0b qact %0b mvm %0b)", dma_spm, qact_spm, mvm_spm);
      assert (!(spm_rd_rsp_valid[TSR_SPM_CL_MVM] && !mvm_busy_q))
        else $error("tsr_core: MVM SPM read response after the MVM unit's done");
      // I4: MVM SPM read accounting
      mvm_rd_out_q <= mvm_rd_out_q + 8'(mvm_rd_take) - 8'(spm_rd_rsp_valid[TSR_SPM_CL_MVM]);
      assert (!(spm_rd_rsp_valid[TSR_SPM_CL_MVM] && mvm_rd_out_q == '0 && !mvm_rd_take))
        else $error("tsr_core: MVM SPM read response without a request");
      if (mvm_done) begin
        assert (!mvm_spm && mvm_rd_out_q == '0)
          else $error("tsr_core: MVM done with SPM traffic (request %0b, reads outstanding %0d)",
                      mvm_spm, mvm_rd_out_q);
      end
      // I4: weight streams, per channel
      for (int c = 0; c < NUM_CH; c++) begin
        automatic logic req = s_rd_req_valid[c][TSR_SRC_WS] && s_rd_req_ready[c][TSR_SRC_WS];
        automatic logic rsp = s_rd_rsp_valid[c][TSR_SRC_WS] && s_rd_rsp_ready[c][TSR_SRC_WS];
        automatic logic [47:0] left = ws_exp_q[c] - ws_req_q[c];
        automatic logic [47:0] want_len = (left < 48'(64 - ws_addr_q[c][5:0])) ? left : 48'(64 - ws_addr_q[c][5:0]);
        assert (!((s_rd_req_valid[c][TSR_SRC_WS] || s_rd_rsp_valid[c][TSR_SRC_WS]) && !mvm_busy_q))
          else $error("tsr_core: weight-stream traffic on channel %0d while no MVM is in flight", c);
        assert (s_rd_rsp_ready[c][TSR_SRC_WS] || !HAS_MVM)
          else $error("tsr_core: WS rd_rsp_ready low on channel %0d (rtl-m3 §4: constant 1)", c);
        if (mvm_take) begin
          ws_exp_q[c]  <= ws_beats_exp(mvm_cmd.n, mvm_cmd.rows_per_ch, mvm_cmd.row_beats, c);
          ws_req_q[c]  <= '0;
          ws_rsp_q[c]  <= '0;
          ws_addr_q[c] <= 64'(mvm_cmd.w_off);
        end else begin
          if (req) begin
            ws_req_q[c]  <= ws_req_q[c] + 48'(s_rd_req_len[c][TSR_SRC_WS]);
            ws_addr_q[c] <= ws_addr_q[c] + 64'(s_rd_req_len[c][TSR_SRC_WS]);
            assert (64'(s_rd_req_addr[c][TSR_SRC_WS]) == ws_addr_q[c] &&
                    48'(s_rd_req_len[c][TSR_SRC_WS]) == want_len && want_len != '0)
              else $error("tsr_core: WS request ch%0d addr %0d len %0d, expected addr %0d len %0d (rtl-m3 §4)",
                          c, s_rd_req_addr[c][TSR_SRC_WS], s_rd_req_len[c][TSR_SRC_WS], ws_addr_q[c], want_len);
          end
          if (rsp) ws_rsp_q[c] <= ws_rsp_q[c] + 48'd1;
        end
        assert (!(rsp && ws_rsp_q[c] >= ws_req_q[c]))
          else $error("tsr_core: WS response beat on channel %0d without a request", c);
        if (mvm_done) begin
          assert (!s_rd_req_valid[c][TSR_SRC_WS] && ws_req_q[c] == ws_exp_q[c] && ws_rsp_q[c] == ws_req_q[c])
            else $error("tsr_core: MVM done on channel %0d with WS request %0b, %0d beats requested, %0d received, %0d expected (rows_c x ROW_BEATS)",
                        c, s_rd_req_valid[c][TSR_SRC_WS], ws_req_q[c], ws_rsp_q[c], ws_exp_q[c]);
        end
      end
    end
  end
`endif

endmodule
