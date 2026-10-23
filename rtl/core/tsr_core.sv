// tsr_core.sv — Tesseract M2 core top: CSR port, NUM_CH channel memory ports, done.
//
// Owner: core agent (docs/rtl-m2.md §2, block (f)). The structure below is the M2 netlist
// contract: it wires the block skeletons exactly as docs/rtl-m2.md §2/§3 describes. The core agent
// keeps this wiring (changes go through docs/rtl-m2.md) and adds integration logic/assertions.
//
// Core boundary (spec 코어 경계 인터페이스):
//   - CSR port: csr_valid/csr_ready/csr_we/csr_addr/csr_wdata/csr_rdata (tsr_csr contract).
//   - done: level = STATUS.DONE.
//   - memory port per channel c (packed [NUM_CH], index c): read request {addr_beat, len_beats, id},
//     read response {data, id, last}, write request {addr_beat, len_beats, id}, write data
//     {data, last}, write response {id}; all valid/ready (docs/rtl-m2.md §4). Every output is
//     driven from a register inside tsr_mem_arb.
// Clock/reset: one clock `clk`; `rst_n` synchronous, active low, already synchronised to clk.
// Scope: NOP END LI ADDI LD ST QACT FENCE (M2) and MVM (M3, HAS_MVM = 1).
//   HAS_MVM = 1 (default: the U200 v1 bitstream executes llama.cpp graphs, which need MVM):
//     tsr_mvm (docs/rtl-m3.md) on the cmdproc MVM dispatch port (mvm_cmd_* / mvm_done), SPM
//     client TSR_SPM_CL_MVM (broadcast reads, gather writes, ACC reads), the weight-stream
//     source TSR_SRC_WS of every channel's tsr_mem_arb (cluster c <-> arbiter c) and
//     perf_mvm_busy -> PERF_MVM_CYCLES (cycles the MVM unit holds an instruction). LANES must be
//     128 (one Q4_K beat; CFG0 reports it).
//   HAS_MVM = 0 (M2 test builds): no MVM unit, MVM is ILLEGAL_OP, the weight-stream sources of
//     the arbiters and SPM client TSR_SPM_CL_MVM are tied off; bit-identical to the M2 core.
// Implementation parameters (results identical for every legal value; tests vary them):
//   SPM_RD_LAT (SPM read latency seen by every SPM client; 1..4, U200 timing wants 4 at 4 MiB,
//   docs/rtl-m2.md §12), FETCH_BEATS (fetch buffer, power of two in 2..64), and for HAS_MVM = 1
//   (docs/rtl-m3.md §1): MVM_WS_FIFO (weight FIFO beats per cluster, power of two >= 64),
//   MVM_BC_PIPE / MVM_RES_PIPE (broadcast / result pipe stages per cluster; U200: >= 2 for
//   clusters in another SLR), MVM_ACT_RD_LAT, MVM_MAC_LAT, MVM_MUL_LAT, MVM_ADD_LAT (1..8),
//   MVM_WS_RD_LAT (1..4).
// Verification: tb/core/test_core.py (trace replay of traces/m2 with HAS_MVM = 0 and of
// traces/m3 + traces/gen with HAS_MVM = 1 per configuration, directed and random programs against
// the ISA simulator), tb/fastreplay (C++ replay of the big trace sets); simulation-only
// integration checks I1..I4 at the end of this file.

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
  parameter int unsigned     NIREG          = TSR_DEF_NIREG,
  parameter int unsigned     NSREG          = TSR_DEF_NSREG,
  parameter int unsigned     LOOP_DEPTH     = TSR_DEF_LOOP_DEPTH,
  parameter int unsigned     RD_OUTSTANDING = TSR_DEF_RD_OUTSTANDING,
  parameter longint unsigned CH_BYTES       = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W    = TSR_DEF_ADDR_BEAT_W,
  parameter bit              HAS_MVM        = 1'b1,
  parameter int unsigned     SPM_RD_LAT     = TSR_DEF_SPM_RD_LAT,
  parameter int unsigned     FETCH_BEATS    = 16,
  // MVM unit implementation parameters (HAS_MVM = 1; docs/rtl-m3.md §1)
  parameter int unsigned     MVM_WS_FIFO    = TSR_DEF_WS_FIFO_BEATS,
  parameter int unsigned     MVM_BC_PIPE    = TSR_DEF_MVM_BC_PIPE,
  parameter int unsigned     MVM_RES_PIPE   = TSR_DEF_MVM_RES_PIPE,
  parameter int unsigned     MVM_ACT_RD_LAT = TSR_DEF_ACT_RD_LAT,
  parameter int unsigned     MVM_MAC_LAT    = TSR_DEF_MAC_LAT,
  parameter int unsigned     MVM_MUL_LAT    = TSR_DEF_FMUL_LAT,
  parameter int unsigned     MVM_ADD_LAT    = TSR_DEF_FADD_LAT,
  parameter int unsigned     MVM_WS_RD_LAT  = TSR_DEF_WS_RD_LAT
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

  // ---- completion ----
  output logic                                   done,

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
    $fatal(1, "tsr_core: BEAT_BITS must be 512 (a beat is one 64-byte SPM line in v1)");
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
  if (NIREG != 16) begin : g_chk_nireg
    $fatal(1, "tsr_core: NIREG must be 16 (4-bit register fields, R8..R15 = ARG0..7)");
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
  if (HAS_MVM && (MVM_ADD_LAT < 1 || MVM_ADD_LAT > 8 || MVM_WS_RD_LAT < 1 || MVM_WS_RD_LAT > 4 ||
                  MVM_ACT_RD_LAT < 1 || MVM_MAC_LAT < 1 || MVM_MUL_LAT < 1)) begin : g_chk_mvmlat
    $fatal(1, "tsr_core: MVM latencies: ADD 1..8, WS_RD 1..4, ACT_RD / MAC / MUL >= 1 (docs/rtl-m3.md §1)");
  end

  localparam logic [31:0] CAPS = 32'd0;   // M2: no optional feature (VU, RLD/LOOP, TOPK, WAITM/JUMP)
  localparam int unsigned SPM_LINES = SPM_BYTES / TSR_LINE_BYTES;
  localparam int unsigned SPM_AW    = (SPM_LINES > 1) ? $clog2(SPM_LINES) : 1;
  localparam int unsigned unused_params = LOOP_DEPTH;   // M8 (LOOP)

  // -------------------------------------------------------------------------------------------
  // CSR block <-> command processor
  // -------------------------------------------------------------------------------------------
  logic                 run_start, run_abort, run_soft_reset, run_finish;
  logic [63:0]          run_cmd_addr;
  logic [31:0]          run_cmd_count;
  logic [7:0][31:0]     run_args;
  logic [TSR_ERR_W-1:0] run_err_code;
  logic [31:0]          run_err_pc, run_err_info, cur_pc;
  logic                 perf_retire, perf_mvm_busy;
  logic [NUM_CH-1:0]    perf_ch_rd_beat, perf_ch_wait;

  tsr_csr #(
    .NUM_CH(NUM_CH), .LANES(LANES), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES),
    .VL(VL), .NSREG(NSREG), .CAPS(CAPS)
  ) u_csr (
    .clk, .rst_n,
    .csr_valid, .csr_ready, .csr_we, .csr_addr, .csr_wdata, .csr_rdata, .done,
    .run_start, .run_cmd_addr, .run_cmd_count, .run_args, .run_abort, .run_soft_reset,
    .run_finish, .run_err_code, .run_err_pc, .run_err_info, .cur_pc,
    .perf_retire, .perf_mvm_busy, .perf_ch_rd_beat, .perf_ch_wait
  );

  // -------------------------------------------------------------------------------------------
  // Command processor, fetch, units
  // -------------------------------------------------------------------------------------------
  logic            fetch_start, fetch_stop, fetch_idle;
  logic [63:0]     fetch_cmd_addr;
  logic [31:0]     fetch_cmd_count;
  logic            insn_valid, insn_ready;
  tsr_fetch_insn_t insn;
  logic            dma_cmd_valid, dma_cmd_ready, dma_done;
  tsr_dma_cmd_t    dma_cmd;
  logic            qact_cmd_valid, qact_cmd_ready, qact_done;
  tsr_qact_cmd_t   qact_cmd;
  logic            mvm_cmd_valid, mvm_cmd_ready, mvm_done;
  tsr_mvm_cmd_t    mvm_cmd;

  tsr_cmdproc #(
    .NUM_CH(NUM_CH), .T_MAX(T_MAX), .K_MAX(K_MAX), .SPM_BYTES(SPM_BYTES), .CH_BYTES(CH_BYTES),
    .ADDR_BEAT_W(ADDR_BEAT_W), .HAS_MVM(HAS_MVM)
  ) u_cmdproc (
    .clk, .rst_n,
    .run_start, .run_cmd_addr, .run_cmd_count, .run_args, .run_abort, .run_soft_reset,
    .run_finish, .run_err_code, .run_err_pc, .run_err_info, .cur_pc, .perf_retire,
    .fetch_start, .fetch_cmd_addr, .fetch_cmd_count, .fetch_stop, .fetch_idle,
    .insn_valid, .insn_ready, .insn,
    .dma_cmd_valid, .dma_cmd_ready, .dma_cmd, .dma_done,
    .qact_cmd_valid, .qact_cmd_ready, .qact_cmd, .qact_done,
    .mvm_cmd_valid, .mvm_cmd_ready, .mvm_cmd, .mvm_done
  );

  // Read-source ports of the per-channel arbiters: [channel][source].
  logic [NUM_CH-1:0][TSR_NSRC-1:0]                   s_rd_req_valid, s_rd_req_ready;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][ADDR_BEAT_W-1:0]  s_rd_req_addr;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][TSR_LEN_W-1:0]    s_rd_req_len;
  logic [NUM_CH-1:0][TSR_NSRC-1:0][TSR_SRC_ID_W-1:0] s_rd_req_id;
  logic [NUM_CH-1:0][TSR_NSRC-1:0]                   s_rd_rsp_valid, s_rd_rsp_ready;
  logic [NUM_CH-1:0][BEAT_BITS-1:0]                  s_rd_rsp_data;
  logic [NUM_CH-1:0]                                 s_rd_rsp_last;
  logic [NUM_CH-1:0][TSR_SRC_ID_W-1:0]               s_rd_rsp_id;

  // Fetch (channel 0, source TSR_SRC_FETCH).
  logic                    f_rd_req_valid;
  logic [ADDR_BEAT_W-1:0]  f_rd_req_addr;
  logic [TSR_LEN_W-1:0]    f_rd_req_len;
  logic [TSR_SRC_ID_W-1:0] f_rd_req_id;
  logic                    f_rd_rsp_ready;

  tsr_fetch #(
    .ADDR_BEAT_W(ADDR_BEAT_W), .CH_BYTES(CH_BYTES), .BEAT_BITS(BEAT_BITS), .BUF_BEATS(FETCH_BEATS)
  ) u_fetch (
    .clk, .rst_n,
    .start(fetch_start), .cmd_addr(fetch_cmd_addr), .cmd_count(fetch_cmd_count), .stop(fetch_stop),
    .idle(fetch_idle),
    .insn_valid, .insn_ready, .insn,
    .rd_req_valid(f_rd_req_valid), .rd_req_ready(s_rd_req_ready[0][TSR_SRC_FETCH]),
    .rd_req_addr(f_rd_req_addr), .rd_req_len(f_rd_req_len), .rd_req_id(f_rd_req_id),
    .rd_rsp_valid(s_rd_rsp_valid[0][TSR_SRC_FETCH]), .rd_rsp_ready(f_rd_rsp_ready),
    .rd_rsp_data(s_rd_rsp_data[0]), .rd_rsp_last(s_rd_rsp_last[0]), .rd_rsp_id(s_rd_rsp_id[0])
  );

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
    .clk, .rst_n,
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
    .clk, .rst_n,
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
      .MAC_LAT(MVM_MAC_LAT), .MUL_LAT(MVM_MUL_LAT), .ADD_LAT(MVM_ADD_LAT), .WS_RD_LAT(MVM_WS_RD_LAT)
    ) u_mvm (
      .clk, .rst_n,
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
    // M2 build: no MVM unit; cmdproc decodes MVM as ILLEGAL_OP and never offers mvm_cmd.
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
    wire unused_mvm = &{1'b0, w_rd_req_ready, w_rd_rsp_valid, spm_rd_req_ready[TSR_SPM_CL_MVM],
                        spm_rd_rsp_valid[TSR_SPM_CL_MVM], spm_wr_ready[TSR_SPM_CL_MVM], mvm_cmd_valid,
                        mvm_cmd};
  end

  tsr_spm #(
    .SPM_BYTES(SPM_BYTES), .NCL(TSR_SPM_NCL), .RD_LAT(SPM_RD_LAT)
  ) u_spm (
    .clk, .rst_n,
    .rd_req_valid(spm_rd_req_valid), .rd_req_ready(spm_rd_req_ready), .rd_req_line(spm_rd_req_line),
    .rd_rsp_valid(spm_rd_rsp_valid), .rd_rsp_data(spm_rd_rsp_data),
    .wr_valid(spm_wr_valid), .wr_ready(spm_wr_ready), .wr_line(spm_wr_line), .wr_data(spm_wr_data),
    .wr_be(spm_wr_be)
  );

  // -------------------------------------------------------------------------------------------
  // Per-channel memory arbiters
  // -------------------------------------------------------------------------------------------
  for (genvar c = 0; c < NUM_CH; c++) begin : g_ch
    // Source FETCH: channel 0 only.
    if (c == 0) begin : g_fetch
      assign s_rd_req_valid[c][TSR_SRC_FETCH] = f_rd_req_valid;
      assign s_rd_req_addr[c][TSR_SRC_FETCH]  = f_rd_req_addr;
      assign s_rd_req_len[c][TSR_SRC_FETCH]   = f_rd_req_len;
      assign s_rd_req_id[c][TSR_SRC_FETCH]    = f_rd_req_id;
      assign s_rd_rsp_ready[c][TSR_SRC_FETCH] = f_rd_rsp_ready;
    end else begin : g_nofetch
      assign s_rd_req_valid[c][TSR_SRC_FETCH] = 1'b0;
      assign s_rd_req_addr[c][TSR_SRC_FETCH]  = '0;
      assign s_rd_req_len[c][TSR_SRC_FETCH]   = '0;
      assign s_rd_req_id[c][TSR_SRC_FETCH]    = '0;
      assign s_rd_rsp_ready[c][TSR_SRC_FETCH] = 1'b1;
    end
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
      .clk, .rst_n,
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

  // Arbiter outputs without a consumer: FETCH ready/valid on channels > 0 (fetch uses channel 0).
  logic unused_fetch;
  always_comb begin
    unused_fetch = 1'b0;
    for (int c = 1; c < NUM_CH; c++) begin
      unused_fetch |= s_rd_req_ready[c][TSR_SRC_FETCH] | s_rd_rsp_valid[c][TSR_SRC_FETCH];
    end
  end

  // -------------------------------------------------------------------------------------------
  // Integration checks (simulation only; docs/rtl-m2.md §3.0, §4 M7, §5; docs/rtl-m3.md §3.0,
  // §6.4, §10.4). Armed by the first clock edge with rst_n low (before it, Verilator's X
  // randomisation makes state arbitrary).
  //   I1 one instruction in flight: at most one unit is offered a command, never while a unit
  //      holds one; a unit pulses done only while it holds an instruction (never in its accepting
  //      cycle); no MVM command with HAS_MVM = 0. MVM: cmd_ready is low while it holds the
  //      instruction, and perf_mvm_busy equals "holds an instruction" in every cycle (from the
  //      cycle after the accepting edge up to and including the cycle of done = PERF_MVM_CYCLES).
  //   I2 a unit touches the SPM only while it holds an instruction (its done means its SPM
  //      effects are complete), so the clients never contend for the 1R1W port: DMA / QACT
  //      (request in the accepting cycle allowed, as in M2), MVM strictly after its accept; an
  //      MVM SPM read response only while it holds the instruction; no two units in one cycle.
  //      Weight-stream requests and responses (source WS of every arbiter) only while the MVM
  //      unit holds an instruction.
  //   I3 memory traffic only during a run; in the first cycle of done (STATUS.DONE) no request
  //      or write data is offered and no read is outstanding on any channel (P4 / M7).
  //   I4 MVM done => quiescent and exact (docs/rtl-m3.md §3.0, §4, §6.4): in the cycle of
  //      mvm_done no WS request is offered on any channel, every WS beat requested has been
  //      received (counted at the arbiter's source port, so fetch reads on channel 0 do not
  //      mask it), no MVM SPM request or write is offered and no MVM SPM read is outstanding;
  //      channel c requested exactly rows_c x ROW_BEATS beats (rows_c = max(0, min(R, N - c*R)))
  //      and never more at any point; WS requests follow the §4 burst rule (len =
  //      min(64 - addr % 64, beats left), contiguous from w_off).
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  logic run_q, done_q1, dma_busy_q, qact_busy_q, mvm_busy_q;
  wire  dma_take  = dma_cmd_valid && dma_cmd_ready;
  wire  qact_take = qact_cmd_valid && qact_cmd_ready;
  wire  mvm_take  = mvm_cmd_valid && mvm_cmd_ready;
  wire  dma_spm   = spm_rd_req_valid[TSR_SPM_CL_DMA] || spm_wr_valid[TSR_SPM_CL_DMA];
  wire  qact_spm  = spm_rd_req_valid[TSR_SPM_CL_QACT] || spm_wr_valid[TSR_SPM_CL_QACT];
  wire  mvm_spm   = spm_rd_req_valid[TSR_SPM_CL_MVM] || spm_wr_valid[TSR_SPM_CL_MVM];
  wire  mem_offer = (|m_rd_req_valid) || (|m_wr_req_valid) || (|m_wr_dat_valid);
  wire  any_busy  = dma_busy_q || qact_busy_q || mvm_busy_q;

  // Same arming scheme as tsr_spm / tsr_mem_arb: the initial block runs before Verilator's first
  // evaluation, which may already see a clock edge with X-randomised rst_n and state.
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;

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
    if (!rst_n) begin
      run_q        <= 1'b0;
      done_q1      <= 1'b0;
      dma_busy_q   <= 1'b0;
      qact_busy_q  <= 1'b0;
      mvm_busy_q   <= 1'b0;
      mvm_rd_out_q <= '0;
      ws_exp_q     <= '0;
      ws_req_q     <= '0;
      ws_rsp_q     <= '0;
      ws_addr_q    <= '0;
    end else if (chk_on_q) begin
      done_q1 <= done;
      if (run_start)  run_q <= 1'b1;
      if (run_finish) run_q <= 1'b0;
      if (dma_take)   dma_busy_q  <= 1'b1;
      if (dma_done)   dma_busy_q  <= 1'b0;
      if (qact_take)  qact_busy_q <= 1'b1;
      if (qact_done)  qact_busy_q <= 1'b0;
      if (mvm_take)   mvm_busy_q  <= 1'b1;
      if (mvm_done)   mvm_busy_q  <= 1'b0;
      // I1
      assert ($countones({dma_cmd_valid, qact_cmd_valid, mvm_cmd_valid}) <= 1 && !(mvm_cmd_valid && !HAS_MVM))
        else $error("tsr_core: more than one unit offered a command (dma %0b qact %0b mvm %0b, HAS_MVM %0d)",
                    dma_cmd_valid, qact_cmd_valid, mvm_cmd_valid, HAS_MVM);
      assert (!((dma_cmd_valid || qact_cmd_valid || mvm_cmd_valid) && any_busy))
        else $error("tsr_core: command offered while a unit holds an instruction");
      assert (!(dma_done && !dma_busy_q) && !(qact_done && !qact_busy_q) && !(mvm_done && !mvm_busy_q))
        else $error("tsr_core: unit done without an instruction in flight (dma %0b qact %0b mvm %0b)",
                    dma_done, qact_done, mvm_done);
      assert (!(mvm_cmd_ready && mvm_busy_q))
        else $error("tsr_core: MVM unit ready while it holds an instruction");
      assert (perf_mvm_busy == mvm_busy_q)
        else $error("tsr_core: perf_mvm_busy %0b while the MVM unit %s an instruction", perf_mvm_busy,
                    mvm_busy_q ? "holds" : "does not hold");
      // I2
      assert (!(dma_spm && !dma_busy_q && !dma_take) && !(qact_spm && !qact_busy_q && !qact_take) &&
              !(mvm_spm && !mvm_busy_q))
        else $error("tsr_core: SPM access by an idle unit (dma %0b qact %0b mvm %0b)", dma_spm, qact_spm, mvm_spm);
      assert ($countones({dma_spm, qact_spm, mvm_spm}) <= 1)
        else $error("tsr_core: two units use the SPM at the same time (dma %0b qact %0b mvm %0b)",
                    dma_spm, qact_spm, mvm_spm);
      assert (!(spm_rd_rsp_valid[TSR_SPM_CL_MVM] && !mvm_busy_q))
        else $error("tsr_core: MVM SPM read response after the MVM unit's done");
      // I3
      assert (!(mem_offer && !run_q))
        else $error("tsr_core: memory request outside a run");
      assert (!(done && !done_q1 && (mem_offer || (|perf_ch_wait))))
        else $error("tsr_core: DONE with memory traffic in flight (offer %0b, waiting %b)",
                    mem_offer, perf_ch_wait);
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
