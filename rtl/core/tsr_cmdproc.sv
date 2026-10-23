// tsr_cmdproc.sv — command processor: run sequencing, decode, ALL architectural checks,
// R0..R15, LI / ADDI, dispatch to the units, error reporting.
//
// Owner: cmdproc agent (docs/rtl-m2.md §2, block (a); per-instruction checks §7).
//
// Sequential execution (spec 실행 모델): one instruction at a time; the next instruction starts
// after the previous one's unit pulsed done. For each pc (tsr_sim.c tsr_sim_run order):
//   1. run_abort set            -> finish ABORTED, ERR_PC = pc, ERR_INFO = 0 (PC CSR unchanged)
//   2. pc == cmd_count          -> finish ILLEGAL_OP, ERR_PC = cmd_count, ERR_INFO = 0
//      (cur_pc := pc from here on, also in this case)
//   3. fetched entry range_err  -> finish DRAM_RANGE, ERR_PC = pc, ERR_INFO = low 32 bits of
//                                  (cmd_addr + pc/2)
//   4. decode + checks in the order ILLEGAL_OP -> BAD_FMT -> K_RANGE -> T_RANGE -> SPM_RANGE ->
//      DRAM_RANGE (operands in field order A1, A2, A3), exactly as tsr_sim.c ex_* (§7 lists them).
//      A failing instruction changes no state (registers, SPM, DRAM): checks complete before any
//      register write or dispatch. Units assume legal operands.
//   5. execute: NOP/FENCE nothing; END finish OK; LI/ADDI write R[rd] (R0 stays 0, 32-bit wrap);
//      LD/ST with n_lines == 0 retire without dispatch; otherwise dispatch tsr_dma_cmd_t /
//      tsr_qact_cmd_t (/ tsr_mvm_cmd_t when HAS_MVM) and wait for the unit's done.
//      HAS_MVM = 0: opcode MVM is ILLEGAL_OP (absent from this bitstream). CAPS = 0: MULI, RLD,
//      LOOP, WAITM, JUMP, VOP, RED, ROPE, TOPK, SOP, DQROW and undefined opcodes are ILLEGAL_OP.
//   6. perf_retire pulses for every instruction that completes without error (END included).
// Run start (run_start pulse): latch cmd_addr / cmd_count, R1..R7 := 0, R8..R15 := run_args,
//   cur_pc := 0, pulse fetch_start. run_soft_reset: R1..R15 := 0, cur_pc := 0 (only while idle;
//   a simultaneous run_start wins, which equals reset-then-start).
// Run end: pulse fetch_stop, wait for fetch_idle (every unit is idle: one instruction in flight and
//   its done was seen), then pulse run_finish with run_err_code (0 for END) / run_err_pc /
//   run_err_info (both 0 for END). STATUS.DONE therefore implies a quiescent memory port.
//
// Micro-architecture: a multi-cycle check pipeline, one instruction at a time (instruction rate
// is not critical; every stage is a short path for 300 MHz):
//   S_NEXT  abort / end-of-buffer / fetch-range checks, pop the instruction into IR (+ per-word
//           nonzero flags)
//   S_D0    opcode class, reserved fields and sub codes -> ILLEGAL_OP; register reads
//   S_D1    A1..A3 = R[x] + imm, ADDI result, BAD_FMT / K_RANGE / T_RANGE, QA line terms,
//           resolved strides                       (NOP END FENCE LI ADDI, illegal -> S_EXEC)
//   S_D2    QA_LINES, (T-1) * stride
//   S_D3    operand extents e (lines / beats, 48 bit)
//   S_D4    end addresses a + e (48 bit, no wrap)  (MVM waits here for R * ROW_BEATS)
//   S_D5    a + e > limit per operand
//   S_EXEC  error priority -> finish, or execute / dispatch
//   S_ISSUE cmd_valid until cmd_ready;  S_WAIT  until the unit's done
//   S_DRAIN fetch_stop issued; wait for fetch_idle, then run_finish
// Extents never wrap: e <= 2^43 (MVM weights) and a < 2^32, so 48-bit sums are exact.
// Cost per instruction (cycles between pops; measured by tb/cmdproc cp_throughput): NOP / FENCE /
// LI / ADDI 4; LD / ST / QACT 10 with a unit that accepts at once and pulses done one cycle later
// (8 + dispatch handshake + unit time); LD / ST with n_lines = 0: 8. insn_ready and every output
// are registers or decode of registers only.
// HAS_MVM = 1 (M3 room, verified against the simulator by tb/cmdproc): MVM decode and all checks of
// docs/rtl-m2.md §7 (BAD_FMT, K/T from w5, dst / x SPM extents, weight extent
// ceil(N/NUM_CH) * ROW_BEATS by a serial divider and shift-add multiplier, ~46 cycles), and
// tsr_mvm_cmd_t with rows_per_ch / row_beats / y_lines / qa_lines / resolved dst_stride (0 when
// N = 0). HAS_MVM = 0 builds none of it and decodes MVM as ILLEGAL_OP.

module tsr_cmdproc
  import tsr_pkg::*;
#(
  parameter int unsigned     NUM_CH      = TSR_DEF_NUM_CH,
  parameter int unsigned     T_MAX       = TSR_DEF_T_MAX,
  parameter int unsigned     K_MAX       = TSR_DEF_K_MAX,
  parameter int unsigned     SPM_BYTES   = TSR_DEF_SPM_BYTES,
  parameter longint unsigned CH_BYTES    = TSR_DEF_CH_BYTES,
  parameter int unsigned     ADDR_BEAT_W = TSR_DEF_ADDR_BEAT_W,
  parameter bit              HAS_MVM     = 1'b0
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // ---- run control from tsr_csr ----
  input  logic                    run_start,
  input  logic [63:0]             run_cmd_addr,
  input  logic [31:0]             run_cmd_count,
  input  logic [7:0][31:0]        run_args,
  input  logic                    run_abort,
  input  logic                    run_soft_reset,

  // ---- run status to tsr_csr ----
  output logic                    run_finish,
  output logic [TSR_ERR_W-1:0]    run_err_code,
  output logic [31:0]             run_err_pc,
  output logic [31:0]             run_err_info,
  output logic [31:0]             cur_pc,
  output logic                    perf_retire,

  // ---- tsr_fetch ----
  output logic                    fetch_start,
  output logic [63:0]             fetch_cmd_addr,
  output logic [31:0]             fetch_cmd_count,
  output logic                    fetch_stop,
  input  logic                    fetch_idle,
  input  logic                    insn_valid,
  output logic                    insn_ready,
  input  tsr_fetch_insn_t         insn,

  // ---- tsr_dma ----
  output logic                    dma_cmd_valid,
  input  logic                    dma_cmd_ready,
  output tsr_dma_cmd_t            dma_cmd,
  input  logic                    dma_done,

  // ---- tsr_qact ----
  output logic                    qact_cmd_valid,
  input  logic                    qact_cmd_ready,
  output tsr_qact_cmd_t           qact_cmd,
  input  logic                    qact_done,

  // ---- MVM unit (M3; tied off by tsr_core when HAS_MVM = 0) ----
  output logic                    mvm_cmd_valid,
  input  logic                    mvm_cmd_ready,
  output tsr_mvm_cmd_t            mvm_cmd,
  input  logic                    mvm_done
);

  // -------------------------------------------------------------------------------------------
  // Constants
  // -------------------------------------------------------------------------------------------
  localparam int unsigned     SPM_LINES  = SPM_BYTES / TSR_LINE_BYTES;
  localparam longint unsigned DRAM_LIM   = tsr_dram_limit(CH_BYTES, ADDR_BEAT_W);
  localparam int unsigned     EW         = 48;                 // extents and end addresses
  localparam logic [EW-1:0]   SPM_LIM_E  = EW'(SPM_LINES);
  localparam logic [EW-1:0]   DRAM_LIM_E = EW'(DRAM_LIM);
  localparam logic [31:0]     K_MAX_W    = 32'(K_MAX);
  localparam logic [31:0]     T_MAX_W    = 32'(T_MAX);
  localparam logic [4:0]      NUM_CH_W   = 5'(NUM_CH);

  if (NUM_CH < 1 || NUM_CH > TSR_NUM_CH_MAX || T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH ||
      K_MAX > 65280 || DRAM_LIM > (64'd1 << 40)) begin : g_chk_params
    $fatal(1, "tsr_cmdproc: parameter out of range (NUM_CH 1..16, T_MAX 1..8, K_MAX <= 65280, DRAM limit <= 2^40)");
  end

  typedef enum logic [3:0] {
    S_IDLE, S_NEXT, S_D0, S_D1, S_D2, S_D3, S_D4, S_D5, S_EXEC, S_ISSUE, S_WAIT, S_DRAIN
  } state_t;

  typedef enum logic [1:0] { U_DMA, U_QACT, U_MVM } unit_t;

  // -------------------------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------------------------
  state_t            st_q;
  logic [63:0]       run_addr_q;
  logic [31:0]       run_count_q;
  logic [31:0]       pc_q, cur_pc_q;
  logic [31:0]       count_m1_q;        // run_count - 1
  logic              at_end_q;          // pc_q == run_count_q
  logic [15:1][31:0] rf_q;              // R1..R15 (packed: visible to testbenches as one vector)
  wire  [15:0][31:0] rf_rd = {rf_q, 32'd0};   // R0..R15 for the read ports (R0 = 0)

  logic [6:0][31:0]  ir_q;              // w0..w6 of the instruction being checked / executed
  logic [7:1]        ir_nz_q;           // ir word i != 0

  // S_D0 results
  logic              op_end_q, op_li_q, op_addi_q, op_ldst_q, op_st_q, op_qact_q, op_mvm_q;
  logic              illegal_q;
  logic [31:0]       rva_q, rvb_q, rvc_q;
  // S_D1 results
  logic [31:0]       a1_q, a2_q, a3_q;
  logic              badfmt_q, kbad_q, tbad_q;
  logic              kind_q;            // QACT QA kind / MVM fmt (0: A32 / Q8_0, 1: A256 / Q4_K)
  logic [15:0]       k16_q;
  logic [3:0]        t4_q;
  logic [2:0]        tm1_q;             // T - 1 (T valid: 1..8)
  logic [15:0]       qs_l_q, d_l_q, bs_l_q;
  logic [31:0]       xl_q;              // QACT: f32 lines per token (K/16); MVM: y_lines
  logic [31:0]       stride_q;          // QACT: src stride; MVM: dst stride (0 -> default)
  // S_D2 .. S_D5 results
  logic [15:0]       qa_l_q;
  logic [34:0]       prod_q;            // (T-1) * stride
  logic [EW-1:0]     e1_q, e2_q;
  logic [EW-1:0]     sum1_q, sum2_q, sumd_q;
  logic              s1_bad_q, s2_bad_q, d_bad_q;

  // dispatch / finish
  unit_t             unit_q;
  logic              dma_v_q, qact_v_q, mvm_v_q;
  tsr_dma_cmd_t      dma_cmd_q;
  tsr_qact_cmd_t     qact_cmd_q;
  tsr_mvm_cmd_t      mvm_cmd_q;
  logic              fetch_start_q, fetch_stop_q, run_finish_q, retire_q;
  logic [TSR_ERR_W-1:0] fin_code_q;
  logic [31:0]       fin_pc_q, fin_info_q;

  // MVM weight extent (HAS_MVM): R = ceil(N / NUM_CH), ROW_BEATS(fmt, K), R * ROW_BEATS
  logic              mx_start, mx_ready;
  logic [31:0]       mx_rows;
  logic [31:0]       mx_row_beats;
  logic [EW-1:0]     mx_ext;

  // -------------------------------------------------------------------------------------------
  // S_D0: opcode class and reserved fields (tsr_sim.c exec_insn / ex_*; docs/rtl-m2.md §7)
  // -------------------------------------------------------------------------------------------
  wire [31:0] w0  = ir_q[0];
  wire [7:0]  opc = w0[7:0];
  wire [3:0]  sub = w0[11:8];

  logic d_end, d_li, d_addi, d_ldst, d_st, d_qact, d_mvm, d_ill;
  always_comb begin
    d_end  = 1'b0;
    d_li   = 1'b0;
    d_addi = 1'b0;
    d_ldst = 1'b0;
    d_st   = 1'b0;
    d_qact = 1'b0;
    d_mvm  = 1'b0;
    d_ill  = 1'b1;                       // MVM without HAS_MVM, CAPS features, undefined opcodes
    case (opc)
      TSR_OP_NOP, TSR_OP_FENCE: begin
        d_ill = (|(w0 & TSR_W0_RSV_NOP)) || (|ir_nz_q[7:1]);
      end
      TSR_OP_END: begin
        d_end = 1'b1;
        d_ill = (|(w0 & TSR_W0_RSV_NOP)) || (|ir_nz_q[7:1]);
      end
      TSR_OP_LI: begin
        d_li  = 1'b1;
        d_ill = (|(w0 & TSR_W0_RSV_LI)) || (|ir_nz_q[7:2]);
      end
      TSR_OP_ADDI: begin
        d_addi = 1'b1;
        d_ill  = (|(w0 & TSR_W0_RSV_ADDI)) || (|ir_nz_q[7:2]);
      end
      TSR_OP_LD, TSR_OP_ST: begin
        d_ldst = 1'b1;
        d_st   = (opc == TSR_OP_ST);
        d_ill  = (|(w0 & TSR_W0_RSV_LDST)) || (|ir_nz_q[7:4]) || ({1'b0, sub} >= NUM_CH_W);
      end
      TSR_OP_QACT: begin
        d_qact = 1'b1;
        d_ill  = (|(w0 & TSR_W0_RSV_QACT)) || (|ir_nz_q[7:6]) || (sub >= 4'd2);
      end
      TSR_OP_MVM: begin
        if (HAS_MVM) begin
          d_mvm = 1'b1;
          d_ill = (|(w0 & TSR_W0_RSV_MVM)) || (|(ir_q[5] & TSR_W5_RSV_MVM)) || ir_nz_q[7];
        end
      end
      default: ;
    endcase
  end

  // -------------------------------------------------------------------------------------------
  // S_D1: addresses, K / T checks, QA line terms, strides
  // -------------------------------------------------------------------------------------------
  logic [31:0] k_full, t_full;
  logic        is_kt;
  always_comb begin
    if (op_mvm_q) begin
      k_full = {16'd0, ir_q[5][15:0]};
      t_full = {28'd0, ir_q[5][19:16]};
    end else begin
      k_full = ir_q[3];
      t_full = ir_q[4];
    end
    is_kt = op_qact_q || op_mvm_q;
  end

  wire        d1_kind   = sub[0];
  wire        d1_bs_bad = d1_kind ? (k_full[7:0] != 8'd0) : (k_full[4:0] != 5'd0);
  wire        d1_kbad   = is_kt && ((k_full == 32'd0) || d1_bs_bad || (k_full > K_MAX_W));
  wire        d1_tbad   = is_kt && ((t_full == 32'd0) || (t_full > T_MAX_W));
  wire        d1_badfmt = op_mvm_q && (sub >= 4'd2);
  wire [15:0] d1_k16    = k_full[15:0];
  wire [15:0] d1_nblk   = d1_kind ? {8'd0, d1_k16[15:8]} : {5'd0, d1_k16[15:5]};
  wire [15:0] d1_qs_l   = 16'(({1'b0, d1_k16} + 17'd63) >> 6);      // ceil(K / 64)
  wire [15:0] d1_d_l    = 16'(({1'b0, d1_nblk} + 17'd15) >> 4);     // ceil(4 * nblk / 64)
  wire [15:0] d1_bs_l   = d1_kind ? 16'(({5'd0, d1_k16[15:4]} + 17'd31) >> 5) : 16'd0;
  wire [31:0] d1_y_l    = 32'(({1'b0, ir_q[4]} + 33'd15) >> 4);     // MVM: ceil(4N / 64)
  wire [31:0] d1_xl     = op_mvm_q ? d1_y_l : {20'd0, d1_k16[15:4]};
  wire [31:0] d1_sfield = op_mvm_q ? ir_q[6] : ir_q[5];               // stride field, 0 = default

  // -------------------------------------------------------------------------------------------
  // S_D2 / S_D3 / S_D4 / S_D5 datapath
  // -------------------------------------------------------------------------------------------
  wire [34:0] d2_prod = (tm1_q[0] ? {3'd0, stride_q}       : 35'd0)
                      + (tm1_q[1] ? {2'd0, stride_q, 1'b0} : 35'd0)
                      + (tm1_q[2] ? {1'd0, stride_q, 2'b0} : 35'd0);
  wire [19:0] d3_tqa  = 20'(t4_q) * 20'(qa_l_q);                     // T * QA_LINES
  wire [EW-1:0] d3_strided = EW'(prod_q) + EW'(xl_q);                 // (T-1) * stride + len

  // Range-checked operands per class: SPM A1 (LD/ST, QACT, MVM), SPM A2 (QACT, MVM),
  // DRAM (LD/ST: A2, MVM: A3).
  wire en1 = op_ldst_q || op_qact_q || op_mvm_q;
  wire en2 = op_qact_q || op_mvm_q;
  wire end_ = op_ldst_q || op_mvm_q;

  // -------------------------------------------------------------------------------------------
  // S_EXEC: error priority
  // -------------------------------------------------------------------------------------------
  logic [TSR_ERR_W-1:0] ex_code;
  logic [31:0]          ex_info;
  always_comb begin
    ex_info = 32'd0;
    if (illegal_q)      ex_code = TSR_ERR_ILLEGAL_OP;
    else if (badfmt_q)  ex_code = TSR_ERR_BAD_FMT;
    else if (kbad_q)    ex_code = TSR_ERR_K_RANGE;
    else if (tbad_q)    ex_code = TSR_ERR_T_RANGE;
    else if (s1_bad_q) begin
      ex_code = TSR_ERR_SPM_RANGE;
      ex_info = a1_q;
    end else if (s2_bad_q) begin
      ex_code = TSR_ERR_SPM_RANGE;
      ex_info = a2_q;
    end else if (d_bad_q) begin
      ex_code = TSR_ERR_DRAM_RANGE;
      ex_info = op_mvm_q ? a3_q : a2_q;
    end else begin
      ex_code = TSR_ERR_NONE;
    end
  end

  wire        ex_wr_en  = (op_li_q || op_addi_q) && (tsr_w0_rd(w0) != 4'd0);
  wire [31:0] ex_wr_val = op_li_q ? ir_q[1] : a1_q;                   // ADDI: R[ra] + w1

  wire unit_done = (unit_q == U_DMA)  ? dma_done :
                   (unit_q == U_QACT) ? qact_done : mvm_done;
  wire unit_take = (dma_v_q && dma_cmd_ready) || (qact_v_q && qact_cmd_ready) ||
                   (mvm_v_q && mvm_cmd_ready);

  // Fetch-range ERR_INFO: low 32 bits of the beat address CMD_ADDR + pc/2.
  wire [31:0] fetch_err_beat = run_addr_q[31:0] + {1'b0, pc_q[31:1]};

  // insn_ready depends on cmdproc registers only. An instruction handed over in a cycle in which
  // ABORT is seen is dropped (the run ends; fetch is stopped and drained).
  wire rdy = (st_q == S_NEXT) && !at_end_q;
  wire pop = rdy && insn_valid && !run_abort;

  // -------------------------------------------------------------------------------------------
  // Control
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q          <= S_IDLE;
      run_addr_q    <= 64'd0;
      run_count_q   <= 32'd0;
      pc_q          <= 32'd0;
      cur_pc_q      <= 32'd0;
      at_end_q      <= 1'b0;
      rf_q          <= '0;
      unit_q        <= U_DMA;
      dma_v_q       <= 1'b0;
      qact_v_q      <= 1'b0;
      mvm_v_q       <= 1'b0;
      fetch_start_q <= 1'b0;
      fetch_stop_q  <= 1'b0;
      run_finish_q  <= 1'b0;
      retire_q      <= 1'b0;
      fin_code_q    <= '0;
      fin_pc_q      <= 32'd0;
      fin_info_q    <= 32'd0;
    end else begin
      fetch_start_q <= 1'b0;
      fetch_stop_q  <= 1'b0;
      run_finish_q  <= 1'b0;
      retire_q      <= 1'b0;

      unique case (st_q)
        S_IDLE: begin
          if (run_soft_reset) begin
            rf_q     <= '0;
            cur_pc_q <= 32'd0;
          end
          if (run_start) begin
            run_addr_q    <= run_cmd_addr;
            run_count_q   <= run_cmd_count;
            count_m1_q    <= run_cmd_count - 32'd1;
            at_end_q      <= (run_cmd_count == 32'd0);
            pc_q          <= 32'd0;
            cur_pc_q      <= 32'd0;
            for (int unsigned i = 1; i < 8; i++) rf_q[i] <= 32'd0;
            for (int unsigned i = 0; i < 8; i++) rf_q[8 + i] <= run_args[i];
            fetch_start_q <= 1'b1;
            st_q          <= S_NEXT;
          end
        end

        S_NEXT: begin
          if (run_abort) begin                                   // 1. ABORT (PC CSR unchanged)
            fin_code_q   <= TSR_ERR_ABORTED;
            fin_pc_q     <= pc_q;
            fin_info_q   <= 32'd0;
            fetch_stop_q <= 1'b1;
            st_q         <= S_DRAIN;
          end else if (at_end_q) begin                           // 2. past CMD_COUNT
            cur_pc_q     <= pc_q;
            fin_code_q   <= TSR_ERR_ILLEGAL_OP;
            fin_pc_q     <= run_count_q;
            fin_info_q   <= 32'd0;
            fetch_stop_q <= 1'b1;
            st_q         <= S_DRAIN;
          end else if (insn_valid) begin
            cur_pc_q <= pc_q;
            if (insn.range_err) begin                            // 3. fetch beyond the DRAM limit
              fin_code_q   <= TSR_ERR_DRAM_RANGE;
              fin_pc_q     <= pc_q;
              fin_info_q   <= fetch_err_beat;
              fetch_stop_q <= 1'b1;
              st_q         <= S_DRAIN;
            end else begin
              st_q <= S_D0;
            end
          end
        end

        S_D0: st_q <= S_D1;

        S_D1: st_q <= (op_ldst_q || op_qact_q || op_mvm_q) ? S_D2 : S_EXEC;

        S_D2: st_q <= S_D3;

        S_D3: st_q <= S_D4;

        S_D4: if (!op_mvm_q || mx_ready) st_q <= S_D5;

        S_D5: st_q <= S_EXEC;

        S_EXEC: begin
          if (ex_code != TSR_ERR_NONE) begin
            fin_code_q   <= ex_code;
            fin_pc_q     <= pc_q;
            fin_info_q   <= ex_info;
            fetch_stop_q <= 1'b1;
            st_q         <= S_DRAIN;
          end else if (op_end_q) begin
            retire_q     <= 1'b1;
            fin_code_q   <= TSR_ERR_NONE;
            fin_pc_q     <= 32'd0;
            fin_info_q   <= 32'd0;
            fetch_stop_q <= 1'b1;
            st_q         <= S_DRAIN;
          end else if (op_ldst_q && ir_nz_q[3]) begin
            dma_v_q <= 1'b1;
            unit_q  <= U_DMA;
            st_q    <= S_ISSUE;
          end else if (op_qact_q) begin
            qact_v_q <= 1'b1;
            unit_q   <= U_QACT;
            st_q     <= S_ISSUE;
          end else if (op_mvm_q) begin
            mvm_v_q <= 1'b1;
            unit_q  <= U_MVM;
            st_q    <= S_ISSUE;
          end else begin                                         // NOP FENCE LI ADDI, LD/ST n = 0
            if (ex_wr_en) rf_q[tsr_w0_rd(w0)] <= ex_wr_val;
            retire_q <= 1'b1;
            pc_q     <= pc_q + 32'd1;
            at_end_q <= (pc_q == count_m1_q);
            st_q     <= S_NEXT;
          end
        end

        S_ISSUE: begin
          if (unit_take) begin
            dma_v_q  <= 1'b0;
            qact_v_q <= 1'b0;
            mvm_v_q  <= 1'b0;
            st_q     <= S_WAIT;
          end
        end

        S_WAIT: begin
          if (unit_done) begin
            retire_q <= 1'b1;
            pc_q     <= pc_q + 32'd1;
            at_end_q <= (pc_q == count_m1_q);
            st_q     <= S_NEXT;
          end
        end

        S_DRAIN: begin
          if (fetch_idle && !fetch_stop_q) begin
            run_finish_q <= 1'b1;
            st_q         <= S_IDLE;
          end
        end

        default: st_q <= S_IDLE;
      endcase
    end
  end

  // -------------------------------------------------------------------------------------------
  // Check pipeline registers (datapath; no reset needed: every field is written before use)
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (pop) begin
      ir_q <= insn.insn[6:0];          // w7 is only needed as ir_nz_q[7]
      for (int unsigned i = 1; i < 8; i++) ir_nz_q[i] <= (insn.insn[i] != 32'd0);
    end
    if (st_q == S_D0) begin
      op_end_q  <= d_end;
      op_li_q   <= d_li;
      op_addi_q <= d_addi;
      op_ldst_q <= d_ldst;
      op_st_q   <= d_st;
      op_qact_q <= d_qact;
      op_mvm_q  <= d_mvm;
      illegal_q <= d_ill;
      rva_q     <= rf_rd[tsr_w0_ra(w0)];
      rvb_q     <= rf_rd[tsr_w0_rb(w0)];
      rvc_q     <= rf_rd[tsr_w0_rc(w0)];
      s1_bad_q  <= 1'b0;                 // range flags are computed in S_D5 (range ops only)
      s2_bad_q  <= 1'b0;
      d_bad_q   <= 1'b0;
    end
    if (st_q == S_D1) begin
      a1_q     <= rva_q + ir_q[1];
      a2_q     <= rvb_q + ir_q[2];
      a3_q     <= rvc_q + ir_q[3];
      badfmt_q <= d1_badfmt;
      kbad_q   <= d1_kbad;
      tbad_q   <= d1_tbad;
      kind_q   <= d1_kind;
      k16_q    <= d1_k16;
      t4_q     <= t_full[3:0];
      tm1_q    <= 3'(t_full[3:0] - 4'd1);
      qs_l_q   <= d1_qs_l;
      d_l_q    <= d1_d_l;
      bs_l_q   <= d1_bs_l;
      xl_q     <= d1_xl;
      stride_q <= (d1_sfield != 32'd0) ? d1_sfield : d1_xl;
    end
    if (st_q == S_D2) begin
      qa_l_q <= qs_l_q + d_l_q + bs_l_q;
      prod_q <= d2_prod;
    end
    if (st_q == S_D3) begin
      if (op_qact_q) begin
        e1_q <= EW'(d3_tqa);             // dst: T * QA_LINES
        e2_q <= d3_strided;              // src: (T-1) * src_stride + K/16
      end else if (op_mvm_q) begin
        e1_q <= d3_strided;              // dst: (T-1) * dst_stride + ceil(4N/64)
        e2_q <= EW'(d3_tqa);             // x:   T * QA_LINES
      end else begin
        e1_q <= EW'(ir_q[3]);            // LD/ST: n_lines on both sides
        e2_q <= EW'(0);
      end
    end
    if (st_q == S_D4) begin
      sum1_q <= EW'(a1_q) + e1_q;
      sum2_q <= EW'(a2_q) + e2_q;
      sumd_q <= op_mvm_q ? (EW'(a3_q) + mx_ext) : (EW'(a2_q) + EW'(ir_q[3]));
    end
    if (st_q == S_D5) begin
      s1_bad_q <= en1  && (sum1_q > SPM_LIM_E);
      s2_bad_q <= en2  && (sum2_q > SPM_LIM_E);
      d_bad_q  <= end_ && (sumd_q > DRAM_LIM_E);
    end
    if (st_q == S_EXEC) begin
      dma_cmd_q.is_st     <= op_st_q;
      dma_cmd_q.ch        <= sub;
      dma_cmd_q.spm_line  <= a1_q;
      dma_cmd_q.dram_beat <= a2_q;
      dma_cmd_q.n_lines   <= ir_q[3];
      qact_cmd_q.kind       <= kind_q;
      qact_cmd_q.dst_line   <= a1_q;
      qact_cmd_q.src_line   <= a2_q;
      qact_cmd_q.k          <= k16_q;
      qact_cmd_q.t          <= t4_q;
      qact_cmd_q.src_stride <= stride_q;
      qact_cmd_q.qa_lines   <= qa_l_q;
      mvm_cmd_q.fmt         <= kind_q;
      mvm_cmd_q.acc         <= w0[12];
      mvm_cmd_q.dst_line    <= a1_q;
      mvm_cmd_q.x_line      <= a2_q;
      mvm_cmd_q.w_off       <= a3_q;
      mvm_cmd_q.n           <= ir_q[4];
      mvm_cmd_q.k           <= k16_q;
      mvm_cmd_q.t           <= t4_q;
      mvm_cmd_q.dst_stride  <= stride_q;
      mvm_cmd_q.y_lines     <= xl_q;
      mvm_cmd_q.rows_per_ch <= mx_rows;
      mvm_cmd_q.row_beats   <= mx_row_beats;
      mvm_cmd_q.qa_lines    <= qa_l_q;
    end
  end

  // -------------------------------------------------------------------------------------------
  // MVM weight extent: R = ceil(N / NUM_CH) (serial restoring division), ROW_BEATS(fmt, K),
  // R * ROW_BEATS (serial shift-add). Started in S_D1 of an MVM; S_D4 waits for mx_ready.
  // -------------------------------------------------------------------------------------------
  assign mx_start = (st_q == S_D1) && op_mvm_q;

  if (HAS_MVM) begin : g_mvm
    localparam int unsigned RB_W = 12;   // ROW_BEATS(Q8_0, 65280) = 1084 < 2^11
    typedef enum logic [1:0] { MX_IDLE, MX_DIV, MX_MUL, MX_DONE } mx_state_t;
    mx_state_t     mx_st_q;
    logic [32:0]   mx_num_q;             // dividend N + NUM_CH - 1, shifted out MSB first
    logic [4:0]    mx_rem_q;
    logic [31:0]   mx_quo_q;
    logic [5:0]    mx_cnt_q;
    logic [RB_W-1:0] mx_rb_q;
    logic [EW-1:0] mx_acc_q;
    logic [5:0]    mx_rem_sh;

    assign mx_rem_sh = {mx_rem_q, mx_num_q[32]};

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        mx_st_q <= MX_IDLE;
      end else begin
        unique case (mx_st_q)
          MX_IDLE: begin
            if (mx_start) begin
              mx_num_q <= {1'b0, ir_q[4]} + 33'(NUM_CH - 1);
              mx_rem_q <= 5'd0;
              mx_quo_q <= 32'd0;
              mx_cnt_q <= 6'd33;
              mx_rb_q  <= RB_W'(tsr_row_beats(sub[0], {16'd0, ir_q[5][15:0]}));
              mx_acc_q <= '0;
              mx_st_q  <= MX_DIV;
            end
          end
          MX_DIV: begin                  // one quotient bit per cycle, MSB first
            mx_num_q <= {mx_num_q[31:0], 1'b0};
            if (mx_rem_sh >= 6'(NUM_CH)) begin
              mx_rem_q <= 5'(mx_rem_sh - 6'(NUM_CH));
              mx_quo_q <= {mx_quo_q[30:0], 1'b1};
            end else begin
              mx_rem_q <= mx_rem_sh[4:0];
              mx_quo_q <= {mx_quo_q[30:0], 1'b0};
            end
            mx_cnt_q <= mx_cnt_q - 6'd1;
            if (mx_cnt_q == 6'd1) begin
              mx_cnt_q <= 6'(RB_W);
              mx_st_q  <= MX_MUL;
            end
          end
          MX_MUL: begin                  // acc += R << i for each set bit i of ROW_BEATS
            if (mx_rb_q[RB_W-1]) mx_acc_q <= {mx_acc_q[EW-2:0], 1'b0} + EW'(mx_quo_q);
            else                 mx_acc_q <= {mx_acc_q[EW-2:0], 1'b0};
            mx_rb_q  <= {mx_rb_q[RB_W-2:0], 1'b0};
            mx_cnt_q <= mx_cnt_q - 6'd1;
            if (mx_cnt_q == 6'd1) mx_st_q <= MX_DONE;
          end
          MX_DONE: begin
            if (st_q == S_EXEC) mx_st_q <= MX_IDLE;
          end
          default: mx_st_q <= MX_IDLE;
        endcase
      end
    end

    // ROW_BEATS for the dispatch struct (mx_rb_q is consumed by the multiply).
    logic [RB_W-1:0] mx_rbk_q;
    always_ff @(posedge clk) begin
      if (mx_start) mx_rbk_q <= RB_W'(tsr_row_beats(sub[0], {16'd0, ir_q[5][15:0]}));
    end

    assign mx_ready     = (mx_st_q == MX_DONE);
    assign mx_rows      = mx_quo_q;
    assign mx_row_beats = 32'(mx_rbk_q);
    assign mx_ext       = mx_acc_q;
  end else begin : g_nomvm
    assign mx_ready     = 1'b1;
    assign mx_rows      = 32'd0;
    assign mx_row_beats = 32'd0;
    assign mx_ext       = '0;
    wire unused_mvm = &{1'b0, mx_start, mvm_cmd_ready, mvm_done};
  end

  // -------------------------------------------------------------------------------------------
  // Outputs
  // -------------------------------------------------------------------------------------------
  assign run_finish      = run_finish_q;
  assign run_err_code    = fin_code_q;
  assign run_err_pc      = fin_pc_q;
  assign run_err_info    = fin_info_q;
  assign cur_pc          = cur_pc_q;
  assign perf_retire     = retire_q;
  assign fetch_start     = fetch_start_q;
  assign fetch_cmd_addr  = run_addr_q;
  assign fetch_cmd_count = run_count_q;
  assign fetch_stop      = fetch_stop_q;
  assign insn_ready      = rdy;
  assign dma_cmd_valid   = dma_v_q;
  assign dma_cmd         = dma_cmd_q;
  assign qact_cmd_valid  = qact_v_q;
  assign qact_cmd        = qact_cmd_q;
  assign mvm_cmd_valid   = mvm_v_q;
  assign mvm_cmd         = HAS_MVM ? mvm_cmd_q : '0;

  // insn.pc is informational (asserted below); k16 / kind feed the dispatch structs only.
  wire unused_ok = &{1'b0, insn.pc};

  // -------------------------------------------------------------------------------------------
  // Simulation checks of the internal protocols
  // -------------------------------------------------------------------------------------------
`ifndef SYNTHESIS
  // Checked only after a reset was applied (the state is random before; the initial block runs
  // before the first clock edge, after Verilator's X randomisation).
  logic rst_seen;
  initial rst_seen = 1'b0;
  always @(posedge clk) if (!rst_n) rst_seen <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && rst_seen) begin
      if (pop) assert (insn.pc == pc_q) else $error("tsr_cmdproc: fetch delivered pc %0d, expected %0d", insn.pc, pc_q);
      if (run_start) assert (st_q == S_IDLE) else $error("tsr_cmdproc: run_start while busy");
      if (dma_done)  assert (st_q == S_WAIT && unit_q == U_DMA)  else $error("tsr_cmdproc: unexpected dma_done");
      if (qact_done) assert (st_q == S_WAIT && unit_q == U_QACT) else $error("tsr_cmdproc: unexpected qact_done");
      if (HAS_MVM && mvm_done) assert (st_q == S_WAIT && unit_q == U_MVM) else $error("tsr_cmdproc: unexpected mvm_done");
    end
  end
`endif

endmodule
