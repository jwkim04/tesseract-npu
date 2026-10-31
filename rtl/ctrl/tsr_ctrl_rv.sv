// tsr_ctrl_rv.sv — the RISC-V control subsystem, clk_rv side (CC §2.1, §3, §4, §5, §12):
// Ibex, the riscv-dbg debug module, ITCM / DTCM, the bus (tsr_rv_bus), the control window
// (tsr_ctl_win), the DMI glue and the RISC-V reset sequencer. Every signal here is in clk_rv; the
// crossings are in tsr_ctrl_cdc, and tsr_ctrl instantiates both.
//
// Owner: ctrl agent (W4 item 3; CC §2.2 `tsr_ctrl_rv`). Out-of-context timing gate T1 (CC §14.3):
// rtl/ctrl/viv/ctrl_ooc.tcl.
//
// Ibex (third_party/ibex, CC §3.1-§3.2, CC-34): RV32IMC + Zicsr + Zifencei, machine mode only;
//   RV32E 0, RV32M = RV32M (default RV32MFast: CC §3.2's fallback 1, applied by gate T1;
//   RV32MSingleCycle is maxperf's), RV32B None, RV32ZC Zca (the pinned commit has
//   RV32ZC: Zca only, CC §3.2), RegFile = REGFILE (RegFileFPGA on the FPGA), WritebackStage /
//   BranchTargetALU = WB_STAGE / BT_ALU (1 / 1: "maxperf"), no ICache / ECC / branch predictor / PMP,
//   MHPMCounterNum 0, DbgTriggerEn 1 with DbgHwBreakNum 2, SecureIbex 0, BaseIsa RV32I (CHERIoT off),
//   DmBaseAddr 0x1000_0000 / mask 0xFFF, DmHaltAddr 0x1000_0800, DmExceptionAddr 0x1000_0808,
//   hart_id 0, boot_addr_i = RV_BOOT (captured at each reset release), fetch_enable_i on,
//   mcounteren read-only 0 (as tsr_rv). debug_req_i = the DM's request OR DB's dbg_req_rv (DB §3.3).
//   The core clock gate (prim_clock_gating) binds to rtl/ctrl/prim/prim_clock_gating.sv
//   (pass-through, CC §3.1).
// Debug module (third_party/riscv-dbg dm_top, CC §3.2, DB §3.1): NrHarts 1, BusWidth 32,
//   DmBaseAddress 0x1000_0000 (two-scratch ROM), SelectableHarts 1; slave port on both Ibex ports
//   (through tsr_rv_bus); system-bus master on the data side (tsr_rv_bus, with CC §4.3's
//   refusals); DMI from the CSR window through tsr_ctrl_cdc (no JTAG DTM). hartinfo = DataAddr
//   0x380, DataCount 2, dataaccess 1, nscratch 2. unavailable = the hart held in reset;
//   ndmreset_ack pulses whenever the hart leaves a reset (dmstatus.havereset, as the simulator's DM).
//   Its reset is the clk_rv domain reset AND rst_ext_n (ap_rst_n_2), never the RISC-V reset, so a
//   debugger stays attached across RISC-V resets (CC §12.2, DB §3.1).
// RISC-V reset sequence (CC §12.2, docs/ctrl-riscv.md §7.3): RV_RUN 1 -> 0 or ndmreset: (2) stop
//   granting on both Ibex ports, wait for the outstanding data access (bounded by the windows'
//   bounds), drop a push waiting for a credit (rv_kill); (3) assert Ibex's reset and clear the
//   control-window state (hart_clr) for 3 cycles; (4) RV_RESET_DONE. Release: RV_RUN = 1 and
//   ndmreset = 0 -> Ibex's reset deasserts 2 cycles later, fetch at RV_BOOT + 0x80. After a
//   hardware reset the hart is held (RV_RESET_DONE = 1). While held the load port owns the TCM write
//   port (load_mode, CC §4.2).
// DB's halt request (DB §3.3): pend is set by dbg_halt_req (a pulse from the core's toggle), and at
//   the reset release with dbg_halt_on_boot; cleared by RV_HALTED or dbg_halt_cancel;
//   dbg_req_rv = pend AND NOT RV_HALTED.
// Resets: rst_n = the clk_rv domain reset (ap_rst_n through tsr_rst_sync, synchronous use here);
// rst_ext_n = ap_rst_n_2 synchronised (AND-ed into the Ibex / DM resets only).

module tsr_ctrl_rv
  import tsr_tables_pkg::*;
  import tsr_ctrl_pkg::*;
#(
  parameter  int unsigned     ITCM_BYTES     = TSR_ITCM_BYTES,
  parameter  int unsigned     DTCM_BYTES     = TSR_DTCM_BYTES,
  parameter  int unsigned     SPM_BYTES      = 4 * 1024 * 1024,
  parameter  logic [31:0]     CAPS           = TSR_CAPS_VU | TSR_CAPS_ATT,
  parameter  logic [5:0][4:0] Q_DEPTH        = {5'd4, 5'd0, 5'd16, 5'd4, 5'd4, 5'd8},
  parameter  int unsigned     FIFO_DEPTH     = TSR_ISSUE_FIFO_DEPTH,
  parameter  int unsigned     LOG_DEPTH      = TSR_LOG_DEPTH,
  parameter  int unsigned     LOG_FIFO_DEPTH = 16,
  parameter  int unsigned     ISSUE_WAIT_MAX = TSR_ISSUE_WAIT_MAX,
  parameter  ibex_pkg::rv32m_e   RV32M       = ibex_pkg::RV32MFast,
  parameter  ibex_pkg::regfile_e REGFILE     = ibex_pkg::RegFileFPGA,
  parameter  bit              WB_STAGE       = 1'b1,
  parameter  bit              BT_ALU         = 1'b1,
  localparam int unsigned     FAW            = $clog2(FIFO_DEPTH),
  localparam int unsigned     LAW            = $clog2(LOG_FIFO_DEPTH)
) (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      rst_ext_n,
  // core state, synchronised into clk_rv (tsr_ctrl_cdc)
  input  logic                      rv_run,
  input  logic [31:0]               rv_boot,
  input  logic                      dbg_stop,
  input  logic                      dbg_halt_req,
  input  logic                      dbg_halt_cancel,
  input  logic                      dbg_halt_on_boot,
  input  logic                      irq_lq,
  input  logic                      irq_lcq,
  input  logic                      perf_time_en,
  input  logic                      perf_evt_en,
  input  logic [3:0]                q_hold,
  input  logic [5:0]                q_err,
  input  logic [5:0]                q_nonempty,
  input  logic [63:0]               mtime,
  input  logic [5:0][27:0]          s_seq,
  input  logic [5:0][27:0]          d_seq,
  input  err_rec_t [5:0]            err_rec,
  // clk_rv state for the core
  output logic [5:0][27:0]          issued_seq,
  output logic                      rv_halted,
  output logic                      rv_sleep,
  output logic                      rv_reset_done,
  output logic                      rv_running,
  output logic                      rv_ndmreset,
  output logic                      rv_dmactive,    // dmcontrol.dmactive of the debug module (DBG_RV_STATE.DMACTIVE)
  output logic                      rv_halt_pend,
  output logic                      push_timeout_ev,
  output logic                      bridge_timeout_ev,
  output logic [31:0]               perf_push_stall,
  output logic [31:0]               perf_bridge_acc,
  output logic [31:0]               perf_bridge_wait,
  // issue FIFO and log FIFO, write sides
  output logic                      if_wr,
  output logic [TSR_IFIFO_BITS-1:0] if_wdata,
  input  logic [FAW:0]              if_free,
  input  logic                      if_empty,
  output logic                      log_wr,
  output logic [31:0]               log_wdata,
  input  logic [LAW:0]              log_free,
  // handshakes: A sides (requests into the core)
  input  logic                      x_ready,
  output logic                      x_req,
  output xreq_t                     x_req_data,
  input  logic                      x_done,
  input  logic                      x_timeout,
  input  logic [XRSP_W-1:0]         x_rsp,
  input  logic                      lq_ready,
  output logic                      lq_req,
  output lqreq_t                    lq_req_data,
  input  logic                      lq_done,
  input  logic                      lq_timeout,
  input  logic [LQRSP_W-1:0]        lq_rsp,
  output logic [31:0]               rt_state,
  output logic [31:0]               rt_info,
  output logic                      rt_wr,
  // handshakes: B sides (requests from the core)
  input  logic                      ld_valid,
  input  ldreq_t                    ld_req,
  output logic                      ld_rsp,
  output logic [LDRSP_W-1:0]        ld_rsp_data,
  input  logic                      dmi_valid,
  input  logic [DMIREQ_W-1:0]       dmi_req,
  output logic                      dmi_rsp,
  output logic [DMIRSP_W-1:0]       dmi_rsp_data,
  // DB window (W4-B)
  output logic                      db_req,
  output logic                      db_we,
  output logic [11:0]               db_off,
  output logic [31:0]               db_wdata,
  input  logic [31:0]               db_rdata,
  input  logic                      db_irq4,
  input  logic                      db_irq6,
  // Ibex taps for the DB block (clk_rv; tsr_dbg_rv with HAS_RV_TAPS, DB §3.7, §6.2, §10.2; W4-B integrate):
  // the data port, ibex_core's crash_dump, the sleep / reset levels and the double-fault pulse
  output logic                      tap_data_req,
  output logic                      tap_data_gnt,
  output logic [31:0]               tap_data_addr,
  output logic                      tap_data_we,
  output logic [3:0]                tap_data_be,
  output logic [31:0]               tap_cd_pc,
  output logic [31:0]               tap_cd_npc,
  output logic [31:0]               tap_cd_daddr,
  output logic [31:0]               tap_cd_epc,
  output logic [31:0]               tap_cd_eaddr,
  output logic                      tap_sleep,
  output logic                      tap_halted,
  output logic                      tap_in_reset,
  output logic                      tap_dfault
);

  localparam int unsigned IAW = $clog2(ITCM_BYTES / 4);
  localparam int unsigned DAW = $clog2(DTCM_BYTES / 4);

  // ------------------------------------------------------------------ RISC-V reset sequencer
  typedef enum logic [2:0] { SQ_HELD, SQ_REL, SQ_RUN, SQ_DRAIN, SQ_CLR } sq_e;
  sq_e         sq_q;
  logic [1:0]  sq_cnt_q;
  logic        hart_rst_n_q;     // Ibex reset (registered: glitch free)
  logic        ibex_run, ibex_busy, rv_kill, hart_clr, load_mode, ndmreset_ack;
  logic [31:0] boot_q;
  logic        ndmreset;
  logic        dm_active;      // dmcontrol.dmactive: exported to the core (DBG_RV_STATE.DMACTIVE, DB §10.1)

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sq_q         <= SQ_HELD;
      sq_cnt_q     <= '0;
      hart_rst_n_q <= 1'b0;
      ndmreset_ack <= 1'b0;
      boot_q       <= '0;
    end else begin
      ndmreset_ack <= 1'b0;
      unique case (sq_q)
        SQ_HELD: if (rv_run && !ndmreset) begin
          sq_q <= SQ_REL; sq_cnt_q <= '0;
        end
        SQ_REL: begin
          if (!rv_run || ndmreset) begin
            sq_q <= SQ_HELD;
          end else if (sq_cnt_q == 2'd1) begin
            boot_q       <= rv_boot;
            hart_rst_n_q <= 1'b1;
            ndmreset_ack <= 1'b1;
            sq_q         <= SQ_RUN;
          end else begin
            sq_cnt_q <= sq_cnt_q + 1'b1;
          end
        end
        SQ_RUN: if (!rv_run || ndmreset) begin
          sq_q <= SQ_DRAIN; sq_cnt_q <= '0;
        end
        SQ_DRAIN: begin
          // The grants stop now; an access granted in this or the next cycle is still accepted.
          if (sq_cnt_q != 2'd2) sq_cnt_q <= sq_cnt_q + 1'b1;
          else if (!ibex_busy) begin
            hart_rst_n_q <= 1'b0;
            sq_q <= SQ_CLR; sq_cnt_q <= '0;
          end
        end
        SQ_CLR: begin
          if (sq_cnt_q == 2'd2) sq_q <= SQ_HELD;
          else sq_cnt_q <= sq_cnt_q + 1'b1;
        end
        default: sq_q <= SQ_HELD;
      endcase
    end
  end

  assign ibex_run      = sq_q == SQ_RUN;
  assign rv_kill       = sq_q == SQ_DRAIN;
  assign hart_clr      = sq_q == SQ_CLR;
  assign load_mode     = !hart_rst_n_q;
  assign rv_reset_done = sq_q == SQ_HELD;
  assign rv_running    = hart_rst_n_q;
  assign rv_ndmreset   = ndmreset;
  assign rv_dmactive   = dm_active;

  logic hart_rst_n, dm_rst_n;
  assign hart_rst_n = hart_rst_n_q && rst_n && rst_ext_n;
  assign dm_rst_n   = rst_n && rst_ext_n;

  // DB halt request (DB §3.3).
  logic pend_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      pend_q <= 1'b0;
    end else if (dbg_halt_cancel || rv_halted) begin
      pend_q <= 1'b0;
    end else if (dbg_halt_req || (ndmreset_ack && dbg_halt_on_boot)) begin
      pend_q <= 1'b1;
    end
  end
  assign rv_halt_pend = pend_q;

  // ------------------------------------------------------------------ Ibex
  logic        instr_req, instr_gnt, instr_rvalid, instr_err;
  logic [31:0] instr_addr, instr_rdata;
  logic        data_req, data_gnt, data_rvalid, data_we, data_err;
  logic [3:0]  data_be;
  logic [31:0] data_addr, data_wdata, data_rdata;
  logic        irq_timer;
  logic [14:0] irq_fast;
  logic        dm_debug_req, debug_req;
  ibex_pkg::crash_dump_t crash_dump;
  logic        core_sleep;

  assign debug_req = dm_debug_req || (pend_q && !rv_halted);

`ifdef RVFI
  // RISC-V Formal Interface (simulation builds; tb/ctrl's lockstep against tsr_rv, CC §15.4).
  logic        rvfi_valid, rvfi_trap, rvfi_halt, rvfi_intr, rvfi_ext_debug_mode;
  logic [63:0] rvfi_order;
  logic [31:0] rvfi_insn, rvfi_rd_wdata, rvfi_pc_rdata, rvfi_pc_wdata, rvfi_mem_addr;
  logic [31:0] rvfi_mem_rdata, rvfi_mem_wdata;
  logic [4:0]  rvfi_rd_addr;
  logic [3:0]  rvfi_mem_rmask, rvfi_mem_wmask;
  logic [1:0]  rvfi_mode;
  // Read by the testbench through hierarchical references (tb/ctrl/sv/tb_ctrl.sv).
  wire unused_rvfi = &{1'b0, rvfi_valid, rvfi_trap, rvfi_halt, rvfi_intr, rvfi_ext_debug_mode, rvfi_order,
                       rvfi_insn, rvfi_rd_wdata, rvfi_pc_rdata, rvfi_pc_wdata, rvfi_mem_addr,
                       rvfi_mem_rdata, rvfi_mem_wdata, rvfi_rd_addr, rvfi_mem_rmask, rvfi_mem_wmask,
                       rvfi_mode};
`endif

  // Ibex outputs this subsystem does not use (shadow / lockstep / alert / integrity ports, the
  // optional RVFI fields) are left open by name; the waiver covers this instance only (rtl/lint.sh
  // rule: targeted, inline; Ibex's own waiver file would switch the rule off for the whole core).
  /* verilator lint_off PINCONNECTEMPTY */
  ibex_top #(
    .BaseIsa         (ibex_pkg::BaseIsaRV32I),
    .PMPEnable       (1'b0),
    .MHPMCounterNum  (0),
    .RV32E           (1'b0),
    .RV32M           (RV32M),
    .RV32B           (ibex_pkg::RV32BNone),
    .RV32ZC          (ibex_pkg::RV32Zca),
    .RegFile         (REGFILE),
    .BranchTargetALU (BT_ALU),
    .WritebackStage  (WB_STAGE),
    .ICache          (1'b0),
    .ICacheECC       (1'b0),
    .BranchPredictor (1'b0),
    .DbgTriggerEn    (1'b1),
    .DbgHwBreakNum   (2),
    .SecureIbex      (1'b0),
    .ICacheScramble  (1'b0),
    .DmBaseAddr      (TSR_RV_DM_BASE),
    .DmAddrMask      (TSR_RV_DM_SIZE - 1),
    .DmHaltAddr      (TSR_RV_DM_BASE + 32'h800),
    .DmExceptionAddr (TSR_RV_DM_BASE + 32'h808)
  ) u_ibex (
    .clk_i                   (clk),
    .rst_ni                  (hart_rst_n),
    .test_en_i               (1'b0),
    .ram_cfg_icache_tag_i    ('0),
    .ram_cfg_icache_tag_o    (),
    .ram_cfg_icache_data_i   ('0),
    .ram_cfg_icache_data_o   (),
    .cheriot_enable_i        (ibex_pkg::IbexMuBiOff),
    .hart_id_i               (32'h0),
    .boot_addr_i             (boot_q),
    .trvk_heap_base_addr_i   (32'h0),
    .instr_req_o             (instr_req),
    .instr_gnt_i             (instr_gnt),
    .instr_rvalid_i          (instr_rvalid),
    .instr_addr_o            (instr_addr),
    .instr_rdata_i           (instr_rdata),
    .instr_rdata_intg_i      (7'h0),
    .instr_err_i             (instr_err),
    .data_req_o              (data_req),
    .data_gnt_i              (data_gnt),
    .data_rvalid_i           (data_rvalid),
    .data_we_o               (data_we),
    .data_be_o               (data_be),
    .data_addr_o             (data_addr),
    .data_wdata_o            (data_wdata),
    .data_wdata_intg_o       (),
    .data_tag_o              (),
    .data_rdata_i            (data_rdata),
    .data_rdata_intg_i       (7'h0),
    .data_tag_i              (1'b0),
    .data_err_i              (data_err),
    .trvk_revbm_req_o        (),
    .trvk_revbm_gnt_i        (1'b0),
    .trvk_revbm_rvalid_i     (1'b0),
    .trvk_revbm_addr_o       (),
    .trvk_revbm_rdata_i      (32'h0),
    .trvk_revbm_rdata_intg_i (7'h0),
    .trvk_revbm_err_i        (1'b0),
    .irq_software_i          (1'b0),
    .irq_timer_i             (irq_timer),
    .irq_external_i          (1'b0),
    .irq_fast_i              (irq_fast),
    .irq_nm_i                (1'b0),
    .scramble_key_valid_i    (1'b0),
    .scramble_key_i          ('0),
    .scramble_nonce_i        ('0),
    .scramble_req_o          (),
    .debug_req_i             (debug_req),
    .crash_dump_o            (crash_dump),
    .double_fault_seen_o     (tap_dfault),
`ifdef RVFI
    .rvfi_valid              (rvfi_valid),
    .rvfi_order              (rvfi_order),
    .rvfi_insn               (rvfi_insn),
    .rvfi_trap               (rvfi_trap),
    .rvfi_halt               (rvfi_halt),
    .rvfi_intr               (rvfi_intr),
    .rvfi_mode               (rvfi_mode),
    .rvfi_ixl                (),
    .rvfi_rs1_addr           (),
    .rvfi_rs2_addr           (),
    .rvfi_rs3_addr           (),
    .rvfi_rs1_rdata          (),
    .rvfi_rs2_rdata          (),
    .rvfi_rs3_rdata          (),
    .rvfi_rs1_rcap           (),
    .rvfi_rs2_rcap           (),
    .rvfi_rd_wcap            (),
    .rvfi_rd_addr            (rvfi_rd_addr),
    .rvfi_rd_wdata           (rvfi_rd_wdata),
    .rvfi_pc_rdata           (rvfi_pc_rdata),
    .rvfi_pc_wdata           (rvfi_pc_wdata),
    .rvfi_mem_addr           (rvfi_mem_addr),
    .rvfi_mem_rmask          (rvfi_mem_rmask),
    .rvfi_mem_wmask          (rvfi_mem_wmask),
    .rvfi_mem_rdata          (rvfi_mem_rdata),
    .rvfi_mem_wdata          (rvfi_mem_wdata),
    .rvfi_mem_is_cap         (),
    .rvfi_mem_rcap           (),
    .rvfi_mem_wcap           (),
    .rvfi_ext_pre_mip        (),
    .rvfi_ext_post_mip       (),
    .rvfi_ext_nmi            (),
    .rvfi_ext_nmi_int        (),
    .rvfi_ext_debug_req      (),
    .rvfi_ext_debug_mode     (rvfi_ext_debug_mode),
    .rvfi_ext_rf_wr_suppress (),
    .rvfi_ext_mcycle         (),
    .rvfi_ext_mhpmcounters   (),
    .rvfi_ext_mhpmcountersh  (),
    .rvfi_ext_ic_scr_key_valid (),
    .rvfi_ext_irq_valid      (),
    .rvfi_ext_expanded_insn_valid (),
    .rvfi_ext_expanded_insn  (),
    .rvfi_ext_expanded_insn_last (),
`endif
    .fetch_enable_i          (ibex_pkg::IbexMuBiOn),
    .mcounteren_writable_i   (ibex_pkg::IbexMuBiOff),
    .alert_minor_o           (),
    .alert_major_internal_o  (),
    .alert_major_bus_o       (),
    .core_sleep_o            (core_sleep),
    .scan_rst_ni             (1'b1),
    .lockstep_cmp_en_o       (),
    .data_req_shadow_o       (),
    .data_we_shadow_o        (),
    .data_be_shadow_o        (),
    .data_addr_shadow_o      (),
    .data_wdata_shadow_o     (),
    .data_wdata_intg_shadow_o (),
    .instr_req_shadow_o      (),
    .instr_addr_shadow_o     ()
  );
  /* verilator lint_on PINCONNECTEMPTY */

  // RV_SLEEP crosses into the core domain: registered here (no logic in front of a synchroniser).
  always_ff @(posedge clk) begin
    if (!rst_n) rv_sleep <= 1'b0;
    else        rv_sleep <= core_sleep && hart_rst_n_q;
  end

  // ------------------------------------------------------------------ debug module
  logic        dm_req, dm_we;
  logic [31:0] dm_addr, dm_wdata, dm_rdata;
  logic [3:0]  dm_be;
  logic        sba_req, sba_we, sba_gnt, sba_rvalid, sba_err;
  logic [31:0] sba_addr, sba_wdata, sba_rdata;
  logic [3:0]  sba_be;
  logic        dmi_req_valid, dmi_req_ready, dmi_resp_valid;
  dm::dmi_resp_t dmi_resp;
  dm::hartinfo_t hartinfo;

  assign hartinfo = '{zero1: '0, nscratch: 4'd2, zero0: '0, dataaccess: 1'b1,
                      datasize: dm::DataCount, dataaddr: dm::DataAddr};

  dm_top #(
    .NrHarts         (1),
    .BusWidth        (32),
    .DmBaseAddress   (TSR_RV_DM_BASE),
    .SelectableHarts (1'b1),
    .ReadByteEnable  (1'b1)
  ) u_dm (
    .clk_i                (clk),
    .rst_ni               (dm_rst_n),
    .next_dm_addr_i       (32'h0),
    .testmode_i           (1'b0),
    .ndmreset_o           (ndmreset),
    .ndmreset_ack_i       (ndmreset_ack),
    .dmactive_o           (dm_active),
    .debug_req_o          (dm_debug_req),
    .unavailable_i        (!hart_rst_n_q),
    .hartinfo_i           (hartinfo),
    .slave_req_i          (dm_req),
    .slave_we_i           (dm_we),
    .slave_addr_i         (dm_addr),
    .slave_be_i           (dm_be),
    .slave_wdata_i        (dm_wdata),
    .slave_rdata_o        (dm_rdata),
    .master_req_o         (sba_req),
    .master_add_o         (sba_addr),
    .master_we_o          (sba_we),
    .master_wdata_o       (sba_wdata),
    .master_be_o          (sba_be),
    .master_gnt_i         (sba_gnt),
    .master_r_valid_i     (sba_rvalid),
    .master_r_err_i       (sba_err),
    .master_r_other_err_i (1'b0),
    .master_r_rdata_i     (sba_rdata),
    .dmi_rst_ni           (1'b1),
    .dmi_req_valid_i      (dmi_req_valid),
    .dmi_req_ready_o      (dmi_req_ready),
    .dmi_req_i            (dmi_req),
    .dmi_resp_valid_o     (dmi_resp_valid),
    .dmi_resp_ready_i     (1'b1),
    .dmi_resp_o           (dmi_resp)
  );

  // DMI glue: one request of the CSR window's handshake at a time (CC §11.2, DB §3.2).
  // dmi_rsp_pend_q keeps a request from being sent twice in the cycle after its answer, while the
  // handshake's valid falls (the DM's response FIFO answers one cycle after the request at the
  // earliest).
  logic dmi_sent_q, dmi_rsp_pend_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      dmi_sent_q <= 1'b0;
    end else if (dmi_rsp) begin
      dmi_sent_q <= 1'b0;
    end else if (dmi_req_valid && dmi_req_ready) begin
      dmi_sent_q <= 1'b1;
    end
  end
  assign dmi_req_valid = dmi_valid && !dmi_sent_q && !dmi_rsp_pend_q;
  assign dmi_rsp       = dmi_sent_q && dmi_resp_valid;
  assign dmi_rsp_data  = dmi_resp;

  always_ff @(posedge clk) begin
    if (!rst_n) dmi_rsp_pend_q <= 1'b0;
    else        dmi_rsp_pend_q <= dmi_rsp;
  end

  // ------------------------------------------------------------------ TCMs (CC §4.2)
  logic           itcm_we, itcm_re, dtcm_we, dtcm_re;
  logic [IAW-1:0] itcm_waddr, itcm_raddr;
  logic [DAW-1:0] dtcm_waddr, dtcm_raddr;
  logic [31:0]    itcm_wdata, itcm_rdata, dtcm_wdata, dtcm_rdata;
  logic [3:0]     itcm_wbe, dtcm_wbe;

  tsr_sram_1r1w #(.DEPTH(ITCM_BYTES / 4), .WIDTH(32), .RD_LAT(1)) u_itcm (
    .clk(clk), .we(itcm_we), .waddr(itcm_waddr), .wdata(itcm_wdata), .wbe(itcm_wbe),
    .re(itcm_re), .raddr(itcm_raddr), .rdata(itcm_rdata)
  );
  tsr_sram_1r1w #(.DEPTH(DTCM_BYTES / 4), .WIDTH(32), .RD_LAT(1)) u_dtcm (
    .clk(clk), .we(dtcm_we), .waddr(dtcm_waddr), .wdata(dtcm_wdata), .wbe(dtcm_wbe),
    .re(dtcm_re), .raddr(dtcm_raddr), .rdata(dtcm_rdata)
  );

  // ------------------------------------------------------------------ bus and windows
  logic        w_req, w_we, w_rsp, w_err;
  master_e     w_master;
  region_e     w_region;
  logic [3:0]  w_be;
  logic [31:0] w_addr, w_wdata, w_pc, w_rdata;

  tsr_rv_bus #(.ITCM_BYTES(ITCM_BYTES), .DTCM_BYTES(DTCM_BYTES), .SPM_BYTES(SPM_BYTES)) u_bus (
    .clk          (clk),
    .rst_n        (rst_n),
    .hart_rst_n   (hart_rst_n_q),
    .ibex_run     (ibex_run),
    .load_mode    (load_mode),
    .ibex_busy    (ibex_busy),
    .rv_halted    (rv_halted),
    .instr_req    (instr_req),
    .instr_addr   (instr_addr),
    .instr_gnt    (instr_gnt),
    .instr_rvalid (instr_rvalid),
    .instr_rdata  (instr_rdata),
    .instr_err    (instr_err),
    .data_req     (data_req),
    .data_we      (data_we),
    .data_be      (data_be),
    .data_addr    (data_addr),
    .data_wdata   (data_wdata),
    .data_pc      (crash_dump.current_pc),
    .data_gnt     (data_gnt),
    .data_rvalid  (data_rvalid),
    .data_rdata   (data_rdata),
    .data_err     (data_err),
    .sba_req      (sba_req),
    .sba_we       (sba_we),
    .sba_addr     (sba_addr),
    .sba_wdata    (sba_wdata),
    .sba_be       (sba_be),
    .sba_gnt      (sba_gnt),
    .sba_rvalid   (sba_rvalid),
    .sba_err      (sba_err),
    .sba_rdata    (sba_rdata),
    .ld_valid     (ld_valid),
    .ld_req       (ld_req),
    .ld_rsp       (ld_rsp),
    .ld_rsp_data  (ld_rsp_data),
    .itcm_we      (itcm_we),
    .itcm_waddr   (itcm_waddr),
    .itcm_wdata   (itcm_wdata),
    .itcm_wbe     (itcm_wbe),
    .itcm_re      (itcm_re),
    .itcm_raddr   (itcm_raddr),
    .itcm_rdata   (itcm_rdata),
    .dtcm_we      (dtcm_we),
    .dtcm_waddr   (dtcm_waddr),
    .dtcm_wdata   (dtcm_wdata),
    .dtcm_wbe     (dtcm_wbe),
    .dtcm_re      (dtcm_re),
    .dtcm_raddr   (dtcm_raddr),
    .dtcm_rdata   (dtcm_rdata),
    .dm_req       (dm_req),
    .dm_we        (dm_we),
    .dm_addr      (dm_addr),
    .dm_be        (dm_be),
    .dm_wdata     (dm_wdata),
    .dm_rdata     (dm_rdata),
    .w_req        (w_req),
    .w_master     (w_master),
    .w_region     (w_region),
    .w_we         (w_we),
    .w_be         (w_be),
    .w_addr       (w_addr),
    .w_wdata      (w_wdata),
    .w_pc         (w_pc),
    .w_rsp        (w_rsp),
    .w_rdata      (w_rdata),
    .w_err        (w_err)
  );

  tsr_ctl_win #(
    .CAPS(CAPS), .Q_DEPTH(Q_DEPTH), .SPM_BYTES(SPM_BYTES), .FIFO_DEPTH(FIFO_DEPTH),
    .LOG_DEPTH(LOG_DEPTH), .LOG_FIFO_DEPTH(LOG_FIFO_DEPTH), .ISSUE_WAIT_MAX(ISSUE_WAIT_MAX)
  ) u_win (
    .clk               (clk),
    .rst_n             (rst_n),
    .hart_clr          (hart_clr),
    .rv_kill           (rv_kill),
    .w_req             (w_req),
    .w_master          (w_master),
    .w_region          (w_region),
    .w_we              (w_we),
    .w_be              (w_be),
    .w_addr            (w_addr),
    .w_wdata           (w_wdata),
    .w_pc              (w_pc),
    .w_rsp             (w_rsp),
    .w_rdata           (w_rdata),
    .w_err             (w_err),
    .if_wr             (if_wr),
    .if_wdata          (if_wdata),
    .if_free           (if_free),
    .if_empty          (if_empty),
    .issued_seq        (issued_seq),
    .s_seq             (s_seq),
    .d_seq             (d_seq),
    .err_rec           (err_rec),
    .q_hold            (q_hold),
    .q_err             (q_err),
    .q_nonempty        (q_nonempty),
    .mtime             (mtime),
    .irq_lq            (irq_lq),
    .irq_lcq           (irq_lcq),
    .dbg_stop          (dbg_stop),
    .perf_time_en      (perf_time_en),
    .perf_evt_en       (perf_evt_en),
    .db_req            (db_req),
    .db_we             (db_we),
    .db_off            (db_off),
    .db_wdata          (db_wdata),
    .db_rdata          (db_rdata),
    .db_irq4           (db_irq4),
    .db_irq6           (db_irq6),
    .x_ready           (x_ready),
    .x_req             (x_req),
    .x_req_data        (x_req_data),
    .x_done            (x_done),
    .x_timeout         (x_timeout),
    .x_rsp             (x_rsp),
    .lq_ready          (lq_ready),
    .lq_req            (lq_req),
    .lq_req_data       (lq_req_data),
    .lq_done           (lq_done),
    .lq_timeout        (lq_timeout),
    .lq_rsp            (lq_rsp),
    .log_wr            (log_wr),
    .log_wdata         (log_wdata),
    .log_free          (log_free),
    .rt_state          (rt_state),
    .rt_info           (rt_info),
    .rt_wr             (rt_wr),
    .push_timeout_ev   (push_timeout_ev),
    .bridge_timeout_ev (bridge_timeout_ev),
    .perf_push_stall   (perf_push_stall),
    .perf_bridge_acc   (perf_bridge_acc),
    .perf_bridge_wait  (perf_bridge_wait),
    .irq_timer         (irq_timer),
    .irq_fast          (irq_fast)
  );

  // Ibex taps (DB): combinational copies of the hart's signals, sampled by tsr_dbg_rv in clk_rv
  assign tap_data_req   = data_req;
  assign tap_data_gnt   = data_gnt;
  assign tap_data_addr  = data_addr;
  assign tap_data_we    = data_we;
  assign tap_data_be    = data_be;
  assign tap_cd_pc      = crash_dump.current_pc;
  assign tap_cd_npc     = crash_dump.next_pc;
  assign tap_cd_daddr   = crash_dump.last_data_addr;
  assign tap_cd_epc     = crash_dump.exception_pc;
  assign tap_cd_eaddr   = crash_dump.exception_addr;
  assign tap_sleep      = core_sleep && hart_rst_n_q;
  assign tap_halted     = rv_halted;
  assign tap_in_reset   = !hart_rst_n_q;

endmodule
