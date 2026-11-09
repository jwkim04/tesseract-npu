// tsr_rv_bus.sv — RISC-V instruction and data buses of the control subsystem (CC §4.1-§4.3, §3.2).
//
// Spec: CC §2.2 `tsr_rv_bus`. clk_rv domain.
//
// Address map (CC §4.1, TSR_RV_* of tsr_tables_pkg; region sizes are parameters, every address of a
// region beyond its size faults): ITCM (fetch, byte / half / word), DTCM (byte / half / word),
// debug module memory (fetch, word), control window (word), core window (word loads), SPM window
// and its line-buffered alias (byte / half / word). Everything else is an access fault.
//
// Bus rules (CC §4.3, docs/ctrl-riscv.md §2.3):
//   - Every grant comes from a register (instr_gnt, data_gnt, sba_gnt and the load port's), never
//     from the address decode of the same cycle. Every wait is a late rvalid.
//   - Data side: one access in flight at a time. A TCM access that needs no read port of the next
//     cycle (any DTCM access, an ITCM store; stores write the SRAM from registers in the next cycle,
//     a DTCM load of a word stored in the cycle before gets the store's bytes forwarded) answers in
//     the next cycle and keeps the grant, so the TCMs take one data access per cycle (Ibex's
//     writeback stage, CC-34). An ITCM load or a
//     debug-module access is "deferred": it uses the ITCM read port or the DM slave port in the
//     next cycle (the instruction port gets no grant then: "data reads steal the fetch port for one
//     cycle"), answers two cycles after its grant, and the grant returns in its answer cycle. A
//     window access (control window, core window, SPM windows) goes to tsr_ctl_win one cycle after
//     its grant; the grant returns the cycle after its answer.
//   - Three masters (CC §4.3, §4.2): Ibex's data port, the DM's system-bus master (SBA) and the host
//     load port (EX's LOAD_* while the hart is held in reset), round robin per access: the master
//     served last has the lowest priority; with nobody else waiting Ibex keeps the grant.
//   - SBA refusals (D15, CC-33): every write to the control or core window, every DM-range access,
//     an ITCM / DTCM write while the hart is held in reset (load_mode: the load port owns the TCM
//     write port, CC §4.2) answer with an error and do nothing. The load port is refused outside
//     load_mode and outside ITCM / DTCM.
//   - Word-only regions (DM, control window, core window) fault on byte enables other than 4'hF
//     (Ibex splits a misaligned access into aligned ones, so a misaligned word faults there too);
//     a core-window store faults. Exception: in debug mode (RV_HALTED: fetching from the DM range)
//     the hart may make byte / half accesses to the DM range, because riscv-dbg's debug ROM reads
//     its per-hart flags with lbu (debug_rom.S); outside debug mode the DM range stays word-only as
//     CC §4.1 and the simulator's bus (tsr_bus.c) have it.
// Instruction side: ITCM (rvalid in the cycle after the grant) and DM (deferred, below); error
// response elsewhere. RV_HALTED (CC §3.2, CC-29) = the last granted fetch address lies in the DM
// range 0x1000_0000-0x1000_0FFF.
// TCMs: tsr_sram_1r1w (read latency 1, byte enables) per CC §4.2, outside this module (ports).
// Reset: rst_n synchronous, active low (the clk_rv domain reset); hart_rst_n additionally clears
// the fetch state (RV_HALTED) when the hart is reset.

module tsr_rv_bus
  import tsr_tables_pkg::*;
  import tsr_ctrl_pkg::*;
#(
  parameter  int unsigned ITCM_BYTES = TSR_ITCM_BYTES,
  parameter  int unsigned DTCM_BYTES = TSR_DTCM_BYTES,
  parameter  int unsigned SPM_BYTES  = 4 * 1024 * 1024,
  localparam int unsigned IAW        = $clog2(ITCM_BYTES / 4),
  localparam int unsigned DAW        = $clog2(DTCM_BYTES / 4)
) (
  input  logic           clk,
  input  logic           rst_n,
  input  logic           hart_rst_n,
  input  logic           ibex_run,
  input  logic           load_mode,
  output logic           ibex_busy,
  output logic           rv_halted,
  // Ibex instruction port
  input  logic           instr_req,
  input  logic [31:0]    instr_addr,
  output logic           instr_gnt,
  output logic           instr_rvalid,
  output logic [31:0]    instr_rdata,
  output logic           instr_err,
  // Ibex data port (data_pc: crash_dump_o.current_pc, the accessing instruction, CC §7.1)
  input  logic           data_req,
  input  logic           data_we,
  input  logic [3:0]     data_be,
  input  logic [31:0]    data_addr,
  input  logic [31:0]    data_wdata,
  input  logic [31:0]    data_pc,
  output logic           data_gnt,
  output logic           data_rvalid,
  output logic [31:0]    data_rdata,
  output logic           data_err,
  // DM system-bus master
  input  logic           sba_req,
  input  logic           sba_we,
  input  logic [31:0]    sba_addr,
  input  logic [31:0]    sba_wdata,
  input  logic [3:0]     sba_be,
  output logic           sba_gnt,
  output logic           sba_rvalid,
  output logic           sba_err,
  output logic [31:0]    sba_rdata,
  // load port (B side of the LOAD_* handshake)
  input  logic           ld_valid,
  input  ldreq_t         ld_req,
  output logic           ld_rsp,
  output logic [32:0]    ld_rsp_data,
  // ITCM / DTCM
  output logic           itcm_we,
  output logic [IAW-1:0] itcm_waddr,
  output logic [31:0]    itcm_wdata,
  output logic [3:0]     itcm_wbe,
  output logic           itcm_re,
  output logic [IAW-1:0] itcm_raddr,
  input  logic [31:0]    itcm_rdata,
  output logic           dtcm_we,
  output logic [DAW-1:0] dtcm_waddr,
  output logic [31:0]    dtcm_wdata,
  output logic [3:0]     dtcm_wbe,
  output logic           dtcm_re,
  output logic [DAW-1:0] dtcm_raddr,
  input  logic [31:0]    dtcm_rdata,
  // DM slave port (riscv-dbg dm_top, read data one cycle after the request)
  output logic           dm_req,
  output logic           dm_we,
  output logic [31:0]    dm_addr,
  output logic [3:0]     dm_be,
  output logic [31:0]    dm_wdata,
  input  logic [31:0]    dm_rdata,
  // window port (tsr_ctl_win): one request at a time, exactly one answer each
  output logic           w_req,
  output master_e        w_master,
  output region_e        w_region,
  output logic           w_we,
  output logic [3:0]     w_be,
  output logic [31:0]    w_addr,
  output logic [31:0]    w_wdata,
  output logic [31:0]    w_pc,
  input  logic           w_rsp,
  input  logic [31:0]    w_rdata,
  input  logic           w_err
);

  if (ITCM_BYTES < 1024 || (ITCM_BYTES & (ITCM_BYTES - 1)) != 0 ||
      ITCM_BYTES > TSR_RV_ITCM_LIMIT - TSR_RV_ITCM_BASE ||
      DTCM_BYTES < 1024 || (DTCM_BYTES & (DTCM_BYTES - 1)) != 0 ||
      DTCM_BYTES > TSR_RV_DTCM_LIMIT - TSR_RV_DTCM_BASE ||
      SPM_BYTES == 0 || (SPM_BYTES % 64) != 0 || SPM_BYTES > TSR_RV_SPM_SIZE ||
      TSR_RV_SPM_SIZE != 32'h0040_0000) begin : g_chk
    $fatal(1, "tsr_rv_bus: TCM sizes must be powers of two within their reserved ranges (CC §4.1); SPM_BYTES a multiple of 64 within the window");
  end

  // Region decode. The regions are aligned to their (power-of-two) sizes, so a region is an
  // equality compare of the upper address bits (no adder in the address -> grant / SRAM-enable
  // path, the clk_rv critical structure, gate T1); a SPM_BYTES that is not a power of two adds a
  // compare of the offset.
  localparam int unsigned IB = $clog2(ITCM_BYTES);
  localparam int unsigned DB = $clog2(DTCM_BYTES);
  localparam int unsigned SB = $clog2(SPM_BYTES);
  localparam bit          SPM_POW2 = (SPM_BYTES & (SPM_BYTES - 1)) == 0;
  function automatic logic in_rg(input logic [31:0] a, input logic [31:0] base, input int unsigned bits);
    return (a >> bits) == (base >> bits);
  endfunction
  function automatic region_e decode(input logic [31:0] a);
    if (in_rg(a, TSR_RV_ITCM_BASE, IB)) return RG_ITCM;
    if (in_rg(a, TSR_RV_DTCM_BASE, DB)) return RG_DTCM;
    if (in_rg(a, TSR_RV_DM_BASE, 12))   return RG_DM;
    if (in_rg(a, TSR_RV_CTL_BASE, 12))  return RG_CW;
    if (in_rg(a, TSR_RV_CORE_BASE, 12)) return RG_CORE;
    if (in_rg(a, TSR_RV_SPM_BASE, SPM_POW2 ? SB : 22) && (SPM_POW2 || a[21:0] < 22'(SPM_BYTES)))
      return RG_SPM;
    if (in_rg(a, TSR_RV_SPM_LB_BASE, SPM_POW2 ? SB : 22) && (SPM_POW2 || a[21:0] < 22'(SPM_BYTES)))
      return RG_SPMLB;
    return RG_NONE;
  endfunction

  // ------------------------------------------------------------------ grants and acceptance
  logic    halted_q;     // RV_HALTED: the last granted fetch was in the DM range
  logic    gnt_ibex_q, gnt_sba_q, gnt_ld_q, igrant_q;
  logic    acc_ibex, acc_sba, acc_ld, acc;
  master_e last_q;
  logic    ld_infl_q, win_busy_q;
  master_e win_m_q;

  assign data_gnt  = gnt_ibex_q;
  assign sba_gnt   = gnt_sba_q;
  assign instr_gnt = igrant_q;

  assign acc_ibex = data_req && gnt_ibex_q;
  assign acc_sba  = sba_req && gnt_sba_q;
  assign acc_ld   = ld_valid && !ld_infl_q && gnt_ld_q;
  assign acc      = acc_ibex || acc_sba || acc_ld;

  // The accepted access. The grants are registered and mutually exclusive, so they (not the
  // requests) select the fields: no request -> select path (gate T1).
  master_e     a_m;
  logic        a_we;
  logic [3:0]  a_be;
  logic [31:0] a_addr, a_wdata;
  always_comb begin
    a_m = MS_IBEX; a_we = data_we; a_be = data_be; a_addr = data_addr; a_wdata = data_wdata;
    if (gnt_sba_q) begin
      a_m = MS_SBA; a_we = sba_we; a_be = sba_be; a_addr = sba_addr; a_wdata = sba_wdata;
    end else if (gnt_ld_q) begin
      a_m = MS_LD; a_we = ld_req.we; a_be = 4'hF; a_addr = ld_req.addr; a_wdata = ld_req.wdata;
    end
  end

  region_e a_rg, rg_ibex, rg_sba, rg_ld;
  logic    a_word, a_fault, a_short, a_def, a_win;
  assign rg_ibex = decode(data_addr);   // the three decodes run in parallel, the grant selects
  assign rg_sba  = decode(sba_addr);
  assign rg_ld   = decode(ld_req.addr);
  assign a_rg    = gnt_sba_q ? rg_sba : gnt_ld_q ? rg_ld : rg_ibex;
  assign a_word = a_be == 4'hF;
  always_comb begin
    a_fault = 1'b0;
    unique case (a_m)
      MS_IBEX: a_fault = (a_rg == RG_NONE) ||
                         (a_rg == RG_DM && !a_word && !halted_q) ||
                         ((a_rg == RG_CW || a_rg == RG_CORE) && !a_word) ||
                         (a_rg == RG_CORE && a_we);
      MS_SBA:  a_fault = (a_rg == RG_NONE) || (a_rg == RG_DM) ||
                         ((a_rg == RG_CW || a_rg == RG_CORE) && (a_we || !a_word)) ||
                         ((a_rg == RG_ITCM || a_rg == RG_DTCM) && a_we && load_mode);
      default: a_fault = !load_mode || !(a_rg == RG_ITCM || a_rg == RG_DTCM) || (a_addr[1:0] != 2'b00);
    endcase
  end
  assign a_short = !a_fault && (a_rg == RG_DTCM || (a_rg == RG_ITCM && a_we));
  assign a_def   = !a_fault && ((a_rg == RG_ITCM && !a_we) || a_rg == RG_DM);
  assign a_win   = !a_fault && (a_rg == RG_CW || a_rg == RG_CORE || a_rg == RG_SPM || a_rg == RG_SPMLB);

  // Next-cycle grants (registered).
  logic blocked, ibex_c, sba_w, ld_w;
  assign blocked = (acc && (a_def || a_win)) || (win_busy_q && !w_rsp);
  assign ibex_c  = ibex_run && data_req;
  assign sba_w   = sba_req && !acc_sba;
  assign ld_w    = ld_valid && !ld_infl_q && !acc_ld;

  // DM fetch deferral state (below), declared before the grant logic that reads fdp_d
  logic        def_dm, fdp_q, fd_go, fdp_d, ird_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      gnt_ibex_q <= 1'b0;
      gnt_sba_q  <= 1'b0;
      gnt_ld_q   <= 1'b0;
      igrant_q   <= 1'b0;
      last_q     <= MS_IBEX;
    end else begin
      gnt_ibex_q <= 1'b0;
      gnt_sba_q  <= 1'b0;
      gnt_ld_q   <= 1'b0;
      igrant_q   <= ibex_run && !(acc && a_def) && !fdp_d;
      if (acc) last_q <= a_m;
      if (!blocked) begin
        if (!sba_w && !ld_w) begin
          gnt_ibex_q <= ibex_run;
        end else begin
          // round robin: start after the master served last
          unique case (acc ? a_m : last_q)
            MS_IBEX: if (sba_w) gnt_sba_q <= 1'b1; else if (ld_w) gnt_ld_q <= 1'b1;
                     else gnt_ibex_q <= ibex_run;
            MS_SBA:  if (ld_w) gnt_ld_q <= 1'b1; else if (ibex_c) gnt_ibex_q <= 1'b1;
                     else gnt_sba_q <= 1'b1;
            default: if (ibex_c) gnt_ibex_q <= 1'b1; else if (sba_w) gnt_sba_q <= 1'b1;
                     else gnt_ld_q <= 1'b1;
          endcase
        end
      end
    end
  end

  // ------------------------------------------------------------------ TCM / DM ports
  // Short accesses use the DTCM ports or the ITCM write port in the grant cycle; deferred ones the
  // ITCM read port or the DM slave port in the next cycle; fetches the ITCM read port or the DM
  // slave port in their grant cycle (never together with a deferred access: no fetch grant then).
  logic        f_acc;
  region_e     f_rg;
  logic        def_q;
  region_e     def_rg_q;
  master_e     def_m_q;
  logic        def_we_q;
  logic [3:0]  def_be_q;
  logic [31:2] def_addr_q;
  logic [31:0] def_wdata_q;

  assign f_acc = instr_req && igrant_q;
  assign f_rg  = decode(instr_addr);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      def_q <= 1'b0;
    end else begin
      def_q <= acc && a_def;
    end
    if (acc && a_def) begin
      def_rg_q    <= a_rg;
      def_m_q     <= a_m;
      def_we_q    <= a_we;
      def_be_q    <= a_be;
      def_addr_q  <= a_addr[31:2];
      def_wdata_q <= a_wdata;
    end
  end

  // ITCM stores reach the SRAM one cycle after their grant, from registers (no Ibex address ->
  // BRAM write-enable path, gate T1). They still answer in the next cycle: a later data read of
  // the ITCM is deferred (reads the cycle after its grant, so it sees the store), and code written
  // to ITCM is fetched only after fence.i (CC §4.1), which flushes Ibex's prefetch.
  logic           iw_q;
  logic [IAW-1:0] iw_addr_q;
  logic [31:0]    iw_data_q;
  logic [3:0]     iw_be_q;
  always_ff @(posedge clk) begin
    if (!rst_n) iw_q <= 1'b0;
    else        iw_q <= acc && a_short && a_rg == RG_ITCM;
    iw_addr_q <= a_addr[IAW+1:2];
    iw_data_q <= a_wdata;
    iw_be_q   <= a_be;
  end
  assign itcm_we    = iw_q;
  assign itcm_waddr = iw_addr_q;
  assign itcm_wdata = iw_data_q;
  assign itcm_wbe   = iw_be_q;
  // The ITCM is read in every cycle the instruction port has a grant, whatever Ibex presents (a
  // read with no fetch, or of an address outside the ITCM, is never used): no request or address
  // decode in the BRAM enable (gate T1, the fetch path Ibex ID -> jump target -> instr_addr).
  assign itcm_re    = igrant_q || (def_q && def_rg_q == RG_ITCM);
  assign itcm_raddr = (def_q && def_rg_q == RG_ITCM) ? def_addr_q[IAW+1:2] : instr_addr[IAW+1:2];

  // DTCM (gate T1: no Ibex address -> BRAM enable path). The read port reads in every cycle a
  // data master has the grant (a read nobody asked for is never used: r1_dtcm_q marks the answers
  // that take it). Stores reach the SRAM one cycle after their grant, from registers; a load of
  // the same word in that cycle reads the SRAM before the store lands (read-first), so its answer
  // takes the store's bytes from fwd_* (one store in flight at most: a store's SRAM write is in
  // the cycle after its grant, before the next store's).
  logic           dw_q, fwd_q;
  logic [DAW-1:0] dw_addr_q;
  logic [31:0]    dw_data_q, fwd_data_q;
  logic [3:0]     dw_be_q, fwd_be_q;
  logic           a_dtcm_rd;
  assign a_dtcm_rd = acc && a_short && a_rg == RG_DTCM && !a_we;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      dw_q  <= 1'b0;
      fwd_q <= 1'b0;
    end else begin
      dw_q  <= acc && a_short && a_rg == RG_DTCM && a_we;
      fwd_q <= a_dtcm_rd && dw_q && dw_addr_q == a_addr[DAW+1:2];
    end
    dw_addr_q  <= a_addr[DAW+1:2];
    dw_data_q  <= a_wdata;
    dw_be_q    <= a_be;
    fwd_data_q <= dw_data_q;
    fwd_be_q   <= dw_be_q;
  end
  assign dtcm_we    = dw_q;
  assign dtcm_waddr = dw_addr_q;
  assign dtcm_wdata = dw_data_q;
  assign dtcm_wbe   = dw_be_q;
  assign dtcm_re    = gnt_ibex_q || gnt_sba_q || gnt_ld_q;
  assign dtcm_raddr = a_addr[DAW+1:2];

  logic [31:0] dtcm_word;   // the SRAM word with a just-written store's bytes forwarded
  for (genvar b = 0; b < 4; b++) begin : g_fwd
    assign dtcm_word[8*b +: 8] = (fwd_q && fwd_be_q[b]) ? fwd_data_q[8*b +: 8] : dtcm_rdata[8*b +: 8];
  end

  // Debug-module fetches are deferred like data accesses to the DM: the fetch granted in cycle t
  // reads the DM slave from registers in the first following cycle the port is free of a deferred
  // data access, and answers the cycle after (no Ibex fetch address -> dm_mem path, gate T1;
  // debug-mode code runs at half the fetch rate). While a DM fetch is pending or answered next,
  // the instruction port gets no grant, so instruction answers stay in order and one per cycle.
  logic [31:2] fd_addr_q;
  assign def_dm   = def_q && def_rg_q == RG_DM;
  assign fd_go    = fdp_q && !def_dm;                      // the pending DM fetch reads now
  assign fdp_d    = (f_acc && f_rg == RG_DM) || (fdp_q && def_dm);
  always_ff @(posedge clk) begin
    if (!rst_n || !hart_rst_n) begin
      fdp_q <= 1'b0;
      ird_q <= 1'b0;
    end else begin
      fdp_q <= fdp_d;
      ird_q <= fd_go;
    end
    if (f_acc && f_rg == RG_DM) fd_addr_q <= instr_addr[31:2];
  end
  assign dm_req   = def_dm || fd_go;
  assign dm_we    = def_dm && def_we_q;
  assign dm_addr  = def_dm ? {def_addr_q, 2'b00} : {fd_addr_q, 2'b00};
  assign dm_be    = def_dm ? def_be_q : 4'hF;
  assign dm_wdata = def_wdata_q;

  // ------------------------------------------------------------------ instruction responses
  typedef enum logic [1:0] { IS_ITCM, IS_ERR } isrc_e;
  logic  ir_valid_q;
  isrc_e ir_src_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ir_valid_q <= 1'b0;
    end else begin
      ir_valid_q <= f_acc && f_rg != RG_DM;          // DM fetches answer through ird_q
    end
    if (f_acc) ir_src_q <= (f_rg == RG_ITCM) ? IS_ITCM : IS_ERR;
  end
  assign instr_rvalid = ir_valid_q || ird_q;
  assign instr_rdata  = ird_q ? dm_rdata : itcm_rdata;
  assign instr_err    = ir_valid_q && ir_src_q == IS_ERR;

  always_ff @(posedge clk) begin
    if (!rst_n || !hart_rst_n) halted_q <= 1'b0;
    else if (f_acc)            halted_q <= f_rg == RG_DM;
  end
  assign rv_halted = halted_q;

  // ------------------------------------------------------------------ window requests
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w_req      <= 1'b0;
      win_busy_q <= 1'b0;
    end else begin
      w_req <= acc && a_win;
      if (acc && a_win)  win_busy_q <= 1'b1;
      else if (w_rsp)    win_busy_q <= 1'b0;
    end
    if (acc && a_win) begin
      win_m_q  <= a_m;
      w_master <= a_m;
      w_region <= a_rg;
      w_we     <= a_we;
      w_be     <= a_be;
      w_addr   <= a_addr;
      w_wdata  <= a_wdata;
      w_pc     <= data_pc;
    end
  end

  // ------------------------------------------------------------------ data responses
  // r1: short accesses and faults (cycle after the grant); r2: deferred (two cycles after); the
  // window answers directly. At most one of the three in any cycle (the grant rules above).
  logic    r1_q, r1_dtcm_q, r1_err_q, r2_q;
  master_e r1_m_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      r1_q <= 1'b0;
      r2_q <= 1'b0;
    end else begin
      r1_q <= acc && (a_short || a_fault);
      r2_q <= def_q;
    end
    if (acc) begin
      r1_m_q    <= a_m;
      r1_err_q  <= a_fault;
      r1_dtcm_q <= a_rg == RG_DTCM && !a_we && !a_fault;
    end
  end

  logic        rsp_v, rsp_err;
  master_e     rsp_m;
  logic [31:0] rsp_rdata;
  always_comb begin
    rsp_v = 1'b0; rsp_m = MS_IBEX; rsp_err = 1'b0; rsp_rdata = '0;
    if (r1_q) begin
      rsp_v = 1'b1; rsp_m = r1_m_q; rsp_err = r1_err_q; rsp_rdata = r1_dtcm_q ? dtcm_word : '0;
    end else if (r2_q) begin
      rsp_v = 1'b1; rsp_m = def_m_q; rsp_err = 1'b0;
      rsp_rdata = def_we_q ? '0 : (def_rg_q == RG_DM) ? dm_rdata : itcm_rdata;
    end else if (w_rsp) begin
      rsp_v = 1'b1; rsp_m = win_m_q; rsp_err = w_err; rsp_rdata = w_rdata;
    end
  end

  assign data_rvalid = rsp_v && rsp_m == MS_IBEX;
  assign data_rdata  = rsp_rdata;
  assign data_err    = rsp_err;
  assign sba_rvalid  = rsp_v && rsp_m == MS_SBA;
  assign sba_rdata   = rsp_rdata;
  assign sba_err     = rsp_err;
  assign ld_rsp      = rsp_v && rsp_m == MS_LD;
  assign ld_rsp_data = {rsp_err, rsp_rdata};

  always_ff @(posedge clk) begin
    if (!rst_n)       ld_infl_q <= 1'b0;
    else if (acc_ld)  ld_infl_q <= 1'b1;
    else if (ld_rsp)  ld_infl_q <= 1'b0;
  end

  // An Ibex data access is in flight (the reset sequence waits for it, CC §12.2).
  assign ibex_busy = (acc_ibex) || (r1_q && r1_m_q == MS_IBEX) || ((def_q || r2_q) && def_m_q == MS_IBEX) ||
                     (win_busy_q && win_m_q == MS_IBEX);

`ifndef SYNTHESIS
  bit arm;   // armed once the reset has been seen
  initial arm = 1'b0;
  always @(posedge clk) begin   // not always_ff: the initial block writes arm too (IEEE 1800 9.2.2.4)
    if (!rst_n) arm <= 1'b1;
    if (rst_n && arm) begin
      if (32'(gnt_ibex_q) + 32'(gnt_sba_q) + 32'(gnt_ld_q) > 1) $error("tsr_rv_bus: two data grants");
      if (32'(r1_q) + 32'(r2_q) + 32'(w_rsp) > 1) $error("tsr_rv_bus: two data answers in one cycle");
      if (w_rsp && !win_busy_q) $error("tsr_rv_bus: window answer without a request");
      if (itcm_re && f_acc && f_rg == RG_ITCM && def_q && def_rg_q == RG_ITCM) $error("tsr_rv_bus: ITCM read port conflict");
      if (ir_valid_q && ird_q) $error("tsr_rv_bus: two instruction answers in one cycle");
    end
  end
`endif

endmodule
