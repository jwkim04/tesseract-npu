// tsr_dbg_rv.sv — the clk_rv part of the DB block and its crossings (docs/design/05-debug.md
// "DB" §3.7, §6.2, §8.2, §10.2; CC §11.2): DB's control-window registers (REC_COUNT, REC_DATA0-3,
// REC_POP, REC_CFG, DBG_FLAGS) behind tsr_ctrl's DB window, the completion-record FIFO (core ->
// clk_rv), the RV_SAMPLE and RVWP_APPLY handshakes, the two RV watchpoints on the Ibex data port
// and the double-fault event.
//
// Owner: DB (W4-B item 6). tsr_core instantiates it next to tsr_ctrl (it has both clocks; its
// core-domain ports go to tsr_dbg in tsr_core_dp, its clk_rv ports to tsr_ctrl's db_* port and
// to the Ibex taps). Oracle: the simulator's DBG model (sw/sim/tsr_dbg.c rec_*, h_cw_read /
// h_cw_write, on_data, rv_sample, rvwp_apply).
//
// Core-domain side (tsr_dbg): rec_wr / rec_wdata write one record per cycle; rec_room = the FIFO
//   can take a record in the next cycle as well; rec_all = REC_CFG.ALL (synchronised). smp_* and
//   wpa_* are tsr_cdc_hs A sides (TIMEOUT 4,096 core cycles; *_local answers at once when clk_rv
//   is not running). wph_v / wph_mask / wph_addr / wph_halt: one RV-watchpoint event per watched
//   access (a 16-entry crossing FIFO, including synchroniser credit-return latency; the core
//   side drains one per cycle). dfault: a pulse per double fault. flags (DBG_FLAGS) and
//   rv_run (CTRL.RV_RUN) cross as levels.
// clk_rv side: db_req / db_we / db_off / db_wdata from tsr_ctl_win (0x160-0x17C except 0x178,
//   0x400-0x7FC), db_rdata valid in the cycle after db_req; db_irq4 = REC_CFG.IRQ_HALF and
//   REC_COUNT >= 32 (irq_fast[4]). The REC FIFO and REC_CFG are clk_rv state that restarts empty
//   with the hart (the simulator's rule): when RV_RUN rises (seen in clk_rv) REC_CFG := 0 and the
//   records present then are drained (one per clk_rv cycle; the hart's boot code reads REC far
//   later). While the hart is in reset records accumulate and, with the FIFO full, are dropped and
//   counted, as in the simulator.
// Ibex taps (HAS_RV_TAPS = 1; inputs in clk_rv): the data request {data_req, data_gnt, data_addr
//   (word aligned), data_we, data_be}, crash_dump {current_pc, next_pc, last_data_addr,
//   exception_pc, exception_addr}, core_sleep, the hart's debug mode (RV_HALTED, clk_rv local),
//   the hart's reset level and double_fault_seen. RV watchpoint j (DB §3.7): an access [a, a + n)
//   (a = word address + the lowest byte enable, n = the number of byte enables) of the enabled
//   kind (LOAD / STORE) overlapping [LO, HI) of the configuration last applied (RVWP_APPLY).
//   RV_SAMPLE latches crash_dump, sleep, RV_HALTED and the reset level. HAS_RV_TAPS = 0 (until
//   tsr_ctrl exports the taps): no watchpoint hits, and RV_SAMPLE / RVWP_APPLY are never answered
//   (the core side times out after 4,096 cycles: RV_SAMPLE_TIMEOUT / RVWP_TIMEOUT, as the
//   simulator without a hart).
// The halt request of a watchpoint with HALT goes through the core side (tsr_dbg toggles
//   dbg_halt_tgl): the hart halts a few clk_rv cycles after the access instead of within one
//   instruction (DB §3.7 *confirm*; a clk_rv-local request needs a tsr_ctrl input).
// CTRL_SYNC = 1 (clk_rv is clk): one synchroniser stage.

module tsr_dbg_rv #(
  parameter int unsigned REC_DEPTH   = tsr_tables_pkg::TSR_REC_DEPTH,
  parameter bit          HAS_RV_TAPS = 1'b0,
  parameter bit          CTRL_SYNC   = 1'b0,
  parameter int unsigned HS_TIMEOUT  = 4096
) (
  // ---- core domain (tsr_dbg) ----
  input  logic          clk,
  // rst_n: the core domain's synchronous reset, also the asynchronous input of this module's
  // clk_rv reset synchroniser (as tsr_ctrl, CC §12.2)
  /* verilator lint_off SYNCASYNCNET */
  input  logic          rst_n,
  /* verilator lint_on SYNCASYNCNET */
  input  logic          rec_wr,
  input  logic [127:0]  rec_wdata,
  output logic          rec_room,
  output logic          rec_all,
  output logic          smp_ready,
  input  logic          smp_req,
  input  logic          smp_local,
  output logic          smp_done,
  output logic          smp_timeout,
  output logic [163:0]  smp_data,
  output logic          wpa_ready,
  input  logic          wpa_req,
  input  logic          wpa_local,
  input  logic [135:0]  wpa_data,
  output logic          wpa_done,
  output logic          wpa_timeout,
  output logic          wph_v,
  output logic [1:0]    wph_mask,
  output logic [31:0]   wph_addr,
  output logic          wph_halt,
  output logic          dfault,
  input  logic [4:0]    flags,
  input  logic          rv_run,

  // ---- clk_rv domain ----
  input  logic          clk_rv,
  input  logic          db_req,
  input  logic          db_we,
  input  logic [11:0]   db_off,
  input  logic [31:0]   db_wdata,
  output logic [31:0]   db_rdata,
  output logic          db_irq4,
  // Ibex taps (HAS_RV_TAPS)
  input  logic          ib_data_req,
  input  logic          ib_data_gnt,
  input  logic [31:0]   ib_data_addr,
  input  logic          ib_data_we,
  input  logic [3:0]    ib_data_be,
  input  logic [31:0]   ib_cd_pc,
  input  logic [31:0]   ib_cd_npc,
  input  logic [31:0]   ib_cd_daddr,
  input  logic [31:0]   ib_cd_epc,
  input  logic [31:0]   ib_cd_eaddr,
  input  logic          ib_sleep,
  input  logic          ib_halted,
  input  logic          ib_in_reset,
  input  logic          ib_dfault
);

  localparam int unsigned ST = CTRL_SYNC ? 1 : 2;
  localparam int unsigned RAW = $clog2(REC_DEPTH);

  if (REC_DEPTH < 2 || (REC_DEPTH & (REC_DEPTH - 1)) != 0) begin : g_chk
    $fatal(1, "tsr_dbg_rv: REC_DEPTH must be a power of two >= 2");
  end

  // clk_rv reset: the core-domain reset synchronised (CC §12.2, as tsr_ctrl)
  // The synchroniser starts from a reset flop of its own: a launch flop that also drives core logic
  // would be a fan-out to the destination clock (report_cdc CDC-11).
  logic rst_rv_n;
  (* keep = "true" *) logic rst_src_q;
  always_ff @(posedge clk) rst_src_q <= rst_n;
  tsr_rst_sync #(.STAGES(2)) u_rst (.clk(clk_rv), .rst_ni(rst_src_q), .rst_no(rst_rv_n));

  // ---------------------------------------------------------------------------------------------
  // Levels into clk_rv: DBG_FLAGS, RV_RUN; REC_CFG.ALL into the core
  // ---------------------------------------------------------------------------------------------
  // rv_run and flags are combinational in the core (registers ORed, gated): they enter the
  // synchroniser from a flop of their own (report_cdc CDC-10); the bits are independent levels.
  logic [4:0] flags_rv;
  logic       run_rv;
  (* dont_touch = "true" *) logic [5:0] lv_q;   // a flop of its own: not merged with tsr_ctrl_cdc's copy of rv_run / dbg_stop (CDC-11)
  always_ff @(posedge clk) begin
    if (!rst_n) lv_q <= '0;
    else        lv_q <= {rv_run, flags};
  end
  tsr_sync #(.W(6), .STAGES(ST), .PER_BIT(1'b1)) u_sync_rv (.clk(clk_rv), .rst_n(rst_rv_n), .d(lv_q),
                                                           .q({run_rv, flags_rv}));
  logic [1:0] rcfg_q;                       // REC_CFG {IRQ_HALF, ALL} (clk_rv)
  tsr_sync #(.W(1), .STAGES(ST)) u_sync_all (.clk(clk), .rst_n(rst_n), .d(rcfg_q[0]), .q(rec_all));

  // ---------------------------------------------------------------------------------------------
  // REC FIFO (core -> clk_rv, DB §8.2)
  // ---------------------------------------------------------------------------------------------
  logic [RAW:0] wfree, rcount;
  logic         wempty, rvalid, rready;
  logic [127:0] rdata;
  tsr_cdc_fifo #(.W(128), .DEPTH(REC_DEPTH), .STAGES(ST)) u_rec (
    .wclk(clk), .wrst_n(rst_n), .wr_en(rec_wr), .wr_data(rec_wdata), .wr_free(wfree), .wr_empty(wempty),
    .rclk(clk_rv), .rrst_n(rst_rv_n), .rd_valid(rvalid), .rd_data(rdata), .rd_ready(rready), .rd_count(rcount)
  );
  // wr_free counts this cycle's write only from the next cycle on
  assign rec_room = (wfree > (RAW + 1)'(rec_wr));
  wire unused_we = &{1'b0, wempty};

  // ---------------------------------------------------------------------------------------------
  // DB window registers (clk_rv)
  // ---------------------------------------------------------------------------------------------
  wire rd_pop = db_req && db_we && db_off == tsr_tables_pkg::TSR_CW_REC_POP;
  logic         run_prev_q;
  logic [RAW:0] drain_q;                   // records still to drain after the hart left reset
  wire          drain = (drain_q != '0);
  assign rready = rvalid && (rd_pop || drain);
  logic [31:0] rdat_q;
  always_ff @(posedge clk_rv) begin
    if (!rst_rv_n) begin
      rcfg_q     <= '0;
      rdat_q     <= '0;
      run_prev_q <= 1'b0;
      drain_q    <= '0;
    end else begin
      run_prev_q <= run_rv;
      if (run_rv && !run_prev_q) begin
        rcfg_q  <= '0;
        drain_q <= rcount;
      end else if (drain && (rvalid || rcount == '0)) begin
        drain_q <= drain_q - 1'b1;
      end
      if (db_req && db_we && db_off == tsr_tables_pkg::TSR_CW_REC_CFG && !(run_rv && !run_prev_q))
        rcfg_q <= db_wdata[1:0];
      if (db_req && !db_we) begin
        case (db_off)
          tsr_tables_pkg::TSR_CW_REC_COUNT: rdat_q <= 32'(rcount);
          12'h164: rdat_q <= rvalid ? rdata[31:0]   : 32'd0;
          12'h168: rdat_q <= rvalid ? rdata[63:32]  : 32'd0;
          12'h16C: rdat_q <= rvalid ? rdata[95:64]  : 32'd0;
          12'h170: rdat_q <= rvalid ? rdata[127:96] : 32'd0;
          tsr_tables_pkg::TSR_CW_REC_CFG:   rdat_q <= {30'd0, rcfg_q};
          tsr_tables_pkg::TSR_CW_DBG_FLAGS: rdat_q <= {27'd0, flags_rv};
          default: rdat_q <= '0;
        endcase
      end
    end
  end
  assign db_rdata = rdat_q;
  assign db_irq4  = rcfg_q[1] && (32'(rcount) >= tsr_tables_pkg::TSR_REC_IRQ_LEVEL);

  // ---------------------------------------------------------------------------------------------
  // RV_SAMPLE and RVWP_APPLY handshakes
  // ---------------------------------------------------------------------------------------------
  logic         s_bv, w_bv, s_brsp, w_brsp;
  logic         s_breq;
  logic [135:0] w_breq;
  logic [163:0] s_bdata;
  logic         w_arsp;
  tsr_cdc_hs #(.REQ_W(1), .RSP_W(164), .TIMEOUT(HS_TIMEOUT), .STAGES(ST)) u_hs_smp (
    .clk_a(clk), .rst_a_n(rst_n), .a_ready(smp_ready), .a_req(smp_req), .a_local(smp_local),
    .a_req_data(1'b0), .a_done(smp_done), .a_timeout(smp_timeout), .a_rsp_data(smp_data),
    .clk_b(clk_rv), .rst_b_n(rst_rv_n), .b_valid(s_bv), .b_req_data(s_breq), .b_rsp(s_brsp),
    .b_rsp_data(s_bdata)
  );
  tsr_cdc_hs #(.REQ_W(136), .RSP_W(1), .TIMEOUT(HS_TIMEOUT), .STAGES(ST)) u_hs_wpa (
    .clk_a(clk), .rst_a_n(rst_n), .a_ready(wpa_ready), .a_req(wpa_req), .a_local(wpa_local),
    .a_req_data(wpa_data), .a_done(wpa_done), .a_timeout(wpa_timeout), .a_rsp_data(w_arsp),
    .clk_b(clk_rv), .rst_b_n(rst_rv_n), .b_valid(w_bv), .b_req_data(w_breq), .b_rsp(w_brsp),
    .b_rsp_data(1'b0)
  );
  wire unused_hs = &{1'b0, s_breq, w_arsp, db_wdata[31:2]};

  // ---------------------------------------------------------------------------------------------
  // Watchpoint hits (clk_rv -> core) and the double-fault event
  // ---------------------------------------------------------------------------------------------
  // Ibex can access once per clk_rv cycle (CC-34). Even when clk is faster, slots remain occupied
  // until read-pointer credits cross back. Allow 3 stages in each direction (CDC jitter), FWFT
  // presentation + consumption, and phase: <= 5 core + 4 RV periods for a returned credit.
  // At F_CORE >= F_RV this is <= 9 RV periods; 16 slots cover a sustained stream without drops.
  // A sustained source faster than the sink cannot be made lossless by any finite FIFO; flag
  // overflow below rather than silently concealing a clock/rate assumption violation in tests.
  localparam int unsigned HIT_DEPTH = 16;
  localparam int unsigned HAW = $clog2(HIT_DEPTH);
  logic [35:0] hf_wdata, hf_rdata;
  logic        hf_wr, hf_rvalid;
  logic [HAW:0] hf_free, hf_count;
  logic        hf_empty;
  tsr_cdc_fifo #(.W(36), .DEPTH(HIT_DEPTH), .STAGES(ST)) u_hit (
    .wclk(clk_rv), .wrst_n(rst_rv_n), .wr_en(hf_wr), .wr_data(hf_wdata), .wr_free(hf_free), .wr_empty(hf_empty),
    .rclk(clk), .rrst_n(rst_n), .rd_valid(hf_rvalid), .rd_data(hf_rdata), .rd_ready(hf_rvalid), .rd_count(hf_count)
  );
  assign wph_v    = hf_rvalid;
  assign wph_mask = hf_rdata[33:32];
  assign wph_addr = hf_rdata[31:0];
  assign wph_halt = hf_rdata[34];
  wire unused_hf = &{1'b0, hf_empty, hf_count, hf_rdata[35]};

  logic df_tgl_q;
  logic df_sync, df_prev_q;
  tsr_sync #(.W(1), .STAGES(ST)) u_sync_df (.clk(clk), .rst_n(rst_n), .d(df_tgl_q), .q(df_sync));
  always_ff @(posedge clk) begin
    if (!rst_n) df_prev_q <= 1'b0;
    else        df_prev_q <= df_sync;
  end
  assign dfault = (df_sync != df_prev_q);

  if (HAS_RV_TAPS) begin : g_taps
    // applied watchpoint configuration {ctl[3:0], hi, lo} x 2
    logic [1:0][3:0]  wctl_q;
    logic [1:0][31:0] wlo_q, whi_q;
    // the access: a = word address + lowest byte enable, n = byte enables set
    logic [31:0] a;
    logic [2:0]  n;
    always_comb begin
      a = {ib_data_addr[31:2], 2'b00};
      n = 3'd0;
      for (int b = 3; b >= 0; b--) if (ib_data_be[b]) a = {ib_data_addr[31:2], 2'(b)};
      for (int b = 0; b < 4; b++) n = n + 3'(ib_data_be[b]);
    end
    wire acc = ib_data_req && ib_data_gnt;
    logic [1:0] hit, hlt;
    always_comb begin
      for (int j = 0; j < 2; j++) begin
        automatic logic kind = ib_data_we ? wctl_q[j][2] : wctl_q[j][1];
        hit[j] = acc && wctl_q[j][0] && kind && (33'(a) < 33'(whi_q[j])) && (33'(a) + 33'(n) > 33'(wlo_q[j]));
        hlt[j] = hit[j] && wctl_q[j][3] && !ib_halted;
      end
    end
    assign hf_wr    = (hit != 2'd0) && (hf_free != '0);
    assign hf_wdata = {1'b0, |hlt, hit, a};
    wire unused_a = &{1'b0, ib_data_addr[1:0]};

`ifndef SYNTHESIS
    always_ff @(posedge clk_rv) begin
      if (rst_rv_n && hit != 2'd0 && hf_free == '0)
        $error("tsr_dbg_rv: watchpoint event FIFO overflow (count/HALT would be lost)");
    end
`endif

    logic [163:0] smp_q;
    always_ff @(posedge clk_rv) begin
      if (!rst_rv_n) begin
        wctl_q   <= '0;
        wlo_q    <= '0;
        whi_q    <= '0;
        df_tgl_q <= 1'b0;
      end else begin
        if (w_bv) {wctl_q[1], whi_q[1], wlo_q[1], wctl_q[0], whi_q[0], wlo_q[0]} <= w_breq;
        if (ib_dfault) df_tgl_q <= !df_tgl_q;
      end
      if (s_bv) smp_q <= {ib_in_reset, ib_sleep, ib_halted, 1'b0, ib_cd_eaddr, ib_cd_epc, ib_cd_daddr, ib_cd_npc,
                          ib_cd_pc};
    end
    // answer one cycle after the request arrived (the sample is taken in that cycle)
    logic s_bv_q;
    always_ff @(posedge clk_rv) begin
      if (!rst_rv_n) s_bv_q <= 1'b0;
      else           s_bv_q <= s_bv && !s_bv_q;
    end
    assign s_brsp  = s_bv && s_bv_q;
    assign s_bdata = smp_q;
    assign w_brsp  = w_bv;
  end else begin : g_notaps
    assign hf_wr    = 1'b0;
    assign hf_wdata = '0;
    assign s_brsp   = 1'b0;
    assign s_bdata  = '0;
    assign w_brsp   = 1'b0;
    always_ff @(posedge clk_rv) begin
      if (!rst_rv_n) df_tgl_q <= 1'b0;
    end
    wire unused_taps = &{1'b0, ib_data_req, ib_data_gnt, ib_data_addr, ib_data_we, ib_data_be, ib_cd_pc,
                         ib_cd_npc, ib_cd_daddr, ib_cd_epc, ib_cd_eaddr, ib_sleep, ib_halted, ib_in_reset,
                         ib_dfault, s_bv, w_bv, w_breq, hf_free};
  end

endmodule
