// tb_ctrl_core.sv — core-domain stub of the tb/ctrl bench (CC §15.2: "tsr_ctrl_rv + tsr_ctrl_cdc
// + a core-domain stub"). Testbench only (Verilator, cocotb).
//
// It stands in for tsr_core_dp on the core side of tsr_ctrl:
//   - dispatcher: pops issue-FIFO entries at random times, checks each ticket against its own
//     per-unit count (the dispatcher's assertion, CC §7.2), and runs a per-unit queue model: S[u]
//     (accept) and D[u] (done) step by at most one per cycle at random times, D <= S <= popped;
//   - SPM bridge: a 4 MiB SPM of lines (initialised to 0xA5 like the simulator's) answering after a
//     random latency; core window: answers 0xC0DE0000 ^ offset; LQ / LCQ / log registers: an empty
//     LQ, an 8-deep LCQ ring (LCQ_RD stays 0), the EX §4.5 faults, LOG_FREE's answer = words
//     received (no drops); log words drained at random;
//   - MTIME (core cycles since reset).
// Every transaction is written to a log file in the run directory (ififo.log, spm.log, core.log,
// lq.log, logw.log) for the Python side to compare with the simulator. The knobs (cfg_*) let the
// Python tests silence a responder (timeouts) or slow it down.

module tb_ctrl_core
  import tsr_tables_pkg::*;
  import tsr_ctrl_pkg::*;
(
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic [31:0]               cfg_seed,
  input  logic [7:0]                cfg_lat,         // max random latency (core cycles)
  input  logic                      cfg_pop_mute,
  input  logic                      cfg_acc_mute,
  input  logic                      cfg_spm_mute,
  input  logic                      cfg_cwin_mute,
  input  logic                      cfg_lq_mute,
  input  logic                      if_valid,
  output logic                      if_ready,
  input  logic [TSR_IFIFO_BITS-1:0] if_data,
  input  logic [5:0][27:0]          issued_seq,
  output logic [5:0][27:0]          s_seq,
  output logic [5:0][27:0]          d_seq,
  input  logic                      spm_req_valid,
  input  logic                      spm_req_we,
  input  logic [25:0]               spm_req_addr,
  input  logic [3:0]                spm_req_be,
  input  logic [31:0]               spm_req_wdata,
  output logic                      spm_rsp_valid,
  input logic spm_rsp_ready,
  output logic [511:0]              spm_rsp_line,
  input  logic                      cwin_req_valid,
  input  logic [11:0]               cwin_req_off,
  output logic                      cwin_rsp_valid,
  output logic [31:0]               cwin_rsp_rdata,
  input  logic                      lq_req_valid,
  input  logic [11:0]               lq_req_off,
  input  logic                      lq_req_we,
  input  logic [31:0]               lq_req_wdata,
  input  logic [7:0][31:0]          lq_req_lcq_w,
  output logic                      lq_rsp_valid,
  output logic [31:0]               lq_rsp_rdata,
  output logic                      lq_rsp_fault,
  input  logic                      log_valid,
  input  logic [31:0]               log_data,
  output logic                      log_ready,
  output logic [63:0]               mtime,
  output logic [31:0]               stub_errors,
  output logic [31:0]               stub_pops
);

  integer fd_if, fd_spm, fd_core, fd_lq, fd_log;
  initial begin
    fd_if   = $fopen("ififo.log", "w");
    fd_spm  = $fopen("spm.log", "w");
    fd_core = $fopen("core.log", "w");
    fd_lq   = $fopen("lq.log", "w");
    fd_log  = $fopen("logw.log", "w");
  end

  // xorshift32 random stream (deterministic from cfg_seed)
  logic [31:0] rnd_q;
  function automatic logic [31:0] xs(input logic [31:0] x);
    logic [31:0] y;
    y = x ^ (x << 13);
    y = y ^ (y >> 17);
    return y ^ (y << 5);
  endfunction
  always_ff @(posedge clk) begin
    if (!rst_n) rnd_q <= cfg_seed | 32'h1;
    else        rnd_q <= xs(rnd_q);
  end

  always_ff @(posedge clk) begin
    if (!rst_n) mtime <= '0;
    else        mtime <= mtime + 1'b1;
  end

  // ------------------------------------------------------------------ dispatcher and queues
  logic [27:0] cnt_q [6];
  logic [27:0] pop_q [6];
  logic [27:0] s_q [6], d_q [6];
  assign if_ready = !cfg_pop_mute && rnd_q[1:0] != 2'b00;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int u = 0; u < 6; u++) begin
        cnt_q[u] <= '0; pop_q[u] <= '0; s_q[u] <= '0; d_q[u] <= '0;
      end
      stub_errors <= '0;
      stub_pops   <= '0;
    end else begin
      if (if_valid && if_ready) begin
        logic [1:0]  kind;
        logic [3:0]  w0;
        logic [31:0] ticket;
        logic [3:0]  tu;
        kind   = if_data[TSR_IFIFO_KIND_LSB +: 2];
        w0     = if_data[TSR_IFIFO_W_LSB +: 4];
        ticket = if_data[TSR_IFIFO_TICKET_LSB +: 32];
        tu     = ticket[31:28];
        stub_pops <= stub_pops + 1;
        $fwrite(fd_if, "%0d %h %h %h %h %h %h %h %h %h %h %h %h %h\n", kind,
                if_data[TSR_IFIFO_W_LSB + 0 +: 32], if_data[TSR_IFIFO_W_LSB + 32 +: 32],
                if_data[TSR_IFIFO_W_LSB + 64 +: 32], if_data[TSR_IFIFO_W_LSB + 96 +: 32],
                if_data[TSR_IFIFO_W_LSB + 128 +: 32], if_data[TSR_IFIFO_W_LSB + 160 +: 32],
                if_data[TSR_IFIFO_W_LSB + 192 +: 32], if_data[TSR_IFIFO_W_LSB + 224 +: 32],
                if_data[TSR_IFIFO_DEP0_LSB +: 32], if_data[TSR_IFIFO_DEP1_LSB +: 32],
                if_data[TSR_IFIFO_REC_LSB], ticket, if_data[TSR_IFIFO_PC_LSB +: 32]);
        $fflush(fd_if);
        if (kind == TSR_IFIFO_KIND_OP || (kind == TSR_IFIFO_KIND_CTRL && w0 == TSR_IFIFO_CTRL_CODE_QCLEAR)) begin
          if (tu > 4'd5 || ticket[27:0] != cnt_q[tu[2:0]]) begin
            stub_errors <= stub_errors + 1;
            $display("tb_ctrl_core: ticket %h out of order (expected seq %h)", ticket, cnt_q[tu[2:0]]);
          end else begin
            cnt_q[tu[2:0]] <= cnt_q[tu[2:0]] + 1'b1;
            pop_q[tu[2:0]] <= pop_q[tu[2:0]] + 1'b1;
          end
        end else if (ticket != TSR_TICKET_NONE) begin
          stub_errors <= stub_errors + 1;
          $display("tb_ctrl_core: control entry with a ticket %h", ticket);
        end
      end
      for (int u = 0; u < 6; u++) begin
        logic [3:0] r;
        r = 4'(rnd_q >> (4 * u + 8));
        if (!cfg_acc_mute && s_q[u] != pop_q[u] && r[1:0] == 2'b00)
          s_q[u] <= s_q[u] + 1'b1;
        if (d_q[u] != s_q[u] && r[3:2] != 2'b11)
          d_q[u] <= d_q[u] + 1'b1;
      end
    end
  end
  for (genvar u = 0; u < 6; u++) begin : g_seq
    assign s_seq[u] = s_q[u];
    assign d_seq[u] = d_q[u];
  end

  // ------------------------------------------------------------------ SPM, core window, LQ
  logic [511:0] spm [65536];
  initial begin
    for (int i = 0; i < 65536; i++) spm[i] = {64{8'hA5}};
  end

  logic [7:0] spm_wait_q, cw_wait_q, lq_wait_q;
  logic       spm_busy_q, cw_busy_q, lq_busy_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      spm_busy_q <= 1'b0; cw_busy_q <= 1'b0; lq_busy_q <= 1'b0;
      spm_rsp_valid <= 1'b0; cwin_rsp_valid <= 1'b0; lq_rsp_valid <= 1'b0;
    end else begin
      spm_rsp_valid  <= spm_rsp_valid && !spm_rsp_ready;
      cwin_rsp_valid <= 1'b0;
      lq_rsp_valid   <= 1'b0;
      // SPM bridge
      if (spm_req_valid && !spm_busy_q && !spm_rsp_valid) begin
        spm_busy_q <= 1'b1;
        spm_wait_q <= (cfg_lat == 0) ? 8'd0 : 8'(rnd_q[15:8] % (32'(cfg_lat) + 1));
      end else if (spm_busy_q && !cfg_spm_mute) begin
        if (spm_wait_q != 0) begin
          spm_wait_q <= spm_wait_q - 1'b1;
        end else begin
          logic [15:0] line;
          logic [3:0]  word;
          line = spm_req_addr[21:6];
          word = spm_req_addr[5:2];
          if (spm_req_we) begin
            for (int b = 0; b < 4; b++)
              if (spm_req_be[b]) spm[line][32*word + 8*b +: 8] <= spm_req_wdata[8*b +: 8];
            $fwrite(fd_spm, "W %h %h %h\n", spm_req_addr, spm_req_be, spm_req_wdata);
          end else begin
            $fwrite(fd_spm, "R %h\n", spm_req_addr);
          end
          $fflush(fd_spm);
          spm_rsp_line  <= spm[line];
          spm_rsp_valid <= 1'b1;
          spm_busy_q    <= 1'b0;
        end
      end
      // core window
      if (cwin_req_valid && !cw_busy_q && !cwin_rsp_valid) begin
        cw_busy_q <= 1'b1;
        cw_wait_q <= (cfg_lat == 0) ? 8'd0 : 8'(rnd_q[23:16] % (32'(cfg_lat) + 1));
        if (cwin_req_off == TSR_CSR_LOAD_DATA || (cwin_req_off >= TSR_CSR_DMI_BASE && cwin_req_off < 12'hE00)) begin
          stub_errors <= stub_errors + 1;
          $display("tb_ctrl_core: core-window request for masked offset %h", cwin_req_off);
        end
      end else if (cw_busy_q && !cfg_cwin_mute) begin
        if (cw_wait_q != 0) begin
          cw_wait_q <= cw_wait_q - 1'b1;
        end else begin
          cwin_rsp_rdata <= 32'hC0DE0000 ^ {20'h0, cwin_req_off};
          cwin_rsp_valid <= 1'b1;
          cw_busy_q      <= 1'b0;
          $fwrite(fd_core, "%h\n", cwin_req_off);
          $fflush(fd_core);
        end
      end
      // LQ / LCQ / log registers
      if (lq_req_valid && !lq_busy_q && !lq_rsp_valid) begin
        lq_busy_q <= 1'b1;
        lq_wait_q <= (cfg_lat == 0) ? 8'd0 : 8'(rnd_q[31:24] % (32'(cfg_lat) + 1));
      end else if (lq_busy_q && !cfg_lq_mute) begin
        if (lq_wait_q != 0) begin
          lq_wait_q <= lq_wait_q - 1'b1;
        end else begin
          logic [31:0] r;
          logic        f;
          r = '0; f = 1'b0;
          if (lq_req_we) begin
            unique case (lq_req_off)
              TSR_CW_LQ_POP: f = 1'b1;                                 // LQ empty
              TSR_CW_LCQ_POST: begin
                if (lcq_wr_q - lcq_rd_q >= 32'd8 || lq_req_wdata[0]) f = 1'b1;   // full, or ANSWER while not ACTIVE
                else begin
                  $fwrite(fd_lq, "%h %h %h %h %h %h %h %h\n", lq_req_lcq_w[0], lq_req_lcq_w[1], lq_req_lcq_w[2],
                          lq_req_lcq_w[3], lq_req_lcq_w[4], lq_req_lcq_w[5], lq_req_lcq_w[6], lq_req_lcq_w[7]);
                  $fflush(fd_lq);
                end
              end
              default: begin
                stub_errors <= stub_errors + 1;
                $display("tb_ctrl_core: LQ write to %h", lq_req_off);
              end
            endcase
          end else begin
            unique case (lq_req_off)
              TSR_CW_LCQ_FREE: r = 32'd8 - (lcq_wr_q - lcq_rd_q);
              TSR_CW_LOG_FREE: r = log_rx_q;
              default:         r = '0;    // empty LQ, no watermarks, not ACTIVE, LOG_CFG 0
            endcase
          end
          lq_rsp_rdata <= r;
          lq_rsp_fault <= f;
          lq_rsp_valid <= 1'b1;
          lq_busy_q    <= 1'b0;
        end
      end
    end
  end

  logic [31:0] lcq_wr_q, lcq_rd_q, log_rx_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      lcq_wr_q <= '0; lcq_rd_q <= '0; log_rx_q <= '0;
    end else begin
      if (lq_busy_q && !cfg_lq_mute && lq_wait_q == 0 && lq_req_we && lq_req_off == TSR_CW_LCQ_POST &&
          !(lcq_wr_q - lcq_rd_q >= 32'd8 || lq_req_wdata[0]))
        lcq_wr_q <= lcq_wr_q + 1'b1;
      if (log_valid && log_ready) begin
        log_rx_q <= log_rx_q + 1'b1;
        $fwrite(fd_log, "%h\n", log_data);
        $fflush(fd_log);
      end
    end
  end
  assign log_ready = rnd_q[5:4] != 2'b00;

  wire unused_ok = &{1'b0, issued_seq, lq_req_wdata[31:1]};

endmodule
