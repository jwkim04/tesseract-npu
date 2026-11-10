// Exercise the real CSR and drain logic while a segmented issue entry is reserved.
module tb_rv_local_idle;
  logic clk = 0;
  always #2ns clk = ~clk;
  logic rst_n = 0;
  logic csr_valid = 0, csr_we = 0, csr_ready;
  logic [11:0] csr_addr;
  logic [31:0] csr_wdata, csr_rdata, status;
  logic [5:0] ifq_count = 0;

  tsr_core_dp #(
    .NUM_CH(1), .SPM_BYTES(4096), .CH_BYTES(262144), .ADDR_BEAT_W(12),
    .HAS_MVM(0), .HAS_VU(0), .HAS_ATT(0), .HAS_DB(0), .RV_LOCAL_TRANSPORT(1)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .csr_valid(csr_valid),
    .csr_we(csr_we),
    .csr_addr(csr_addr),
    .csr_wdata(csr_wdata),
    .m_rd_req_ready('0),
    .m_rd_rsp_valid('0),
    .m_rd_rsp_data('0),
    .m_rd_rsp_id('0),
    .m_rd_rsp_last('0),
    .mb_rd_req_ready('0),
    .mb_rd_rsp_valid('0),
    .mb_rd_rsp_data('0),
    .mb_rd_rsp_id('0),
    .mb_rd_rsp_last('0),
    .cfg_dual_en('0),
    .m_wr_req_ready('0),
    .m_wr_dat_ready('0),
    .m_wr_rsp_valid('0),
    .m_wr_rsp_id('0),
    .ifq_valid('0),
    .ifq_entry('0),
    .ifq_count(ifq_count),
    .issued_seq('0),
    .err_ack('0),
    .rec_resync('0),
    .spm_req_valid('0),
    .spm_req_we('0),
    .spm_req_addr('0),
    .spm_req_be('0),
    .spm_req_wdata('0),
    .spm_rsp_ready('0),
    .cwin_req_valid('0),
    .cwin_req_off('0),
    .lq_req_valid('0),
    .lq_req_off('0),
    .lq_req_we('0),
    .lq_req_wdata('0),
    .lq_req_lcq_w('0),
    .log_valid('0),
    .log_data('0),
    .rt_state('0),
    .rt_info('0),
    .ld_ready('0),
    .ld_rsp_valid('0),
    .ld_rsp_rdata('0),
    .ld_rsp_err('0),
    .ld_rsp_timeout('0),
    .dmi_ready('0),
    .dmi_rsp_valid('0),
    .dmi_rsp_rdata('0),
    .dmi_rsp_resp('0),
    .dmi_rsp_timeout('0),
    .rv_cyc('0),
    .perf_push_stall('0),
    .perf_bridge_acc('0),
    .perf_bridge_wait('0),
    .push_timeout('0),
    .bridge_timeout('0),
    .rv_halted('0),
    .rv_sleep('0),
    .rv_running('0),
    .rv_reset_done(1'b1),
    .rv_clk_ok('0),
    .rv_ndmreset('0),
    .rv_dmactive('0),
    .plat_err_i('0),
    .rv_halt_pend('0),
    .rec_room('0),
    .rec_all('0),
    .smp_ready('0),
    .smp_done('0),
    .smp_timeout('0),
    .smp_data('0),
    .wpa_ready('0),
    .wpa_done('0),
    .wpa_timeout('0),
    .wph_v('0),
    .wph_mask('0),
    .wph_addr('0),
    .wph_halt('0),
    .dfault('0),
    .csr_ready, .csr_rdata
  );

  task automatic tick();
    @(posedge clk);
    #1ns;
  endtask

  task automatic access(input bit we, input logic [11:0] addr, input logic [31:0] data,
                        output logic [31:0] result);
    csr_valid = 1;
    csr_we = we;
    csr_addr = addr;
    csr_wdata = data;
    do tick(); while (!csr_ready);
    result = csr_rdata;
    tick();
    csr_valid = 0;
    tick();
  endtask

  initial begin
    repeat (4) tick();
    rst_n = 1;
    repeat (4) tick();
    access(0, 12'h024, '0, status);
    assert (status[7:6] == 2'b11) else $fatal(1, "empty status %h", status);
    // Models the reserved FIFO head while no complete entry is visible at
    // the dispatcher, with the actual datapath CSR gates under test.
    ifq_count = 1;
    tick();
    access(0, 12'h024, '0, status);
    assert (status[7:6] == 2'b00) else $fatal(1, "pending status %h", status);
    access(0, 12'he60, '0, status);
    assert (status == 1) else $fatal(1, "Q_FIFO %h", status);
    access(1, 12'h020, 32'h200, status);
    access(0, 12'h024, '0, status);
    assert (status[19]) else $fatal(1, "QCLEAR allowed with pending packet");
    access(1, 12'h020, 32'h400, status);
    repeat (100) tick();
    access(0, 12'h024, '0, status);
    assert (status[9] && status[7:6] == 2'b00) else $fatal(1, "SOFT_RESET did not wait %h", status);
    ifq_count = 0;
    repeat (8) tick();
    access(0, 12'h024, '0, status);
    assert (!status[9] && status[7:6] == 2'b11) else $fatal(1, "drained status %h", status);
    $display("PASS Q_FIFO, QIDLE/QQUIESCENT, QCLEAR and SOFT_RESET with a reserved but incomplete issue entry");
    $finish;
  end
  initial begin
    #5000ns;
    $fatal(1, "timeout");
  end
endmodule
