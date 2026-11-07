// Check the segmented links with the same held-input contracts as the core.
module tb_rv_local_transport #(
  parameter bit XSLR_REG = 1'b0       // the SPM link end's request stage (tsr_rv_spm_tx)
);
  timeunit 1ns;
  timeprecision 1ps;
  logic clk = 0;
  always #2 clk = ~clk;
  logic rst_n = 0;

  logic spm_req, bridge_req, bridge_v, bridge_ready, bridge_err, spm_v, spm_err, spm_done, spm_done_err;
  logic [511:0] bridge_line, spm_done_line;
  logic [127:0] spm_data;
  logic [62:0] spm_fields, bridge_fields;
  tsr_rv_spm_tx #(.XSLR_REG(XSLR_REG)) u_spm_tx (
    .clk, .rst_n, .req_valid(spm_req), .req_fields(spm_fields), .bridge_req_valid(bridge_req),
    .bridge_req_fields(bridge_fields), .bridge_rsp_valid(bridge_v),
    .bridge_rsp_ready(bridge_ready), .bridge_rsp_line(bridge_line), .bridge_rsp_err(bridge_err),
    .link_valid(spm_v), .link_data(spm_data), .link_err(spm_err)
  );
  tsr_rv_spm_rx u_spm_rx (
    .clk, .rst_n, .link_valid(spm_v), .link_data(spm_data), .link_err(spm_err),
    .rsp_valid(spm_done), .rsp_line(spm_done_line), .rsp_err(spm_done_err)
  );

  logic fifo_valid, fifo_ready, link_valid, link_accepted, issue_valid, issue_ready;
  logic [386:0] fifo_entry, issue_entry;
  logic [128:0] link_data;
  tsr_rv_issue_tx u_issue_tx (.clk, .rst_n, .fifo_valid, .fifo_ready, .fifo_entry, .link_valid, .link_data, .link_accepted);
  tsr_rv_issue_rx u_issue_rx (.clk, .rst_n, .link_valid, .link_data, .link_accepted, .issue_valid, .issue_entry, .issue_ready);

  logic lq_req, lq_we, lq_v, lq_link_we, lq_link_end, csr_valid, csr_we;
  logic [11:0] lq_off, lq_link_off, csr_off;
  logic [31:0] lq_wdata, csr_wdata;
  logic [127:0] lq_data;
  logic [255:0] lq_record, csr_record;
  tsr_rv_lq_tx u_lq_tx (
    .clk, .rst_n, .req_valid(lq_req), .req_off(lq_off), .req_we(lq_we), .req_wdata(lq_wdata), .req_lcq_w(lq_record),
    .link_valid(lq_v), .link_data(lq_data), .link_off(lq_link_off), .link_we(lq_link_we), .link_end(lq_link_end)
  );
  tsr_rv_lq_rx u_lq_rx (
    .clk, .rst_n, .link_valid(lq_v), .link_data(lq_data), .link_off(lq_link_off), .link_we(lq_link_we), .link_end(lq_link_end),
    .req_valid(csr_valid), .req_off(csr_off), .req_we(csr_we), .req_wdata(csr_wdata), .req_lcq_w(csr_record)
  );

  int cycles, spm_added, issue_added, lq_added, lcq_post_added;
  task automatic tick();
    @(posedge clk);
    #1ns;
    cycles++;
  endtask

  task automatic flush_reset();
    rst_n = 0;
    spm_req = 0;
    bridge_v = 0;
    fifo_valid = 0;
    issue_ready = 0;
    lq_req = 0;
    tick();
    assert (!spm_done && !issue_valid && !csr_valid && !fifo_ready) else $fatal;
    rst_n = 1;
    repeat (5) begin
      tick();
      assert (!spm_done && !issue_valid && !csr_valid && !fifo_ready) else $fatal;
    end
  endtask

  task automatic do_spm(input int latency, input int late_hold);
    int response_cycle;
    for (int i = 0; i < 16; i++) bridge_line[32*i +: 32] = $urandom;
    bridge_err = 1'($urandom);
    spm_fields = 63'({$urandom, $urandom});
    spm_req = 1;
    #1ps;
    // XSLR_REG: the request pulse and its fields come one cycle later, still with the fields held.
    if (XSLR_REG) begin
      assert (!bridge_req) else $fatal;
      tick();
    end
    assert (bridge_req && bridge_fields == spm_fields) else $fatal;
    tick();
    repeat (latency) begin
      assert (!bridge_req && !spm_done) else $fatal;
      tick();
    end
    bridge_v = 1;
    response_cycle = cycles;
    for (int part = 0; part < 4; part++) begin
      assert (bridge_ready == (part == 3)) else $fatal;
      tick();
      assert (!bridge_req && !spm_done) else $fatal;
    end
    bridge_v = 0;
    tick();
    assert (spm_done && spm_done_line == bridge_line && spm_done_err == bridge_err) else $fatal;
    spm_added = cycles - response_cycle;
    repeat (late_hold) begin
      tick();
      assert (!spm_done && !bridge_req) else $fatal;
    end
    spm_req = 0;
    tick();
  endtask

  task automatic do_issue(input int stall);
    int source_cycle;
    for (int i = 0; i < 12; i++) fifo_entry[32*i +: 32] = $urandom;
    fifo_entry[386:384] = 3'($urandom);
    fifo_valid = 1;
    source_cycle = cycles;
    repeat (3) begin
      tick();
      assert (!issue_valid && !fifo_ready) else $fatal;
    end
    tick();
    assert (issue_valid && issue_entry == fifo_entry && !fifo_ready) else $fatal;
    issue_added = cycles - source_cycle;
    repeat (stall) begin
      tick();
      assert (issue_valid && issue_entry == fifo_entry && !fifo_ready) else $fatal;
    end
    issue_ready = 1;
    tick();
    assert (!issue_valid && fifo_ready) else $fatal;
    tick();
    fifo_valid = 0;
    issue_ready = 0;
    assert (!issue_valid && !fifo_ready) else $fatal;
  endtask

  task automatic do_issue_burst();
    int first_cycle;
    fifo_valid  = 1;
    issue_ready = 1;
    for (int entry = 0; entry < 8; entry++) begin
      // Model a FWFT head changing immediately after the preceding pop.
      fifo_entry = {129'(entry + 32), 129'(entry + 16), 129'(entry)};
      first_cycle = cycles;
      repeat (3) begin
        tick();
        assert (!issue_valid && !fifo_ready) else $fatal;
      end
      tick();
      assert (issue_valid && issue_entry == fifo_entry && !fifo_ready) else $fatal;
      tick();
      assert (!issue_valid && fifo_ready) else $fatal;
      tick();
      assert (cycles - first_cycle == 6 && !issue_valid && !fifo_ready) else $fatal;
    end
    fifo_valid = 0;
    issue_ready = 0;
    repeat (4) begin
      tick();
      assert (!issue_valid && !fifo_ready) else $fatal;
    end
  endtask

  task automatic do_lq(input bit post, input int late_hold);
    int source_cycle;
    lq_off = post ? 12'h260 : 12'h228;
    lq_we = post || 1'($urandom);
    lq_wdata = $urandom;
    for (int i = 0; i < 8; i++) lq_record[32*i +: 32] = $urandom;
    lq_req = 1;
    source_cycle = cycles;
    tick();
    assert (!csr_valid) else $fatal;
    if (post) begin
      tick();
      assert (!csr_valid) else $fatal;
    end
    tick();
    assert (csr_valid && csr_off == lq_off && csr_we == lq_we) else $fatal;
    if (post) lcq_post_added = cycles - source_cycle;
    else      lq_added = cycles - source_cycle;
    if (post) begin
      assert (csr_record == lq_record && csr_wdata == {31'b0, lq_wdata[0]}) else $fatal;
    end else assert (csr_wdata == lq_wdata) else $fatal;
    repeat (late_hold) begin
      tick();
      assert (!csr_valid) else $fatal;
    end
    lq_req = 0;
    tick();
  endtask

  initial begin
    flush_reset();
    for (int iteration = 0; iteration < 100; iteration++) begin
      do_spm(int'($urandom_range(10, 0)), int'($urandom_range(8, 1)));
      do_issue(int'($urandom_range(20, 0)));
      do_lq(1'(iteration), int'($urandom_range(8, 1)));
    end
    do_issue_burst();
    // Reset with a request, each partial flit position, a complete stalled
    // issue entry, or an acceptance acknowledgement still in flight.
    for (int phase = 0; phase <= 5; phase++) begin
      spm_req = 1;
      fifo_valid = 1;
      lq_off = 12'h260;
      lq_we = 1;
      lq_req = 1;
      tick();
      bridge_v = 1;
      repeat (phase) begin
        tick();
        if (bridge_ready) begin
          tick();
          bridge_v = 0;
        end
      end
      if (phase == 5) begin
        issue_ready = 1;
        tick();
      end
      flush_reset();
      do_spm(0, 1);
      do_issue(0);
      do_lq(1, 1);
    end
    $display("PASS 100 randomized SPM/issue/LQ transfers per link, six reset phases and recovery");
    // The direct path is sampled on the first edge after its source valid.
    // A registered endpoint's valid above is sampled one edge later.
    assert (spm_added == 5 && issue_added == 4 && lq_added == 2 && lcq_post_added == 3) else $fatal;
    $display("Added core cycles at consumer edge: SPM=%0d issue=%0d LQ=%0d LCQ_POST=%0d",
             spm_added, issue_added, lq_added, lcq_post_added);
    $finish;
  end

  initial begin
    #50000ns;
    $fatal(1, "timeout");
  end
endmodule
