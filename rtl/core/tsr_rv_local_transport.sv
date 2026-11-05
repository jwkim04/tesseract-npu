// Same-core-clock links between the datapath and the RV island. Each link
// retains the existing upstream transaction until its downstream endpoint
// consumes it. Only the segment stream and its control cross the boundary.
/* verilator lint_off DECLFILENAME */

module tsr_rv_spm_tx (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         req_valid,
  output logic         bridge_req_valid,
  input  logic         bridge_rsp_valid,
  output logic         bridge_rsp_ready,
  input  logic [511:0] bridge_rsp_line,
  input  logic         bridge_rsp_err,
  output logic         link_valid,
  output logic [127:0] link_data,
  output logic         link_err
);
  logic req_seen_q;
  logic [1:0] part_q;

  // The CDC holds req_valid through the reassembled response, including
  // after an RV-side timeout. Issue once even after the bridge returns idle.
  assign bridge_req_valid = req_valid && !req_seen_q;
  assign bridge_rsp_ready = part_q == 2'd3;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      req_seen_q <= 1'b0;
      part_q     <= '0;
      link_valid <= 1'b0;
    end else begin
      req_seen_q <= req_valid;
      link_valid <= bridge_rsp_valid;
      if (bridge_rsp_valid) begin
        link_data <= bridge_rsp_line[128*part_q +: 128];
        link_err  <= bridge_rsp_err;
        part_q    <= part_q + 2'd1;
      end
    end
  end
endmodule

module tsr_rv_spm_rx (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         link_valid,
  input  logic [127:0] link_data,
  input  logic         link_err,
  output logic         rsp_valid,
  output logic [511:0] rsp_line,
  output logic         rsp_err
);
  logic [1:0] part_q;

  // There is one pending bridge request and one reserved line at this end.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      part_q    <= '0;
      rsp_valid <= 1'b0;
    end else begin
      rsp_valid <= link_valid && part_q == 2'd3;
      if (link_valid) begin
        rsp_line[128*part_q +: 128] <= link_data;
        if (part_q == 2'd3) rsp_err <= link_err;
        part_q <= part_q + 2'd1;
      end
    end
  end
endmodule

module tsr_rv_issue_tx (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         fifo_valid,
  input  logic [386:0] fifo_entry,
  output logic         fifo_ready,
  output logic         link_valid,
  output logic [128:0] link_data,
  input  logic         link_accepted
);
  typedef enum logic [1:0] {IDLE, SECOND, THIRD, ACK} state_t;
  state_t state_q;

  // Keep the FIFO head reserved through every segment and the receiver's
  // stall. Q_FIFO, reset draining and RV write credits use this same pop.
  assign fifo_ready = link_accepted;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state_q    <= IDLE;
      link_valid <= 1'b0;
    end else begin
      link_valid <= 1'b0;
      case (state_q)
        IDLE: if (fifo_valid) begin
          link_data  <= fifo_entry[0 +: 129];
          link_valid <= 1'b1;
          state_q    <= SECOND;
        end
        SECOND: begin
          link_data  <= fifo_entry[129 +: 129];
          link_valid <= 1'b1;
          state_q    <= THIRD;
        end
        THIRD: begin
          link_data  <= fifo_entry[258 +: 129];
          link_valid <= 1'b1;
          state_q    <= ACK;
        end
        ACK: if (link_accepted) state_q <= IDLE;
      endcase
    end
  end
endmodule

module tsr_rv_issue_rx (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         link_valid,
  input  logic [128:0] link_data,
  output logic         link_accepted,
  output logic         issue_valid,
  output logic [386:0] issue_entry,
  input  logic         issue_ready
);
  logic [1:0] part_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      part_q        <= '0;
      issue_valid   <= 1'b0;
      link_accepted <= 1'b0;
    end else begin
      link_accepted <= issue_valid && issue_ready;
      if (issue_valid && issue_ready) issue_valid <= 1'b0;
      if (link_valid) begin
        issue_entry[129*part_q +: 129] <= link_data;
        part_q <= part_q == 2'd2 ? 2'd0 : part_q + 2'd1;
        if (part_q == 2'd2) issue_valid <= 1'b1;
      end
    end
  end
endmodule

module tsr_rv_lq_tx (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         req_valid,
  input  logic [11:0]  req_off,
  input  logic         req_we,
  input  logic [31:0]  req_wdata,
  input  logic [255:0] req_lcq_w,
  output logic         link_valid,
  output logic [127:0] link_data,
  output logic [11:0]  link_off,
  output logic         link_we,
  output logic         link_end
);
  logic req_seen_q, second_q;
  wire  post = req_we && req_off == 12'(tsr_tables_pkg::TSR_CW_LCQ_POST);

  // A POST sends two record halves and its end flag. Other accesses send
  // the ordinary 32-bit write data once. The CDC holds the request until
  // the CSR answers, so neither the payload nor the header needs staging.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      req_seen_q <= 1'b0;
      second_q   <= 1'b0;
      link_valid <= 1'b0;
    end else begin
      req_seen_q <= req_valid;
      second_q   <= req_valid && !req_seen_q && post;
      link_valid <= (req_valid && !req_seen_q) || second_q;
      if ((req_valid && !req_seen_q) || second_q) begin
        link_off <= req_off;
        link_we  <= req_we;
        link_end <= post && req_wdata[0];
        if (second_q) link_data <= req_lcq_w[255:128];
        else          link_data <= post ? req_lcq_w[127:0] : {96'b0, req_wdata};
      end
    end
  end
endmodule

module tsr_rv_lq_rx (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         link_valid,
  input  logic [127:0] link_data,
  input  logic [11:0]  link_off,
  input  logic         link_we,
  input  logic         link_end,
  output logic         req_valid,
  output logic [11:0]  req_off,
  output logic         req_we,
  output logic [31:0]  req_wdata,
  output logic [255:0] req_lcq_w
);
  logic second_q;
  wire  post = link_we && link_off == 12'(tsr_tables_pkg::TSR_CW_LCQ_POST);

  // The CSR sees exactly one request pulse, after the entire POST record
  // has arrived. Its existing answer is still the only completion.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      second_q  <= 1'b0;
      req_valid <= 1'b0;
    end else begin
      req_valid <= 1'b0;
      if (link_valid) begin
        second_q <= post && !second_q;
        if (post && !second_q) begin
          req_lcq_w[127:0] <= link_data;
        end else begin
          req_valid <= 1'b1;
          req_off   <= link_off;
          req_we    <= link_we;
          req_wdata <= post ? {31'b0, link_end} : link_data[31:0];
          if (post) req_lcq_w[255:128] <= link_data;
        end
      end
    end
  end
endmodule

/* verilator lint_on DECLFILENAME */
