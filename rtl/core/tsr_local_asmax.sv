// Consume score/P buffers beside one four-lane VU bank. Physical owners are
// {PAIR, PAIR+2}. The common row maxima arrive before the start pulse.
module tsr_local_asmax
  import tsr_pkg::*;
  import tsr_att_pkg::*;
  import tsr_vu_pkg::*;
#(parameter int unsigned PAIR = 0)
(
  input  logic clk, rst_n,
  input  tsr_local_asmax_cmd_t cmd,
  output logic done,
  output tsr_local_score_req_t [1:0] mem_req,
  input  logic [1:0][TSR_LINE_BITS-1:0] mem_data,
  output tsr_vu_local_req_t lane_req,
  input  tsr_vu_local_rsp_t lane_rsp
);
  tsr_local_asmax_cmd_t cmd_q;
  always_ff @(posedge clk) begin
    if (!rst_n) cmd_q <= '0;
    else cmd_q <= cmd;
  end

  typedef logic [TSR_LOCAL_SCORE_AW-1:0] address_t;
  typedef struct packed {
    logic owner;
    address_t address;
    logic [1:0] quarter;
    logic last;
  } result_tag_t;

  logic [31:0] maxima [TSR_LOCAL_MAX_ROWS];
  logic active_q, reading_q, owner_q;
  logic [1:0] slot_q, quarter_q;
  logic [1:0][5:0] groups_q;
  logic [5:0] group_q, stride_q;
  logic [7:0] row_q, rows_q;
  logic [31:0] scale_q, row_max_q;
  address_t row_base_q;
  logic [TSR_LINE_BITS-1:0] line_q;
  logic [2:0][127:0] results_q;
  logic [TSR_VU_LF:0] issued_q;
  result_tag_t [TSR_VU_LF:0] tags_q;
  result_tag_t result_tag;

  wire issue_lanes = active_q && !reading_q;
  wire finish_group = group_q + 6'd1 == groups_q[owner_q];
  wire finish_row = finish_group && (owner_q || groups_q[1] == 0);
  wire finish_issue = finish_row && row_q + 8'd1 == rows_q;
  wire write_result = (|lane_rsp.valid) && result_tag.quarter == 2'd3;
  wire prefetch = issue_lanes && quarter_q == 2'd3 && !finish_issue;
  wire next_owner = finish_row ? groups_q[0] == 0 : finish_group ? 1'b1 : owner_q;
  wire [5:0] next_group = finish_group ? 6'd0 : group_q + 6'd1;
  wire address_t next_row_base = finish_row ? row_base_q + address_t'(stride_q) : row_base_q;
  wire [TSR_LINE_BITS-1:0] read_line = quarter_q == 0 ? mem_data[owner_q] : line_q;

  function automatic logic [5:0] owner_groups(logic [7:0] groups, logic [1:0] off);
    return 6'((9'(groups) + 9'd3 - 9'(off)) >> 2);
  endfunction
  wire [5:0] groups0 = owner_groups(cmd_q.groups, 2'(PAIR) - cmd_q.first_owner);
  wire [5:0] groups1 = owner_groups(cmd_q.groups, 2'(PAIR + 2) - cmd_q.first_owner);

  always_comb begin
    lane_req = '0;
    if (issue_lanes) begin
      lane_req.valid = '1;
      lane_req.a = read_line[quarter_q*128 +: 128];
      lane_req.max_value = row_max_q;
      lane_req.scale = scale_q;
    end
    for (int c = 0; c < 2; c++) begin
      mem_req[c] = '0;
      mem_req[c].slot = slot_q;
      mem_req[c].rd = (active_q && reading_q && owner_q == 1'(c)) ||
                      (prefetch && next_owner == 1'(c));
      mem_req[c].ra = prefetch ? next_row_base + address_t'(next_group) : row_base_q + address_t'(group_q);
      mem_req[c].wr = write_result && result_tag.owner == 1'(c);
      mem_req[c].wa = result_tag.address;
      mem_req[c].data = {lane_rsp.y, results_q};
    end
  end

  assign result_tag = tags_q[TSR_VU_LF];
  wire unused_back = &{1'b0, lane_rsp.back_valid, lane_rsp.back_y};

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      active_q <= 1'b0;
      done <= 1'b0;
      issued_q <= '0;
    end else begin
      done <= write_result && result_tag.last;
      issued_q <= {issued_q[TSR_VU_LF-1:0], issue_lanes};
      if (cmd_q.start) begin
        active_q <= groups0 != 0 || groups1 != 0;
        done <= groups0 == 0 && groups1 == 0;
        reading_q <= 1'b1;
        owner_q <= groups0 == 0;
        groups_q <= {groups1, groups0};
        slot_q <= cmd_q.slot;
        stride_q <= owner_groups(cmd_q.groups, 2'd0);
        scale_q <= cmd_q.value;
        row_max_q <= maxima[0];
        row_q <= '0;
        rows_q <= cmd_q.row;
        group_q <= '0;
        row_base_q <= '0;
        quarter_q <= '0;
      end else if (active_q && reading_q) begin
        reading_q <= 1'b0;
      end else if (issue_lanes) begin
        quarter_q <= quarter_q + 2'd1;
        if (quarter_q == 0) line_q <= read_line;
        if (quarter_q == 2'd3) begin
          if (finish_row) begin
            active_q <= !finish_issue;
            owner_q <= groups_q[0] == 0;
            row_q <= row_q + 8'd1;
            if (!finish_issue) row_max_q <= maxima[row_q + 8'd1];
            row_base_q <= row_base_q + address_t'(stride_q);
            group_q <= '0;
          end else if (finish_group) begin
            owner_q <= 1'b1;
            group_q <= '0;
          end else begin
            group_q <= group_q + 6'd1;
          end
        end
      end
      if (|lane_rsp.valid && result_tag.quarter != 2'd3)
        results_q[result_tag.quarter] <= lane_rsp.y;
    end
    if (cmd_q.max_valid) maxima[cmd_q.row] <= cmd_q.value;
    tags_q[0] <= '{owner: owner_q, address: row_base_q + address_t'(group_q),
                   quarter: quarter_q, last: finish_issue};
    for (int k = 1; k <= TSR_VU_LF; k++) tags_q[k] <= tags_q[k-1];
  end

`ifndef SYNTHESIS
  logic checks_on_q;
  initial checks_on_q = 1'b0;
  always_ff @(posedge clk) begin
    if (!rst_n) checks_on_q <= 1'b1;
    else if (checks_on_q) begin
      assert (issued_q[TSR_VU_LF] == (|lane_rsp.valid));
      if (cmd_q.start) assert (!active_q && issued_q == '0 && cmd_q.slot != 0);
      if (issue_lanes) assert (int'(row_base_q) + int'(group_q) < TSR_LOCAL_SCORE_LINES);
    end
  end
`else
  wire unused_issued = &{1'b0, issued_q};
`endif
endmodule
