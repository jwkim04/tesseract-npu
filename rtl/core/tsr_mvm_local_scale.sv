// Scale completed gather jobs beside their owner pair's existing VU lanes.
// Metadata and data are ordered. Each accepted line reserves a slot until
// the scaled line has transferred to the gather link.
module tsr_mvm_local_scale
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
#(
  parameter int unsigned W = 562,
  parameter int unsigned DEPTH = 16,
  localparam int unsigned AW = $clog2(DEPTH)
) (
  input logic clk, rst_n,
  input logic [31:0] scale, post_scale,
  input logic softcap,
  input logic s_valid,
  output logic s_ready,
  input logic [W-1:0] s_data,
  output logic m_valid,
  input logic m_ready,
  output logic [W-1:0] m_data,
  output tsr_vu_local_req_t lane_req,
  input tsr_vu_local_rsp_t lane_rsp
);
  if (W < TSR_LINE_BITS + 32 || DEPTH < 2 || (1 << AW) != DEPTH) begin : g_cfg
    $fatal(1, "tsr_mvm_local_scale: job metadata and power-of-two depth required");
  end

  logic [W-TSR_LINE_BITS-1:0] meta [DEPTH];
  logic [TSR_LINE_BITS-1:0] raw [DEPTH], result [DEPTH], tanh_result [DEPTH];
  logic [AW:0] alloc_q, pop_q;
  // Each phase's admitted and returned lines remain in producer order.
  logic [2:0][AW:0] issue_pos_q, return_pos_q;
  logic held_q;
  logic [TSR_LINE_BITS-1:0] input_q;
  logic [2:0] issue_q, back_return_q;
  logic [1:0] return_q, phase_q;
  logic [TSR_VU_LF:0][1:0] phase_pipe_q;
  logic [2:0][127:0] quarters_q;
  logic [6:0][63:0] back_parts_q;
  logic next_valid;
  logic [1:0] next_phase;
  wire [1:0] return_phase = phase_pipe_q[TSR_VU_LF];
  wire send = m_valid && m_ready;
  wire receive = s_valid && s_ready;
  wire result_valid = |lane_rsp.valid;
  wire last_result = result_valid && return_q == 2'd3;
  wire back_valid = |lane_rsp.back_valid;
  wire last_issue = issue_q == (phase_q == 2'd1 ? 3'd7 : 3'd3);

  // An accepted job reserves all its phases and its final return slot.
  assign s_ready = alloc_q - pop_q != (AW+1)'(DEPTH);
  assign m_valid = return_pos_q[2] != pop_q;
  // Only owned words participate in the epilogue. The original gather
  // uses tail byte enables with +0.0 data beyond the final logical row.
  wire [W-TSR_LINE_BITS-1:0] out_meta = meta[pop_q[AW-1:0]];
  logic [TSR_LINE_BITS-1:0] out_line;
  for (genvar word_idx = 0; word_idx < 16; word_idx++) begin : g_mask
    assign out_line[32*word_idx +: 32] = out_meta[16+word_idx] ?
                                       result[pop_q[AW-1:0]][32*word_idx +: 32] : 32'd0;
  end
  assign m_data = {out_meta, out_line};
  assign lane_req = '{valid: !held_q ? 4'd0 : phase_q == 2'd1 ? 4'b0011 : 4'hf,
                      scale_only: 1'b1, tanh_op: phase_q == 2'd1,
                      a: phase_q == 2'd1 ? {64'd0, input_q[64*issue_q +: 64]} :
                                          input_q[128*issue_q[1:0] +: 128],
                      max_value: 32'd0, scale: phase_q == 2'd2 ? post_scale : scale};

  always_comb begin
    // Drain later phases first to free the head of the output queue. Each
    // dispatch still issues four consecutive quarters of one local line.
    next_phase = 2'd0;
    next_valid = issue_pos_q[0] != alloc_q;
    if (softcap && issue_pos_q[1] != return_pos_q[0]) begin
      next_phase = 2'd1;
      next_valid = 1'b1;
    end
    if (softcap && issue_pos_q[2] != return_pos_q[1]) begin
      next_phase = 2'd2;
      next_valid = 1'b1;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      alloc_q <= '0;
      pop_q <= '0;
      issue_pos_q <= '0;
      return_pos_q <= '0;
      held_q <= 1'b0;
      issue_q <= '0;
      return_q <= '0;
      back_return_q <= '0;
      phase_q <= '0;
    end else begin
      if (send) pop_q <= pop_q + 1'b1;
      if (receive) begin
        alloc_q <= alloc_q + 1'b1;
        raw[alloc_q[AW-1:0]] <= s_data[TSR_LINE_BITS-1:0];
        meta[alloc_q[AW-1:0]] <= s_data[W-1:TSR_LINE_BITS];
      end
      if (!held_q || last_issue) begin
        held_q <= next_valid;
        issue_q <= '0;
        if (next_valid) begin
          phase_q <= next_phase;
          issue_pos_q[next_phase] <= issue_pos_q[next_phase] + 1'b1;
          case (next_phase)
            2'd2: input_q <= tanh_result[issue_pos_q[2][AW-1:0]];
            2'd1: input_q <= result[issue_pos_q[1][AW-1:0]];
            default: input_q <= raw[issue_pos_q[0][AW-1:0]];
          endcase
        end
      end else begin
        issue_q <= issue_q + 3'd1;
      end
      if (result_valid) begin
        return_q <= return_q + 2'd1;
        if (last_result) begin
          return_pos_q[return_phase] <= return_pos_q[return_phase] + 1'b1;
          result[return_pos_q[return_phase][AW-1:0]] <= {lane_rsp.y, quarters_q};
          // Keep cursors aligned across consecutive SCALE and SOFTCAP ops.
          if (!softcap) begin
            issue_pos_q[1] <= return_pos_q[0] + 1'b1;
            issue_pos_q[2] <= return_pos_q[0] + 1'b1;
            return_pos_q[1] <= return_pos_q[0] + 1'b1;
            return_pos_q[2] <= return_pos_q[0] + 1'b1;
          end
        end else begin
          quarters_q[return_q] <= lane_rsp.y;
        end
      end
      if (back_valid) begin
        back_return_q <= back_return_q + 3'd1;
        if (back_return_q == 3'd7) begin
          return_pos_q[1] <= return_pos_q[1] + 1'b1;
          tanh_result[return_pos_q[1][AW-1:0]] <= {lane_rsp.back_y, back_parts_q};
        end else begin
          back_parts_q[back_return_q] <= lane_rsp.back_y;
        end
      end
    end
    phase_pipe_q[0] <= phase_q;
    for (int i = 1; i <= TSR_VU_LF; i++) phase_pipe_q[i] <= phase_pipe_q[i-1];
  end

`ifndef SYNTHESIS
  logic checks_on_q;
  initial checks_on_q = 1'b0;
  always_ff @(posedge clk) if (!rst_n) checks_on_q <= 1'b1;
  else if (checks_on_q) begin
    assert ((AW+1)'(alloc_q - pop_q) <= (AW+1)'(DEPTH));
    if (result_valid) assert (alloc_q != return_pos_q[2] && return_phase != 2'd1);
    if (back_valid) assert (softcap && return_pos_q[1] != issue_pos_q[1]);
  end
`endif
endmodule
