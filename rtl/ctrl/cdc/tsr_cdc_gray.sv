// tsr_cdc_gray.sv — counter crossing by gray code (CC §11.2: ISSUED_SEQ, S[u], D[u], MTIME, the
// clk_rv cycle count and the clk_rv PERF counters; docs/ctrl-riscv.md §7.1 rules).
//
// Spec: CC §2.2 `tsr_cdc_gray`.
//
// Contract:
//   - src_cnt is a W-bit binary counter (modulo 2^W) in the source domain that moves by at most
//     one step (+1) per src_clk cycle (asserted in simulation: CC §11.2 "≤ 1 step per core cycle",
//     mutation "step a gray counter by 2", CC §15.5).
//   - dst_cnt is a copy in the destination domain that lags and never leads: every value it takes
//     is a value src_cnt held (gray code: only one bit changes per step, so a sample taken during a
//     change is the old or the new value). Latency: 1 src_clk + STAGES + 1 dst_clk cycles.
//   - The gray register and the synchroniser input are the only crossing path: constrain it with
//     set_max_delay -datapath_only and set_bus_skew (rtl/ctrl/viv/tsr_ctrl_cdc.xdc, CC §11.3).
// Resets: both synchronous, active low; both sides reset to 0 (src_cnt must be 0 in reset). The two
// resets must come from the same hardware reset (CDC endpoints are never reset on their own,
// CC §12.2).

module tsr_cdc_gray #(
  parameter int unsigned W      = 28,
  parameter int unsigned STAGES = 2
) (
  input  logic         src_clk,
  input  logic         src_rst_n,
  input  logic [W-1:0] src_cnt,
  input  logic         dst_clk,
  input  logic         dst_rst_n,
  output logic [W-1:0] dst_cnt
);

  function automatic logic [W-1:0] bin2gray(input logic [W-1:0] b);
    return b ^ (b >> 1);
  endfunction

  function automatic logic [W-1:0] gray2bin(input logic [W-1:0] g);
    logic [W-1:0] b;
    b[W-1] = g[W-1];
    for (int i = int'(W) - 2; i >= 0; i--) b[i] = b[i+1] ^ g[i];
    return b;
  endfunction

  logic [W-1:0] gray_q;     // source domain, the only signal that crosses
  always_ff @(posedge src_clk) begin
    if (!src_rst_n) gray_q <= '0;
    else            gray_q <= bin2gray(src_cnt);
  end

  logic [W-1:0] gray_s;
  tsr_sync #(.W(W), .STAGES(STAGES)) u_sync (
    .clk(dst_clk), .rst_n(dst_rst_n), .d(gray_q), .q(gray_s)
  );

  always_ff @(posedge dst_clk) begin
    if (!dst_rst_n) dst_cnt <= '0;
    else            dst_cnt <= gray2bin(gray_s);
  end

`ifndef SYNTHESIS
  // CC §11.2: a gray counter steps by at most one per source cycle.
  logic [W-1:0] prev_q;
  bit           prev_v;   // set after the reset was seen (initial 0 below; the state is random)
  initial prev_v = 1'b0;
  always @(posedge src_clk) begin   // not always_ff: the initial block writes prev_v too (IEEE 1800 9.2.2.4)
    if (!src_rst_n) begin
      prev_v <= 1'b1;
      prev_q <= '0;
    end else if (prev_v) begin
      prev_q <= src_cnt;
      if (W'(src_cnt - prev_q) > W'(1))
        $error("tsr_cdc_gray: source counter stepped by %0d (> 1) in one cycle", W'(src_cnt - prev_q));
    end
  end
`endif

endmodule
