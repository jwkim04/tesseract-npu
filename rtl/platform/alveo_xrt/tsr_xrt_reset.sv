// tsr_xrt_reset.sv — kernel reset distribution for the Alveo XRT platform adapter.
//
// Spec: docs/platform-u200.md §5.
//
// ap_rst_n (Vitis: synchronous to ap_clk, active low, driven by the platform's processor-system
// reset of the dynamic region) is registered once (rst1_q, the only flop on the input net) and
// then distributed as separate register copies, one per consumer, so that each copy can be placed
// next to its logic (the channel bridges sit in the SLR of their DDR4 controller):
//   rst_ctrl_n       control bridge (AXI4-Lite slave, sequencer, CSR arbiter)     stage 2
//   rst_ch_n[c][k]   channel c, part k < CH_COPIES (the bridge's parts: request
//                    slices, AXI-side stages, read / write path, counter groups)    stage 2
//   rst_core_n       tsr_core                                               stage CORE_STAGES
// The CH_COPIES copies of a channel are equal (same stage, same value in every cycle); one copy
// per part keeps each reset net local to its part (timing: in the k1 link one copy per
// bridge, driving all of its ~450 reset loads, gave the second-worst path, 0 levels and 3.07 ns of
// routing to the write counters; docs/platform-u200.md §8.2).
// Assertion reaches every copy one cycle after rst1_q (each stage is AND-ed with rst1_q, which
// also keeps the core chain from being packed into a shift-register primitive); deassertion
// ripples down the chain, so the core leaves reset CORE_STAGES - 2 cycles after the bridges and
// the control block (never earlier: a bridge slice in reset reports s_ready = 1 and would drop a
// request of a core that is already running). The control block and the bridges therefore see
// their reset 2 cycles after ap_rst_n falls; their AXI outputs (AxVALID, WVALID) are low from
// the third cycle of a platform reset on and stay low until at least the first edge after
// ap_rst_n rises (the AXI rule; the platform holds ap_rst_n for many cycles).
// The registers start at 0 (configuration INIT; simulation: no random start state), so the
// kernel is held in reset from configuration until ap_rst_n is first seen high.
// The `keep` attribute (Xilinx synthesis) stops the equivalent copies from being merged back
// into one register; Verilator ignores it. This is the only vendor attribute of the adapter
// (target-specific layer, docs/platform-u200.md §5).

module tsr_xrt_reset #(
  parameter int unsigned N_CH        = 4,
  parameter int unsigned CH_COPIES   = 1,   // >= 1: equal stage-2 copies per channel (one per part)
  parameter int unsigned CORE_STAGES = 3    // >= 2: core reset pipeline depth (rst1_q is stage 1)
) (
  input  logic                           clk,
  input  logic                           rst_in_n,
  output logic                           rst_ctrl_n,
  output logic [N_CH-1:0][CH_COPIES-1:0] rst_ch_n,
  output logic                           rst_core_n
);

  if (CORE_STAGES < 2) begin : g_chk_stages
    $fatal(1, "tsr_xrt_reset: CORE_STAGES must be >= 2 (the core leaves reset with or after the bridges)");
  end
  if (CH_COPIES < 1) begin : g_chk_copies
    $fatal(1, "tsr_xrt_reset: CH_COPIES must be >= 1");
  end

  // Initial value 0 (FPGA: register INIT after configuration): every consumer starts in reset
  // until ap_rst_n has been seen high, also in simulation (where flops start random). The
  // registers load in a plain always: IEEE 1800 9.2.2.4 forbids a second writer (the initial
  // block) of an always_ff variable, which XSim (hw_emu) rejects; Verilator and synthesis accept
  // either form.
  logic rst1_q;
  (* keep = "true" *) logic                           ctrl_q;
  (* keep = "true" *) logic [N_CH-1:0][CH_COPIES-1:0] ch_q;
  (* keep = "true" *) logic [CORE_STAGES-2:0]         core_q;   // core_q[k] = stage k + 2

  initial begin
    rst1_q = 1'b0;
    ctrl_q = 1'b0;
    ch_q   = '0;
    core_q = '0;
  end

  always @(posedge clk) rst1_q <= rst_in_n;

  always @(posedge clk) begin
    ctrl_q <= rst1_q;
    ch_q   <= {(N_CH * CH_COPIES){rst1_q}};
    core_q[0] <= rst1_q;
    for (int unsigned k = 1; k <= CORE_STAGES - 2; k++) core_q[k] <= core_q[k-1] & rst1_q;
  end

  assign rst_ctrl_n = ctrl_q;
  assign rst_ch_n   = ch_q;
  assign rst_core_n = core_q[CORE_STAGES-2];

endmodule
