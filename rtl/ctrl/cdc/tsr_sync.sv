// tsr_sync.sv — W-bit level synchronizer into the clock `clk` (CC §11.2 "Levels", §11.3).
//
// Spec: CC §2.2, §11. Generic and vendor-independent: the Xilinx
// ASYNC_REG property is set by the constraints (rtl/ctrl/viv/tsr_ctrl_cdc.xdc: every cell named
// *u_sync*/s_q*), not by an attribute here (docs/rtl-core.md §1: vendor attributes only in rtl/tech).
// CC §2.2 places tsr_sync in rtl/tech/<target>/; it lives here until the technology layer takes it
// (the module and port names are the ones CC §2.2 uses).
//
// Contract: q follows d after STAGES rising edges of clk (each bit on its own: a multi-bit value
// that changes in several bits at once is only safe if it is gray coded or quasi-static).
//   STAGES >= 1. CC §11.1 `CTRL_SYNC` = 1 builds pass STAGES = 1 (a plain register stage).
//   Simulation jitter mode (CC-27, CC §15.4): with `TSR_CDC_JITTER` defined, a transition takes
//   STAGES or STAGES + 1 stages, chosen at random. PER_BIT = 1: each bit chooses on its own, which
//   emulates skew between the bits of a bus of independent levels. PER_BIT = 0: the whole bus
//   chooses together; this is the mode for gray-coded buses, whose source may step faster than
//   the destination clock, so that a per-bit choice would model a skew of a whole destination
//   period, which set_bus_skew (min period, CC §11.3) excludes, and could show a code the source
//   never had. The choice changes only while the last two stages agree, so no transition is ever
//   undone (no glitch).
// Reset: rst_n synchronous, active low, to RST_VAL (the value the source shows in reset).

module tsr_sync #(
  parameter int unsigned W       = 1,
  parameter int unsigned STAGES  = 2,
  parameter logic [W-1:0] RST_VAL = '0,
  parameter bit          PER_BIT = 1'b0
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);

  if (STAGES < 1) begin : g_chk
    $fatal(1, "tsr_sync: STAGES >= 1");
  end

  logic [W-1:0] s_q [STAGES+1];   // s_q[0] samples d; s_q[STAGES] is the extra jitter stage

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int unsigned i = 0; i <= STAGES; i++) s_q[i] <= RST_VAL;
    end else begin
      s_q[0] <= d;
      for (int unsigned i = 1; i <= STAGES; i++) s_q[i] <= s_q[i-1];
    end
  end

`ifdef TSR_CDC_JITTER
  // Per-bit stage count STAGES or STAGES + 1 (CC-27). The choice for the next cycle may change only
  // when stages STAGES - 1 and STAGES will agree after this edge (prev = what stage STAGES - 1
  // loads now).
  logic [W-1:0] late_q, prev;
  if (STAGES >= 2) begin : g_prev
    assign prev = s_q[STAGES-2];
  end else begin : g_prev_d
    assign prev = d;
  end
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      late_q <= '0;
    end else if (PER_BIT) begin
      for (int unsigned b = 0; b < W; b++) begin
        if (prev[b] == s_q[STAGES-1][b]) late_q[b] <= 1'($urandom_range(1, 0));
      end
    end else if (prev == s_q[STAGES-1]) begin
      late_q <= {W{1'($urandom_range(1, 0))}};
    end
  end
  for (genvar b = 0; b < W; b++) begin : g_q
    assign q[b] = late_q[b] ? s_q[STAGES][b] : s_q[STAGES-1][b];
  end
`else
  assign q = s_q[STAGES-1];
  wire unused_ok = &{1'b0, s_q[STAGES], PER_BIT};
`endif

endmodule
