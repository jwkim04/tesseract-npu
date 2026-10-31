// tsr_vu_dly.sv — fixed delay line of D register stages (D = 0: a wire), no reset, no enable.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §7.3: "SRL delay lines carry a, b, the clamp decision
// and the TANH small-path value alongside"). Without a reset or an enable, synthesis maps deep lines
// to SRL16 / SRL32 shift registers (one LUT per bit per 32 stages). Lines of at most FF_MAX stages stay
// in flip-flops (shreg_extract = "no"): LUTs are SLR1's scarce resource (CC §14.1), flip-flops are not,
// and a short SRL costs a whole LUT per bit. Datapath only: whatever a stage holds after power-up is
// never used (the consumers qualify it with their own valid pipelines).

module tsr_vu_dly #(
  parameter int unsigned W      = 32,
  parameter int unsigned D      = 1,
  parameter int unsigned FF_MAX = 6          // lines up to this depth in flip-flops
) (
  input  logic         clk,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  if (D == 0) begin : g_wire
    assign q = d;
    wire unused_ok = &{1'b0, clk};
  end else if (D <= FF_MAX) begin : g_ff
    (* shreg_extract = "no" *) logic [W-1:0] sr [D];
    always_ff @(posedge clk) begin
      sr[0] <= d;
      for (int i = 1; i < D; i++) sr[i] <= sr[i-1];
    end
    assign q = sr[D-1];
  end else begin : g_sr
    // the first stage stays a flip-flop, so the SRL's input is a register (an SRL data input has a
    // longer setup than a flip-flop, and synthesis would otherwise pull the driving register in)
    (* shreg_extract = "no" *) logic [W-1:0] s0;
    logic [W-1:0] sr [D-1];
    always_ff @(posedge clk) begin
      s0    <= d;
      sr[0] <= s0;
      for (int i = 1; i < D - 1; i++) sr[i] <= sr[i-1];
    end
    assign q = sr[D-2];
  end
endmodule
