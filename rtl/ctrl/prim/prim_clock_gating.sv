// prim_clock_gating.sv — the binding of Ibex's abstract clock gate (lowRISC prim_clock_gating) for
// Tesseract (CC §3.1, docs/ctrl-riscv.md §2.3): a pass-through, as on the FPGA, where no clock is
// gated (Ibex's sleep then idles its pipeline with the clock running; architecturally the same).
//
// CC §3.1 binds prim_clock_gating to
// rtl/tech/<target>/tsr_clk_gate.sv; until the technology layer has that module this file is the
// binding for every target (an ASIC target replaces it with a latch-based gate). It replaces
// third_party/ibex/vendor/lowrisc_ip/ip/prim_generic/rtl/prim_clock_gating.sv, which is not in the
// file list. Ports and parameters are lowRISC's.

module prim_clock_gating #(
  parameter bit NoFpgaGate    = 1'b0,
  parameter bit FpgaBufGlobal = 1'b1
) (
  input  logic clk_i,
  input  logic en_i,
  input  logic test_en_i,
  output logic clk_o
);
  assign clk_o = clk_i;
  wire unused_ok = &{1'b0, en_i, test_en_i, NoFpgaGate, FpgaBufGlobal};
endmodule
