// tsr_rst_sync.sv — reset synchronizer: asynchronous assert, synchronous deassert (CC §12.2,
// docs/ctrl-riscv.md §7.3).
//
// CC §2.2 places it in rtl/tech/<target>/; it lives here until the
// technology layer takes it. Generic: ASYNC_REG comes from the constraints (tsr_ctrl_cdc.xdc).
//
// Contract: rst_no = 0 at once while rst_ni = 0 (no clock needed); rst_no rises STAGES rising
// edges of clk after rst_ni has risen. Used for the clk_rv domain's reset (ap_rst_n into clk_rv)
// and for the asynchronous resets of Ibex and riscv-dbg, which keep their own reset style
// (CC §3.1, a documented exception to docs/rtl-core.md §1).

module tsr_rst_sync #(
  parameter int unsigned STAGES = 2
) (
  input  logic clk,
  input  logic rst_ni,
  output logic rst_no
);

  if (STAGES < 1) begin : g_chk
    $fatal(1, "tsr_rst_sync: STAGES >= 1");
  end

  logic [STAGES:0] s_q;   // s_q[0] is the constant 1 shifted in

  assign s_q[0] = 1'b1;
  for (genvar i = 1; i <= STAGES; i++) begin : g_s
    always_ff @(posedge clk or negedge rst_ni) begin
      if (!rst_ni) s_q[i] <= 1'b0;
      else         s_q[i] <= s_q[i-1];
    end
  end

  assign rst_no = s_q[STAGES];

endmodule
