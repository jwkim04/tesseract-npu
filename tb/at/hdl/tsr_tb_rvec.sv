// Two row-vector emitters (tsr_mvm_rvec, PIPE 0 and PIPE 1) on one op description and one value table, each with its own
// job_ready: the job streams and the done pulses must be equal (tb/at/test_rvec_pipe.py).
module tsr_tb_rvec
  import tsr_pkg::*;
#(
  parameter int unsigned SPM_AW = 16,
  localparam int unsigned JW    = 1 + SPM_AW + 16 + 16 + TSR_LINE_BITS
) (
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   start,
  input  logic [3:0]             tq,
  input  logic [3:0]             gs,
  input  logic [3:0]             tact,
  input  logic [8:0]             nseg,
  input  logic [6:0]             g_heads,
  input  logic [5:0]             hg0,
  input  logic [14:0]            n_head,
  input  logic [SPM_AW-1:0]      base,
  input  logic                   last_en,
  input  logic [127:0][31:0]     tbl,
  input  logic [1:0]             job_ready,
  output logic [1:0]             job_valid,
  output logic [1:0][JW-1:0]     job,
  output logic [1:0]             done
);
  logic [1:0][6:0] val_addr;
  for (genvar p = 0; p < 2; p++) begin : g_dut
    tsr_mvm_rvec #(.SPM_AW(SPM_AW), .PIPE(p == 1)) u_rv (
      .clk, .rst_n, .start, .tq, .gs, .tact, .nseg, .g_heads, .hg0, .n_head, .base, .last_en,
      .val_addr(val_addr[p]), .val_data(tbl[val_addr[p]]),
      .job_valid(job_valid[p]), .job_ready(job_ready[p]), .job(job[p]), .done(done[p])
    );
  end
endmodule
