// tb/mvm/hdl/dbl/tsr_mvm_actbuf.sv — TEST DOUBLE of rtl/core/tsr_mvm_actbuf.sv (same module name,
// parameters and ports; selected by tb/mvm/test_mvm.py instead of the rtl/core file). Behavioural, simulation only; not part of any filelist
// and not linted by rtl/lint.sh (it lives below tb/mvm/hdl/, one directory deeper on purpose: the
// module name collides with the real one).
//
// Behaviour = the contract of docs/rtl-mvm.md §3.4 / §5: per token a QS array of 1024-bit words
// (line 2w in [511:0], line 2w+1 in [1023:512]) and a DBS array of 512-bit lines; one write per
// cycle (no backpressure); reads of all tokens with exactly RD_LAT cycles latency.
// Extra checks (they test the cluster wrapper that drives it):
//   - writes in range (tok < T_MAX, QS line < 2 * QS_WORDS, DBS line < DBS_LINES);
//   - no read in a cycle with a write (the wrapper's activation gating keeps the back end's
//     reads after the broadcast); read addresses in range.

// Behavioural test double: index / width conversions are intentional (not a synthesis model).
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
module tsr_mvm_actbuf
  import tsr_pkg::*;
#(
  parameter  int unsigned K_MAX     = TSR_DEF_K_MAX,
  parameter  int unsigned T_MAX     = TSR_DEF_T_MAX,
  parameter  int unsigned RD_LAT    = TSR_DEF_ACT_RD_LAT,
  localparam int unsigned QS_WORDS  = tsr_actbuf_qs_words(K_MAX),
  localparam int unsigned DBS_LINES = tsr_actbuf_dbs_lines(K_MAX),
  localparam int unsigned QS_AW     = (QS_WORDS > 1) ? $clog2(QS_WORDS) : 1,
  localparam int unsigned DBS_AW    = (DBS_LINES > 1) ? $clog2(DBS_LINES) : 1
) (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          wr_valid,
  input  logic [T_MAX-1:0]              wr_tmask,    // tokens written
  input  tsr_mvm_bcline_t               wr_line,
  input  logic                          qs_re,
  input  logic [T_MAX-1:0][QS_AW-1:0]   qs_addr,     // per token
  output logic [T_MAX-1:0][1023:0]      qs_data,
  input  logic                          dbs_re,
  input  logic [T_MAX-1:0][DBS_AW-1:0]  dbs_addr,
  output logic [T_MAX-1:0][511:0]       dbs_data
);

  logic [1023:0] qs_mem  [T_MAX][QS_WORDS];
  logic [511:0]  dbs_mem [T_MAX][DBS_LINES];
  logic [T_MAX-1:0][1023:0] qp [RD_LAT];
  logic [T_MAX-1:0][511:0]  dp [RD_LAT];

  always_ff @(posedge clk) begin
    for (int unsigned t = 0; t < T_MAX; t++) begin
      if (wr_valid && wr_tmask[t]) begin
        if (!wr_line.dbs) begin
          if (wr_line.addr[0]) qs_mem[t][wr_line.addr >> 1][1023:512] <= wr_line.data;
          else                 qs_mem[t][wr_line.addr >> 1][511:0]    <= wr_line.data;
        end else begin
          dbs_mem[t][wr_line.addr] <= wr_line.data;
        end
      end
    end
    for (int unsigned t = 0; t < T_MAX; t++) begin
      if (qs_re)  qp[0][t] <= qs_mem[t][qs_addr[t]];
      if (dbs_re) dp[0][t] <= dbs_mem[t][dbs_addr[t]];
    end
    for (int unsigned i = 1; i < RD_LAT; i++) begin
      qp[i] <= qp[i-1];
      dp[i] <= dp[i-1];
    end
  end
  assign qs_data  = qp[RD_LAT-1];
  assign dbs_data = dp[RD_LAT-1];

  // ---- checks ----
  logic on_q;
  initial on_q = 1'b0;
  always @(posedge clk) if (!rst_n) on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && on_q) begin
      if (wr_valid) begin
        assert (32'(wr_line.tok) < T_MAX && wr_tmask == (T_MAX'(1) << wr_line.tok))
          else $error("%m: write of token %0d >= T_MAX or token mask %b", wr_line.tok, wr_tmask);
        if (!wr_line.dbs) begin
          assert (32'(wr_line.addr) < 2 * QS_WORDS) else $error("%m: QS line %0d out of range", wr_line.addr);
        end else begin
          assert (32'(wr_line.addr) < DBS_LINES) else $error("%m: DBS line %0d out of range", wr_line.addr);
        end
        assert (!qs_re && !dbs_re) else $error("%m: read in a cycle with a broadcast write");
      end
      if (qs_re)  assert (32'(qs_addr[0]) < QS_WORDS && qs_addr == {T_MAX{qs_addr[0]}})
        else $error("%m: QS read address %0d", qs_addr[0]);
      if (dbs_re) assert (32'(dbs_addr[0]) < DBS_LINES && dbs_addr == {T_MAX{dbs_addr[0]}})
        else $error("%m: DBS read address %0d", dbs_addr[0]);
    end
  end

endmodule
