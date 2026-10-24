// tsr_mvm_actbuf.sv — MVM cluster activation buffer (spec MVM 클러스터 "클러스터당 활성값 버퍼"):
// holds the T_MAX QA images of the current MVM (QS + D + BS per token), written by the activation
// broadcast, read by the back end (spec stage 5).
//
// Owner: back agent (docs/rtl-m3.md §2 block (back); organisation §5).
//
// Organisation (docs/rtl-m3.md §5): per token t
//   QS : QS_WORDS = ceil(K_MAX/128) words of 1024 bits = two QS lines: word w = {line 2w+1,
//        line 2w} (line 2w in bits [511:0]); built as two 512-bit banks (even / odd lines).
//   DBS: DBS_LINES = tsr_actbuf_dbs_lines(K_MAX) lines of 512 bits: the QA image's lines from
//        ceil(K/64) on (D lines at 0 .. d_lines-1, then the A256 BS lines).
//   All T_MAX tokens are read together (one address, T_MAX x the width): the tokens share the
//   weight beat (spec 자원 추정 "1024비트 x 292 per token").
// Contract:
//   write: wr_valid with wr_line (tsr_mvm_bcline_t; `last` is ignored here): token wr_line.tok,
//          QS line (dbs = 0) wr_line.addr -> bank addr[0], word addr >> 1; D/BS line (dbs = 1)
//          wr_line.addr. One line per cycle, no backpressure (addresses always in range).
//   read : qs_re / dbs_re with an address; qs_data / dbs_data hold the words exactly RD_LAT cycles
//          later (fully pipelined, undefined in other cycles), for every token t < T_MAX.
//          Reads and writes of the same MVM never overlap in time (the broadcast completes before
//          the back end reads). Write-to-read distance: a write presented in cycle c is in the
//          RAM after the edge ending cycle c + 1 (one write input register, see below), so a
//          read presented in cycle c + 2 or later returns it; a read in cycle c + 1 returns the
//          old content. The cluster gating (docs/rtl-m3.md §3.2: act_ready := 1 at the edge that
//          takes the last bc_line) plus the back end's input register give >= 2: the first beat
//          is accepted in cycle c + 1 at the earliest and its reads are presented in cycle c + 2.
//   no reset of the arrays (contents undefined until written; tokens >= T of an MVM keep stale
//   data, whose results the gather ignores).
// Implementation: one write input register (docs/rtl-m3.md §9.2 item 1: the 3 * T_MAX RAMs share
//   the 512 write-data bits; the per-RAM write enables are decoded before the register, so each
//   RAM's WE pin is a flop), then tsr_sram_1r1w (rtl/tech) per (token, bank) and per token for
//   DBS, RD_LAT = RD_LAT; read address / enable go straight to the RAMs (the back end drives them
//   from flops). U200 (K_MAX 32768, T_MAX 1): QS 2 x 256 x 512 bits + DBS 72 x 512 bits
//   (2 x 8 + 8 = 24 RAMB36, docs/rtl-m3.md §5; the RAM mapping is M4's).
//   Cost (generic): 527 + 3 * T_MAX write-register flops, no other logic than the WE decode.

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

  // ---- write side (activation broadcast) ----
  input  logic                          wr_valid,
  input  tsr_mvm_bcline_t               wr_line,

  // ---- read side (back end) ----
  input  logic                          qs_re,
  input  logic [QS_AW-1:0]              qs_addr,
  output logic [T_MAX-1:0][1023:0]      qs_data,     // [t][511:0] = line 2w, [1023:512] = line 2w+1
  input  logic                          dbs_re,
  input  logic [DBS_AW-1:0]             dbs_addr,
  output logic [T_MAX-1:0][511:0]       dbs_data
);

  if (T_MAX < 1 || T_MAX > TSR_T_MAX_ARCH || RD_LAT < 1) begin : g_chk
    $fatal(1, "tsr_mvm_actbuf: T_MAX in [1, 8] and RD_LAT >= 1 required");
  end
  if (K_MAX < 256 || K_MAX > 65280 || (K_MAX % 256) != 0) begin : g_chk_k
    $fatal(1, "tsr_mvm_actbuf: K_MAX must be a multiple of 256 in [256, 65280]");
  end

  // ---- write input register (per-RAM write enables decoded in front of it) ----
  logic [T_MAX-1:0][1:0] qs_we_q;    // [token][bank]
  logic [T_MAX-1:0]      dbs_we_q;   // [token]
  logic [QS_AW-1:0]      qs_wa_q;
  logic [DBS_AW-1:0]     dbs_wa_q;
  logic [511:0]          wd_q;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      qs_we_q  <= '0;
      dbs_we_q <= '0;
    end else begin
      for (int we_t = 0; we_t < int'(T_MAX); we_t++) begin
        qs_we_q[we_t][0] <= wr_valid & (32'(wr_line.tok) == 32'(we_t)) & ~wr_line.dbs & ~wr_line.addr[0];
        qs_we_q[we_t][1] <= wr_valid & (32'(wr_line.tok) == 32'(we_t)) & ~wr_line.dbs &  wr_line.addr[0];
        dbs_we_q[we_t]   <= wr_valid & (32'(wr_line.tok) == 32'(we_t)) &  wr_line.dbs;
      end
    end
  end

  // Address / data load every cycle (no clock enable: M3 part C timing, docs/rtl-m3.md §14);
  // the RAMs write only with the registered enables above, so the content is unchanged.
  always_ff @(posedge clk) begin
    qs_wa_q  <= QS_AW'(wr_line.addr[9:1]);
    dbs_wa_q <= DBS_AW'(wr_line.addr);
    wd_q     <= wr_line.data;
  end

  // ---- RAMs ----
  for (genvar t = 0; t < T_MAX; t++) begin : g_tok
    for (genvar b = 0; b < 2; b++) begin : g_bank
      tsr_sram_1r1w #(.DEPTH(QS_WORDS), .WIDTH(512), .RD_LAT(RD_LAT)) u_qs (
        .clk,
        .we(qs_we_q[t][b]), .waddr(qs_wa_q), .wdata(wd_q), .wbe({64{1'b1}}),
        .re(qs_re), .raddr(qs_addr), .rdata(qs_data[t][512*b +: 512])
      );
    end
    tsr_sram_1r1w #(.DEPTH(DBS_LINES), .WIDTH(512), .RD_LAT(RD_LAT)) u_dbs (
      .clk,
      .we(dbs_we_q[t]), .waddr(dbs_wa_q), .wdata(wd_q), .wbe({64{1'b1}}),
      .re(dbs_re), .raddr(dbs_addr), .rdata(dbs_data[t])
    );
  end

  // `last` is the cluster's (activation gating); addr bits above the RAM depth are always 0.
  wire unused_ok = &{1'b0, wr_line.last};

  // ---- simulation checks (armed by the first reset edge) ----
`ifndef SYNTHESIS
  logic a_on_q;
  initial a_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) a_on_q <= 1'b1;

  always_ff @(posedge clk) begin
    if (rst_n && a_on_q) begin
      if (wr_valid) begin
        assert (32'(wr_line.tok) < T_MAX) else $error("%m: broadcast line for token %0d >= T_MAX", wr_line.tok);
        assert (wr_line.dbs ? (32'(wr_line.addr) < DBS_LINES) : (32'(wr_line.addr) < 2 * QS_WORDS))
          else $error("%m: broadcast line address %0d out of range (dbs %0d)", wr_line.addr, wr_line.dbs);
      end
      if (qs_re)  assert (32'(qs_addr)  < QS_WORDS)  else $error("%m: QS read address %0d out of range", qs_addr);
      if (dbs_re) assert (32'(dbs_addr) < DBS_LINES) else $error("%m: DBS read address %0d out of range", dbs_addr);
    end
  end
`endif

endmodule
