// tb/mvm/hdl/dbl/tsr_mvm_back.sv — TEST DOUBLE of rtl/core/tsr_mvm_back.sv (same module name,
// parameters and ports; selected by tb/mvm/test_mvm.py instead of the rtl/core file, which the
// back-end agent implements in parallel). Behavioural, simulation only; not part of any filelist
// and not linted by rtl/lint.sh (the module name collides with the real one).
//
// Behaviour = the contract of docs/rtl-mvm.md §3.3-§3.5 / §7.1, computed from what the double
// receives through its ports: the tagged beats of the real front end and the activations read
// from the (double) activation buffer through qs_* / dbs_* (one QS word and one DBS line per beat:
// Q8_0 D line b >> 4; Q4_K half 0 the D line s >> 4, half 1 the BS line cfg_d_lines + s/2).
// Integer sums in SV (exact); every fp operation through DPI-C into libtsr_ref (tb_fmul, tb_fadd,
// tb_fsub, tb_i32_to_f32 in tsr_mvm_dbl.cpp), in the order of tsr_term_* / tsr_dot_row:
//   Q8_0 term(b+g) = fmul(fmul(d_g, Dx[b+g]), i32_to_f32(isum_g)), partial (b+g) % 16;
//   Q4_K term(s)   = fsub(fmul(fmul(d, Dx[s]), f32(isum)), fmul(fmul(dmin, Dx[s]), f32(msum)));
//   partials from +0.0 per row, p[i % 16] = fadd(p, term); y = tree16(p) at row_last.
// Timing knobs (variables written by cocotb through VPI; defaults = fast and ready):
//   knob_stall  percent of cycles in which beat_ready is forced to 0 (random)
//   knob_lat    each row result is held back a random 0..knob_lat cycles after its last beat
//   knob_cap    rows in flight + results queued that the double accepts (>= 1): the result
//               "FIFO" whose space it reserves before accepting a row's last beat
// Pipeline: beat accepted (beat_ready is a flop) -> register -> activation read (qs_re/dbs_re)
// -> ACT_RD_LAT -> per-token arithmetic -> result queue (random release) -> res_valid/res_y
// output register (R2). idle = no beat in flight, no row started, no result pending.

// Behavioural test double: index / width conversions are intentional (not a synthesis model).
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
module tsr_mvm_back
  import tsr_pkg::*;
  import tsr_att_pkg::*;
#(
  parameter  bit          HAS_ATT    = 1'b0,          // W4-B at: the double models the MVM only
  parameter  int unsigned K_MAX      = TSR_DEF_K_MAX,
  parameter  int unsigned T_MAX      = TSR_DEF_T_MAX,
  parameter  int unsigned ACT_RD_LAT = TSR_DEF_ACT_RD_LAT,
  parameter  int unsigned MAC_LAT    = TSR_DEF_MAC_LAT,
  parameter  int unsigned MUL_LAT    = TSR_DEF_FMUL_LAT,
  parameter  int unsigned ADD_LAT    = TSR_DEF_FADD_LAT,
  parameter  bit          FAST_CLK   = 1'b0,          // the real back end's PV mode copies (ignored: the double has no PV)
  localparam int unsigned QS_WORDS   = tsr_actbuf_qs_words(K_MAX),
  localparam int unsigned DBS_LINES  = tsr_actbuf_dbs_lines(K_MAX),
  localparam int unsigned QS_AW      = (QS_WORDS > 1) ? $clog2(QS_WORDS) : 1,
  localparam int unsigned DBS_AW     = (DBS_LINES > 1) ? $clog2(DBS_LINES) : 1,
  localparam int unsigned NS         = (T_MAX + 1) / 2
) (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic [T_MAX-1:0]              rst_tok_n,      // the real back end's per-token reset copies (unused here)
  input  logic                          cfg_fmt,
  input  logic [7:0]                    cfg_d_lines,
  // W4-B at: attention ports (ignored / 0 in the double)
  input  tsr_att_bcfg_t                 att_cfg,
  input  logic [T_MAX-1:0][10:0]        att_tbase,
  input  logic                          att_start,
  output logic                          seg_valid,
  output logic [T_MAX-1:0][31:0]        seg_y,
  input  tsr_att_rd_t                   pv_rd,
  input  logic                          pv_mac_v,
  input  logic                          pv_mac_first,
  input  logic [63:0][7:0]              pv_mac_q,
  input  logic [TSR_MVM_GROUPS-1:0][NS-1:0][24:0] pv_mac_a,
  output logic [TSR_MVM_GROUPS-1:0][NS-1:0][31:0][37:0] pv_acc,
  input  logic                          beat_valid,
  output logic                          beat_ready,
  input  tsr_mvm_beat_t                 beat,
  output logic                          qs_re,
  output logic [T_MAX-1:0][QS_AW-1:0]   qs_addr,
  input  logic [T_MAX-1:0][1023:0]      qs_data,
  output logic                          dbs_re,
  output logic [T_MAX-1:0][DBS_AW-1:0]  dbs_addr,
  input  logic [T_MAX-1:0][511:0]       dbs_data,
  output logic                          res_valid,
  input  logic                          res_ready,
  output logic [T_MAX-1:0][31:0]        res_y,
  output logic                          idle
);

  import "DPI-C" function int tb_fadd(input int a, input int b);
  import "DPI-C" function int tb_fsub(input int a, input int b);
  import "DPI-C" function int tb_fmul(input int a, input int b);
  import "DPI-C" function int tb_i32_to_f32(input int v);

  // ---- knobs (cocotb writes them; see header) ----
  int unsigned knob_stall = 0;
  int unsigned knob_lat   = 0;
  int unsigned knob_cap   = 4;

  // ---- input register and activation read ----
  logic          rdy_q;
  logic          s0_v;
  tsr_mvm_beat_t s0_b;
  logic          dv [ACT_RD_LAT];
  tsr_mvm_beat_t db [ACT_RD_LAT];

  wire acc = beat_valid & rdy_q;

  logic [QS_AW-1:0]  qa1;
  logic [DBS_AW-1:0] da1;
  always_comb begin
    qs_re    = s0_v;
    dbs_re   = s0_v;
    qa1      = '0;
    da1      = '0;
    if (!cfg_fmt) begin
      qa1 = QS_AW'(s0_b.blk >> 2);                          // word b/4 (lines b/2 .. )
      da1 = DBS_AW'(s0_b.blk >> 4);                         // D line of block b (and b+1)
    end else begin
      qa1 = QS_AW'({s0_b.blk, s0_b.half});                  // word 2s + half
      da1 = s0_b.half ? DBS_AW'(32'(cfg_d_lines) + 32'(s0_b.blk >> 1))   // BS line
                      : DBS_AW'(s0_b.blk >> 4);             // D line of s
    end
    qs_addr  = {T_MAX{qa1}};
    dbs_addr = {T_MAX{da1}};
  end
  assign seg_valid = 1'b0;
  assign seg_y     = '0;
  always_comb begin                     // (per lane: one '0 fill would exceed the replication limit)
    for (int g = 0; g < TSR_MVM_GROUPS; g++)
      for (int s = 0; s < int'(NS); s++)
        for (int k = 0; k < 32; k++) pv_acc[g][s][k] = '0;
  end
  wire unused_att = &{1'b0, att_cfg, att_tbase, att_start, pv_rd, pv_mac_v, pv_mac_first, pv_mac_q, pv_mac_a, rst_tok_n};

  // ---- per-token row state and result queue ----
  int unsigned p    [T_MAX][16];      // partials (f32 bits)
  int          part0[T_MAX];          // Q4_K: isum of half 0
  int unsigned dx0  [T_MAX];          // Q4_K: Dx[s] (read with half 0)
  typedef struct { logic [T_MAX-1:0][31:0] y; longint unsigned at; } res_t;
  res_t          rq [$];
  longint unsigned cyc;
  int unsigned   infl_q;              // rows whose last beat was accepted, result not yet taken
  logic          in_row_q;
  logic          rv_q;
  logic [T_MAX-1:0][31:0] ry_q;

  function automatic int signed s8(logic [7:0] v);  return int'(signed'(v)); endfunction
  function automatic int signed s16(logic [15:0] v); return int'(signed'(v)); endfunction

  function automatic int unsigned tree16(int unsigned q[16]);
    int unsigned l[16];
    for (int i = 0; i < 16; i++) l[i] = q[i];
    for (int w = 8; w >= 1; w = w / 2) begin
      for (int k = 0; k < w; k++) l[k] = tb_fadd(l[2*k], l[2*k+1]);
    end
    return l[0];
  endfunction

  // one beat (activations already read) for token t
  task automatic do_beat(input int t, input tsr_mvm_beat_t b, input logic [1023:0] qs,
                         input logic [511:0] dbs);
    int signed S, part, msum, isum;
    int unsigned dx, d, term, t1, t2, i;
    int hs;
    if (!cfg_fmt) begin
      hs = int'(b.blk[1]);                                  // line (b/2) % 2 of the word
      for (int g = 0; g < 2; g++) begin
        if (g == 1 && !b.two) break;
        S = 0;
        for (int L = 0; L < 32; L++) S += s8(b.w[32*g + L]) * s8(qs[512*hs + 8*(32*g + L) +: 8]);
        i  = 32'(b.blk) + 32'(g);
        dx = dbs[32*(i & 15) +: 32];
        d  = (g == 0) ? b.d_a : b.d_b;
        term = tb_fmul(tb_fmul(d, dx), tb_i32_to_f32(S));
        p[t][i % 16] = tb_fadd(p[t][i % 16], term);
      end
    end else begin
      part = 0;
      for (int g = 0; g < 4; g++) begin
        S = 0;
        for (int L = 0; L < 32; L++) S += s8(b.w[32*g + L]) * s8(qs[8*(32*g + L) +: 8]);
        part += int'(b.sc[g]) * S;
      end
      if (!b.half) begin
        part0[t] = part;
        dx0[t]   = dbs[32*(b.blk & 15) +: 32];
      end else begin
        msum = 0;
        for (int k = 0; k < 8; k++) begin
          msum += int'(b.m[k]) * (s16(dbs[16*(16*int'(b.blk[0]) + 2*k) +: 16])
                                + s16(dbs[16*(16*int'(b.blk[0]) + 2*k + 1) +: 16]));
        end
        isum = part0[t] + part;
        t1 = tb_fmul(tb_fmul(b.d_a, dx0[t]), tb_i32_to_f32(isum));
        t2 = tb_fmul(tb_fmul(b.d_b, dx0[t]), tb_i32_to_f32(msum));
        term = tb_fsub(t1, t2);
        i = 32'(b.blk);
        p[t][i % 16] = tb_fadd(p[t][i % 16], term);
      end
    end
  endtask

  logic any_d;
  always_comb begin
    any_d = s0_v;
    for (int i = 0; i < ACT_RD_LAT; i++) any_d |= dv[i];
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rdy_q    <= 1'b0;
      s0_v     <= 1'b0;
      for (int i = 0; i < ACT_RD_LAT; i++) dv[i] <= 1'b0;
      rq.delete();
      infl_q   <= 0;
      in_row_q <= 1'b0;
      rv_q     <= 1'b0;
      cyc      <= 0;
      for (int t = 0; t < T_MAX; t++) for (int k = 0; k < 16; k++) p[t][k] = 0;
    end else begin
      automatic int unsigned infl_n = infl_q;
      cyc <= cyc + 1;
      // input
      s0_v <= acc;
      if (acc) s0_b <= beat;
      if (acc) in_row_q <= ~beat.row_last;
      if (acc && beat.row_last) infl_n++;
      // read pipe
      dv[0] <= s0_v;
      db[0] <= s0_b;
      for (int i = 1; i < ACT_RD_LAT; i++) begin
        dv[i] <= dv[i-1];
        db[i] <= db[i-1];
      end
      // arithmetic on the beat whose activations arrive now
      if (dv[ACT_RD_LAT-1]) begin
        automatic tsr_mvm_beat_t b = db[ACT_RD_LAT-1];
        for (int t = 0; t < T_MAX; t++) do_beat(t, b, qs_data[t], dbs_data[t]);
        if (b.row_last) begin
          automatic res_t r;
          for (int t = 0; t < T_MAX; t++) begin
            r.y[t] = tree16(p[t]);
            for (int k = 0; k < 16; k++) p[t][k] = 0;
          end
          r.at = cyc + longint'($urandom_range(knob_lat, 0));
          rq.push_back(r);
        end
      end
      // output register
      if (rv_q && res_ready) begin
        rv_q <= 1'b0;
        infl_n--;
      end
      if ((!rv_q || res_ready) && rq.size() != 0 && rq[0].at <= cyc) begin
        rv_q <= 1'b1;
        ry_q <= rq[0].y;
        void'(rq.pop_front());
      end
      infl_q <= infl_n;
      rdy_q  <= (infl_n < (knob_cap == 0 ? 1 : knob_cap)) && ($urandom_range(99, 0) >= knob_stall);
    end
  end

  assign beat_ready = rdy_q;
  assign res_valid  = rv_q;
  assign res_y      = ry_q;
  assign idle       = !any_d && !in_row_q && infl_q == 0;

  // ---- checks: R2 on the (gated) beat input, results only when rows are in flight ----
  logic          on_q, hold_q;
  tsr_mvm_beat_t hold_b;
  initial on_q = 1'b0;
  always @(posedge clk) if (!rst_n) on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      hold_q <= 1'b0;
    end else begin
      if (on_q && hold_q) begin
        assert (beat_valid && beat == hold_b) else $error("%m: beat dropped or changed before acceptance (R2)");
      end
      hold_q <= beat_valid & ~rdy_q;
      hold_b <= beat;
    end
  end

endmodule
