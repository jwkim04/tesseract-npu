// tsr_fp_funcs_tb.sv — testbench-only wrapper: tsr_fp_pkg functions on LANES parallel lanes
// (combinational), for the cocotb tests (tb/fp/test_fp.py) and the Verilator C++ sweep
// (tb/fp/cpp/fp_sweep.cpp via tsr_fp_sweep_tb.sv). Not part of the core.
//
// Per lane i: x[i] (f32 bits / int32) and h[i] (fp16 bits) in; every helper's result out.
// amax over all lanes (element order = lane order):
//   tree_*  balanced binary tree of tsr_f32_amax_merge (how a QACT unit would reduce a line)
//   chain_* left-to-right fold (the reference scan order)
// Both must equal tsr_ref.c's sequential scan (amax_candidate + "a > amax").

module tsr_fp_funcs_tb
  import tsr_fp_pkg::*;
#(
  parameter int unsigned LANES = 16   // power of two
) (
  input  logic [LANES-1:0][31:0] x,
  input  logic [LANES-1:0][15:0] h,
  output logic [LANES-1:0][31:0] daz,
  output logic [LANES-1:0]       is_nan,
  output logic [LANES-1:0][30:0] amag,
  output logic [LANES-1:0][31:0] cand,       // tsr_f32_amax_t {mag, neg}
  output logic [LANES-1:0][31:0] h2f,        // tsr_fp16_to_f32(h)
  output logic [LANES-1:0][15:0] f2h,        // tsr_f32_to_fp16(x)
  output logic [LANES-1:0][7:0]  q8a,        // tsr_f32_to_q8(x, 0)
  output logic [LANES-1:0][7:0]  q8e,        // tsr_f32_to_q8(x, 1)
  output logic [LANES-1:0][31:0] i2f,        // tsr_i32_to_f32(x)
  output logic [30:0]            tree_amax,
  output logic [31:0]            tree_max,
  output logic [30:0]            chain_amax,
  output logic [31:0]            chain_max
);

  tsr_f32_amax_t [LANES-1:0] c;

  for (genvar i = 0; i < int'(LANES); i++) begin : g_lane
    always_comb begin
      daz[i]    = tsr_f32_daz(x[i]);
      is_nan[i] = tsr_f32_is_nan(x[i]);
      amag[i]   = tsr_f32_amag(x[i]);
      c[i]      = tsr_f32_amax_cand(x[i]);
      cand[i]   = c[i];
      h2f[i]    = tsr_fp16_to_f32(h[i]);
      f2h[i]    = tsr_f32_to_fp16(x[i]);
      q8a[i]    = tsr_f32_to_q8(x[i], 1'b0);
      q8e[i]    = tsr_f32_to_q8(x[i], 1'b1);
      i2f[i]    = tsr_i32_to_f32(x[i]);
    end
  end

  // Balanced tree, in place: round n merges v[2j], v[2j+1] -> v[j] for j < n (reads of
  // 2j, 2j+1 >= j happen before they are overwritten).
  always_comb begin
    tsr_f32_amax_t v [LANES];
    for (int i = 0; i < int'(LANES); i++) v[i] = c[i];
    for (int n = int'(LANES) / 2; n >= 1; n = n / 2) begin
      for (int j = 0; j < n; j++) v[j] = tsr_f32_amax_merge(v[2*j], v[2*j+1]);
    end
    tree_amax = v[0].mag;
    tree_max  = tsr_f32_amax_val(v[0]);
  end

  // Left-to-right fold starting from {0, +0} (the reference's initial amax / max).
  always_comb begin
    tsr_f32_amax_t acc;
    acc = '0;
    for (int i = 0; i < int'(LANES); i++) acc = tsr_f32_amax_merge(acc, c[i]);
    chain_amax = acc.mag;
    chain_max  = tsr_f32_amax_val(acc);
  end

endmodule
