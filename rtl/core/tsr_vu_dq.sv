// tsr_vu_dq.sv — the DQROW front end of the VU: the per-block constants of one chunk of a repacked
// row, from the chunk's meta line, into a 32-entry table the injector reads.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §5.6 DQROW, §6.8; VU-15). Layout: docs/spec.md §5
// (tsr_repack_row, sw/sim/tsr_ref.c block_offsets): a chunk is one meta line and its quant lines,
// 1024 elements = 64 output lines in both formats (Q8_0: 32 blocks, 16 quant lines; Q4_K: 4
// superblocks, 8 quant lines).
// Contract: tsr_dequant_row bit for bit, with the lanes' existing micro-ops (no DQROW datapath there):
//   Q8_0   y = FM(d, q) as U_ADDMULC FM(FA(a, b), c1): a = 2^23 + (q + 128) (the bits 0x4B000000 |
//          (q ^ 0x80), built by the injector), b = -(2^23 + 128), so FA(a, b) = q exactly; c1 = d =
//          F32(fp16 d). Table entry b (block in the chunk): {c1 = d, -, -}.
//   Q4_K   y = FS(FM(d1, q), FM(dmin, m)) with d1 = FM(d, sc) (tsr_get_scale_min_k4 sc, m) as U_ROT with
//          neg: FS(FM(a, c1), FM(b, c2)), a = F32(q) (4-bit, the injector), c1 = d1, b = F32(m), c2 =
//          F32(dmin). Table entry 8·s + j (superblock s, sub-block j): {c1 = d1, b = F32(m), c2 = dmin}.
//   FM is commutative bit for bit here (q and m are never NaN, every NaN result is the canonical one),
//   so the operand order inside the lanes does not matter.
// Timing: start (the meta pseudo-slot's fire in the injector, with its SPM line) clears the table's
//   fill count; entries are computed in order, one per cycle, LAT cycles after start + their index,
//   and prod = the entries written so far. The injector holds a DQROW slot until prod > its entry. A
//   new start cancels the entries still in flight (the next chunk's meta comes only after every line
//   of the current chunk has been injected).

module tsr_vu_dq
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
(
  input  logic                     clk,
  input  logic                     rst_n,
  input  logic                     start,
  input  logic                     start_fmt,      // 1: Q4_K
  input  logic [TSR_LINE_BITS-1:0] start_meta,
  output logic [5:0]               prod,           // table entries ready (0..32)
  input  logic [4:0]               rd_e,
  output logic [31:0]              rd_c1,
  output logic [31:0]              rd_b,
  output logic [31:0]              rd_c2
);
  import tsr_fp_pkg::*;

  localparam int unsigned FL = TSR_VU_FMUL_LAT;   // d1 multiplier

  logic                     run_q, fmt_q;
  logic [4:0]               e_q;
  logic [TSR_LINE_BITS-1:0] meta_q;

  function automatic logic [7:0] mb(input logic [TSR_LINE_BITS-1:0] x, input int unsigned i);
    return x[8*i +: 8];
  endfunction

  // -------------------------------------------------------------------------------------------
  // S1: field extraction of entry e_q
  // -------------------------------------------------------------------------------------------
  logic        v1_q;
  logic [4:0]  e1_q;
  logic [15:0] d16_q, dm16_q;
  logic [5:0]  sc_q, m_q;
  always_ff @(posedge clk) begin
    int unsigned b, j;
    logic [7:0] q [12];
    b = 32'(e_q[4:3]);
    j = 32'(e_q[2:0]);
    for (int i = 0; i < 12; i++) q[i] = mb(meta_q, 16 * b + 4 + i);
    e1_q <= e_q;
    if (!fmt_q) begin
      d16_q  <= {mb(meta_q, 2 * 32'(e_q) + 1), mb(meta_q, 2 * 32'(e_q))};
      dm16_q <= '0;
      sc_q   <= '0;
      m_q    <= '0;
    end else begin
      d16_q  <= {mb(meta_q, 16 * b + 1), mb(meta_q, 16 * b)};
      dm16_q <= {mb(meta_q, 16 * b + 3), mb(meta_q, 16 * b + 2)};
      if (j < 4) begin                                         // tsr_get_scale_min_k4
        sc_q <= q[j][5:0];
        m_q  <= q[j + 4][5:0];
      end else begin
        sc_q <= {q[j - 4][7:6], q[j + 4][3:0]};
        m_q  <= {q[j][7:6], q[j + 4][7:4]};
      end
    end
  end

  // -------------------------------------------------------------------------------------------
  // S2: conversions
  // -------------------------------------------------------------------------------------------
  logic        v2_q;
  logic [4:0]  e2_q;
  logic [31:0] d32_q, dm32_q, sc32_q, m32_q;
  always_ff @(posedge clk) begin
    e2_q   <= e1_q;
    d32_q  <= tsr_fp16_to_f32(d16_q);
    dm32_q <= tsr_fp16_to_f32(dm16_q);
    sc32_q <= tsr_i32_to_f32(32'(sc_q));
    m32_q  <= tsr_i32_to_f32(32'(m_q));
  end

  // -------------------------------------------------------------------------------------------
  // FM: d1 = FM(d, sc) (Q4_K); the other values delayed alongside
  // -------------------------------------------------------------------------------------------
  logic [31:0] d1, dw, dmw, mw;
  logic [4:0]  ew;
  logic        fm_ov;
  tsr_fp_mul #(.LAT(FL)) u_fm (
    .clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(d32_q), .b(sc32_q), .out_valid(fm_ov), .y(d1)
  );
  tsr_vu_dly #(.W(32), .D(FL)) d_d  (.clk, .d(d32_q),  .q(dw));
  tsr_vu_dly #(.W(32), .D(FL)) d_dm (.clk, .d(dm32_q), .q(dmw));
  tsr_vu_dly #(.W(32), .D(FL)) d_m  (.clk, .d(m32_q),  .q(mw));
  tsr_vu_dly #(.W(5),  .D(FL)) d_e  (.clk, .d(e2_q),   .q(ew));

  // valid pipeline (cleared by start: entries of a cancelled chunk are not written)
  logic [FL-1:0] vf_q;
  always_ff @(posedge clk) begin
    if (!rst_n || start) begin
      v1_q <= 1'b0;
      v2_q <= 1'b0;
      vf_q <= '0;
    end else begin
      v1_q <= run_q;
      v2_q <= v1_q;
      vf_q <= {vf_q[FL-2:0], v2_q};
    end
  end
  wire wv = vf_q[FL-1];

  // -------------------------------------------------------------------------------------------
  // Sequencer and table
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      run_q <= 1'b0;
      e_q   <= '0;
      prod  <= '0;
    end else if (start) begin
      run_q <= 1'b1;
      e_q   <= '0;
      prod  <= '0;
    end else begin
      if (run_q) begin
        e_q <= e_q + 5'd1;
        if (e_q == 5'd31) run_q <= 1'b0;
      end
      if (wv) prod <= 6'(ew) + 6'd1;
    end
  end
  always_ff @(posedge clk) begin
    if (start) begin
      fmt_q  <= start_fmt;
      meta_q <= start_meta;
    end
  end

  logic [95:0] tab [32];
  always_ff @(posedge clk) begin
    if (wv) tab[ew] <= fmt_q ? {d1, mw, dmw} : {dw, 64'd0};
  end
  assign {rd_c1, rd_b, rd_c2} = tab[rd_e];

  wire unused_ok = &{1'b0, fm_ov};

endmodule
