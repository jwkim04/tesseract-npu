// tsr_gdn_dota.sv — AREA REFERENCE ONLY (docs/design/02-vector-unit.md §7.12.6): the dot-product engine
// of scheme (A), f32 adders in a fixed tree, for the same job as tsr_vu_gdn_dot + the converter: 8 lanes
// deliver one element of the row and one of the vector per cycle each.
//   lane l: p = FM(s_l, k_l)                    8 tsr_fp_mul
//           acc_l = FA(acc_l, p)                8 tsr_fp_add (the row accumulation of lane l)
//   row end: the 8 lane accumulators are added in the tree ((0+1)+(2+3))+((4+5)+(6+7)) on one shared
//           tsr_fp_add (7 additions over the row period of 16 cycles or more; 9 adders per engine)
// The accumulator loop of a real design needs 5 interleaved rows (the adder latency is 5) and a 7-step
// schedule for the tree; neither changes the instance count or the arithmetic cost, so they are not
// modeled. The engine is not functional; it is not part of any filelist or test. Two engines are needed
// (u' and w), so a design of scheme (A) has 16 multipliers and 18 adders in its dot products.
// Ports are registered (the units under test in tb/fp/viv/fp_area.tcl have registered ports).

module tsr_gdn_dota #(
  parameter int unsigned LANES = 8
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    in_valid,
  input  logic [LANES*32-1:0]     s,
  input  logic [LANES*32-1:0]     k,
  input  logic [2:0]              sel,
  output logic                    out_valid,
  output logic [31:0]             y
);

  logic                v_q;
  logic [LANES*32-1:0] s_q, k_q;
  logic [2:0]          sel_q;
  always_ff @(posedge clk) begin
    if (!rst_n) v_q <= 1'b0; else v_q <= in_valid;
  end
  always_ff @(posedge clk) begin
    s_q <= s; k_q <= k; sel_q <= sel;
  end

  logic [LANES-1:0]    pv, av;
  logic [LANES*32-1:0] p, acc;
  logic [LANES*32-1:0] acc_fb;
  for (genvar l = 0; l < LANES; l++) begin : g
    tsr_fp_mul #(.LAT(4)) u_m (.clk, .rst_n, .ce(1'b1), .in_valid(v_q), .a(s_q[32*l +: 32]), .b(k_q[32*l +: 32]),
                               .out_valid(pv[l]), .y(p[32*l +: 32]));
    tsr_fp_add #(.LAT(5)) u_a (.clk, .rst_n, .ce(1'b1), .in_valid(pv[l]), .a(acc_fb[32*l +: 32]), .b(p[32*l +: 32]),
                               .sub(1'b0), .out_valid(av[l]), .y(acc[32*l +: 32]));
    always_ff @(posedge clk) acc_fb[32*l +: 32] <= acc[32*l +: 32];
  end

  // the shared tree adder: operands chosen from the lane accumulators and its own result
  logic [31:0] ta, tb, ty;
  logic        tv;
  logic [31:0] tr_q;
  logic [2:0] ia, ib;
  assign ia = sel_q;
  assign ib = sel_q ^ 3'd1;
  always_comb begin
    ta = acc[32*int'(ia) +: 32];
    tb = acc[32*int'(ib) +: 32];
    if (sel_q == 3'd7) begin
      ta = tr_q;
      tb = acc[0 +: 32];
    end
  end
  tsr_fp_add #(.LAT(5)) u_t (.clk, .rst_n, .ce(1'b1), .in_valid(&av), .a(ta), .b(tb), .sub(1'b0), .out_valid(tv), .y(ty));
  always_ff @(posedge clk) begin
    tr_q <= ty;
    out_valid <= tv;
    y <= ty;
  end

endmodule
