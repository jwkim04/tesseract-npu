// tsr_gdn_tb.sv — testbench-only toplevel of the GDN engine bench (tb/gdn/test_gdn.py): tsr_vu_gdn with a model of
// what tsr_vu does around it. The bench plays the injector (it drives the slot fields and the data, one cycle per
// sub-slot, and honours cfg_ok / vec_ok / row_ok) and the writer's credit (oc_ok, oc_take); this wrapper plays the
// lanes: the update beat goes through FA(FM(a, c1), FM(b, c2)) built from tsr_fp_mul (LAT 4) and tsr_fp_add (LAT 5),
// the same primitives as the lane micro-op U_ROT, and leaves with its write descriptor LAT cycles later (y_*).
module tsr_gdn_tb
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
#(
  parameter  int unsigned LANES  = 8,
  parameter  int unsigned NSLOT  = 64,
  parameter  int unsigned NB_MAX = 32,
  parameter  int unsigned R_MAX  = 1024,
  parameter  int unsigned RD_LAT = 3,
  parameter  int unsigned FM_LAT = 4,
  parameter  int unsigned FA_LAT = 5,
  parameter                STYLE  = "ultra"
) (
  input  logic                      clk,
  input  logic                      rst_n,
  // injector side
  output logic                      cfg_ok,
  output logic                      vec_ok,
  output logic                      row_ok,
  input  logic                      g_v,
  input  logic [2:0]                g_gsub,
  input  logic [4:0]                g_tail,
  input  logic [23:0]               g_d0,
  input  logic [23:0]               g_d1,
  input  logic [23:0]               g_bline,
  input  logic [15:0]               g_rrow,
  input  logic                      g_tag,
  input  logic [511:0]              g_line,
  input  logic [31:0]               g_alpha,
  input  logic [31:0]               g_beta,
  // writer credit
  input  logic                      oc_ok,
  output logic [4:0]                oc_take,
  // the lanes' results with their descriptors
  output logic                      y_v,
  output logic [LANES*32-1:0]       y_data,
  output logic [23:0]               y_line,
  output logic                      y_part,
  output logic                      y_tag,
  // o lines
  output logic                      o_line_v,
  input  logic                      o_line_ready,
  output logic                      o_line_tag,
  output logic [23:0]               o_line_addr,
  output logic [511:0]              o_line_data,
  output logic                      act
);
  localparam int unsigned LAT = FM_LAT + FA_LAT;

  tsr_vu_slot_t g_slot;
  always_comb begin
    g_slot       = '0;
    g_slot.tag   = g_tag;
    g_slot.gsub  = tsr_vu_gs_e'(g_gsub);
    g_slot.tail  = g_tail;
    g_slot.d0    = g_d0;
    g_slot.d1    = g_d1;
    g_slot.bline = g_bline;
    g_slot.r_row = g_rrow;
  end

  logic                u_v, u_part, u_tag;
  logic [LANES*32-1:0] u_s, u_k;
  logic [31:0]         u_alpha, u_e;
  logic [23:0]         u_line;
  tsr_vu_gdn #(.LANES(LANES), .NSLOT(NSLOT), .NB_MAX(NB_MAX), .R_MAX(R_MAX), .RD_LAT(RD_LAT), .FM_LAT(FM_LAT),
               .FA_LAT(FA_LAT), .STYLE(STYLE)) u_gdn (
    .clk, .rst_n, .cfg_ok, .vec_ok, .row_ok,
    .g_v, .g_slot, .g_line, .g_alpha, .g_beta,
    .u_v, .u_s, .u_k, .u_alpha, .u_e, .u_line, .u_part, .u_tag,
    .oc_ok, .oc_take,
    .o_line_v, .o_line_ready, .o_line_tag, .o_line_addr, .o_line_data, .act
  );

  // the lanes: FA(FM(alpha, s), FM(e, k)) per word, and the descriptor delayed alike
  logic [LANES-1:0] mv1, mv2, av;
  for (genvar l = 0; l < LANES; l++) begin : g_lane
    logic [31:0] p1, p2;
    tsr_fp_mul #(.LAT(FM_LAT)) u_m1 (.clk, .rst_n, .ce(1'b1), .in_valid(u_v), .a(u_s[32*l +: 32]), .b(u_alpha),
                                     .out_valid(mv1[l]), .y(p1));
    tsr_fp_mul #(.LAT(FM_LAT)) u_m2 (.clk, .rst_n, .ce(1'b1), .in_valid(u_v), .a(u_k[32*l +: 32]), .b(u_e),
                                     .out_valid(mv2[l]), .y(p2));
    tsr_fp_add #(.LAT(FA_LAT)) u_a (.clk, .rst_n, .ce(1'b1), .in_valid(mv1[l]), .a(p1), .b(p2), .sub(1'b0),
                                    .out_valid(av[l]), .y(y_data[32*l +: 32]));
  end
  logic [LAT-1:0]     dv;
  logic [25:0]        dd [LAT];
  always_ff @(posedge clk) begin
    if (!rst_n) dv <= '0; else dv <= LAT'({dv, u_v});
    dd[0] <= {u_tag, u_part, u_line};
    for (int i = 1; i < LAT; i++) dd[i] <= dd[i-1];
  end
  assign y_v    = dv[LAT-1];
  assign {y_tag, y_part, y_line} = dd[LAT-1];

  wire unused_ok = &{1'b0, mv2, av, g_slot};
endmodule
